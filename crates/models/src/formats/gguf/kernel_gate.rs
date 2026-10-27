//! **GGUF 内核族的 GPU 对拍门**(2026-10-19 刀 6b;设计 §四 L3 / REQ-GGUF-05)。
//!
//! 门 = `owl_gguf_quant_act_*` + `owl_gguf_gemm_*`(GPU)**对拍** CPU 参考路径
//! (刀 5 的解量化 + host 点积)。三条纪律写在判据里:
//!
//! 1. **参考侧复刻同一激活量化**(q8_1:每 32 元素块 `d = amax/127`、
//!    `q = round(x/d)`)—— MMVQ 的量化损失是**设计本身**,拿未量化的参考去比
//!    会把"算法固有的 8bit 损失"记成"内核错";复刻后剩下的差异只有
//!    f32 累加序 + f16 输出舍入 ⇒ 阈可以收得很紧(2% 已能抓住任何真错:
//!    错列/错块/错位移一律 O(1) 相对误差);
//! 2. **参考用 f64 累加** ⇒ 判据不掺 host 侧的累加噪声;
//! 3. **权重字节 = 文件原样**(本门只考内核算术,**不含**布局变换 —— 变换的
//!    门在刀 4c 的源侧,对拍 HF 快照)。
//!
//! 覆盖:Q4_K(4B `attn_gate`)/ Q6_K(4B `attn_qkv`)/ Q8_0(4B `ssm_out`)——
//! 三型三块几何(QK_K 超块 + QK8_0 块)都在门内。

#[cfg(test)]
mod tests {
    use crate::contract::Dtype;
    use crate::formats::gguf::dequant::dequant_all;
    use crate::formats::gguf::reader::Gguf;
    use crate::formats::gguf::GgmlTy;
    use crate::ops::SemanticKernel;
    use crate::testkit::harvest_f16;
    use crate::TensorOps;
    use owl_macros::owl_test;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;

    /// 32 元素块:q8_1 的**块字节数**(单一事实源 = `family::gguf::Q8_1_BLOCK_BYTES`;
    /// ⚠️ 不是 GGUF 张量型 `Q8_1` 的 40B —— 两者同名不同物,见该常量文档)
    const XQ_BLOCK: usize = crate::formats::gguf::Q8_1_BLOCK_BYTES;

    /// host 参考:q8_1 量化(与核同式独立副本) → 逐元素乘累加(f64)
    ///
    /// 返回 `(y[m, n], 量化损失的相对幅度)`(后者入账:证明"差异确实来自量化",
    /// 而不是内核错)。
    fn reference(x_f16: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> (Vec<f64>, f64) {
        assert_eq!(k % 32, 0, "k 必须是 32 的倍数");
        // ① 激活 → q8_1(块内 amax;与核内 warp 归约逐式一致)
        let mut xq = vec![0f32; m * k];
        for mi in 0..m {
            for b in 0..k / 32 {
                let blk = &x_f16[mi * k + b * 32..mi * k + b * 32 + 32];
                let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                for j in 0..32 {
                    let q = if amax == 0.0 { 0.0 } else { (blk[j] / d).round() };
                    xq[mi * k + b * 32 + j] = q * d;
                }
            }
        }
        let loss: f64 = {
            let num: f64 = x_f16
                .iter()
                .zip(&xq)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();
            let den: f64 = x_f16.iter().map(|a| (*a as f64).powi(2)).sum();
            (num / den.max(1e-30)).sqrt()
        };
        // ② y = xq · Wᵀ(f64 累加;W 行 = 权重外维 = 输出列)
        let mut y = vec![0f64; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += xq[mi * k + ki] as f64 * w[ni * k + ki] as f64;
                }
                y[mi * n + ni] = acc;
            }
        }
        (y, loss)
    }

    /// **GPU 门**:三型 × 三种块几何,逐元素对拍(含 f16 输出舍入余量)
    #[owl_test(gpu)]
    async fn gpu_gguf_gemm_matches_cpu_reference() {
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        // (张量, 期望型, m)
        let cases: &[(&str, &str, usize)] = &[
            ("blk.0.attn_gate.weight", "Q4_K", 3),
            ("blk.0.ssm_out.weight", "Q8_0", 4),
            // decode 形状 = m1(VDR=4 的 q8_0 MMVQ 必须单独过门;2026-10-21)
            ("blk.0.ssm_out.weight", "Q8_0", 1),
            ("blk.0.attn_qkv.weight", "Q6_K", 2),
            // 4B 的 **Q5_K 两张**(`attn_k` / `attn_output`)—— 2026-10-19 补:
            // 原先四型门漏了 Q5_K,而它正是 4B 与 27B 的**唯一型差**
            // (27B 无 Q5_K)⇒ 端到端一分歧就该先问"是不是这张没验过"
            ("blk.3.attn_k.weight", "Q5_K", 2),
            ("blk.3.attn_output.weight", "Q5_K", 3),
        ];
        let mut gpu = crate::testkit::gpu_client().await;
        for (tname, want_ty, m) in cases.iter().copied() {
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
            let ix = g.index_of(tname).unwrap();
            assert_eq!(ti.ty.name(), want_ty, "{tname} 型不符(用例表过期?)");
            let bytes = g.bytes(ix);
            let (rows, inner): (usize, usize) = (ti.rows() as usize, ti.inner() as usize);
            let (be, bb) = ti.ty.block();
            let row_bytes = inner / be * bb;
            assert_eq!(row_bytes as u64, ti.row_bytes());

            // 参考:CPU 解量化 + host 点积
            let w = dequant_all(ti.ty, bytes, rows, inner).expect("dequant");
            let k = inner;
            let xv: Vec<f32> = (0..m * k)
                .map(|i| {
                    let f = half::f16::from_f32(((i as f32 * 0.0173).sin() * 1.7).clamp(-2.5, 2.5));
                    f.to_f32()
                })
                .collect();
            let (yref, qloss) = reference(&xv, &w, m, rows, k);

            // 设备链:激活 → q8_1 → GEMM
            let xb: Vec<u8> = xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
            let x = TensorOps::from_host(Dtype::F16, vec![m, k], &xb);
            let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
            let xq_bytes = m * (k / 32) * XQ_BLOCK;
            let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                .arg(&x)
                .arg_i32(k as i32)
                .arg_i32(k as i32)
                .aux(&[m, k])
                .with_shape(Dtype::U8, vec![m, xq_bytes / m]);
            let y = TensorOps::call(SemanticKernel::GgufGemm)
                .arg(&wblk)
                .arg(&xq)
                .arg_i32(m as i32)
                .arg_i32(rows as i32)
                .arg_i32(k as i32)
                .arg_i32(k as i32)
                .arg_i32(0)
                .arg_i32(rows as i32)
                .aux(&[want_ty_code(want_ty), m, rows])
                .with_shape(Dtype::F16, vec![m, rows]);
            let got = harvest_f16(&mut gpu, &y).await;
            assert_eq!(got.len(), m * rows, "{tname} 输出元素数");

            let scale = yref.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-12);
            let mut maxd = 0f64;
            let mut worst = 0usize;
            for (i, (a, b)) in got.iter().zip(&yref).enumerate() {
                let d = (*a as f64 - *b).abs() / scale;
                if d > maxd {
                    maxd = d;
                    worst = i;
                }
            }
            eprintln!(
                "[gguf-gpu] {tname}({want_ty}, {rows}×{inner}, m={m}) max|Δ|/max|ref| = {maxd:.3e} \
                 (量化损失 {qloss:.3e};@ {worst}: gpu {} vs ref {:.6})",
                got[worst],
                yref[worst]
            );
            assert!(
                maxd < 0.02,
                "{tname}({want_ty}) 超阈:max|Δ|/max|ref| = {maxd:.4e} @ {worst}(错列/错块/错位移都会 O(1) 相对误差)"
            );
        }
    }

    /// **官方 MMQ ↔ MMVQ 对拍门**(W-2c;2026-10-19)。
    ///
    /// 两臂**同一份 q8_1 激活**(同 `GgufQuantAct`),权重同一份块字节 ⇒ 差异只来自
    /// 归约序(MMQ 先在 tile 里重排再做点积;MMVQ 是 warp 归约)⇒ 判据取
    /// `max|Δ|/max|ref| < 2e-2`(与现有 GEMM 门口径同一量级;错 tile/错网格会给 O(1))。
    /// 覆盖 `m = 32`(整 tile 倍数)与 `m = 71`(非整倍数,走 `need_check`)两档 × 四型。
    #[owl_test(gpu)]
    async fn gpu_gguf_mmq_matches_mmvq() {
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        let cases: &[(&str, &str)] = &[
            ("blk.0.attn_gate.weight", "Q4_K"),
            ("blk.0.ssm_out.weight", "Q8_0"),
            ("blk.0.attn_qkv.weight", "Q6_K"),
            ("blk.3.attn_k.weight", "Q5_K"),
        ];
        let mut gpu = crate::testkit::gpu_client().await;
        for (tname, want_ty) in cases.iter().copied() {
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
            let (be, bb) = ti.ty.block();
            let row_bytes = inner / be * bb;
            let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
            for m in [32usize, 71] {
                let xv: Vec<f32> = (0..m * inner)
                    .map(|i| {
                        half::f16::from_f32(((i as f32 * 0.0119).sin() * 1.4).clamp(-2.5, 2.5))
                            .to_f32()
                    })
                    .collect();
                let xb: Vec<u8> =
                    xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
                let x = TensorOps::from_host(Dtype::F16, vec![m, inner], &xb);
                let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                    .arg(&x)
                    .arg_i32(inner as i32)
                    .arg_i32(inner as i32)
                    .aux(&[m, inner])
                    .with_shape(Dtype::U8, vec![m, inner / 32 * XQ_BLOCK]);
                let a = harvest_f16(
                    &mut gpu,
                    &TensorOps::call(SemanticKernel::GgufGemm)
                        .arg(&wblk)
                        .arg(&xq)
                        .arg_i32(m as i32)
                        .arg_i32(rows as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(0)
                        .arg_i32(rows as i32)
                        .aux(&[want_ty_code(want_ty), m, rows])
                        .with_shape(Dtype::F16, vec![m, rows]),
                )
                .await;
                let b = harvest_f16(
                    &mut gpu,
                    &TensorOps::call(SemanticKernel::GgufMmq)
                        .arg(&wblk)
                        .arg(&xq)
                        .arg_i32(inner as i32)
                        .arg_i32(rows as i32)
                        .arg_i32(m as i32)
                        .aux(&[want_ty_code(want_ty), m, rows])
                        .with_shape(Dtype::F16, vec![m, rows]),
                )
                .await;
                let scale = a.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-12);
                let mut maxd = 0f32;
                let mut worst = 0usize;
                for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                    let d = (p - q).abs();
                    if d > maxd {
                        maxd = d;
                        worst = i;
                    }
                }
                let rel = maxd / scale;
                eprintln!(
                    "[gguf-gpu] {tname}({want_ty}, {rows}×{inner}, m={m}):MMQ vs MMVQ max|Δ|/max = {rel:.3e} @ {worst}"
                );
                assert!(
                    rel < 2e-2,
                    "{tname}({want_ty}) m={m}:MMQ 与 MMVQ 差 {rel:.4e}(错 tile/错网格会给 O(1))"
                );
            }
        }
    }

    /// **smem 复用臂 ↔ MMVQ 对拍门**(W-2b;2026-10-19)。
    ///
    /// 两臂**不位级相同**(本臂喂原始 f16 激活,MMVQ 先量化成 q8_1)⇒ 判据 =
    /// `max|Δ|/max|ref| < 2e-2`(q8_1 的量化噪声量级;错窗/错块/错分块会给 O(1))。
    /// 覆盖 `m = MT`(整窗)与 `m = MT + 37`(尾窗 + 非整块 token 数)两档 × 四型。
    #[owl_test(gpu)]
    async fn gpu_gguf_smem_gemm_matches_mmvq() {
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        const MT: usize = owl_kernels::family::gguf::SMEM_MT;
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        let cases: &[(&str, &str)] = &[
            ("blk.0.attn_gate.weight", "Q4_K"),
            ("blk.0.ssm_out.weight", "Q8_0"),
            ("blk.0.attn_qkv.weight", "Q6_K"),
            ("blk.3.attn_k.weight", "Q5_K"),
        ];
        let mut gpu = crate::testkit::gpu_client().await;
        for (tname, want_ty) in cases.iter().copied() {
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
            let (be, bb) = ti.ty.block();
            let row_bytes = inner / be * bb;
            let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
            for m in [MT, MT + 37] {
                let xv: Vec<f32> = (0..m * inner)
                    .map(|i| {
                        half::f16::from_f32(((i as f32 * 0.0137).cos() * 1.3).clamp(-2.5, 2.5))
                            .to_f32()
                    })
                    .collect();
                let xb: Vec<u8> =
                    xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
                let x = TensorOps::from_host(Dtype::F16, vec![m, inner], &xb);
                let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                    .arg(&x)
                    .arg_i32(inner as i32)
                    .arg_i32(inner as i32)
                    .aux(&[m, inner])
                    .with_shape(Dtype::U8, vec![m, inner / 32 * XQ_BLOCK]);
                let a = harvest_f16(
                    &mut gpu,
                    &TensorOps::call(SemanticKernel::GgufGemm)
                        .arg(&wblk)
                        .arg(&xq)
                        .arg_i32(m as i32)
                        .arg_i32(rows as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(0)
                        .arg_i32(rows as i32)
                        .aux(&[want_ty_code(want_ty), m, rows])
                        .with_shape(Dtype::F16, vec![m, rows]),
                )
                .await;
                let b = harvest_f16(
                    &mut gpu,
                    &TensorOps::call(SemanticKernel::GgufGemmSmem)
                        .arg(&wblk)
                        .arg(&x)
                        .arg_i32(m as i32)
                        .arg_i32(rows as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(0)
                        .arg_i32(rows as i32)
                        .aux(&[want_ty_code(want_ty), m, rows])
                        .with_shape(Dtype::F16, vec![m, rows]),
                )
                .await;
                let scale = a.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-12);
                let mut maxd = 0f32;
                let mut worst = 0usize;
                for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                    let d = (p - q).abs();
                    if d > maxd {
                        maxd = d;
                        worst = i;
                    }
                }
                let rel = maxd / scale;
                eprintln!(
                    "[gguf-gpu] {tname}({want_ty}, {rows}×{inner}, m={m}):smem vs MMVQ max|Δ|/max = {rel:.3e} @ {worst}"
                );
                assert!(
                    rel < 2e-2,
                    "{tname}({want_ty}) m={m}:smem 臂与 MMVQ 差 {rel:.4e}(超出 q8_1 量化噪声量级 ⇒ 错窗/错分块)"
                );
            }
        }
    }

    /// **复用臂 ↔ MMVQ 位级门**(W-2;2026-10-19):同一权重/激活/形状下,两臂
    /// 的输出必须**逐位相同**。
    ///
    /// 为什么可以要求"位级"而不是 allclose:复用臂只改了**调度**(一行权重算
    /// `TILE_MT` 个 token),每个 token 的 `vec_dot` 序与 warp reduce **一字未动**
    /// ⇒ 数值路径完全相同。位级门一旦红,就是几何/越界/串行化写坏了 —— 这类错
    /// 在 allclose 下可能被"平均值掩盖"。
    ///
    /// 覆盖:`m = TILE_MT`(整窗)与 `m = TILE_MT + 4`(尾窗)两档 × 四型。
    #[owl_test(gpu)]
    async fn gpu_gguf_tile_gemm_matches_mmvq_bitwise() {
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        const MT: usize = owl_kernels::family::gguf::TILE_MT;
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        let cases: &[(&str, &str)] = &[
            ("blk.0.attn_gate.weight", "Q4_K"),
            ("blk.0.ssm_out.weight", "Q8_0"),
            ("blk.0.attn_qkv.weight", "Q6_K"),
            ("blk.3.attn_k.weight", "Q5_K"),
        ];
        let mut gpu = crate::testkit::gpu_client().await;
        for (tname, want_ty) in cases.iter().copied() {
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
            let (be, bb) = ti.ty.block();
            let row_bytes = inner / be * bb;
            let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
            for m in [MT, MT + 4] {
                let xv: Vec<f32> = (0..m * inner)
                    .map(|i| half::f16::from_f32(((i as f32 * 0.0137).cos() * 1.3).clamp(-2.5, 2.5)).to_f32())
                    .collect();
                let xb: Vec<u8> =
                    xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
                let x = TensorOps::from_host(Dtype::F16, vec![m, inner], &xb);
                let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                    .arg(&x)
                    .arg_i32(inner as i32)
                    .arg_i32(inner as i32)
                    .aux(&[m, inner])
                    .with_shape(Dtype::U8, vec![m, inner / 32 * XQ_BLOCK]);
                let build = |op: SemanticKernel| {
                    TensorOps::call(op)
                        .arg(&wblk)
                        .arg(&xq)
                        .arg_i32(m as i32)
                        .arg_i32(rows as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(inner as i32)
                        .arg_i32(0)
                        .arg_i32(rows as i32)
                        .aux(&[want_ty_code(want_ty), m, rows])
                        .with_shape(Dtype::F16, vec![m, rows])
                };
                let a = harvest_f16(&mut gpu, &build(SemanticKernel::GgufGemm)).await;
                let b = harvest_f16(&mut gpu, &build(SemanticKernel::GgufGemmTile)).await;
                assert_eq!(a.len(), b.len(), "{tname} m={m} 输出长度");
                let mut nbit = 0usize;
                let mut worst = 0usize;
                for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                    if p.to_bits() != q.to_bits() {
                        nbit += 1;
                        if nbit == 1 {
                            worst = i;
                        }
                    }
                }
                assert_eq!(
                    nbit, 0,
                    "{tname}({want_ty}) m={m}:复用臂与 MMVQ 有 {nbit} 项不逐位相同(首个 @ {worst}:{} vs {})",
                    a[worst], b[worst]
                );
                eprintln!(
                    "[gguf-gpu] {tname}({want_ty}, {rows}×{inner}, m={m}):复用臂 ≡ MMVQ 逐位相同 ✓"
                );
            }
        }
    }

    /// **量化表行解量化门**(查表臂;`token_embd` 在 GGUF 里是量化块)。
    ///
    /// 覆盖两个靶位的真表:4B = Q6_K(`tie_word_embeddings`)· 27B = Q4_K。
    /// 判据 = 与 `dequant_row`(刀 5 的 CPU 参考,已与 `gguf-py` 位级对拍)逐元素
    /// 相等;容差取本核特有的误差源:**数值在 `half` 域里算**(`gguf.cuh` 的
    /// 块解量化即如此)⇒ 记录为 f16 舍入上界的几倍,远离“错行/错块”的 O(1)。
    #[owl_test(gpu)]
    async fn gpu_gguf_dequant_rows_matches_reference() {
        use crate::formats::gguf::dequant::dequant_row;
        let mut gpu = crate::testkit::gpu_client().await;
        let mut checked = 0usize;
        for (res_key, want_ty) in [("gguf:4b", "Q6_K"), ("gguf:27b", "Q4_K")] {
            if !resource_ok(res_key) {
                owl_shared::testkit::report::skip("资源清单缺 \"{res_key}\"");
                continue;
            }
            let path = PathBuf::from(resource(res_key).expect("resource_ok 已过"));
            let g = Gguf::open(&path).expect("open");
            let tname = "token_embd.weight";
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("[{res_key}] 缺 {tname}"));
            assert_eq!(ti.ty.name(), want_ty, "[{res_key}] {tname} 型不符(用例表过期?)");
            let bytes = g.bytes(g.index_of(tname).unwrap());
            let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
            let (be, bb) = ti.ty.block();
            let row_bytes = inner / be * bb;
            assert_eq!(row_bytes as u64, ti.row_bytes());

            // 三个 id(首/中/末;末行专抓“块内最后一行的越界”)
            let ids: [usize; 3] = [0, 7, rows - 1];
            let ids_b: Vec<u8> = ids.iter().flat_map(|i| (*i as f32).to_le_bytes()).collect();
            let tbl = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
            let idt = TensorOps::from_host(Dtype::F32, vec![ids.len()], &ids_b);
            let out = TensorOps::call(SemanticKernel::GgufDequantRows)
                .arg(&tbl)
                .arg(&idt)
                .arg_i32(inner as i32)
                .arg_i32(ids.len() as i32)
                .arg_i32(rows as i32)
                .aux(&[want_ty_code(want_ty), ids.len(), inner])
                .with_shape(Dtype::F16, vec![ids.len(), inner]);
            let got = harvest_f16(&mut gpu, &out).await;
            assert_eq!(got.len(), ids.len() * inner, "[{res_key}] 输出元素数");

            for (r, id) in ids.iter().copied().enumerate() {
                let mut want = vec![0f32; inner];
                dequant_row(ti.ty, &bytes[id * row_bytes..(id + 1) * row_bytes], &mut want)
                    .unwrap_or_else(|e| panic!("[{res_key}] dequant_row: {e}"));
                let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6) as f64;
                let mut maxd = 0f64;
                let mut worst = 0usize;
                for i in 0..inner {
                    let d = (got[r * inner + i] as f64 - want[i] as f64).abs() / scale;
                    if d > maxd {
                        maxd = d;
                        worst = i;
                    }
                }
                eprintln!(
                    "[gguf-rows] {res_key} {tname}({want_ty}, {rows}×{inner}) id={id}: \
                     max|Δ|/max|ref| = {maxd:.3e} @ {worst}"
                );
                assert!(
                    maxd < 5e-3,
                    "[{res_key}] id={id} 超阈:{maxd:.3e} @ {worst}(错行/错块会给 O(1))"
                );
                checked += 1;
            }
        }
        assert!(checked >= 3, "对拍行数太少({checked})");
    }

    /// U8 缓冲原样回读(位级判据用;不做任何元素解码)
    async fn dtoh_raw(
        face: &mut owl_cuda::GpuClient,
        t: &TensorOps,
        n: usize,
        tag: &str,
    ) -> Vec<u8> {
        use owl_iface::contract::DeviceClient as _;
        let bytes = crate::interpreters::eval_ops(t.step(), face).await.expect("eval");
        eprintln!("  [{tag}] 输出块 id={} (声明 {n}B)", bytes.id);
        let mut buf = vec![0u8; n];
        face.dtoh(&bytes, &mut buf).await.expect("dtoh");
        buf
    }

    /// **A2 门**:`ssm_out` 的**激活侧列置换**(设计 §11.4 落点 A2)。
    ///
    /// 背景:GGUF 转换器把 `ssm_out` 的**内维**(v-head 轴)写成 ggml **tiled** 序
    /// (`conversion/qwen.py::_reorder_v_heads`),而 owl 的 GDN 核产出 **grouped** 序。
    /// 该置换跨量化块(`hv = 128` < `QK_K = 256`)⇒ 权重字节不可逆 ⇒ 在激活侧
    /// 等价置换,且**折进激活量化核**(零额外发射 —— `owl_gguf_quant_act_*_tiled`)。
    ///
    /// 两条判据:
    /// ① **位级**:`tiled(x, tbl)` ≡ `plain(host 按同表置换后的 x)` —— 逐字节相同
    ///    (置换单元 = 32 元素块,块内容不变 ⇒ `d`/`sum`/`q` 全不变);
    /// ② **数值**:tiled 路 + **文件权重字节** ≡ CPU 参考(同文件权重**解量化 +
    ///    内维反重排**成 grouped 序后,host f64 点积)—— 这条闭合整链:
    ///    文件(tiled 序)× 激活(tiled 序) ≡ HF 语义(grouped × grouped)。
    #[owl_test(gpu)]
    async fn gpu_gguf_ssm_out_tiled_act_matches_grouped_reference() {
        use crate::formats::gguf::transform;
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        let tname = "blk.0.ssm_out.weight";
        let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
        assert_eq!(ti.ty.name(), "Q8_0", "{tname} 型不符(用例表过期?)");
        let bytes = g.bytes(g.index_of(tname).unwrap());
        let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
        let (be, bb) = ti.ty.block();
        let row_bytes = inner / be * bb;

        // 维度事实(与 `spec_from_gguf` 同源口径;此处硬写 4B 实数并自证)
        let (nk, nv, hv) = (16usize, 32usize, 128usize);
        assert_eq!(inner, nv * hv, "内维 {inner} ≠ nv·hv = {nv}·{hv}(用例表过期?)");
        let tbl_v = transform::vhead_q8_blocks(nk, nv, hv).expect("列序表");
        assert_eq!(tbl_v.len(), inner / 32, "表长 ≠ 内维块数");
        let tbl_b: Vec<u8> = tbl_v.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tbl = TensorOps::from_host(Dtype::U32, vec![tbl_v.len()], &tbl_b);

        // 激活(确定性生成器;f16 域即设备域)
        let m = 3usize;
        let k = inner;
        let xv: Vec<f32> = (0..m * k)
            .map(|i| {
                let f = half::f16::from_f32(((i as f32 * 0.0173).sin() * 1.7).clamp(-2.5, 2.5));
                f.to_f32()
            })
            .collect();
        let xb: Vec<u8> = xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let x = TensorOps::from_host(Dtype::F16, vec![m, k], &xb);

        // host 侧同表置换(逐 32 元素块;参考臂的输入)
        let mut xp: Vec<f32> = vec![0f32; xv.len()];
        for r in 0..m {
            for (j, s) in tbl_v.iter().enumerate() {
                let d = r * k + j * 32;
                let so = r * k + *s as usize * 32;
                xp[d..d + 32].copy_from_slice(&xv[so..so + 32]);
            }
        }
        let xpb: Vec<u8> = xp.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let xp_t = TensorOps::from_host(Dtype::F16, vec![m, k], &xpb);

        let xq_row = k / 32 * XQ_BLOCK;
        let xq_tiled = TensorOps::call(SemanticKernel::GgufQuantActTiledF16)
            .arg(&x)
            .arg(&tbl)
            .arg_i32(k as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(Dtype::U8, vec![m, xq_row]);
        let xq_plain = TensorOps::call(SemanticKernel::GgufQuantActF16)
            .arg(&xp_t)
            .arg_i32(k as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(Dtype::U8, vec![m, xq_row]);

        let mut gpu = crate::testkit::gpu_client().await;
        // ① 位级:两条路的 q8_1 缓冲**逐字节**相同
        let a = dtoh_raw(&mut gpu, &xq_tiled, m * xq_row, "tiled").await;
        let b = dtoh_raw(&mut gpu, &xq_plain, m * xq_row, "plain").await;
        assert_eq!(a.len(), b.len());
        let diff = a.iter().zip(&b).filter(|(u, v)| u != v).count();
        eprintln!("  [cmp] a vs b 分歧字节 = {diff}");
        // 归因面:失败时列出分歧落在哪些块/哪一段(d/sum/qs)
        let cmp = |got: &[u8], want: &[u8], tag: &str| -> usize {
            let n = got.iter().zip(want).filter(|(g, w)| g != w).count();
            eprintln!("  [{tag}] vs host 期望:分歧 {n} 字节");
            n
        };
        let host = |xv: &[f32], map: Option<&[u32]>| -> Vec<u8> {
            let mut out = Vec::with_capacity(m * xq_row);
            for r in 0..m {
                for j in 0..k / 32 {
                    let sb = map.map(|t| t[j] as usize).unwrap_or(j);
                    let blk = &xv[r * k + sb * 32..r * k + sb * 32 + 32];
                    let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let d = amax / 127.0;
                    let sum: f32 = blk.iter().sum();
                    out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    out.extend_from_slice(&half::f16::from_f32(sum).to_le_bytes());
                    for v in blk {
                        let q = if amax == 0.0 { 0i8 } else { (v / d).round() as i8 };
                        out.push(q as u8);
                    }
                }
            }
            out
        };
        let eh_t = host(&xv, Some(&tbl_v));
        let eh_p = host(&xv, None);
        if diff > 0 {
            // 判据失败时先定位“哪一侧错”:a 应与 tiled 期望等、b 应与 plain 期望等
            let (ra_t, rb_p) = (cmp(&a, &eh_t, "a vs tiled 期望"), cmp(&b, &eh_p, "b vs plain 期望"));
            let (ra_p, rb_t) = (cmp(&a, &eh_p, "a vs plain 期望"), cmp(&b, &eh_t, "b vs tiled 期望"));
            panic!(
                "tiled 量化 ≠ 先置换再量化({diff} 字节不同);\n  \
                 a: tiled 期望残 {ra_t} / plain 期望残 {ra_p};b: plain 期望残 {rb_p} / tiled 期望残 {rb_t}\n  \
                 (若 a 只与 tiled 期望相符而 b 只与 plain 期望相符 ⇒ 两侧各自算对了但“表方向”或“块粒度”错)"
            );
        }
        // 双保险:a 必须与 tiled 期望**完全相等**,且与 plain 期望**必须不等**
        // (后者证明本测真的测到了置换 —— 否则表若退化成恒等,① 会假绿)
        assert_eq!(cmp(&a, &eh_t, "a vs tiled 期望"), 0, "tiled 输出与 host 期望不符");
        assert_ne!(cmp(&a, &eh_p, "a vs plain 期望"), 0, "置换没生效?tiled 输出竟等于自然序期望");
        eprintln!("[gguf-tiled] ① 位级绿:m={m} k={k} 的 q8_1 缓冲 {}B 逐字节相同(且 ≠ 自然序)", a.len());
        // ② 数值:tiled + 文件字节 vs grouped 参考(解量化 + 内维反重排)
        let mut w = dequant_all(ti.ty, bytes, rows, inner).expect("dequant");
        transform::col_deperm_in_place(&mut w, rows, inner, nk, nv, hv).expect("列反重排");
        let (yref, qloss) = reference(&xv, &w, m, rows, k);
        let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
        let y = TensorOps::call(SemanticKernel::GgufGemm)
            .arg(&wblk)
            .arg(&xq_tiled)
            .arg_i32(m as i32)
            .arg_i32(rows as i32)
            .arg_i32(k as i32)
            .arg_i32(k as i32)
            .arg_i32(0)
            .arg_i32(rows as i32)
            .aux(&[want_ty_code("Q8_0"), m, rows])
            .with_shape(Dtype::F16, vec![m, rows]);
        let got = harvest_f16(&mut gpu, &y).await;
        assert_eq!(got.len(), m * rows);
        let scale = yref.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-12);
        let mut maxd = 0f64;
        let mut worst = 0usize;
        for (i, (a, b)) in got.iter().zip(&yref).enumerate() {
            let d = (*a as f64 - *b).abs() / scale;
            if d > maxd {
                maxd = d;
                worst = i;
            }
        }
        eprintln!(
            "[gguf-tiled] ② 数值绿:{tname}(Q8_0, {rows}×{inner}, m={m}) max|Δ|/max|ref| = {maxd:.3e} \
             (量化损失 {qloss:.3e};@ {worst}: gpu {} vs ref {:.6})",
            got[worst], yref[worst]
        );
        assert!(
            maxd < 0.02,
            "tiled 路与 grouped 参考超阈:{maxd:.4e} @ {worst}(错置换会给 O(1);方向反了同样 O(1))"
        );
    }

    /// **`ssm_out` 列序 gather 门**(W-2e;2026-10-19)。
    ///
    /// 背景:这条置换此前**只**折进激活量化核(`GgufQuantActTiled*`)⇒ cuBLAS/MMQ
    /// 两臂对 `ssm_out` 不可用(直吃 f16/q8_1,不做块置换)。W-2e 把置换抽成
    /// 独立一发 `GgufGatherCols`(f16 块级 gather)⇒ 之后与普通层同路。
    ///
    /// 钉三件事(任一错都会静默错值):
    /// ① gather 的字节 == host 同表置换(逐字节);
    /// ② `gather + 普通量化` 与 `tiled 量化` 的 q8_1 缓冲**逐字节相同**(块内容不变);
    /// ③ 端到端 GEMM 两路**位级相同**(同一 MMVQ 归约序)。
    #[owl_test(gpu)]
    async fn gpu_gguf_ssm_out_gather_matches_tiled_act() {
        use crate::formats::gguf::transform;
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        let tname = "blk.0.ssm_out.weight";
        let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
        assert_eq!(ti.ty.name(), "Q8_0", "{tname} 型不符(用例表过期?)");
        let bytes = g.bytes(g.index_of(tname).unwrap());
        let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
        let (be, bb) = ti.ty.block();
        let row_bytes = inner / be * bb;

        let (nk, nv, hv) = (16usize, 32usize, 128usize);
        assert_eq!(inner, nv * hv, "内维 {inner} ≠ nv·hv(用例表过期?)");
        let tbl_v = transform::vhead_q8_blocks(nk, nv, hv).expect("列序表");
        assert_eq!(tbl_v.len(), inner / 32, "表长 ≠ 内维块数");
        let tbl_b: Vec<u8> = tbl_v.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tbl = TensorOps::from_host(Dtype::U32, vec![tbl_v.len()], &tbl_b);

        // m 取 64:同时覆盖两档(≥64 走 cuBLAS,<32 走 MMVQ)之外的最小公共面
        let m = 64usize;
        let k = inner;
        let xv: Vec<f32> = (0..m * k)
            .map(|i| {
                let f = half::f16::from_f32(((i as f32 * 0.0173).sin() * 1.7).clamp(-2.5, 2.5));
                f.to_f32()
            })
            .collect();
        let xb: Vec<u8> = xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let x = TensorOps::from_host(Dtype::F16, vec![m, k], &xb);

        // host 参考置换(与 tiled 量化门同一公式)
        let mut xp: Vec<f32> = vec![0f32; xv.len()];
        for r in 0..m {
            for (j, s) in tbl_v.iter().enumerate() {
                let d = r * k + j * 32;
                let so = r * k + *s as usize * 32;
                xp[d..d + 32].copy_from_slice(&xv[so..so + 32]);
            }
        }
        let xpb: Vec<u8> = xp.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();

        let xg = TensorOps::call(SemanticKernel::GgufGatherCols)
            .arg(&x)
            .arg(&tbl)
            .arg_i32(m as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(Dtype::F16, vec![m, k]);

        let mut gpu = crate::testkit::gpu_client().await;
        // ① gather 字节 == host 置换
        let gb = dtoh_raw(&mut gpu, &xg, m * k * 2, "gather").await;
        let diff = gb.iter().zip(&xpb).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "`GgufGatherCols` 与 host 置换分歧 {diff} 字节(块粒度/表方向错)");
        eprintln!("[gguf-gather] ① 位级绿:{tname}({rows}×{inner}, m={m}) gather == host 置换");

        // ② 量化缓冲:gather + plain  vs  tiled
        let xq_row = k / 32 * XQ_BLOCK;
        let xq_tiled = TensorOps::call(SemanticKernel::GgufQuantActTiledF16)
            .arg(&x)
            .arg(&tbl)
            .arg_i32(k as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(Dtype::U8, vec![m, xq_row]);
        let xq_plain = TensorOps::call(SemanticKernel::GgufQuantActF16)
            .arg(&xg)
            .arg_i32(k as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(Dtype::U8, vec![m, xq_row]);
        let a = dtoh_raw(&mut gpu, &xq_tiled, m * xq_row, "tiled").await;
        let b = dtoh_raw(&mut gpu, &xq_plain, m * xq_row, "gather+plain").await;
        let diff = a.iter().zip(&b).filter(|(u, v)| u != v).count();
        assert_eq!(diff, 0, "gather+plain 与 tiled 的 q8_1 缓冲分歧 {diff} 字节");
        eprintln!("[gguf-gather] ② 位级绿:q8_1 缓冲 {}B 逐字节相同", a.len());

        // ③ 端到端 GEMM 两路位级相同(MMVQ 归约序不变)
        let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
        let mk = |xq: &TensorOps| -> TensorOps {
            TensorOps::call(SemanticKernel::GgufGemm)
                .arg(&wblk)
                .arg(xq)
                .arg_i32(m as i32)
                .arg_i32(rows as i32)
                .arg_i32(k as i32)
                .arg_i32(k as i32)
                .arg_i32(0)
                .arg_i32(rows as i32)
                .aux(&[want_ty_code("Q8_0"), m, rows])
                .with_shape(Dtype::F16, vec![m, rows])
        };
        let y1 = mk(&xq_tiled);
        let y2 = mk(&xq_plain);
        let g1 = harvest_f16(&mut gpu, &y1).await;
        let g2 = harvest_f16(&mut gpu, &y2).await;
        let d = g1.iter().zip(&g2).filter(|(a, b)| a != b).count();
        assert_eq!(d, 0, "两路 GEMM 输出分歧 {d} 项(gather 路径改变了归约序?)");
        eprintln!("[gguf-gather] ③ 位级绿:GEMM 输出 {m}×{rows} 全同 ⇒ 分派可无缝切路");
    }

    /// **head 版 int8-MMA MMQ 门**(W-2f;2026-10-19)。
    ///
    /// 三判据:
    /// ① **激活量化逐字节**:`owl_mmq2_quant_*` vs host 参考([k/128][token] × 144B,
    ///    ds4/d4 按型)—— 布局一旦错位 ⇒ 数值门会给垃圾,但这条先钉"错在哪"。
    /// ② **数值**:MMQ2 臂 vs **cuBLAS 臂**(f16 输入 × f32 累加,参考更准)
    ///    4B+27B 真实张量 × m=128/512,判据 2e-2(§19.5 同口径)。
    /// ③ 「q 窗 J」覆盖:`m=128` 走 J=128,`m=64` 走 J=64 两档都跑。
    #[owl_test(gpu)]
    async fn gpu_gguf_mmq2_matches_cublas() {
        use owl_kernels::family::gguf as famg;
        if !resource_ok("gguf:4b") {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        }
        let mut gpu = crate::testkit::gpu_client().await;
        for (res_key, cases) in [
            ("gguf:4b", &[
                ("blk.0.attn_gate.weight", "Q4_K"),
                ("blk.0.ssm_out.weight", "Q8_0"),
                ("blk.0.attn_qkv.weight", "Q6_K"),
                ("blk.3.attn_k.weight", "Q5_K"),
            ][..]),
            ("gguf:27b", &[
                ("blk.0.ffn_down.weight", "Q4_K"),
                ("blk.0.attn_gate.weight", "Q8_0"),
            ][..]),
            // IQ 族 MMQ2(2026-10-20;REQ-GGUF-04):int8-MMA 臂的 IQ 分支
            ("gguf:0.8b-iq4xs", &[("blk.0.attn_gate.weight", "IQ4_XS")][..]),
            ("gguf:27b-iq3s", &[
                ("blk.4.attn_gate.weight", "IQ3_S"),
                ("blk.1.attn_gate.weight", "IQ3_XXS"),
            ][..]),
        ] {
            if !resource_ok(res_key) {
                owl_shared::testkit::report::skip("资源清单缺 \"{res_key}\"");
                continue;
            }
            let g = Gguf::open(&PathBuf::from(resource(res_key).expect("列了就有"))).expect("open");
            for (tname, want_ty) in cases.iter().copied() {
                let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
                let bytes = g.bytes(g.index_of(tname).unwrap());
                let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
                let (be, bb) = ti.ty.block();
                let row_bytes = inner / be * bb;
                assert!(inner % famg::MMQ2_QK == 0 && rows % famg::MMQ2_I == 0, "{tname} 形状不进本门");
                let ty = famg::ALL
                    .iter()
                    .copied()
                    .find(|t| t.name() == want_ty)
                    .unwrap_or_else(|| panic!("未知型 {want_ty}"));
                let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
                for m in [64usize, 128, 512] {
                    let xv: Vec<f32> = (0..m * inner)
                        .map(|i| {
                            half::f16::from_f32(((i as f32 * 0.0097).cos() * 1.5).clamp(-2.5, 2.5))
                                .to_f32()
                        })
                        .collect();
                    let xb: Vec<u8> =
                        xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
                    let x = TensorOps::from_host(Dtype::F16, vec![m, inner], &xb);

                    // ---- ① 激活量化 vs host 参考(逐字节) ----
                    let ds4 = famg::mmq2_quant_entry_for(ty) == Some(famg::MMQ2_QUANT_DS4);
                    let qk = famg::MMQ2_QUANT_DS4;
                    let quant = if ds4 {
                        TensorOps::call(SemanticKernel::GgufMmq2QuantDs4)
                    } else {
                        TensorOps::call(SemanticKernel::GgufMmq2QuantD4)
                    };
                    let m_pad = famg::mmq2_xq_tokens(m);
                    let xq = quant
                        .arg(&x)
                        .arg_i32(m_pad as i32)
                        .arg_i32(m as i32)
                        .arg_i32(inner as i32)
                        .aux(&[m, inner])
                        .with_shape(
                            Dtype::U8,
                            vec![inner / famg::MMQ2_QK, m_pad * famg::MMQ2_Q8_1_BLOCK_BYTES],
                        );
                    let got = dtoh_raw(&mut gpu, &xq, famg::mmq2_xq_bytes(m, inner), "mmq2-xq").await;
                    let mut want = vec![0u8; got.len()];
                    let nkb = inner / famg::MMQ2_QK;
                    for kb in 0..nkb {
                        for row in 0..m {
                            let base = (kb * m + row) * famg::MMQ2_Q8_1_BLOCK_BYTES;
                            for g4 in 0..4 {
                                let off = kb * famg::MMQ2_QK + g4 * 32;
                                let blk = &xv[row * inner + off..row * inner + off + 32];
                                let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
                                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                                let sum: f32 = blk.iter().sum();
                                if ds4 {
                                    want[base + g4 * 4..base + g4 * 4 + 2]
                                        .copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                                    want[base + g4 * 4 + 2..base + g4 * 4 + 4]
                                        .copy_from_slice(&half::f16::from_f32(sum).to_le_bytes());
                                } else {
                                    want[base + g4 * 4..base + g4 * 4 + 4]
                                        .copy_from_slice(&d.to_le_bytes());
                                }
                                for (j, v) in blk.iter().enumerate() {
                                    want[base + 16 + g4 * 32 + j] = (v * inv).round() as i8 as u8;
                                }
                            }
                        }
                    }
                    let ndiff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        ndiff, 0,
                        "{tname}({want_ty}, m={m}): q8_1_mmq 与 host 参考分歧 {ndiff} 字节(布局/公式错)"
                    );
                    let _ = qk;
                    if std::env::var("OWL_MMQ2_Q_ONLY").is_ok() {
                        eprintln!("[gguf-mmq2] ① 位级绿({tname}, {want_ty}, m={m})");
                        continue;
                    }

                    // ---- ② 数值:MMQ2 vs cuBLAS 臂 ----
                    let y2 = TensorOps::call(SemanticKernel::GgufMmq2)
                        .arg(&wblk)
                        .arg(&xq)
                        .arg_i32(rows as i32)
                        .arg_i32(m as i32)
                        .arg_i32(inner as i32)
                        .aux(&[ty.code() as usize, m, rows])
                        .with_shape(Dtype::F16, vec![m, rows]);
                    let got2 = harvest_f16(&mut gpu, &y2).await;
                    // 参考 = 先解量化全窗再 cuBLAS(f32 累加)
                    let s = TensorOps::uninit(Dtype::F16, vec![rows, inner]);
                    let dq = TensorOps::call_dst(&s, SemanticKernel::GgufCvtRange)
                        .arg(&wblk)
                        .arg_i32(0)
                        .arg_i32(rows as i32)
                        .arg_i32(inner as i32)
                        .aux(&[ty.code() as usize, rows, inner])
                        .with_shape(Dtype::F16, vec![rows, inner]);
                    let yref = TensorOps::call(SemanticKernel::GgufCublasDst)
                        .arg(&x)
                        .arg(&dq)
                        .arg_usize(rows)
                        .arg_usize(inner)
                        .arg_usize(m)
                        .arg_usize(1)
                        .arg_usize(rows)
                        .arg_usize(0)
                        .with_shape(Dtype::F16, vec![m, rows]);
                    let r = harvest_f16(&mut gpu, &yref).await;
                    let scale = r.iter().fold(0f64, |a, v| a.max((*v as f64).abs())).max(1e-12);
                    let mut maxd = 0f64;
                    let mut worst = 0usize;
                    for (i, (a, b)) in got2.iter().zip(&r).enumerate() {
                        let d = (*a as f64 - *b as f64).abs() / scale;
                        if d > maxd {
                            maxd = d;
                            worst = i;
                        }
                    }
                    eprintln!(
                        "[gguf-mmq2] {tname}({want_ty}, {m}×{inner}) MMQ2 vs cuBLAS: max|Δ|/max = {maxd:.3e} @ {worst}"
                    );
                    assert!(maxd < 2e-2, "{tname}(m={m}) 超阈 {maxd:.3e}");
                }
            }
        }
    }

    /// 型名 → ggml 型码(用例表写法;与 `family::gguf::entry_for` 同一词表)
    fn want_ty_code(name: &str) -> usize {
        use owl_kernels::family::gguf::ALL;
        let t = ALL
            .iter()
            .copied()
            .find(|t| t.name() == name)
            .unwrap_or_else(|| panic!("未知 ggml 型名 `{name}`"));
        // 型码 0 = F32;本门三型皆 >0(与 driver 的 `from_code` 语义一致,零/未知都拒)
        assert!(t.code() > 0, "{name} 型码为 0(与本门 aux 约定冲突)");
        t.code() as usize
    }

    /// **dequant+cuBLAS 臂 ↔ MMVQ 对拍门**(W-2d;2026-10-19)。
    ///
    /// 本臂走新引擎机制(`Op::Uninit` scratch + `Op::CallDst` 写列窗)⇒ 这条门同时
    /// 钉三件事:①行窗解量化(含 Q5_K/Q6_K 两趟覆盖)②cuBLAS `ldc`/`off` 列窗
    /// ③f16 累加口径的误差量级(`CUBLAS_COMPUTE_16F`,上游同路)。
    ///
    /// 覆盖:**4B 与 27B 的真实张量**(27B 的 `ffn_down` k=17408 是最长行窗;
    /// `attn_q` n=12288) × `m = 128 / 512`(512 = 上游 `ne11` 档,也是引擎
    /// chunk 512 的形状)。
    #[owl_test(gpu)]
    async fn gpu_gguf_cublas_dst_arm_matches_mmvq() {
        let mut gpu = crate::testkit::gpu_client().await;
        for (res_key, cases) in [
            (
                "gguf:4b",
                &[
                    ("blk.0.attn_gate.weight", "Q4_K"),
                    ("blk.0.ssm_out.weight", "Q8_0"),
                    ("blk.0.attn_qkv.weight", "Q6_K"),
                    ("blk.3.attn_k.weight", "Q5_K"),
                ][..],
            ),
            (
                "gguf:27b",
                &[
                    ("blk.0.ffn_down.weight", "Q4_K"),
                    ("blk.0.attn_gate.weight", "Q8_0"),
                    ("blk.3.attn_q.weight", "Q8_0"),
                ][..],
            ),
            // ---- IQ 族行窗(2026-10-20;REQ-GGUF-04)----
            // IQ4_NL 是最难的一档:组宽 256 = **8 个 32 元素块**(其余 IQ = 1 个超块)
            ("gguf:0.8b-iq4xs", &[("blk.0.attn_gate.weight", "IQ4_XS")][..]),
            (
                "gguf:27b-iq3s",
                &[
                    ("blk.11.attn_k.weight", "IQ4_NL"),
                    ("blk.4.attn_gate.weight", "IQ3_S"),
                    ("blk.1.attn_gate.weight", "IQ3_XXS"),
                    ("blk.19.attn_q.weight", "IQ2_XXS"),
                    ("blk.0.ffn_up.weight", "IQ1_S"),
                ][..],
            ),
            (
                "gguf:27b-iq4xs",
                &[
                    ("blk.15.attn_q.weight", "Q2_K"),
                    ("blk.12.ssm_out.weight", "Q3_K"),
                    ("blk.0.ffn_gate.weight", "IQ2_XS"),
                    ("blk.0.ffn_up.weight", "IQ2_S"),
                ][..],
            ),
        ] {
            if !resource_ok(res_key) {
                owl_shared::testkit::report::skip("资源清单缺 \"{res_key}\"");
                continue;
            }
            let g = Gguf::open(&PathBuf::from(resource(res_key).expect("列了就有"))).expect("open");
            for (tname, want_ty) in cases.iter().copied() {
                let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
                let ix = g.index_of(tname).unwrap();
                let bytes = g.bytes(ix);
                let (rows, inner) = (ti.rows() as usize, ti.inner() as usize);
                let (be, bb) = ti.ty.block();
                let row_bytes = inner / be * bb;
                let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
                for m in [128usize, 512] {
                    let xv: Vec<f32> = (0..m * inner)
                        .map(|i| {
                            half::f16::from_f32(((i as f32 * 0.0097).cos() * 1.5).clamp(-2.5, 2.5))
                                .to_f32()
                        })
                        .collect();
                    let xb: Vec<u8> =
                        xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
                    let x = TensorOps::from_host(Dtype::F16, vec![m, inner], &xb);
                    // 参考:MMVQ(q8_1 激活)
                    let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                        .arg(&x)
                        .arg_i32(inner as i32)
                        .arg_i32(inner as i32)
                        .aux(&[m, inner])
                        .with_shape(Dtype::U8, vec![m, inner / 32 * XQ_BLOCK]);
                    let a = harvest_f16(
                        &mut gpu,
                        &TensorOps::call(SemanticKernel::GgufGemm)
                            .arg(&wblk)
                            .arg(&xq)
                            .arg_i32(m as i32)
                            .arg_i32(rows as i32)
                            .arg_i32(inner as i32)
                            .arg_i32(inner as i32)
                            .arg_i32(0)
                            .arg_i32(rows as i32)
                            .aux(&[want_ty_code(want_ty), m, rows])
                            .with_shape(Dtype::F16, vec![m, rows]),
                    )
                    .await;
                    // 新臂:首 tile 普通 Call(分配 y),其后 CallDst 链式写同块
                    let tile = (32usize << 20) / (inner.max(1) * 2);
                    let tile = tile.clamp(128, 2048) / 64 * 64;
                    let mut last: Option<TensorOps> = None;
                    let mut n0 = 0usize;
                    while n0 < rows {
                        let r = tile.min(rows - n0);
                        let s = TensorOps::uninit(Dtype::F16, vec![r, inner]);
                        let dq = TensorOps::call_dst(&s, SemanticKernel::GgufCvtRange)
                            .arg(&wblk)
                            .arg_i32(n0 as i32)
                            .arg_i32(r as i32)
                            .arg_i32(inner as i32)
                            .aux(&[want_ty_code(want_ty), r, inner])
                            .with_shape(Dtype::F16, vec![r, inner]);
                        let node = match last.as_ref() {
                            None => TensorOps::call(SemanticKernel::GgufCublasDst),
                            Some(prev) => TensorOps::call_dst(prev, SemanticKernel::GgufCublasDst),
                        };
                        last = Some(
                            node.arg(&x)
                                .arg(&dq)
                                .arg_usize(r)
                                .arg_usize(inner)
                                .arg_usize(m)
                                .arg_usize(1)
                                .arg_usize(rows)
                                .arg_usize(n0)
                                .with_shape(Dtype::F16, vec![m, rows]),
                        );
                        n0 += r;
                    }
                    let b = harvest_f16(&mut gpu, &last.expect("至少一个 tile")).await;
                    let scale = a.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-12);
                    let mut maxd = 0f32;
                    let mut worst = 0usize;
                    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                        let d = (p - q).abs();
                        if d > maxd {
                            maxd = d;
                            worst = i;
                        }
                    }
                    let rel = maxd / scale;
                    eprintln!(
                        "[gguf-gpu] {res_key} {tname}({want_ty}, {rows}×{inner}, m={m}, tile={tile}):cuBLAS 臂 vs MMVQ max|Δ|/max = {rel:.3e} @ {worst}"
                    );
                    assert!(
                        rel < 3e-2,
                        "{tname}({want_ty}, m={m}):cuBLAS 臂与 MMVQ 差 {rel:.4e}(行窗/ldc/off 错位会给 O(1))"
                    );
                }
            }
        }
    }


    /// **IQ 族 GPU 门**(2026-10-20;REQ-GGUF-04):MMVQ vs CPU 参考(GPU 域)。
    ///
    /// 判据与 k-quant 门同口径(`max|Δ|/max|ref| < 2e-2`):错码本/错符号表/
    /// 错尺度/错块位移一律 O(1) 相对误差,而 q8_1 激活量化噪声只有 ~1e-3。
    /// 样本 = 本机存量(`0.8B-IQ4_XS` / `27B-UD-IQ3_S`)。
    #[owl_test(gpu)]
    async fn gpu_gguf_iq_gemm_matches_cpu_reference() {
        // (资源键, 张量, 期望型, m)
        let cases: &[(&str, &str, &str, usize)] = &[
            ("gguf:0.8b-iq4xs", "blk.0.attn_gate.weight", "IQ4_XS", 2),
            ("gguf:27b-iq3s", "blk.1.attn_gate.weight", "IQ3_XXS", 1),
            ("gguf:27b-iq3s", "blk.4.attn_gate.weight", "IQ3_S", 1),
            ("gguf:27b-iq3s", "blk.11.attn_k.weight", "IQ4_NL", 2),
            // 27B UD 档(2026-10-20):Q2_K/Q3_K/IQ2_*/IQ1_S
            ("gguf:27b-iq4xs", "blk.15.attn_q.weight", "Q2_K", 1),
            ("gguf:27b-iq4xs", "blk.12.ssm_out.weight", "Q3_K", 1),
            ("gguf:27b-iq4xs", "blk.0.ffn_gate.weight", "IQ2_XS", 1),
            ("gguf:27b-iq4xs", "blk.0.ffn_up.weight", "IQ2_S", 1),
            ("gguf:27b-iq3s", "blk.19.attn_q.weight", "IQ2_XXS", 1),
            ("gguf:27b-iq3s", "blk.0.ffn_up.weight", "IQ1_S", 1),
        ];
        let mut gpu = crate::testkit::gpu_client().await;
        for (res_key, tname, want_ty, m) in cases.iter().copied() {
            if !resource_ok(res_key) {
                owl_shared::testkit::report::skip("资源清单缺 \"{res_key}\"");
                continue;
            }
            let g = Gguf::open(&PathBuf::from(resource(res_key).expect("列了就有")))
                .expect("open");
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
            assert_eq!(ti.ty.name(), want_ty, "[{res_key}] {tname} 型不符(用例表过期?)");
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let (rows, inner): (usize, usize) = (ti.rows() as usize, ti.inner() as usize);
            let (be, bb) = ti.ty.block();
            let row_bytes = inner / be * bb;
            assert_eq!(row_bytes as u64, ti.row_bytes());

            let w = dequant_all(ti.ty, bytes, rows, inner).expect("dequant");
            let k = inner;
            let xv: Vec<f32> = (0..m * k)
                .map(|i| {
                    let f = half::f16::from_f32(((i as f32 * 0.0173).sin() * 1.7).clamp(-2.5, 2.5));
                    f.to_f32()
                })
                .collect();
            let (yref, qloss) = reference(&xv, &w, m, rows, k);

            let xb: Vec<u8> = xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
            let x = TensorOps::from_host(Dtype::F16, vec![m, k], &xb);
            let wblk = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], bytes);
            let xq_bytes = m * (k / 32) * XQ_BLOCK;
            let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                .arg(&x)
                .arg_i32(k as i32)
                .arg_i32(k as i32)
                .aux(&[m, k])
                .with_shape(Dtype::U8, vec![m, xq_bytes / m]);
            let y = TensorOps::call(SemanticKernel::GgufGemm)
                .arg(&wblk)
                .arg(&xq)
                .arg_i32(m as i32)
                .arg_i32(rows as i32)
                .arg_i32(k as i32)
                .arg_i32(k as i32)
                .arg_i32(0)
                .arg_i32(rows as i32)
                .aux(&[want_ty_code(want_ty), m, rows])
                .with_shape(Dtype::F16, vec![m, rows]);
            let got = harvest_f16(&mut gpu, &y).await;
            assert_eq!(got.len(), m * rows, "{tname} 输出元素数");

            let scale = yref.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-12);
            let mut maxd = 0f64;
            let mut worst = 0usize;
            for (i, (a, b)) in got.iter().zip(&yref).enumerate() {
                let d = (*a as f64 - *b).abs() / scale;
                if d > maxd {
                    maxd = d;
                    worst = i;
                }
            }
            eprintln!(
                "[gguf-iq] {res_key} {tname}({want_ty}, {rows}×{inner}, m={m}) \
                 max|Δ|/max|ref| = {maxd:.3e}(量化损失 {qloss:.3e};@ {worst};\
                 gpu/ref @0 = {:.6}/{:.6} = {:.4})",
                got[0] as f64,
                yref[0],
                if yref[0].abs() > 0.0 { got[0] as f64 / yref[0] } else { f64::NAN }
            );
            assert!(
                maxd < 0.02,
                "[{res_key}] {tname}({want_ty}) 超阈:{maxd:.4e} @ {worst}"
            );
        }
    }

    /// **IQ 族行解量化门**(合成表;覆盖本机 `token_embd` 未用到的型)。
    ///
    /// 判据 = 与 `dequant_row`(已与 gguf-py 位级对拍)逐元素相等;容差取 f16 舍入
    /// 量级(核内用 half 算),错行/错块/错码本会给 O(1)。
    #[owl_test(gpu)]
    async fn gpu_gguf_iq_rows_matches_reference() {
        use crate::formats::gguf::dequant::dequant_row;
        let mut gpu = crate::testkit::gpu_client().await;
        for ty in [
            GgmlTy::Iq4Nl,
            GgmlTy::Iq4Xs,
            GgmlTy::Iq3Xxs,
            GgmlTy::Iq3S,
            GgmlTy::Q2K,
            GgmlTy::Q3K,
            GgmlTy::Iq2Xxs,
            GgmlTy::Iq2Xs,
            GgmlTy::Iq2S,
            GgmlTy::Iq1S,
        ] {
            let (be, bb) = ty.block();
            let inner = 512usize;
            let rows = 4usize;
            let row_bytes = inner / be * bb;
            // 确定性伪随机字节(固定 LCG):块内每个 bit 都被真实覆盖
            let mut st = 0x1234_5678u32;
            let mut bytes = vec![0u8; rows * row_bytes];
            for b in bytes.iter_mut() {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (st >> 24) as u8;
            }
            // **尺度必须是有限 half**:随机字节的指数位会落进 inf/NaN 区,
            // 参考侧(d = inf)与核侧同坏 ⇒ 判据变 inf/inf(实测踩到)。
            for r in 0..rows {
                for b in 0..(inner / be) {
                    let d = half::f16::from_f32(0.02 + 0.001 * ((r + b) % 7) as f32);
                    // **d 的位置随型不同**:Q3_K/Q6_K 在块尾,其余在块首
                    let doff = match ty {
                        GgmlTy::Q3K | GgmlTy::Q6K => bb - 2,
                        _ => 0,
                    };
                    let o = r * row_bytes + b * bb + doff;
                    bytes[o..o + 2].copy_from_slice(&d.to_le_bytes());
                }
            }
            let ids: [usize; 3] = [0, 1, rows - 1];
            let ids_b: Vec<u8> = ids.iter().flat_map(|i| (*i as f32).to_le_bytes()).collect();
            let tbl = TensorOps::from_host(Dtype::U8, vec![rows, row_bytes], &bytes);
            let idt = TensorOps::from_host(Dtype::F32, vec![ids.len()], &ids_b);
            let out = TensorOps::call(SemanticKernel::GgufDequantRows)
                .arg(&tbl)
                .arg(&idt)
                .arg_i32(inner as i32)
                .arg_i32(ids.len() as i32)
                .arg_i32(rows as i32)
                .aux(&[ty.code() as usize, ids.len(), inner])
                .with_shape(Dtype::F16, vec![ids.len(), inner]);
            let got = harvest_f16(&mut gpu, &out).await;
            for (r, id) in ids.iter().copied().enumerate() {
                let mut want = vec![0f32; inner];
                dequant_row(ty, &bytes[id * row_bytes..(id + 1) * row_bytes], &mut want)
                    .unwrap_or_else(|e| panic!("{} dequant_row: {e}", ty.name()));
                let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6) as f64;
                let mut maxd = 0f64;
                let mut worst = 0usize;
                for i in 0..inner {
                    let d = (got[r * inner + i] as f64 - want[i] as f64).abs() / scale;
                    if d > maxd {
                        maxd = d;
                        worst = i;
                    }
                }
                eprintln!(
                    "[gguf-iq-rows] {}(id={id}) max|Δ|/max|ref| = {maxd:.3e} @ {worst}",
                    ty.name()
                );
                assert!(maxd < 5e-3, "{} id={id} 超阈:{maxd:.3e} @ {worst}", ty.name());
            }
        }
    }


}

#[cfg(test)]
mod m_scan {
    use crate::contract::Dtype;
    use crate::formats::gguf::reader::Gguf;
    use crate::formats::gguf::GgmlTy;
    use crate::module::{Module as _, QuantPlan, Taken, WeightSource};
    use owl_iface::contract::DeviceClient as _;
    use crate::TensorOps;
    use owl_macros::owl_test;
    use std::path::PathBuf;

    /// 单张量源(诊断台;只服务 m 扫描 —— 键恒 `w`)
    struct OneTensor {
        key: &'static str,
        bytes: Vec<u8>,
    }

    impl WeightSource for OneTensor {
        fn take(&self, key: &str) -> Taken<Vec<f32>> {
            if key == self.key { Ok(None) } else { Ok(None) }
        }
        fn elem_len(&self, key: &str) -> Taken<usize> {
            Ok((key == self.key).then_some(self.bytes.len()))
        }
        fn native_dtype(&self, key: &str) -> Taken<Dtype> {
            Ok((key == self.key).then_some(Dtype::U8))
        }
        fn convert_chunk_into_bytes(
            &self,
            key: &str,
            offset: usize,
            len: usize,
            dst: &mut [u8],
            dtype: Dtype,
        ) -> Taken<()> {
            if key != self.key {
                return Ok(None);
            }
            if dtype != Dtype::U8 {
                return Err(crate::contract::ModelError::Msg(format!(
                    "OneTensor: 只出 U8(实得 {dtype:?})"
                )));
            }
            let Some(src) = self.bytes.get(offset..offset + len) else {
                return Ok(None);
            };
            dst[..len].copy_from_slice(src);
            Ok(Some(()))
        }
    }

    /// **IQ / K-quant 复用臂位级门**(2026-10-21):UD 混型档的每种量化型,
    /// m = 8 / 17(TILE_MT=16 两侧)下 `GgufGemm`(MMVQ) ≡ `GgufGemmTile`。
    ///
    /// 判据 = **逐位相同**(两核的 vec_dot 序逐 token 不变;错型/错块会给 O(1))。
    #[owl_test(gpu)]
    async fn gpu_gguf_tile_matches_mmvq_iq_family() {
        use crate::formats::gguf::Q8_1_BLOCK_BYTES;
        use crate::ops::SemanticKernel;
        // 2026-10-27(worktree iq3-perf):本门默认仍验 IQ4_XS;`OWL_MMQ2_GATE_PATH`
        // 可指向 UD-IQ3_S(它带 IQ2_XXS;IQ4_XS 里没有),用于覆盖新挂的型。
        // 仅测试面用 env(生产路径零 env)。
        let path = std::env::var("OWL_MMQ2_GATE_PATH").unwrap_or_else(|_| {
            "/home/div/Documents/codes/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf".to_string()
        });
        let g = Gguf::open(&PathBuf::from(path)).expect("open GGUF(MMQ2 门)");
        let mut gpu = crate::testkit::gpu_client().await;
        // 每种型取第一张非嵌入张量(embed/lm_head 走同一臂但太大,不必入此门)
        let mut seen = std::collections::BTreeSet::new();
        let mut cases: Vec<(String, GgmlTy, usize, usize)> = Vec::new();
        for t in &g.tensors {
            if t.ty.native_dtype().is_some() || t.name.starts_with("token_embd") || t.name.starts_with("output.") {
                continue;
            }
            if crate::formats::gguf::require_portable(&g).is_err() {
                break;
            }
            if seen.insert(format!("{:?}", t.ty)) {
                cases.push((t.name.clone(), t.ty, t.rows() as usize, t.inner() as usize));
            }
        }
        let mut checked = 0usize;
        for (tname, ty, rows, inner) in &cases {
            if owl_kernels::family::gguf::tile_entry_for(*ty).is_none() {
                eprintln!("[tile-gate] {tname}({}) 无复用臂(跳过)", ty.name());
                continue;
            }
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let (be, bb) = ty.block();
            let wblk = TensorOps::from_host(Dtype::U8, vec![*rows, inner / be * bb], bytes);
            for m in [8usize, 17] {
                let xb: Vec<u8> = (0..m * inner)
                    .flat_map(|i| {
                        let v = (((i as f32) * 0.013).sin() * 1.4) as f32;
                        half::f16::from_f32(v).to_le_bytes()
                    })
                    .collect();
                let x = TensorOps::from_host(Dtype::F16, vec![m, *inner], &xb);
                let xq = TensorOps::call(SemanticKernel::GgufQuantActF16)
                    .arg(&x)
                    .arg_i32(*inner as i32)
                    .arg_i32(*inner as i32)
                    .aux(&[m, *inner])
                    .with_shape(Dtype::U8, vec![m, inner / 32 * Q8_1_BLOCK_BYTES]);
                let build = |op: SemanticKernel| {
                    TensorOps::call(op)
                        .arg(&wblk)
                        .arg(&xq)
                        .arg_i32(m as i32)
                        .arg_i32(*rows as i32)
                        .arg_i32(*inner as i32)
                        .arg_i32(*inner as i32)
                        .arg_i32(0)
                        .arg_i32(*rows as i32)
                        .aux(&[ty.code() as usize, m, *rows])
                        .with_shape(Dtype::F16, vec![m, *rows])
                };
                let a = crate::testkit::harvest_f16(&mut gpu, &build(SemanticKernel::GgufGemm)).await;
                let b = crate::testkit::harvest_f16(&mut gpu, &build(SemanticKernel::GgufGemmTile)).await;
                let mut nbit = 0usize;
                let mut worst = 0usize;
                for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                    if p.to_bits() != q.to_bits() {
                        nbit += 1;
                        if nbit == 1 { worst = i; }
                    }
                }
                assert_eq!(nbit, 0, "[{tname} {} m={m}] 复用臂与 MMVQ 有 {nbit} 项不同(首个 @{worst}:{} vs {})",
                    ty.name(), a[worst], b[worst]);
                checked += 1;
            }
        }
        eprintln!("[tile-gate] IQ/K-quant 复用臂位级门绿:{checked} 组(m=8/17)");
        gpu.close().await.expect("关机");
    }

    /// **小 m 的 MMQ2(int8-MMA)可行性门**(2026-10-21):m ∈ {2, 8, 17, 32}
    /// 时 MMQ2 与 MMVQ 的输出接近(容差 = int8 量化噪声,非逐位),且能被
    /// nsys 读到耗时(用来决定 verify 形状改走哪条臂)。
    #[owl_test(gpu)]
    async fn gpu_gguf_mmq2_small_m() {
        use crate::formats::gguf::Q8_1_BLOCK_BYTES;
        use crate::ops::SemanticKernel;
        // 2026-10-27(worktree iq3-perf):本门默认仍验 IQ4_XS;`OWL_MMQ2_GATE_PATH`
        // 可指向 UD-IQ3_S(它带 IQ2_XXS;IQ4_XS 里没有),用于覆盖新挂的型。
        // 仅测试面用 env(生产路径零 env)。
        let path = std::env::var("OWL_MMQ2_GATE_PATH").unwrap_or_else(|_| {
            "/home/div/Documents/codes/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf".to_string()
        });
        let g = Gguf::open(&PathBuf::from(path)).expect("open GGUF(MMQ2 门)");
        let mut gpu = crate::testkit::gpu_client().await;
        // 每种 MMQ2-可用型各取一张(UD 档 IQ4_XS / Q8_0 / Q4_K / Q5_K / Q6_K /
        // IQ3_S / IQ3_XXS 都在场;按文件里首个命中取)
        let mut seen = std::collections::BTreeSet::new();
        let mut cases: Vec<(String, usize, usize)> = Vec::new();
        for t in &g.tensors {
            if t.name.starts_with("token_embd") || t.name.starts_with("output.") {
                continue;
            }
            if owl_kernels::family::gguf::mmq2_entry_for(t.ty, 8).is_none() {
                continue;
            }
            if t.rows() % 128 != 0 || t.inner() % 128 != 0 {
                continue;
            }
            if seen.insert(t.ty.name().to_string()) {
                cases.push((t.name.clone(), t.rows() as usize, t.inner() as usize));
            }
        }
        for (tname, out_dim, in_dim) in cases.iter() {
            let tname = tname.as_str();
            let (out_dim, in_dim) = (*out_dim, *in_dim);
            let ti = g.get_tensor(tname).expect("tensor");
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let (be, bb) = ti.ty.block();
            let wblk = TensorOps::from_host(Dtype::U8, vec![out_dim, in_dim / be * bb], bytes);
            let ds4 = matches!(ti.ty, GgmlTy::Q4K | GgmlTy::Q5K);
            for m in [8usize, 64, 128, 256, 2048] {
                // 2026-10-21 第二刀:Q3_K / IQ2_S / IQ2_XS / Q2_K 只挂 J=16
                // (`m ≤ 16`)⇒ 大 m 交回 tile/MMVQ,本门不覆盖该档
                if owl_kernels::family::gguf::mmq2_entry_for(ti.ty, m).is_none() {
                    continue;
                }
                let xb: Vec<u8> = (0..m * in_dim)
                    .flat_map(|i| {
                        let v = (((i as f32) * 0.013).sin() * 1.4) as f32;
                        half::f16::from_f32(v).to_le_bytes()
                    })
                    .collect();
                let x = TensorOps::from_host(Dtype::F16, vec![m, in_dim], &xb);
                // MMVQ(参考)
                let xq1 = TensorOps::call(SemanticKernel::GgufQuantActF16)
                    .arg(&x)
                    .arg_i32(in_dim as i32)
                    .arg_i32(in_dim as i32)
                    .aux(&[m, in_dim])
                    .with_shape(Dtype::U8, vec![m, in_dim / 32 * Q8_1_BLOCK_BYTES]);
                let y_ref = TensorOps::call(SemanticKernel::GgufGemm)
                    .arg(&wblk)
                    .arg(&xq1)
                    .arg_i32(m as i32)
                    .arg_i32(out_dim as i32)
                    .arg_i32(in_dim as i32)
                    .arg_i32(in_dim as i32)
                    .arg_i32(0)
                    .arg_i32(out_dim as i32)
                    .aux(&[ti.ty.code() as usize, m, out_dim])
                    .with_shape(Dtype::F16, vec![m, out_dim]);
                // MMQ2(int8-MMA;J=64 或 128)
                let j = if m >= 128 { 128 } else { 64 };
                let q = if ds4 {
                    SemanticKernel::GgufMmq2QuantDs4
                } else {
                    SemanticKernel::GgufMmq2QuantD4
                };
                let m_pad = owl_kernels::family::gguf::mmq2_xq_tokens(m);
                let xq2 = TensorOps::call(q)
                    .arg(&x)
                    .arg_i32(m_pad as i32)
                    .arg_i32(m as i32)
                    .arg_i32(in_dim as i32)
                    .aux(&[m, in_dim])
                    .with_shape(Dtype::U8, vec![in_dim / 128, m_pad * 144]);
                let _ = j;
                let y2 = TensorOps::call(SemanticKernel::GgufMmq2)
                    .arg(&wblk)
                    .arg(&xq2)
                    .arg_i32(out_dim as i32)
                    .arg_i32(m as i32)
                    .arg_i32(in_dim as i32)
                    .aux(&[ti.ty.code() as usize, m, out_dim])
                    .with_shape(Dtype::F16, vec![m, out_dim]);
                // BF16 激活通道(2026-10-21;草稿域):同一重量化手 + MMQ2
                // vs MMVQ(BF16 激活),判据同 5e-2
                {
                    // 与 f16 侧同源同值(同一 seed 公式),只换 bf16 编码
                    let xb16: Vec<u8> = (0..m * in_dim)
                        .flat_map(|i| {
                            let v = (((i as f32) * 0.013).sin() * 1.4) as f32;
                            half::bf16::from_f32(v).to_le_bytes()
                        })
                        .collect();
                    let x16 = TensorOps::from_host(Dtype::BF16, vec![m, in_dim], &xb16);
                    let xq_ref = TensorOps::call(SemanticKernel::GgufQuantActBf16)
                        .arg(&x16)
                        .arg_i32(in_dim as i32)
                        .arg_i32(in_dim as i32)
                        .aux(&[m, in_dim])
                        .with_shape(Dtype::U8, vec![m, in_dim / 32 * Q8_1_BLOCK_BYTES]);
                    let y_ref_b = TensorOps::call(SemanticKernel::GgufGemm)
                        .arg(&wblk)
                        .arg(&xq_ref)
                        .arg_i32(m as i32)
                        .arg_i32(out_dim as i32)
                        .arg_i32(in_dim as i32)
                        .arg_i32(in_dim as i32)
                        .arg_i32(0)
                        .arg_i32(out_dim as i32)
                        .aux(&[ti.ty.code() as usize, m, out_dim])
                        .with_shape(Dtype::F16, vec![m, out_dim]);
                    let qb = if ds4 {
                        SemanticKernel::GgufMmq2QuantDs4Bf16
                    } else {
                        SemanticKernel::GgufMmq2QuantD4Bf16
                    };
                    let xq2b = TensorOps::call(qb)
                        .arg(&x16)
                        .arg_i32(m_pad as i32)
                        .arg_i32(m as i32)
                        .arg_i32(in_dim as i32)
                        .aux(&[m, in_dim])
                        .with_shape(Dtype::U8, vec![in_dim / 128, m_pad * 144]);
                    let y2b = TensorOps::call(SemanticKernel::GgufMmq2)
                        .arg(&wblk)
                        .arg(&xq2b)
                        .arg_i32(out_dim as i32)
                        .arg_i32(m as i32)
                        .arg_i32(in_dim as i32)
                        .aux(&[ti.ty.code() as usize, m, out_dim])
                        .with_shape(Dtype::F16, vec![m, out_dim]);
                    let ab = crate::testkit::harvest_f16(&mut gpu, &y_ref_b).await;
                    let bb = crate::testkit::harvest_f16(&mut gpu, &y2b).await;
                    let sc = ab.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
                    let db = ab.iter().zip(&bb).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / sc;
                    eprintln!("[mmq2-small-m] {tname} m={m}: MMQ2(BF16 激活) vs MMVQ 相对最大差 {db:.3e}");
                    assert!(db < 5e-2, "[{tname} m={m} bf16] MMQ2 与 MMVQ 相对差 {db} > 5e-2");
                }
                let a = crate::testkit::harvest_f16(&mut gpu, &y_ref).await;
                let b = crate::testkit::harvest_f16(&mut gpu, &y2).await;
                let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
                let d = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
                eprintln!("[mmq2-small-m] {tname} m={m}: MMQ2 vs MMVQ 相对最大差 {d:.3e}");
                assert!(d < 5e-2, "[{tname} m={m}] MMQ2 与 MMVQ 相对差 {d} > 5e-2");
                // MMQ(dp4a)对照臂:A/B 小 m 最优臂(Q4_K/Q5_K/Q6_K/Q8_0 才有)
                if owl_kernels::family::gguf::mmq_entry_for(ti.ty).is_some() {
                    let xqrow = in_dim / 32 * Q8_1_BLOCK_BYTES;
                    let y3 = TensorOps::call(SemanticKernel::GgufMmq)
                        .arg(&wblk)
                        .arg(&xq1)
                        .arg_i32(in_dim as i32)
                        .arg_i32(out_dim as i32)
                        .arg_i32(m as i32)
                        .aux(&[ti.ty.code() as usize, m, out_dim])
                        .with_shape(Dtype::F16, vec![m, out_dim]);
                    let _ = xqrow;
                    let c = crate::testkit::harvest_f16(&mut gpu, &y3).await;
                    let d3 = a.iter().zip(&c).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
                    eprintln!("[mmq2-small-m] {tname} m={m}: MMQ(dp4a) vs MMVQ 相对最大差 {d3:.3e}");
                }
            }
        }
        gpu.close().await.expect("关机");
    }

    /// **m 扫描(GPU;诊断台)**:同一张真实量化权重,`Linear::forward` 在
    /// m = 1..128 的耗时 —— 定位 verify(m=8) 为何 8 倍于 decode(m=1)。
    ///
    /// 理想锚:m ≤ 8 应**带宽平坦**(读一遍权重)⇒ ms(8)/ms(1) ≤ 2;
    /// 实测比 ≈ m ⇒ 该臂**零权重复用**(每 token 读一遍)。
    #[owl_test(gpu)]
    async fn gpu_gguf_gemm_m_scan() {
        let Some(path) = owl_shared::testkit::resource("gguf:27b-iq4xs") else {
            owl_shared::testkit::report::skip("资源清单缺 gguf:27b-iq4xs");
            return;
        };
        let path = PathBuf::from(path);
        let g = Gguf::open(&path).expect("open 27B IQ4_XS");
        let mut gpu = crate::testkit::gpu_client().await;
        // 27B UD-IQ4_XS 是**重混型**(IQ4_XS 211 / Q8_0 98 / Q4_K 55 / Q5_K 51 …)
        // ⇒ 取 IU 占比最大的 IQ4_XS 两张 + IQ2_XS 最大张量
        let cases: &[(&str, &'static str, usize, usize)] = &[
            ("blk.0.attn_qkv.weight", "IQ4_XS", 10240, 5120),
            ("blk.0.attn_gate.weight", "IQ4_XS", 6144, 5120),
            ("blk.0.ffn_gate.weight", "IQ2_XS", 17408, 5120),
        ];
        for (tname, want_ty, out_dim, in_dim) in cases.iter().copied() {
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("缺张量 {tname}"));
            if ti.ty.name() != want_ty {
                eprintln!("[m-scan] {tname} 型 {} ≠ 用例表 {want_ty}(跳过)", ti.ty.name());
                continue;
            }
            let ix = g.index_of(tname).unwrap();
            let src = OneTensor { key: "w", bytes: g.bytes(ix).to_vec() };
            let lin = crate::layers::linear::Linear::new("w", out_dim, in_dim, QuantPlan::Gguf(ti.ty));
            let opts = crate::formats::load::LoadOpts::plain(Dtype::F16);
            crate::formats::load::load_component(&mut gpu, &lin, &src, opts)
                .await
                .expect("单张量装载");
            let mut prev = 0f64;
            for m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
                let xv: Vec<f32> = (0..m * in_dim)
                    .map(|i| ((i as f32 * 0.017).sin() * 1.5) as f32)
                    .collect();
                let xb: Vec<u8> = xv.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
                let x = TensorOps::from_host(Dtype::F16, vec![m, in_dim], &xb);
                let ctx = crate::module::ForwardCtx::minimal(m);
                let y = lin.forward(&x, &ctx);
                let reps = if m <= 8 { 20 } else { 8 };
                let _ = crate::interpreters::eval_ops(y.clone().step(), &mut gpu).await.expect("warm");
                gpu.sync().await.expect("sync");
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = crate::interpreters::eval_ops(y.clone().step(), &mut gpu).await.expect("eval");
                }
                gpu.sync().await.expect("sync");
                let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                let ratio = if prev > 0.0 { ms / prev } else { 0.0 };
                eprintln!("[m-scan] {tname}({want_ty}, {out_dim}×{in_dim}) m={m:3}: {ms:8.3} ms  (×{ratio:.2} vs 上一档)");
                prev = ms;
            }
        }
        gpu.close().await.expect("关机");
    }
}
