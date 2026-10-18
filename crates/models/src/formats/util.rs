//! **格式无关的共享助手**(2026-10-17 审查 §四轮-C2 提层)。
//!
//! 为什么存在:这些函数原先**寄生在 `w4a16.rs`** 里(`pub(crate)`),
//! 而 `awq.rs` 要横过来引用它们(6 处 `marlin_eligible`、十余处字节转换),
//! `layers/linear.rs` 也走 `formats::w4a16::marlin_eligible` ——
//! **兄弟模块反向依赖 + 两个源各存一份同义实现**(`f32_bytes_to_f16`
//! 在 awq/w4a16 逐字重复)。它们与 g128 族无关(awq 同样用),故家应在本文件。
//!
//! 分层纪律不变:`formats/` 各源只管「键 → 字节」,本文件只放**纯函数**
//! (无 I/O、无源状态、无 mmap)。
//!
//! `marlin_*` 的语义锚:与 [`crate::layers`](../../layers) 的 Linear 构造期
//! QuantPlan 门控**同一谓词**(单一来源,禁止第二份)。

// ============================================================================
// §1 marlin 形态谓词(与 Linear 构造期门控同源)
// ============================================================================

/// **marlin 可重排谓词**(单一来源:装载侧形态判定 + 层侧 QuantPlan 门控)。
///
/// 判据回归 kernel/pack 真实约束:**`n%256==0`**(repack tile)且
/// **`in%128==0`**(g128 分组)。历史暗雷:形状 global_reduce_fp32 写穿 c_tmp
/// 踩内存,呈「n=2^k 才安全」假象;按 `a_is_s8` 分派修复后 sweep 全绿
/// 512..17408 含全部非 2^k 档与 27B 维度族,m∈{1,32,4096}。
pub fn marlin_eligible(out_dim: usize, in_dim: usize) -> bool {
    out_dim.is_multiple_of(256) && in_dim.is_multiple_of(128)
}

/// 打包 n 档位(暗雷根治后 = **n 直通**):eligibility 已保证 `n%256==0`
/// (repack tile 约束),历史 pad 路线(n → 512×2^k)系暗雷误诊产物,已废
/// (pad 实测仍挂;真凶 = use_fp32_reduce/c_tmp,见 [`marlin_eligible`] 文档)。
/// 保留函数 = **语义锚点**,防再引入 pad。
pub fn marlin_n_pack(n: usize) -> usize {
    debug_assert!(n.is_multiple_of(256), "marlin pack 要求 n%256==0(得 {n})");
    n
}

// ============================================================================
// §2 字节 ↔ 位型(源无关;装载热路径,rayon 按 64K 元素分块)
// ============================================================================

/// i32 平行读(`u8` 字节 → i32;LE)
pub(crate) fn i32s_par(bytes: &[u8]) -> Vec<i32> {
    use rayon::prelude::*;
    let n = bytes.len() / 4;
    let mut v = vec![0i32; n];
    v.par_chunks_mut(1 << 16)
        .enumerate()
        .for_each(|(t, chunk)| {
            for (i, cell) in chunk.iter_mut().enumerate() {
                let o = (t * (1 << 16) + i) * 4;
                *cell = i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            }
        });
    v
}

/// BF16 位型字节 → f32(精确展宽)
pub(crate) fn bf16_par(bytes: &[u8]) -> Vec<f32> {
    use rayon::prelude::*;
    let n = bytes.len() / 2;
    let mut v = vec![0f32; n];
    v.par_chunks_mut(1 << 16)
        .enumerate()
        .for_each(|(t, chunk)| {
            for (i, cell) in chunk.iter_mut().enumerate() {
                let o = (t * (1 << 16) + i) * 2;
                *cell =
                    f32::from_bits((u16::from_le_bytes([bytes[o], bytes[o + 1]]) as u32) << 16);
            }
        });
    v
}

/// F16 平行读。
///
/// 病史(2026-10-07 E5-DF3 定谳):syvai W4A16 的 `weight_scale` 是 **F16**
/// 标注 + F16 字节,曾全走 `bf16_par` 按 BF16 位型读 ⇒ 草稿全部 W4A16 权重
/// = 噪声 ⇒ 草稿前向废 ⇒ AL=0。**scale 读法必须按 index 声明 dtype 分派**
/// (对照:两种读法对 z-lab BF16 源 fc 权重做相关,F16 读 0.977 / BF16 读 0.016)。
pub(crate) fn f16_par(bytes: &[u8]) -> Vec<f32> {
    use rayon::prelude::*;
    let n = bytes.len() / 2;
    let mut v = vec![0f32; n];
    v.par_chunks_mut(1 << 16)
        .enumerate()
        .for_each(|(t, chunk)| {
            for (i, cell) in chunk.iter_mut().enumerate() {
                let o = (t * (1 << 16) + i) * 2;
                *cell = half::f16::from_le_bytes([bytes[o], bytes[o + 1]]).to_f32();
            }
        });
    v
}

/// scale 平行读分派:按 **RawEntry 声明 dtype** 选 F16/BF16 位型
pub(crate) fn scales_par(dt: safetensors::Dtype, bytes: &[u8]) -> Vec<f32> {
    match dt {
        safetensors::Dtype::F16 => f16_par(bytes),
        _ => bf16_par(bytes),
    }
}

/// f16 位型字节 → BF16 位型字节(E5-DF3 同日十二:syvai W4A16 的 scales
/// = F16,而 BF16 激活的 marlin 实例要求 `s_type = BF16`;值经 f32 中转,
/// 尾数 10→7 位 ≈ 0.4% 相对误差,远小于量化噪声)。
pub(crate) fn f16_bytes_to_bf16_bytes(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .flat_map(|c| {
            half::bf16::from_f32(half::f16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes()
        })
        .collect::<Vec<u8>>()
}

/// f32 字节 → f16 字节(**唯一实现**)。
///
/// 三处同义实现收成一处(2026-10-17 审查):awq / w4a16 各有一份逐字重复的
/// rayon 版,`f16c::f32_bytes_to_f16_bytes` 另有一份 SIMD 版 ⇒ 本函数 =
/// **语义锚**(RNE 舍入与 `half::f16::from_f32` 逐位一致),实现委托给
/// `f16c` 的**未对齐安全**字节口(mmap 条目偏移不保证 4 对齐,见坑册 §56)。
/// 调用面 = 源里的 F32 passthrough 键(GDN 标量小张量)⇒ 单线程 SIMD 足够,
/// 不再上 rayon(小张量下 rayon 的分派开销 > 收益)。
pub(crate) fn f32_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; (bytes.len() / 4) * 2];
    crate::f16c::f32_bytes_to_f16_bytes(bytes, &mut out);
    out
}

/// i32 切片 → LE 字节
pub(crate) fn i32s_le_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u16 切片 → LE 字节
pub(crate) fn u16s_le_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 元素数 × 元素宽 → 字节数(**checked**;越界 = `None`)。边界判定的
/// 统一写法(2026-10-17 审查 §四轮-C5:两源原有 checked / 普通算式两套)。
pub(crate) fn checked_len(len: usize, esz: usize) -> Option<usize> {
    len.checked_mul(esz)
}
