//! **FI(FlashInfer FA2)paged prefill 单算子探针** —— 生产 prefill 注意力核的基准尺。
//!
//! 为什么要有这个文件(2026-10-14 立,用户点破):
//! `flashinfer_prefill_paged_fp8kv` 是**生产 t≥128 的 prefill 注意力主核**,
//! 但 `crates/kernels/tests/` 下此前**只有邻居核的单测**(`chunked_prefill_fp8_probe`
//! = xinfer 血统回落核;`xqa_prefill_probe` = XQA 候选)—— 于是 2026-10-14 的
//! "换核" 选型拿**回落核**当了基线,探针全绿而结论方向反了(详录
//! `docs/arch/prefill-graph-施工.md` §14.1)。**基线核必须自己有单测**。
//!
//! 形状 = 生产 chunked prefill 一个 chunk:q = T 行(全局位置 `[ctx−T, ctx)`),
//! GQA 24q/4kv,hd256,page 32,kNHD `[nb, page, hkv, hd]` **fp8 e4m3** 池
//! (FI fp8kv 变体直读主池,adapter 内部布局 = 同一 kNHD)。
//!
//! 三条判据:
//!   1) **数值**:FI vs host 参考(行 i 可见 `[0, ctx−T+i]` 闭区间)误差 < 1e-2;
//!   2) **时间**:T=512 × ctx{512,2048,4096,8192,16384} per-call ms(与
//!      `xqa_prefill_probe` 同机同形状 ⇒ 换核选项在**同一把尺子**上比);
//!   3) **调度**:plan 选的 `cta_tile_q` / `split_kv`(FI_DEBUG=1 打到 stderr)——
//!      `FI_SPLIT_KV=1` 打开被适配器写死的 split-kv,同 harness 直接 A/B。
//!
//! 运行:
//! `CUDA_VISIBLE_DEVICES=<UUID> OWL_TEST_DEVICE=0 cargo test --release -p owl-kernels \
//!  --features device,flashinfer --test fi_prefill_probe -- --test-threads=1 --nocapture`
//! 可选:`FI_PF_ONLY=4096` 只跑该 ctx;`FI_PF_HOST=1` 大形状也跑 host 参考(慢);
//! `FI_DEBUG=1` 打宿主调度向量;`FI_SPLIT_KV=1` 开 split-kv(探针旋钮,缺省关)。
//!
//! ⚠️ **跑法**:本探针是基准 —— host 参考/建池是纯 CPU 大循环,debug 慢
//! 10–30×(2026-10-15 gdb 定谳:卡点在 `lcg_u8` 的逐元素 e4m3 编码与 host 参考)。
//! 建议:`cargo test --release -p owl-kernels --features device,<族> --test <本文件>
//!       -- --test-threads=1 --nocapture`;debug 下只用 `*_PF_ONLY=<ctx>` 跳 host 检查。
#![cfg(feature = "device")]

mod fi_support;
use fi_support::{f16_to_f32, TestRes, H, HD, HKV, PAGE};

use std::time::Instant;

use owl_kernels::contract::{Arg, Dtype, KernelSource, LaunchMsg};
use owl_kernels::device::Exec;

// ============================================================================
// host 参考 + 数据生成(与 xqa_prefill_probe 同一把尺子:逐字同序)
// ============================================================================
// ---------------------------------------------------------------------------
// e4m3 编解码:表化 + 二分(2026-10-15 **测试用例优化**)
//
// 病根(实测,见 gdb 栈):原 `e4m3_encode` 是**每元素 256 次线性扫**、`e4m3_decode`
// 每元素调 `powi`;而本探针建池要 8.4M(16384 档 16.8M)个元素、host 参考里 decode
// 又要被调用十亿次 ⇒ **debug 构建下单核跑分钟级**(表现为"卡住",GPU util 0%)。
// 修法(语义不变):
//   ① `e4m3_decode` → 256 项**一次性表**(数值与原式逐位相同);
//   ② `e4m3_encode` → 按值排序表 + 二分 + 小窗取最小 (err, code)
//      —— 与原 256 线性扫**逐位等价**(含"同误差取最小 code"的 tie 规则),
//      由本文件单测 `e4m3_encode_matches_bruteforce` 机器锁死;
//   ③ host 参考**预解码**一次(池 → f32),内层不再逐元素 decode。
// ---------------------------------------------------------------------------
fn e4m3_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (code, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_decode_exact(code as u8);
        }
        t
    })
}
/// 原始解码式(仅建表时用一次/项;NaN 槽保留原样)
fn e4m3_decode_exact(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m = (code & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
    s * v
}
fn e4m3_decode(code: u8) -> f32 {
    e4m3_table()[code as usize]
}
/// 编码表:按值升序,同值按 code 升序(⇒ tie 自动取最小 code)
fn e4m3_sorted() -> &'static [(f32, u8)] {
    static T: std::sync::OnceLock<Vec<(f32, u8)>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut v: Vec<(f32, u8)> = (0..=255u8)
            .filter(|c| c & 0x7F != 0x7F) // NaN 两种位型不参与编码(原式同)
            .map(|c| (e4m3_decode_exact(c), c))
            .collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        v
    })
}
/// 最近码编码(与原 256 线性扫逐位等价;见文件头注 ②)
fn e4m3_encode(v: f32) -> u8 {
    let tbl = e4m3_sorted();
    let pos = tbl.partition_point(|&(x, _)| x < v);
    let lo = pos.saturating_sub(2);
    let hi = (pos + 2).min(tbl.len());
    let mut best_err = f32::INFINITY;
    let mut best = 0u8;
    for &(x, code) in &tbl[lo..hi] {
        let err = (x - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}
/// 暴力参照(仅单测用;原实现原样保留)
#[cfg(test)]
fn e4m3_encode_bruteforce(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let err = (e4m3_decode_exact(code) - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}

#[cfg(test)]
mod encode_opt_tests {
    use super::*;
    /// 二分编码 ≡ 原 256 线性扫(含 tie/±0/饱和)
    #[test]
    fn e4m3_encode_matches_bruteforce() {
        let mut cases: Vec<f32> = Vec::new();
        for code in 0..=255u8 {
            let v = e4m3_decode_exact(code);
            if v.is_finite() {
                cases.push(v);
                cases.push(v + 1e-3);
                cases.push(v - 1e-3);
            }
        }
        cases.extend([0.0f32, -0.0, 1e-9, -1e-9, 448.0, -448.0, 1e9, -1e9, 1.0, -1.0]);
        for i in 0..20_000u32 {
            let x = i.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B);
            cases.push(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0);
        }
        for v in cases {
            if v.is_nan() {
                continue;
            }
            assert_eq!(
                e4m3_encode(v),
                e4m3_encode_bruteforce(v),
                "v={v} 二分编码与原线性扫不等价"
            );
        }
    }
}

fn lcg_u8(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            e4m3_encode(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0)
        })
        .collect()
}
fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half::f16::from_f32(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0).to_bits()
        })
        .collect()
}

/// host 参考:kNHD + fp8 池,chunked prefill 行 i 看 `[0, ctx−T+i]`(闭区间)。
fn host_ref_knhd(q: &[u16], kc: &[u8], vc: &[u8], t: usize, ctx: usize) -> Vec<f32> {
    // **预解码一次**(2026-10-15 测试用例优化):原实现在最内层逐元素
    // `e4m3_decode`,T=512/ctx=512 单 case 就是 ~1.6e9 次调用(debug 分钟级);
    // 预解码后内层只剩 f32 FMA —— 语义不变(同一批值,同一归约序)。
    let kf: Vec<f32> = kc.iter().map(|&c| e4m3_decode(c)).collect();
    let vf: Vec<f32> = vc.iter().map(|&c| e4m3_decode(c)).collect();
    let qf: Vec<f32> = q.iter().map(|&b| f16_to_f32(b)).collect();
    let mut out = vec![0f32; t * H * HD];
    let kbs = HKV * HD * PAGE;
    let scale = 1.0 / (HD as f32).sqrt();
    for s in 0..t {
        let len = ctx - t + s + 1;
        for h in 0..H {
            let kvh = h / (H / HKV);
            let mut qk = vec![0f32; len];
            #[allow(clippy::needless_range_loop)]
            for b in 0..len {
                let (pb, bb) = (b / PAGE, b % PAGE);
                let mut acc = 0f32;
                for d in 0..HD {
                    let kidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                    acc += qf[(s * H + h) * HD + d] * kf[kidx];
                }
                qk[b] = acc * scale;
            }
            let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                #[allow(clippy::needless_range_loop)]
                for b in 0..len {
                    let (pb, bb) = (b / PAGE, b % PAGE);
                    let vidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                    acc += (qk[b] - m).exp() * vf[vidx];
                }
                out[(s * H + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

// ============================================================================
// 一次全量测量
// ============================================================================
fn run_case(t: usize, ctx: usize, host_check: bool, iters: usize) {
    let nb = ctx.div_ceil(PAGE);
    // 过供给探针(缺省 = 与真值同 ⇒ 历史口径零变化)
    let plan_ctx = owl_shared::env_reader::parse::<usize>("FI_PF_PLAN_CTX").unwrap_or(ctx);
    let mut res = TestRes::new(owl_shared::env_reader::parse_or("OWL_TEST_DEVICE", 0));
    let mut exec = Exec::new();
    // 激活集 = 本探针要发射的那一个算子(A20/A16 之后:白名单外的声明不审
    // 孤儿 —— 见 registry::init_all_inner 的射程注)
    // 读+写都要在场:池审计 E10(读该池实例的算子必须有 ≥1 个写者)——
    // 本探针的池由 host 直灌,写者算子只为审计在场(白名单 = 生产档同款口径)
    let active: &[&str] = &[
        owl_kernels::family::flashinfer::PREFILL_FI_FP8KV,
        "owl_reshape_and_cache_fp8kv_knhd",
    ];
    let mut reg = owl_kernels::registry::OpRegistry::init_all_with(
        owl_kernels::server::family_factories(),
        &mut res,
        &mut exec,
        Some(active),
    )
    .expect("注册表装配");

    let pool = nb * HKV * HD * PAGE;
    let k_fp8 = lcg_u8(pool, 701);
    let v_fp8 = lcg_u8(pool, 702);
    let q_f16 = lcg_f16(t * H * HD, 703);

    let q_b = res.upload(&q_f16.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let k_b = res.upload(&k_fp8);
    let v_b = res.upload(&v_fp8);
    let q_cu = res.upload(&[0u32, t as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let indices: Vec<u32> = (0..nb as u32).collect();
    let idx_b = res.upload(&indices.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let indptr = res.upload(&[0u32, nb as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let last_len = ctx - (nb - 1) * PAGE;
    let ll_b = res.upload(&(last_len as u32).to_le_bytes());
    let wr_b = res.zeros(8);
    let out_b = res.zeros(t * H * HD * 2);

    let msg = LaunchMsg {
        kernel: KernelSource {
            name: owl_kernels::client::flashinfer::PREFILL_FI_FP8KV.to_string(),
            source: String::new(),
        },
        args: vec![
            Arg::Block { id: q_b.id, dtype: Dtype::F16, rank: 2 },
            Arg::Block { id: k_b.id, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: v_b.id, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: q_cu.id, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: idx_b.id, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: indptr.id, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: ll_b.id, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: wr_b.id, dtype: Dtype::F16, rank: 1 },
            Arg::Block { id: out_b.id, dtype: Dtype::F16, rank: 2 },
            Arg::U64(t as u64),      // total_rows
            // ⚠️ **过供给腿**(2026-10-18 图档桶化立案):`FI_PF_PLAN_CTX=<大 ctx>`
            // ⇒ plan 按大上界算(表布局/瓦片数按上界),而**池表四件套仍按真 ctx**
            // ⇒ 核内用真 kv_len 夹取、多出来的 kv_tile 落空 ⇒ 正是"信封上界 +
            // 运行期夹取"的模拟(图档 plan 烘图后的形态)。判据 = host oracle。
            Arg::U64(plan_ctx as u64), // ctx_total(plan 面)

            Arg::U64(t as u64),      // T
            Arg::U64(H as u64),      // hq
            Arg::U64(HKV as u64),    // hkv
            Arg::U64(HD as u64),     // hd
            Arg::U64(PAGE as u64),   // page
            Arg::U64(((1.0f32 / (HD as f32).sqrt()).to_bits()) as u64), // sm_scale 位型
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: t * H * HD,
    };
    reg.execute(&msg, &mut res, &mut exec).expect("FI run");
    res.stream.synchronize().expect("sync");

    // 计时(即发即计时,同 xqa_prefill_probe 口径)
    let mut bench = |n: usize| -> f64 {
        for _ in 0..3 {
            reg.execute(&msg, &mut res, &mut exec).expect("FI warm");
        }
        res.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..n {
            reg.execute(&msg, &mut res, &mut exec).expect("FI bench");
        }
        res.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / n as f64
    };
    let ms = bench(iters);

    let o = res.take_u16(out_b.id);
    let fx: Vec<f32> = o.iter().map(|&h| f16_to_f32(h)).collect();
    let mut line = format!("=== FI T={t} ctx={ctx} (nb={nb}) ===\n  FI: {ms:7.3} ms/call");
    if host_check {
        let host = host_ref_knhd(&q_f16, &k_fp8, &v_fp8, t, ctx);
        let d = fx
            .iter()
            .zip(host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // 逐行体检(独立抓 "row>0 产出空" 那条历史指控)
        let mut bad_rows = 0usize;
        for s in 0..t {
            let rd = fx[s * H * HD..(s + 1) * H * HD]
                .iter()
                .zip(&host[s * H * HD..(s + 1) * H * HD])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if rd > 1e-2 {
                bad_rows += 1;
            }
        }
        let nz = fx.iter().filter(|&&x| x == 0.0).count();
        line += &format!(
            "\n  max|FI−host| = {d:.6}   超门行数 = {bad_rows}/{t}   零值 = {nz}/{}",
            fx.len()
        );
        // 先打印再断言(2026-10-15:split 排查要看到"哪一行/哪一头错")
        println!("{line}");
        if d >= 1e-2 {
            for s in 0..t.min(8) {
                let rd = fx[s * H * HD..(s + 1) * H * HD]
                    .iter()
                    .zip(&host[s * H * HD..(s + 1) * H * HD])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let per_head: Vec<String> = (0..H.min(6))
                    .map(|h| {
                        let a = &fx[s * H * HD + h * HD..s * H * HD + (h + 1) * HD];
                        let b = &host[s * H * HD + h * HD..s * H * HD + (h + 1) * HD];
                        let m = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                        format!("{m:.4}")
                    })
                    .collect();
                println!("    row {s}: max|Δ|={rd:.6}  per-head(前6)={per_head:?}");
            }
            panic!("FI vs host 超门:{d:.6}(FI_SPLIT_KV={:?})", std::env::var("FI_SPLIT_KV"));
        }
    }
    if !host_check {
        println!("{line}");
    }
}

#[test]
fn fi_prefill_speed_and_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let only = owl_shared::env_reader::str("FI_PF_ONLY").unwrap_or_default();
    let host_all = owl_shared::env_reader::flag("FI_PF_HOST");
    println!(
        "[fi-probe] FI_SPLIT_KV={:?} FI_DEBUG={:?}",
        std::env::var("FI_SPLIT_KV").ok(),
        std::env::var("FI_DEBUG").ok()
    );
    // (T, ctx, host_check, iters)
    let cases: &[(usize, usize, bool, usize)] = &[
        // ── 小 T(FI_MIN_TOKENS=128 的缺口面:尾块 / 短 prompt)──
        // 这一档是 2026-10-14 的**真问题面**:现役尾块回落 `chunked knhd`
        // 在引擎里实测 6.7 ms/层(T=54, ctx 3126),而 XQA 只要 0.91 ms/层。
        (1, 3072, true, 50),
        (8, 3072, true, 50),
        // ── **末页半满**(2026-10-16 V1c 立案)──
        // 上面几档 ctx 都是 PAGE 的整数倍 ⇒ `last_len == PAGE`(整页)⇒
        // "末页半满"从未进过 host 校验档。而引擎的每一步 decode(T=1)都是
        // **末页半满**,且 27B+fp8 下 FI ≢ v2(0.8B f16 单页 ≡ ⇒ 唯一未覆盖
        // 组合 = fp8 × T=1 × 多页/半满)⇒ 本组是那一格的核级判据。
        (1, 53, true, 50),   // 2 页,last_len=21(引擎首步形状)
        (1, 3073, true, 50), // 97 页,last_len=1(极端半满)
        (1, 3080, true, 50), // 97 页,last_len=8
        (8, 3080, true, 50), // verify 形状 + 半满(T=8 也判一处)
        (1, 1057, true, 50), // 34 页,last_len=1
        (54, 3072, true, 50),
        (128, 3072, true, 50),
        // ── 大 T(生产主路径)──
        (128, 512, true, 30),
        (512, 512, false, 30),
        (512, 2048, false, 20),
        (512, 4096, false, 20),
        (512, 8192, false, 20),
        (512, 16384, false, 10),
        // ── **verify 形状**(2026-10-15 V1 立案:T=block 扩散草稿的 1 锚+K 行)──
        // 目标:把 XQA SPEC_DEC 在该形状的 1.54 ms/发(ctx32k)打下来。
        // 判据(探针面):≤0.3 ms/发;数值在短 ctx 档由 host oracle 背(见上方
        // (1,3072)/(8,3072) 两例,`FI_SPLIT_KV=1` 复跑即验 split 归约)。
        // decode 形状(q=1):现役 XQA spec-dec 在 ctx32k 是 0.91 ms/发(CUPTI 逐核)
        (1, 8192, false, 30),
        (1, 32768, false, 10),
        (2, 32768, false, 10),
        (4, 32768, false, 10),
        (8, 8192, false, 30),
        (8, 32768, false, 10),
        (16, 32768, false, 10),
        (32, 32768, false, 10),
    ];
    for &(t, ctx, hc, it) in cases {
        if !only.is_empty() && only != ctx.to_string() {
            continue;
        }
        let hc = hc || (host_all && t.saturating_mul(ctx) * H * HD <= 8_000_000_000);
        run_case(t, ctx, hc, it);
    }
}
