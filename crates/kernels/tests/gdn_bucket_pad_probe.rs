//! **P-6b 红门探针**:桶 padding 下 `gdn_chunked` 的**假行污染**(2026-10-15 立)。
//!
//! # 病(设计 §4.3e P-6b)
//! `gdn_chunked` 六核(FLA AOT)的行区间**完全由三张设备表**决定,而三张表今天
//! 由 runtime **按传入的 `t`** 在内部生成:
//!
//! ```text
//! cu   = [0, t]                       ← 序列边界
//! coff = [0, nt],  nt = ceil(t/64)    ← 每序列块数前缀和(chunk_delta_h 的 h 缓冲偏移)
//! idx  = (0, i) for i < nt            ← (seq_id, chunk_id_in_seq),FLA prepare_chunk_indices 语义
//! grid = (nt, hv, 1)
//! ```
//!
//! eager 路 `t = 真行数` ⇒ 一切正常。**桶几何下 `t = t_pad`** ⇒ 假行被当成序列的
//! 一部分走完 ⇒ ①`chunk_delta_h` 把**递推终态**落在假行末端 ②假行的 k/v/β/g 参与
//! 递推 ⇒ **下一块读到的 rec 态是脏的**(跨 chunk 累积、静默)。
//!
//! # 判据(本探针要绿的命题)
//! 同一批真行(前 64 行)在两种布局下跑:
//! - **真行布局**:`t = 64`(nt=1),64 行全真;
//! - **桶布局**:`t = 128`(nt=2),前 64 行真、后 64 行 = **有限垃圾**(仿真:桶
//!   padding 的假行来自被 padding 的激活,不是零),
//! 则 **真行输出** 与 **终态 rec** 必须逐位一致。
//!
//! ⚠️ 修复前本探针**必须红**(终态不一致)—— 红=病在,绿=修好。修复 = 把三张表
//! 从 runtime 私有缓存变成**调用入参**(引擎按 `t_real` 写持久槽),与 FLA 原始
//! API(`cu_seqlens`/`chunk_indices` 入参)一致。
#![cfg(feature = "device")]

mod fi_support;
use fi_support::TestRes;

use owl_kernels::client::gdn_chunked::{GdnChunkedCall, GdnShape};
use owl_kernels::contract::Bytes;
use owl_kernels::device::Exec;
use owl_kernels::registry::KernelFamily;
use owl_kernels::server::gdn_chunked::GdnChunkedRuntime;

const NV: usize = 48;
const NK: usize = 16;
const KD: usize = 128;
const T_REAL: usize = 64;
const T_PAD: usize = 128;

fn lcg(seed: u32, i: usize) -> f32 {
    let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    ((x >> 8) as f32) / 16_777_216.0
}

fn norm_rows(xs: &mut [f32], rows: usize, heads: usize, d: usize) {
    for r in 0..rows {
        for h in 0..heads {
            let base = r * heads * d + h * d;
            let n: f32 = xs[base..base + d].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for i in base..base + d {
                xs[i] /= n;
            }
        }
    }
}

fn f16le(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()
}

/// 一次"桶布局"实验:真行 [0,64) 与布局 [0,t) 的其余行分别由 `real`/`fake` 填充
struct Case {
    res: TestRes,
    exec: Exec,
    rt: Box<dyn KernelFamily>,
    q: Bytes,
    k: Bytes,
    v: Bytes,
    beta: Bytes,
    g: Bytes,
    state: Bytes,
    out: Bytes,
    /// 序列表块(cu|coff|idx;内容按 **t_real** —— P-6b 的核心)
    tables: Bytes,
    nt: usize,
    t: usize,
}

impl Case {
    /// `nt_layout`:false = 表件按**真**块数铸(nt=ceil(t_real/BT);P-6b 原判据);
    /// true = 表件按**布局**块数铸(nt_layout=ceil(t/BT),coff/idx 铺满布局 ——
    /// 这正是**图内 `nt` 烘死**时引擎必须写的口径:核的 grid = nt_layout,
    /// 多出来的尾部 chunk 是"越界 chunk"(行区间 > t_real ⇒ 读写都被 T 夹回)。
    fn new(dev: usize, t: usize, fake: bool, nt_layout: bool) -> Self {
        let mut res = TestRes::new(dev);
        let mut exec = Exec::new();
        let rt = <GdnChunkedRuntime as KernelFamily>::assemble(&mut res, &mut exec)
            .expect("gdn_chunked runtime assemble");

        // 真行(与布局无关 ⇒ 两种布局下逐位相同)
        let mut q = vec![0f32; t * NK * KD];
        let mut k = vec![0f32; t * NK * KD];
        let mut v = vec![0f32; t * NV * KD];
        let mut beta = vec![0f32; t * NV];
        let mut g = vec![0f32; t * NV];
        for i in 0..t * NK * KD {
            q[i] = (lcg(11, i) - 0.5) * 0.2;
            k[i] = (lcg(23, i) - 0.5) * 0.2;
        }
        for i in 0..t * NV * KD {
            v[i] = (lcg(37, i) - 0.5) * 0.2 * 2.0;
        }
        for i in 0..t * NV {
            beta[i] = 0.1 + 0.8 * lcg(53, i);
            g[i] = -(0.01 + 8.0 * lcg(71, i));
        }
        norm_rows(&mut q, t, NK, KD);
        norm_rows(&mut k, t, NK, KD);
        if fake {
            // 假行 = **有限垃圾**(仿真桶 padding:假行来自被 padding 的激活,
            // 不是零 ⇒ 会真参与递推)—— 用另一条 LCG 流,量级与真行同阶
            for i in T_REAL * NK * KD..t * NK * KD {
                q[i] = (lcg(911, i) - 0.5) * 0.2;
                k[i] = (lcg(913, i) - 0.5) * 0.2;
            }
            norm_rows(&mut q[T_REAL * NK * KD..], t - T_REAL, NK, KD);
            norm_rows(&mut k[T_REAL * NK * KD..], t - T_REAL, NK, KD);
            for i in T_REAL * NV * KD..t * NV * KD {
                v[i] = (lcg(937, i) - 0.5) * 0.2 * 2.0;
            }
            for i in T_REAL * NV..t * NV {
                beta[i] = 0.1 + 0.8 * lcg(953, i);
                g[i] = -(0.01 + 8.0 * lcg(971, i));
            }
        }

        // 句柄账长口径:builder 的 check 认"元素数",而 TestRes::upload 记的是
        // 字节数 ⇒ 用 **len=0 叶子占位**(家族侧硬律:len=0 = 裸 id 占位,放行;
        // 见 client/gdn_chunked.rs::check 注释),与 ops_gdn_chunked_chain 同款。
        let leaf = |mut b: Bytes, dtype, rank| -> Bytes {
            b.len = 0;
            b.dtype = dtype;
            b.rank = rank;
            b
        };
        let q_b = leaf(res.upload(&f16le(&q)), owl_kernels::contract::Dtype::F16, 3);
        let k_b = leaf(res.upload(&f16le(&k)), owl_kernels::contract::Dtype::F16, 3);
        let v_b = leaf(res.upload(&f16le(&v)), owl_kernels::contract::Dtype::F16, 3);
        let beta_b = leaf(res.upload(&f16le(&beta)), owl_kernels::contract::Dtype::F16, 2);
        let g_b = leaf(res.upload(&f16le(&g)), owl_kernels::contract::Dtype::F16, 2);
        // state 池 [K,V] f32(零态起点;runtime 内部 trans-in/out)
        let state = leaf(res.zeros(NV * KD * KD * 4), owl_kernels::contract::Dtype::F32, 4);
        let out = leaf(res.zeros(t * NV * KD * 2), owl_kernels::contract::Dtype::F16, 3);
        // **序列表按真长度铸**(P-6b):cu[0,t_real] 恒真;`nt`(= coff/idx 的
        // 铺开长度)在两种口径间切:真块数(P-6b 判据)或布局块数(图姿态)。
        let bt = owl_kernels::family::gdn_chunked::BT;
        let nt = if nt_layout { t.div_ceil(bt) } else { T_REAL.div_ceil(bt) };
        let tables = leaf(
            res.upload(&owl_kernels::family::gdn_chunked::table_host_bytes(T_REAL, nt)),
            owl_kernels::contract::Dtype::U32,
            1,
        );
        Self { res, exec, rt, q: q_b, k: k_b, v: v_b, beta: beta_b, g: g_b, state, out, tables, nt, t }
    }

    fn run(&mut self) {
        let shape = GdnShape { t: self.t as u32, nv: NV as u32, nk: NK as u32, kd: KD as u32 };
        let call = GdnChunkedCall::builder(shape)
            .q(&self.q).expect("q 契约")
            .k(&self.k).expect("k 契约")
            .v(&self.v).expect("v 契约")
            .beta(&self.beta).expect("beta 契约")
            .gate_raw(&self.g).expect("gate 契约")
            .state(&self.state, 0).expect("state 契约")
            .tables(&self.tables, self.nt).expect("tables 契约")
            .build().expect("家族调用装配");
        let msg = call.to_launch(self.out.clone());
        let _ = self.rt.run(&msg, &mut self.res, &mut self.exec).expect("gdn_chunked run");
        self.res.stream.synchronize().expect("sync");
    }

    fn state_f32(&self) -> Vec<f32> {
        self.res.take_f32(self.state.id)
    }

    fn out_f32(&self, rows: usize) -> Vec<f32> {
        self.res.take_u16(self.out.id)[..rows * NV * KD]
            .iter()
            .map(|&h| half::f16::from_bits(h).to_f32())
            .collect()
    }
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn gdn_bucket_pad_does_not_pollute_state() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = owl_cuda::test_device_ordinal();

    // 真行布局(t = 64,nt = 1)
    let mut real = Case::new(dev, T_REAL, false, false);
    real.run();
    let real_out = real.out_f32(T_REAL);
    let real_state = real.state_f32();

    // 桶布局(t = 128,nt = 2;后 64 行 = 有限垃圾)
    let mut padded = Case::new(dev, T_PAD, true, false);
    padded.run();
    let pad_out = padded.out_f32(T_REAL);
    let pad_state = padded.state_f32();

    let d_out = maxdiff(&real_out, &pad_out);
    let d_state = maxdiff(&real_state, &pad_state);
    println!(
        "=== P-6b 判据(t_real={T_REAL} 桶={T_PAD})===\n  真行输出 |Δ| = {d_out:.3e}\n  终态 rec  |Δ| = {d_state:.3e}"
    );

    // 真行输出:两种布局下都应由"本块内"决定(row-local)⇒ 期望一致
    assert!(d_out < 1e-3, "P-6b:真行输出被假行影响(Δ={d_out:.3e})");
    // **终态**:假行参与了递推 ⇒ 修复前这里必红
    assert!(
        d_state < 1e-3,
        "P-6b 红:假行推进了 GDN 递推态(终态 Δ={d_state:.3e})—— \
         桶几何下 t_pad 被当成序列长度;修法 = 三张表(cu/coff/idx)从 runtime 私有缓存 \
         改为调用入参,由引擎按 t_real 写持久槽(设计 §4.3e)"
    );
}

/// **图姿态判据(M2b)**:布局 `t` 更大 **且 `nt` = 布局块数**(表件 coff/idx
/// 铺满布局,`cu` 仍是 `[0, t_real]`)—— 这是 prefill 桶图**必须**用的口径
/// (图内 `nt` 是烘死的标量,不可能随 chunk 变)。
///
/// 与 `nt = 真块数` 的差别:grid 里多出 `nt_layout − nt_real` 个 chunk CTA,
/// 它们的 `i_t*BT ≥ t_real`(行区间越界)⇒ 必须**读被零填/写被丢**。本测就是
/// 把这条"白跑不得污染"钉死:真行输出与终态仍须 ≡ 真块数口径。
#[test]
fn gdn_bucket_pad_graph_posture_nt_layout_is_inert() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = owl_cuda::test_device_ordinal();

    // 参照:真行布局(t=64,nt=1,无假行)
    let mut real = Case::new(dev, T_REAL, false, false);
    real.run();
    let real_out = real.out_f32(T_REAL);
    let real_state = real.state_f32();

    // 图姿态:t=128 布局、后 64 行假行、nt = 2(布局块数)
    let mut padded = Case::new(dev, T_PAD, true, true);
    padded.run();
    let pad_out = padded.out_f32(T_REAL);
    let pad_state = padded.state_f32();

    let d_out = maxdiff(&real_out, &pad_out);
    let d_state = maxdiff(&real_state, &pad_state);
    println!(
        "=== 图姿态判据(t_real={T_REAL} 桶={T_PAD} nt_layout={})===\n  真行输出 |Δ| = {d_out:.3e}\n  终态 rec  |Δ| = {d_state:.3e}",
        T_PAD.div_ceil(owl_kernels::family::gdn_chunked::BT)
    );
    assert!(d_out < 1e-3, "图姿态:真行输出被越界 chunk 影响(Δ={d_out:.3e})");
    assert!(d_state < 1e-3, "图姿态:终态被越界 chunk 改写(Δ={d_state:.3e})—— 桶图不能这么烘 nt");
}
