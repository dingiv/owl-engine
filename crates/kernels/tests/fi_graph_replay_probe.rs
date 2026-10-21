//! **M2a 探针:FI 图档 plan + 捕获/回放对拍**(设计 `docs/arch/prefill-graph-设计与实现.md`
//! §4.1c / §4.3d;2026-10-15 立)。
//!
//! 为什么必须有它:prefill 图(M2b)的**唯一未实测假设**是"FI 的核在线性捕获里
//! 能按**设备表**回放" —— plan 在 split 关时与 ctx 无关(源码级 F-5),回放期
//! ctx/页链全从设备表读。这条不成立就没必要往下做整段单图(退 piecewise)。
//! 本探针把它拆成三个可判定的子实验:
//!
//! | # | 实验 | 判据 |
//! |---|---|---|
//! | ① | **表件换 ctx 回放**:表件 A(ctx=512)捕获 ⇒ 回放前把表件改成 B(ctx=2048) | 回放(B) ≡ eager(B)(同几何 ⇒ **逐位**);且 B ≠ A(证明 ctx 真生效,不是"两边都没读") |
//! | ② | **假行 NaN**:桶 512 行里真行只有 64、其余填 **NaN**,`q_cu=[0,512]` 常量 | 真行**有限**且 ≡ eager(t=64)(容差)⇒ 假行不泄 NaN/不污染真行 |
//! | ③ | **尾部 CTA 掩码**(P-5):t=128 图档 plan ⇒ `padded=max(2·SM/hkv, tiles) > tiles` | 掩码后 ≡ eager(逐位);`FI_NO_MASK=1` 再回放 ⇒ **真行仍一致**(F-8:尾部 CTA 不写坏真行)+ 耗时对比(白跑取证) |
//!
//! 运行(**独立二进制**:图档 plan 由 env `FI_CUDA_GRAPH` 决定,本探针自带该
//! env,避免与 `fi_prefill_probe` 的基线档互相污染):
//! ```
//! OWL_TEST_DEVICE=0 cargo test -p owl-kernels --features device,flashinfer \
//!   --test fi_graph_replay_probe -- --nocapture
//! ```
//!
//! ⚠️ 捕获流程与生产同序:**先窗外预热**(plan 生成 ⇒ 捕获窗内 plan 命中;
//! 窗内 plan 未命中会被 `server/flashinfer.rs` 的硬门结构化拒 —— 本探针顺带
//! 验证该硬门**不在正常流程里误伤**)。
//!
//! ## 三条"不这么写就必失败"的姿态(全部实测踩过;生产侧早有护栏)
//! 1. **不能用 legacy 默认流**:`CudaContext::default_stream()` = stream 0,
//!    CUDA **不允许对它开捕获** ⇒ `begin_capture` 报
//!    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`(措辞误导:读起来像"已经在捕获");
//!    生产用三固定流,测试面自建 `ctx.new_stream()`。
//! 2. **必须关 cudarc event-tracking**(生产 `GpuCtx::new` 的 G0 护栏同款):
//!    开着它,`device_ptr()` 会在块操作上插 **event** ⇒ 窗内那次 FI run 直接
//!    返回 **901 `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`**(窗外同一次调用正常)。
//! 3. **窗内不许 sync**:`run()` 里那句 `stream.synchronize()` 是墙外收尾;
//!    窗内必须走 `exec()`(只发不等)——否则同样 invalidate 捕获。
//!
//! ## ③ 的时间口径(诚实标注)
//! ③ 打的 mask/`no_mask` 两档耗时是"`graph_launch` + `stream.synchronize`"
//! **单发同步口径** ⇒ 量到的是**回放开销地板**(~30µs),不是核时间;尾部 CTA
//! 的白跑成本请用 nsys/CUPTI 单量(设计 §4.1c F-6:本档 `padded=42 > tiles=12`)。
//! ③ 的**正确性**判据(掩码不改真行 / 关掩码也不写坏真行 = F-8)才是本步要钉的。
#![cfg(all(feature = "device", feature = "flashinfer"))]

mod fi_support;
use fi_support::{f16_to_f32, rewrite_block, RawGraph, TestRes, H, HD, HKV, PAGE};

use std::time::Instant;

use owl_kernels::contract::{Arg, Bytes, Dtype, KernelSource, LaunchMsg};
use owl_kernels::device::Exec;

const FI_OP: &str = owl_kernels::client::flashinfer::PREFILL_FI_FP8KV;

/// 有限 e4m3 字节(0x7F/0xFF 是 e4m3 的 NaN ⇒ 随机池里必须剔掉,否则数值比较
/// 会被 NaN 吞掉)
fn finite_e4m3(n: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let mut b = (s >> 24) as u8;
        if b == 0x7F || b == 0xFF {
            b = 0x01;
        }
        v.push(b);
    }
    v
}

fn rand_f16(n: usize, seed: u32) -> Vec<u16> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        // [−1,1) 内的小数:取 f16 位型的一个安全子集(指数 0x3800 = 2^-1 附近)
        v.push((0x3800 | ((s >> 16) as u16 & 0x03FF)) as u16);
    }
    v
}

/// 一个 FI 调用点:池/表件/输出块 + 可换表件
struct Fi {
    res: TestRes,
    exec: Exec,
    reg: owl_kernels::registry::OpRegistry,
    /// 池按 **max_ctx** 建一次(图指向它,回放期不许换块)
    k: Bytes,
    v: Bytes,
    q: Bytes,
    out: Bytes,
    q_cu: Bytes,
    /// 树序依赖边哑块(K0 写先于 FI 读池;FI 不解引用,见家族声明表)
    wr: Bytes,
    idx: Bytes,
    indptr: Bytes,
    ll: Bytes,
    rows: usize,
    /// 构造期上传的 q 位型(毒化假行时以它为底 —— 乱重建会把**真行**也换掉,
    /// 那样②就变成"两次不同的输入在比"(实测踩过:真行对不上,e+2 级))
    q_bits: Vec<u16>,
}

impl Fi {
    /// `out_bytes` = 输出块**字节数**(f16 ⇒ 元素 × 2;漏乘 = 核写出界)
    fn new(dev: usize, rows: usize, max_ctx: usize, out_bytes: usize, seed: u32) -> Self {
        use owl_kernels::registry::OpRegistry;
        let mut res = TestRes::new(dev);
        let mut exec = Exec::new();
        let active: &[&str] = &[FI_OP, "owl_reshape_and_cache_fp8kv_knhd"];
        let reg = OpRegistry::init_all_with(
            owl_kernels::server::family_factories(),
            &mut res,
            &mut exec,
            Some(active),
        )
        .expect("注册表装配");

        let nb = max_ctx.div_ceil(PAGE);
        let pool = nb * HKV * HD * PAGE;
        let k = res.upload(&finite_e4m3(pool, seed));
        let v = res.upload(&finite_e4m3(pool, seed.wrapping_add(1)));
        let q_bits = rand_f16(rows * H * HD, seed.wrapping_add(2));
        let q = res.upload(&q_bits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let out = res.zeros(out_bytes);
        let q_cu = res.upload(&[0u32, rows as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let wr = res.zeros(8);
        // 页链槽按 **max_ctx** 定容(回放期只改内容不换块 ⇒ 图里指针恒定)
        let idx = res.upload(
            &vec![0u32; nb].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        );
        let indptr = res.upload(&[0u32, 0].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let ll = res.upload(&0u32.to_le_bytes());
        Self { res, exec, reg, k, v, q, out, q_cu, wr, idx, indptr, ll, rows, q_bits }
    }

    /// 换表件到某个 ctx(= 桶回放期写输入槽;块地址不变、内容变)
    fn tables(&mut self, ctx: usize) {
        let nb = ctx.div_ceil(PAGE);
        let indices: Vec<u32> = (0..nb as u32).collect();
        rewrite_block(&mut self.res, &self.idx, &indices.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        rewrite_block(&mut self.res, &self.indptr, &[0u32, nb as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let last_len = (ctx - (nb - 1) * PAGE) as u32;
        rewrite_block(&mut self.res, &self.ll, &last_len.to_le_bytes());
        rewrite_block(&mut self.res, &self.q_cu, &[0u32, self.rows as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }

    /// 覆写 q 的行区间(假行实验:把 `[from, rows)` 填 NaN)
    fn poison_q_rows(&mut self, from: usize, to: usize) {
        let mut bits: Vec<u16> = self.q_bits.clone();
        for r in from..to.min(self.rows) {
            for e in 0..H * HD {
                bits[r * H * HD + e] = 0x7E00; // f16 NaN
            }
        }
        let bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        rewrite_block(&mut self.res, &self.q, &bytes);
    }

    fn msg(&self, ctx: usize) -> LaunchMsg {
        self.msg_geom(self.rows, self.rows, ctx)
    }

    /// 几何可变的消息构造(`total_rows`/`T` 进 plan 键 ⇒ 换它们 = 换 plan;
    /// 块尺寸仍按本 fixture 的 `rows` —— 探针④只需要"另一条 plan 被算出来
    /// 并推荐 int_ws",不需要那个形状本身几何自洽)。
    fn msg_geom(&self, total_rows: usize, t: usize, ctx: usize) -> LaunchMsg {
        LaunchMsg {
            kernel: KernelSource { name: FI_OP.to_string(), source: String::new() },
            args: vec![
                Arg::Block { id: self.q.id, dtype: Dtype::F16, rank: 2 },
                Arg::Block { id: self.k.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: self.v.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: self.q_cu.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: self.idx.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: self.indptr.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: self.ll.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: self.wr.id, dtype: Dtype::F16, rank: 1 },
                Arg::Block { id: self.out.id, dtype: Dtype::F16, rank: 1 },
                Arg::U64(total_rows as u64), // total_rows
                Arg::U64(ctx as u64),        // ctx_total
                Arg::U64(t as u64),          // T
                Arg::U64(H as u64),
                Arg::U64(HKV as u64),
                Arg::U64(HD as u64),
                Arg::U64(PAGE as u64),
                Arg::U64((1.0f32 / (HD as f32).sqrt()).to_bits() as u64),
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: t * H * HD,
        }
    }

    /// **只发不等**(捕获窗内**必须**走这条 —— 窗内 `cudaStreamSynchronize`
    /// 非法,会把捕获直接 invalidate:实测 `run err 901`)。
    fn exec(&mut self, ctx: usize) {
        let msg = self.msg(ctx);
        self.reg.execute(&msg, &mut self.res, &mut self.exec).expect("FI run");
    }

    /// 发一条**别的几何**的消息(同 registry ⇒ 同 FiRuntime/int_ws;
    /// 探针④用它把 int_ws 刷成另一条 plan)
    fn exec_geom(&mut self, total_rows: usize, t: usize, ctx: usize) {
        let msg = self.msg_geom(total_rows, t, ctx);
        self.reg.execute(&msg, &mut self.res, &mut self.exec).expect("FI run(几何变体)");
    }

    /// 窗外:eager 参考 / 捕获后的收割
    fn run(&mut self, ctx: usize) {
        self.exec(ctx);
        self.res.stream.synchronize().expect("sync");
    }

    /// 只读真行的前 `rows` 行(桶实验里 out 的布局是 t_pad 行;真行在前)
    fn read(&self, rows: usize) -> Vec<f32> {
        let o = self.res.take_u16(self.out.id);
        o[..rows * H * HD].iter().map(|&h| f16_to_f32(h)).collect()
    }
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

#[test]
fn fi_graph_replay_m2a() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = owl_cuda::test_device_ordinal();
    // 本探针**自带图档 env**(独立二进制 ⇒ 无并行踩踏;与 C++ 侧同源同读法)
    std::env::set_var("FI_CUDA_GRAPH", "1");
    owl_kernels::server::flashinfer::set_plan_mode(true);
    std::env::remove_var("FI_NO_MASK");
    println!("=== M2a:FI 图档 plan + 捕获/回放(FI_CUDA_GRAPH=1)===");

    // ── ① 表件换 ctx 回放 ────────────────────────────────────────────
    {
        let t = 512usize;
        let mut f = Fi::new(dev, t, 2048, t * H * HD * 2, 701);
        f.tables(512);
        eprintln!("[①] fixture 就绪;窗外预热");
        f.run(512); // 窗外预热(生产 = GraphPlan warmup;窗内 plan 未命中会被硬门拒)
        eprintln!("[①] 预热完成;开捕获");
        RawGraph::begin(&f.res.ctx, &f.res.stream).expect("begin_capture");
        eprintln!("[①] 捕获中:发 FI");
        f.exec(512); // 窗内:plan 命中(零 host 分配);**不 sync**(窗内非法)
        eprintln!("[①] 收捕获");
        let g = RawGraph::end(&f.res.ctx, &f.res.stream).expect("end_capture");
        g.launch(&f.res.ctx, &f.res.stream).expect("launch");
        f.res.stream.synchronize().expect("sync");
        let a_replay = f.read(t);
        f.run(512);
        let a_eager = f.read(t);
        let d_same = maxdiff(&a_replay, &a_eager);

        // 换表件到 ctx=2048,回放(host 零参与 —— 图里读的就是这块设备表)
        f.tables(2048);
        g.launch(&f.res.ctx, &f.res.stream).expect("launch B");
        f.res.stream.synchronize().expect("sync B");
        let b_replay = f.read(t);
        f.run(2048);
        let b_eager = f.read(t);
        let d_swap = maxdiff(&b_replay, &b_eager);
        let d_ctx = maxdiff(&b_eager, &a_eager);

        println!(
            "[①] 表件换 ctx:回放(A)≡eager(A) {:+.2e} | 回放(B)≡eager(B) {:+.2e} | B≠A {:.2e}",
            d_same, d_swap, d_ctx
        );
        assert!(d_same < 1e-6, "①a 同 ctx 回放应逐位:{d_same:.3e}");
        assert!(d_swap < 1e-6, "①b 换表件回放应≡eager(F-5 未成立?):{d_swap:.3e}");
        assert!(d_ctx > 1e-3, "①c ctx 必须真生效(否则本实验是空转):{d_ctx:.3e}");
    }

    // ── ② 假行 NaN(桶 512 装 64 真行)──────────────────────────────
    {
        let t_pad = 512usize;
        let t_real = 64usize;
        let mut f = Fi::new(dev, t_pad, 512, t_pad * H * HD * 2, 711);
        f.tables(512);
        f.poison_q_rows(t_real, t_pad); // 假行 = NaN(最恶劣的污染源)
        f.run(512); // 预热 + 捕获前参考
        RawGraph::begin(&f.res.ctx, &f.res.stream).expect("begin");
        f.exec(512);
        let g = RawGraph::end(&f.res.ctx, &f.res.stream).expect("end");
        g.launch(&f.res.ctx, &f.res.stream).expect("launch");
        f.res.stream.synchronize().expect("sync");
        let padded = f.read(t_pad); // 全 512 行(含假行)
        let real_rows = &padded[..t_real * H * HD];

        // 参考:真行 t=64、ctx=64(掩码算术在两档下对真行都给出 kv ≤ i ⇒ 同可见集)
        let mut r = Fi::new(dev, t_real, 512, t_real * H * HD * 2, 711);
        r.tables(64);
        r.run(64);
        let ref_rows = r.read(t_real);
        let d_real = maxdiff(real_rows, &ref_rows);

        println!(
            "[②] 假行 NaN:t_real={t_real} 桶={t_pad} ⇒ 真行 finite={} 与 eager(t={t_real}) {:+.2e}",
            finite(real_rows),
            d_real
        );
        assert!(finite(real_rows), "② 假行 NaN 泄进了真行(违反 design §4.2 律①)");
        assert!(d_real < 1e-2, "② 真行与真 t 参考不符:{d_real:.3e}");
    }

    // ── ③ 尾部 CTA 掩码(P-5)────────────────────────────────────────
    {
        let t = 128usize;
        let mut f = Fi::new(dev, t, 128, t * H * HD * 2, 721);
        f.tables(128);
        f.run(128); // 预热(plan 含 mask 生成)
        RawGraph::begin(&f.res.ctx, &f.res.stream).expect("begin");
        f.exec(128);
        let g = RawGraph::end(&f.res.ctx, &f.res.stream).expect("end");

        let t_masked = Instant::now();
        g.launch(&f.res.ctx, &f.res.stream).expect("launch");
        f.res.stream.synchronize().expect("sync");
        let masked_ms = t_masked.elapsed().as_secs_f64() * 1e3;
        let out_masked = f.read(t);
        f.run(128);
        let out_ref = f.read(t);
        let d_mask = maxdiff(&out_masked, &out_ref);

        // 取证档:关掩码 ⇒ 尾部 CTA 真跑(白跑);真行仍不得被写坏(F-8)
        std::env::set_var("FI_NO_MASK", "1");
        let t_unmasked = Instant::now();
        g.launch(&f.res.ctx, &f.res.stream).expect("launch(no mask)");
        f.res.stream.synchronize().expect("sync");
        let unmasked_ms = t_unmasked.elapsed().as_secs_f64() * 1e3;
        let out_unmasked = f.read(t);
        std::env::remove_var("FI_NO_MASK");
        let d_nomask = maxdiff(&out_unmasked, &out_ref);

        println!(
            "[③] 尾掩码:t=128 图档(padded>tiles)⇒ 掩码回放≡eager {:+.2e} | 关掩码真行≡eager {:+.2e} | 回放耗时 mask={masked_ms:.3}ms no_mask={unmasked_ms:.3}ms",
            d_mask, d_nomask
        );
        assert!(d_mask < 1e-6, "③a 掩码回放应≡eager:{d_mask:.3e}");
        assert!(d_nomask < 1e-6, "③b 尾部 CTA 写坏了真行(F-8 被否证):{d_nomask:.3e}");
        assert!(unmasked_ms >= 0.0 && masked_ms >= 0.0, "③c 计时口径自检(地板口径,见头注)");
    }

    // ── ④ int_ws 自携(多 plan 共存;M2b 2026-10-15)─────────────────────
    //
    // 病:int_ws 是**唯一共享区**(每次 `owl_fi_prefill_plan` 都把调度表
    // memcpy 进同一个 int_ws),而图里烘的是 `int_ws + offset` 指针 ⇒ "最后
    // 一个 plan"决定所有图读到什么(多桶/草稿面共存皆可踩)。
    // 判据:A 捕获后,用**别的形状**把 int_ws 刷掉,再回放 A ⇒ 仍 ≡ A 的
    // eager 基线(即 A 的图自携了 A 的 plan)。
    // 取证档:`owl_kernels::server::flashinfer::set_plan_restore(false)` 下本
    // 判据**必须红**(机制承重证据)。⚠️ 该档 2026-10-18 从 env 收编成
    // **进程级 setter**(配置 `[dispatch] fi_plan_restore` 为唯一写者);
    // 探针要跑红档就手动 `set_plan_restore(false)` 再跑本腿(见 README/注释)。
    {
        let t = 128usize;
        let mut f = Fi::new(dev, t, 128, t * H * HD * 2, 731);
        f.tables(128);
        f.run(128); // 预热:plan A 入条目表
        f.res.capture_flag = true; // 窗内:激活 upload 会被烘进图(= 自携)
        RawGraph::begin(&f.res.ctx, &f.res.stream).expect("begin");
        f.exec(128);
        let g = RawGraph::end(&f.res.ctx, &f.res.stream).expect("end");
        f.res.capture_flag = false;
        g.launch(&f.res.ctx, &f.res.stream).expect("launch base");
        f.res.stream.synchronize().expect("sync");
        let base = f.read(t);
        // **结构判据**(权威):int_ws 内容 —— 回放后必须被刷回 A 的 plan 快照。
        // 为什么不用"输出相等"当唯一判据:本档(split 关 + batch=1)的 plan 数组
        // 内容是**形状无关**的(request_indices 全 0、qo_tile_indices 恒等),
        // 输出在中毒时也可能碰巧相等 —— 那正是"靠陈旧表项恰好无害"的静默依赖
        // (设计 §3.2 明列不做)。直接读 int_ws 才能把机制证成承重件。
        let int_ws = f
            .res
            .block_id_of_len(32 * 1024 * 1024)
            .expect("int_ws(32MiB)在册");
        let before = f.res.take_bytes(int_ws);

        // 另一形状(**同 registry** ⇒ 同 int_ws):64 行几何 ⇒ 新 plan ⇒ 冲 int_ws
        // (选与 A **同 padded** 的形状:计划数组的"值"会变(o_indptr/kv_chunk_size),
        //  但**不会**出现越界/重排索引 ⇒ 取证档红的时候也不会跑成病理慢速)
        f.exec_geom(64, 64, 64);
        f.res.stream.synchronize().expect("sync B");
        let after_b = f.res.take_bytes(int_ws);
        let first_diff = (0..2048).find(|&i| after_b[i] != before[i]);
        let b_differs = first_diff.is_some();

        g.launch(&f.res.ctx, &f.res.stream).expect("launch after switch");
        f.res.stream.synchronize().expect("sync");
        let after = f.read(t);
        let restored = f.res.take_bytes(int_ws);
        let ws_restored = restored[..2048] == before[..2048];
        let d = maxdiff(&base, &after);
        println!(
            "[④] 多 plan 共存:B 调用后 int_ws 变了={b_differs}(首个差异字节 {first_diff:?}) | 回放 A 后 int_ws 复位={ws_restored} | 输出 |Δ| = {d:.3e}(取证档 set_plan_restore(false) 应红)"
        );
        assert!(b_differs, "④ 前提不成立:B 的 plan 没写进 int_ws(本实验是空转)");
        assert!(ws_restored, "④ int_ws 自携失败(回放读到了别的 plan 的调度表)");
        assert!(d < 1e-6, "④ 回放输出被别的 plan 影响:{d:.3e}");
    }

    println!("=== M2a 全绿:整段单图(§4.1 的 F-5/F-6/F-8)三条假设实测成立 ===");
}
