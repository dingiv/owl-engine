//! Linear:仿射层(容器 + LoaderOps 装载 + Matmul;无 bias —— Qwen3.5
//! 全系 attention/mlp 权重无 bias)。
//!
//! 权重槽 = **原生布局直读**(2026-09-26 F5-2:检查点 [out, in] 行主序
//! 原样装载,forward matmul_nt —— cuBLAS OP_T 免费转置,host 转置路径
//! 全删;旧"装载期转置"是 naive-matmul 时代产物)。

use crate::contract::Dtype;
use crate::ops::SemanticKernel;
use crate::layers::narrow_strided;
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, QuantPlan, Weight};
use crate::formats::gguf::GgmlTy;
use crate::formats::util::marlin_n_pack;
use crate::TensorOps;

/// **GGUF 块型 GEMM**(两发、零解量化中间态;设计 §四 L5 calltree)
///
/// ① `GgufQuantAct{_F16,_Bf16}`(激活 → q8_1 块:块内 warp 归约出 scale+sum);
/// ② `GgufGemm`(权重块字节原样读,`vec_dot` 在核内解量化)。
/// 权重**压缩驻留**(18G 的 27B Q4_K_M 可上单卡,而 f16 需 54G)。
///
/// **唯一实现 · 两个消费者**:[`Linear::forward_inner`](普通权重)与
/// [`Embedding::lm_head_matmul`](tied 量化表兼输出头 —— 同一张 `[vocab, hidden]`
/// 表既查表又做头 GEMM,设计 §11.5 ②)。两份实现 = 接口不唯一 ⇒ 这里只有一个。
///
/// 槽序/几何见 `ops/gguf.rs` 与 `family/gguf.rs::gemm_geom`;
/// 激活 dtype 分档(输出恒 `U8`,`dt` 载不了输入档 ⇒ 两个 op_id 各一档)。
/// **prefill 复用臂的最小 token 数**(W-2;`m ≥ 本值` 才切调度)。
///
/// 为什么是 8:decode(m = 1)与投机 verify(m = 2..7)都留在 MMVQ —— 那条路径是
/// 带宽下限(读一遍权重),复用臂的小窗只会多几条 guard;prefill 的 chunk
/// (128/512/1024)才谈得上复用。
pub(crate) const GGUF_TILE_MIN_M: usize = 8;

/// `true` = **小张量 + 未用满 token 窗** ⇒ 让 MMVQ 赢。
///
/// 判据两条(缺一不切):
/// 1. `m ≤ TILE_MT` —— 否则 tile 的权重复用是真的省带宽(MMVQ 要重读 `m` 遍);
/// 2. 整张量 `n·k·(块字节/块元素)` ≤ 配置的 L2 上限 —— 重读落在 L2 上。
fn gguf_prefers_mmvq_small_n(ty: GgmlTy, m: usize, n: usize, k: usize, l2_mib: usize) -> bool {
    use owl_kernels::family::gguf::TILE_MT;
    if l2_mib == 0 || m > TILE_MT {
        return false;
    }
    let (blk_elems, blk_bytes) = ty.block();
    if blk_elems == 0 {
        return false;
    }
    n.saturating_mul(k) / blk_elems * blk_bytes <= (l2_mib << 20)
}

/// **复用臂开关**(`[kernel] gguf_tile`;缺省开)。
///
/// 关 ⇒ 全档回 MMVQ(与 W-2 之前逐位相同)—— A/B 测 prefill
/// 收益、以及"怀疑复用臂"时的一键回退。**不影响图形态之外的任何语义**:
/// 两臂位级相同(门:`kernel_gate::gpu_gguf_tile_gemm_matches_mmvq_bitwise`)。
/// 2026-10-21:原 `OWL_GGUF_TILE` env 收编入配置,`env.gguf.tile`。
/// **dequant+cuBLAS 臂的最小 token 数**(W-2d)。上游按 `ne11 ≥ 64` 分流
/// (`ggml_cuda_should_use_mmq` 的 NVIDIA 条件)⇒ 本仓同门槛。
pub(crate) const GGUF_CUBLAS_MIN_M: usize = 64;

/// dequant+cuBLAS 臂开关(`[dispatch] gguf_cublas`;缺省开)。
/// 原 `OWL_GGUF_CUBLAS` env(2026-10-21 收编⇒`env.gguf.cublas`)。

/// **head 版 int8-MMA MMQ 的最小 token 数**(W-2f;2026-10-21 重定)。
///
/// 原 = 64(J=64 打底,只服务 prefill)。实测(27B UD-IQ4_XS,29 MB IQ4_XS 张量):
/// MMVQ 的权重流量 = `m × 字节数` ⇒ m=8 时 238 µs,而复用臂 153 µs、
/// **MMQ2 81 µs(m=8..64 近乎恒定)**;m=2 时 MMVQ 66 µs 仍胜 ⇒ 交叉点 ≈ 4。
/// 详见 `family::gguf::MMQ2_SMALL_M` 的实测表。decode(m=1)不变。
pub(crate) const GGUF_MMQ2_MIN_M: usize = owl_kernels::family::gguf::MMQ2_SMALL_M;

/// int8-MMA MMQ 开关(`[dispatch] gguf_mmq2`;缺省开)。
/// 原 `OWL_GGUF_MMQ2` env(2026-10-21 收编⇒`env.gguf.mmq2`)。

/// **官方 MMQ 的最小 token 数**(W-2c)。
///
/// 上游的批分界:MMVQ 用于小批、MMQ 用于大批(`MMVQ_MAX_BATCH_SIZE = 8`)。
/// 本仓取 32:tile 是 `x=4, y=32`(每 CTA 4 个 token),`m = 32` 时 m 方向 8 个
/// CTA、每 CTA 打满 4 token —— 再小就退化成"小批走大 tile",不如 MMVQ。
pub(crate) const GGUF_MMQ_MIN_M: usize = 32;

/// MMQ 臂开关(`[dispatch] gguf_mmq`;缺省开)。
/// 原 `OWL_GGUF_MMQ` env(2026-10-21 收编⇒`env.gguf.mmq`)。

/// **smem 复用臂的最小 token 数**(W-2b)。窗 = `MT = 64`,故 `m ≥ 64` 才吃满;
/// 8..63 仍是 vec_dot 复用臂(1.65×),decode/verify 仍 MMVQ。
pub(crate) const GGUF_SMEM_MIN_M: usize = 64;

/// smem 臂开关(`[dispatch] gguf_smem`;缺省开)。
/// 原 `OWL_GGUF_SMEM` env(2026-10-21 收编⇒`env.gguf.smem`)。
///
/// ⚠️ 两臂**不位级相同**(smem 臂喂原始 f16 激活,MMVQ/复用臂喂 q8_1)⇒
/// 切换只影响数值末位;门 = `kernel_gate::gpu_gguf_smem_gemm_matches_mmvq`。

#[allow(clippy::too_many_arguments)]
pub(crate) fn gguf_quant_gemm(
    w: &Weight,
    ty: GgmlTy,
    xs: &TensorOps,
    out_dim: usize,
    in_dim: usize,
    col_index: Option<&Weight>,
    arms: &crate::env::GgufEnv,
) -> TensorOps {
    // 形状律:xs = [.., in](空形状 = 声明失守 ⇒ **毒化**,不 panic)
    if xs.shape().is_empty() {
        return TensorOps::poisoned(Dtype::F16, vec![out_dim], "GGUF GEMM: xs 形状为空(需 [.., in])");
    }
    let m: usize = xs.shape()[..xs.shape().len() - 1].iter().product();
    let k = in_dim;
    // **内维列序 gather**(W-2e;`ssm_out` tiled → grouped)
    //
    // `ssm_out` 的内维(v-head 轴)是 ggml **tiled** 序。此前这条置换折进
    // 激活量化核(`GgufQuantActTiled*`),但那只覆盖 MMVQ/tile 两臂 ——
    // cuBLAS/MMQ 臂直吃 f16/q8_1 激活 ⇒ `ssm_out` 只能退到自研 tile 臂
    // (实测 2.67 ms/次、占 prefill 19%;设计 §19.5 定位 ①)。
    // 现在改成:在 f16 激活上做**一次块级 gather**(m×k×2B 拷贝 ≈ 20 µs),
    // 置换后与普通层**完全同路** ⇒ 臂集不再分叉(各臂的门原样适用)。
    // bf16 档仍走既有 tiled 量化折叠(不新增 bf16 gather 臂)。
    let permuted;
    let (xs, col_index) = match col_index {
        Some(t) if xs.dtype == Dtype::F16 => {
            permuted = TensorOps::call(SemanticKernel::GgufGatherCols)
                .arg(xs)
                .arg(&t.decl())
                .arg_i32(m as i32)
                .arg_i32(k as i32)
                .aux(&[m, k])
                .with_shape(Dtype::F16, xs.shape().to_vec());
            (&permuted, None)
        }
        other => (xs, other),
    };
    // **head 版 int8-MMA MMQ**(W-2f;§21 之后的唯一大杠杆):权重块字节直读 +
    // 激活量化成 `block_q8_1_mmq` ⇒ `mma.sync.m16n8k16.s8`。
    //
    // 为什么排在 cuBLAS 臂之前:同机同形状实测本臂 **103 TFLOPS 等效**
    // (llama.cpp)对 cuBLAS 的 66,而且**省掉"权重→f16 解量化"那 10.4%**
    // (设计 §21.2)。`OWL_GGUF_MMQ2=0` 一键回退。
    if m >= GGUF_MMQ2_MIN_M
        // 激活 ∈ {F16, BF16}(BF16 = DFlash2 草稿域;量化手有 `*_bf16` 变体)
        && matches!(xs.dtype, Dtype::F16 | Dtype::BF16)
        && col_index.is_none()
        && arms.mmq2
        && owl_kernels::family::gguf::mmq2_ok(ty, m, out_dim, k)
    {
        use owl_kernels::family::gguf as famg;
        let bf16 = xs.dtype == Dtype::BF16;
        let quant = match famg::mmq2_quant_entry_for_act(ty, bf16) {
            Some(famg::MMQ2_QUANT_DS4) => SemanticKernel::GgufMmq2QuantDs4,
            Some(famg::MMQ2_QUANT_DS4_BF16) => SemanticKernel::GgufMmq2QuantDs4Bf16,
            Some(famg::MMQ2_QUANT_D4_BF16) => SemanticKernel::GgufMmq2QuantD4Bf16,
            _ => SemanticKernel::GgufMmq2QuantD4,
        };
        // 激活 → q8_1_mmq([k/128][token] × 144B;展平成 U8 二秩)
        // 行向 = **m_pad**(ceil(m/J)·J;尾行由量化核写 0)——
        // 见 `family::gguf::mmq2_xq_tokens`:核每 tile 无条件读 J 行,
        // 按 m 铺缓冲会让最后一个 k 块的尾 tile 越界读。
        let m_pad = famg::mmq2_xq_tokens(m);
        let xq = TensorOps::call(quant)
            .arg(xs)
            .arg_i32(m_pad as i32)
            .arg_i32(m as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(
                Dtype::U8,
                vec![k / famg::MMQ2_QK, m_pad * famg::MMQ2_Q8_1_BLOCK_BYTES],
            );
        return TensorOps::call(SemanticKernel::GgufMmq2)
            .arg(&w.decl())
            .arg(&xq)
            .arg_i32(out_dim as i32)
            .arg_i32(m as i32)
            .arg_i32(k as i32)
            .aux(&[ty.code() as usize, m, out_dim])
            .with_shape(Dtype::F16, vec![m, out_dim]);
    }
    // **dequant + cuBLAS 臂**(W-2d;llama.cpp 的 prefill 正路 —— 见设计 §十九)
    //
    // 形态(与上游同构):把权重按**行窗**反量化成 f16 scratch,再逐窗 cuBLAS
    // 写进 `y[:, n0..]`(`ldc` = y 行宽)。适用门(缺一回退下一臂):
    // 1. `m ≥ GGUF_CUBLAS_MIN_M`(上游对 `ne11 ≥ 64` 才走这条路);
    // 2. 激活 = F16(本臂直吃 f16,不做量化);
    // 3. 型有行窗解量化入口且 `k % 256 == 0`(解量化组的几何判据);
    // 4. 列序表已在上面 gather 掉(`col_index` 已归零;bf16 档除外)。
    if m >= GGUF_CUBLAS_MIN_M
        && xs.dtype == Dtype::F16
        && col_index.is_none()
        && arms.cublas
        && owl_kernels::family::gguf::range_ok(ty, k)
    {
        // 行窗预算(W-2e 加宽):cuBLAS 的 CTA 数 = `ceil(rows/128) × ceil(m/128)`
        // —— 每个窗独立一发 cuBLAS(同流串行),故窗小于一波就会白丢 SM。
        // 旧档 32 MiB ⇒ 27B 的 17408 行张量被切成 19 块、每次只 30-67 CTA
        // ⇒ 实测 26 TFLOPS(cuBLAS 同形状可达 55-65;见设计 §19.5 定位 ③)。
        // 现档 96 MiB ⇒ 大张量 1-2 块(每发几百 CTA);峰值 = 单块 scratch
        // (2·rows·k 字节,不进双缓冲 —— 发射序串行),27B 最大 96 MiB。
        // `[dispatch] gguf_cublas_tile` 可强制行窗(诊断/调参用;None/0 = 预算档)
        let budget_rows = (96usize << 20) / (k.max(1) * 2);
        let mut tile = budget_rows.clamp(128, 8 * 1024) / 64 * 64;
        tile = tile.min(out_dim).max(64);
        if let Some(v) = arms.cublas_tile {
            if v > 0 {
                tile = v;
            }
        }
        // 首 tile 用**普通 Call**(由 eval 分配并 `track_new` 整块 y —— 账本上
        // "这个块是新分配的"必须只说一次);后续 tile 用 `CallDst` 写同一块
        // (同块别名,链式)。若首 tile 也用 CallDst,块从未被 `track_new`,
        // 其"使用者计数"起不来 ⇒ 早期释放/复用(实测 27B chunk=512 出垃圾 token)。
        let mut last: Option<TensorOps> = None;
        let mut n0 = 0usize;
        while n0 < out_dim {
            let r = tile.min(out_dim - n0);
            // ① 行窗解量化 → scratch(dest = scratch,`CallDst`)
            let s = TensorOps::uninit(Dtype::F16, vec![r, k]);
            let dq = TensorOps::call_dst(&s, SemanticKernel::GgufCvtRange)
                .arg(&w.decl())
                .arg_i32(n0 as i32)
                .arg_i32(r as i32)
                .arg_i32(k as i32)
                .aux(&[ty.code() as usize, r, k])
                .with_shape(Dtype::F16, vec![r, k]);
            // ② cuBLAS 写列窗(dest = **上一 tile 的节点值** —— 同块别名;
            //    `eval` 是树形求值,同一节点**只能有一个消费者** ⇒ 用链式既保
            //    "写进同一个 y" 的语义,又给出显式顺序边)
            let node = match last.as_ref() {
                None => TensorOps::call(SemanticKernel::GgufCublasDst),
                Some(prev) => TensorOps::call_dst(prev, SemanticKernel::GgufCublasDst),
            };
            last = Some(
                node.arg(xs)
                    .arg(&dq)
                    .arg_usize(r)
                    .arg_usize(k)
                    .arg_usize(m)
                    .arg_usize(1)
                    .arg_usize(out_dim)
                    .arg_usize(n0)
                    .with_shape(Dtype::F16, vec![m, out_dim]),
            );
            n0 += r;
        }
        return last.expect("out_dim > 0 ⇒ 至少一个 tile");
    }
    // **官方 MMQ**(W-2c;tensor-core prefill):llama.cpp 的 `mul_mat_q`。
    // 适用门(缺一即回退到下一臂,不静默):
    // 1. `m ≥ GGUF_MMQ_MIN_M`(小批让位给 MMVQ —— 上游同款分界思想);
    // 2. `k % 32 == 0`(q8_1 块轴;GGUF 计划面已保证);
    // 3. 列序表已在上面 gather 掉(MMQ 不做 tiled 置换)。
    if m >= GGUF_MMQ_MIN_M
        && k % 32 == 0
        && col_index.is_none()
        && arms.mmq
        && owl_kernels::family::gguf::mmq_entry_for(ty).is_some()
    {
        let m_us = m;
        let xq_row = k / 32 * crate::formats::gguf::Q8_1_BLOCK_BYTES;
        let act = match xs.dtype {
            Dtype::F16 => SemanticKernel::GgufQuantActF16,
            Dtype::BF16 => SemanticKernel::GgufQuantActBf16,
            other => {
                return TensorOps::poisoned(
                    Dtype::F16,
                    vec![m, out_dim],
                    format!("GGUF MMQ: 激活 dtype {other:?} 无对应量化臂(需 F16/BF16)"),
                )
            }
        };
        let xq = TensorOps::call(act)
            .arg(xs)
            .arg_i32(k as i32)
            .arg_i32(k as i32)
            .aux(&[m, k])
            .with_shape(Dtype::U8, vec![m_us, xq_row]);
        return TensorOps::call(SemanticKernel::GgufMmq)
            .arg(&w.decl())
            .arg(&xq)
            .arg_i32(k as i32)
            .arg_i32(out_dim as i32)
            .arg_i32(m_us as i32)
            .aux(&[ty.code() as usize, m_us, out_dim])
            .with_shape(Dtype::F16, vec![m_us, out_dim]);
    }
    // **smem 复用臂**(W-2b;prefill 的算术杠杆):权重窗解量化进 smem **每窗一次**,
    // 激活直吃 f16(不再 q8_1),寄存器 4×4 分块点积 —— 少一发核且更准。
    //
    // 三条适用门(缺一即回退,不静默):
    // 1. `k % SMEM_KC == 0`(窗 = 整个超块;尾块处理留到需要时再加);
    // 2. 激活 = **F16**(本臂不做 bf16 臂;bf16 档回退 vec_dot 复用臂);
    // 3. **无内维列序表**(A2 表是单卡 `ssm_out` 的 tiled 列序;本臂的 x 暂存
    //    不做置换 ⇒ 有表就回退,别静默算错)。
    if m >= GGUF_SMEM_MIN_M
        && k % owl_kernels::family::gguf::SMEM_KC == 0
        && xs.dtype == Dtype::F16
        && col_index.is_none()
        && arms.smem
        // **能力表门**(2026-10-20):smem 臂只对 `smem_entry_for` 已移植的型成立;
        // 不查表 = IQ 族走到这里必 `panic!("未移植")`(实测 IQ4_XS 0.8B 起机即崩)
        && owl_kernels::family::gguf::smem_entry_for(ty).is_some()
    {
        return TensorOps::call(SemanticKernel::GgufGemmSmem)
            .arg(&w.decl())
            .arg(xs)
            .arg_i32(m as i32)
            .arg_i32(out_dim as i32)
            .arg_i32(k as i32)
            .arg_i32(0)
            .arg_i32(out_dim as i32)
            .aux(&[ty.code() as usize, m, out_dim])
            .with_shape(Dtype::F16, vec![m, out_dim]);
    }
    if k % 32 != 0 {
        return TensorOps::poisoned(
            Dtype::F16,
            vec![m, out_dim],
            format!("GGUF GEMM: in_dim {k} 不是 q8_1 块(32)的整数倍"),
        );
    }
    // 激活量化核:按 **(激活 dtype, 是否有列序表)** 四选一
    // (列序表 = 本层权重的**内维**处于 ggml tiled 序 ⇒ 激活必须按 32 元素块
    // 置换着喂;置换折进量化核本身 —— 零额外发射,与“先 gather 再量化”逐位等)
    let act = match (xs.dtype, col_index.is_some()) {
        (Dtype::F16, false) => SemanticKernel::GgufQuantActF16,
        (Dtype::BF16, false) => SemanticKernel::GgufQuantActBf16,
        (Dtype::F16, true) => SemanticKernel::GgufQuantActTiledF16,
        (Dtype::BF16, true) => SemanticKernel::GgufQuantActTiledBf16,
        (other, _) => {
            return TensorOps::poisoned(
                Dtype::F16,
                vec![m, out_dim],
                format!("GGUF GEMM: 激活 dtype {other:?} 无对应量化臂(需 F16/BF16)"),
            )
        }
    };
    let xq_row = k / 32 * crate::formats::gguf::Q8_1_BLOCK_BYTES; // q8_1 块 = half2 ds(2+2) + qs(32) = 36B
    let act_call = match col_index {
        // 槽序对位 `.cu`:tiled 变体把表插在 x 与输出之间
        Some(t) => TensorOps::call(act).arg(xs).arg(&t.decl()),
        None => TensorOps::call(act).arg(xs),
    };
    let xq = act_call
        .arg_i32(k as i32)
        .arg_i32(k as i32)
        .aux(&[m, k])
        .with_shape(Dtype::U8, vec![m, xq_row]);
    // **调度分派**(W-2,设计 §十六):prefill(m ≥ 8)走**权重复用臂** ——
    // 一行权重算一窗 token,权重流量 ÷ `TILE_MT`;decode 仍 MMVQ(本来就只该
    // 读一遍权重)。两臂**位级相同**(同一 `vec_dot` 序 + 同一 warp reduce;
    // 门 = `kernel_gate::tile_gemm_matches_mmvq_bitwise`)⇒ 切换零数值风险。
    let gemm = if m >= GGUF_TILE_MIN_M
        // 小张量 + 未用满窗 ⇒ MMVQ(实测快 6.7×)
        && !gguf_prefers_mmvq_small_n(ty, m, out_dim, k, arms.mmvq_l2_mib)
        && arms.tile
        // 同上:复用臂的能力集是独立的(`tile_entry_for`),不查 = IQ 族崩
        && owl_kernels::family::gguf::tile_entry_for(ty).is_some()
    {
        SemanticKernel::GgufGemmTile
    } else {
        SemanticKernel::GgufGemm
    };
    TensorOps::call(gemm)
        .arg(&w.decl())
        .arg(&xq)
        .arg_i32(m as i32)
        .arg_i32(out_dim as i32)
        .arg_i32(k as i32)
        .arg_i32(k as i32)
        .arg_i32(0)
        .arg_i32(out_dim as i32)
        .aux(&[ty.code() as usize, m, out_dim])
        .with_shape(Dtype::F16, vec![m, out_dim])
}

pub struct Linear {
    /// 权重槽 [out, in](检查点原生布局,零转置)
    w: Weight,
    /// E3 量化臂尺寸(声明期常量)
    out_dim: usize,
    in_dim: usize,
    /// 量化组大小(Some = 已启用;None = f16 直读;构造期定形)
    quant_group: Option<usize>,
    /// 量化四件套(qweight marlin-packed / scales / zeros(仅 AWQ) /
    /// workspace;键 = `{w.key}.qweight/.scales/.zeros/.ws`)
    qw: Option<Weight>,
    sc: Option<Weight>,
    zs: Option<Weight>,
    ws: Option<Weight>,
    ctmp: Option<Weight>,
    /// **W4A8 全局因子**(f32[1];装载期由该层权重尺度算出 ——
    /// `pack_marlin_s_a8` 第二返回值 `max/4096`)。`Some` = 本层走 a8 臂
    /// (**单一事实源**:因子在则 a8 在)。
    a8_factor: Option<Weight>,
    /// **GGUF 块型**(2026-10-19):`Some` = 本层权重以 GGUF 块字节驻留
    /// (槽 dtype=U8、形状 [out, row_bytes]);forward 走 quant_act + gemm 两发
    gguf_ty: Option<crate::formats::gguf::GgmlTy>,
    /// **GGUF 内维列序表**(设计 §11.4 落点 A2):`Some` = 本层权重的**内维**
    /// 处于 ggml **tiled** 序(`ssm_out` 的 v-head 反重排)⇒ 喂它的激活须按
    /// 32 元素块置换 —— 折进激活量化核(零额外发射)。表 = `U32[in_dim/32]`,
    /// 由源侧**虚拟键**生成并常驻显存(见 `GgufSource::virtual_perm`)。
    gguf_col_index: Option<Weight>,
    /// **TP 角色标记**(S4 ③';构造期从规则表算一次 —— **纯标注,零决策**:
    /// 层不构造通信节点,只把它透传到图节点上,解释器统一追加 allreduce)
    tp_role: crate::plan::rules::WeightRule,
    /// **输出 dtype 钉定**(`Some` = 本层产出恒铸到该 dtype;2026-10-20)。
    ///
    /// 唯一动机:GGUF 块型臂的 GEMM 核写 `half*`(F16),而 BF16 执行域
    /// (DFlash2 草稿)下游一律按 BF16 读 —— 不铸就是**静默错值**。
    /// HF 臂(`matmul` 声明随激活)本就不错,`cast_16` 同 dtype 直通 ⇒
    /// 统一挂上无成本。缺省 `None` = 既有一切档位逐字节不变。
    out_dt: Option<Dtype>,
}

impl Linear {
    /// 权重槽叶子声明(刀3a':gdn b/a 直发自研 gemv 需绕过 Linear::forward
    /// 的 cublas 臂;仅 f16 直读态合法 —— 量化态调用方须走 forward)
    pub fn weight_decl(&self) -> TensorOps {
        self.w.decl()
    }

    /// 准备容器(`key` = 数据源槽键;`plan` = 量化计划构造期注入 ——
    /// 零数据零副作用,零突变面)。量化计划下按尺寸门控当场定形:
    /// marlin tile 约束(n%256==0 且 k%128==0,与装载源同源)不满足的
    /// 小线性保持 f16 直读 —— 装载源侧对同款判定输出反量化 `.weight`。
    pub fn new(key: &'static str, out_dim: usize, in_dim: usize, plan: QuantPlan) -> Linear {
        let mut lin = Linear {
            w: Weight::new(key, vec![out_dim, in_dim]),
            out_dim,
            in_dim,
            quant_group: None,
            qw: None,
            sc: None,
            zs: None,
            ws: None,
            ctmp: None,
            a8_factor: None,
            gguf_ty: None,
            gguf_col_index: None,
            tp_role: crate::plan::rules::rule_of(key),
            out_dt: None,
        };
        let eligible = crate::formats::util::marlin_eligible(out_dim, in_dim);
        match plan {
            QuantPlan::Gguf(ty) => lin.shape_gguf(ty),
            QuantPlan::W4A16 if eligible => lin.quantize_g128(),
            QuantPlan::W4A16Awq if eligible => lin.quantize_awq32(),
            QuantPlan::W4A8Awq if eligible => lin.quantize_awq_a8(),
            _ => {}
        }
        lin
    }

    /// 权重�@ f16 直读态(**逐层旁路的合法性判据**):未量化且非 GGUF 块型
    /// ⇒ 槽里真的是 f16 位型。
    ///
    /// 为什么要有这个门:自研 GEMV 旁路(`weight_decl()` + `GemvDual`)
    /// 只吃 f16 位型 —— 量化档若误走旁路,核会把块字节当 f16 读(静默错值)。
    /// 旧世界量化档的 `b`/`a`(out 极小)因尺寸门控总落 f16 ⇒ 雷未爆;
    /// **GGUF 是第一个 b/a 也量化的档位**(`ssm_alpha`/`ssm_beta`)。
    pub fn is_f16_direct(&self) -> bool {
        self.quant_group.is_none() && self.gguf_ty.is_none()
    }

    /// 输出行数(= 本 rank 的 logits 宽度;词表切档 = `vocab / world`)
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// 本层的 TP 规则(**只读**;判据/门用 —— 决策仍只在构造期做一次)
    pub fn tp_rule(&self) -> crate::plan::rules::WeightRule {
        self.tp_role
    }

    /// **TP 规则覆盖**(2026-10-19;唯一消费者 = lm_head 的**词表切**)
    ///
    /// 默认规则来自键表([`crate::plan::rules::rule_of`];`lm_head` = `Replicate`)。
    /// 词表切要 `UniformCol`(切 out 维 = 词表行)。**层不猜**:调用方
    /// ([`crate::layers::embedding::Embedding::new_untied_head`])按
    /// "分片行数 < 全量行数"这一个判据决定 —— 单源 = `ModelSpec::head_vocab`。
    pub fn with_tp_rule(mut self, rule: crate::plan::rules::WeightRule) -> Self {
        self.tp_role = rule;
        self
    }

    /// **输出 dtype 钉定**(唯一消费者 = DFlash2 草稿的 GGUF 块型臂;
    /// 见字段 `out_dt` 的判据)。同 dtype 直通 ⇒ 挂上零成本。
    pub fn with_out_dtype(mut self, dt: Dtype) -> Self {
        self.out_dt = Some(dt);
        self
    }

    /// 取出权重槽(单权重基本函数 `load_weight` 的入口;消费层容器)
    pub fn into_weight(self) -> Weight {
        self.w
    }

    /// **挂内维列序表**(GGUF `ssm_out` 的 v-head 反重排;设计 §11.4 落点 A2)。
    ///
    /// 两条不变式(**响亮**,不静默):
    /// 1. 只在 **GGUF 量化臂**下合法 —— f16 直读臂没有激活量化这一步,
    ///    表会被静默忽略 ⇒ 那种场景必须由**源侧装载期**把列序扭回来
    ///    (native 件的列置换,`GgufSource::read_elements`);
    /// 2. 表长必须 = `in_dim / 32`(与激活量化核的块数同口径;不等 = 静默错值)。
    pub fn set_gguf_col_index(&mut self, t: Weight) {
        assert!(
            self.gguf_ty.is_some(),
            "Linear::set_gguf_col_index: 本层不是 GGUF 量化臂(gguf_ty = None)—— \
             f16 臂没有激活量化,列序表会被静默忽略(native 件请走源侧装载期列置换)"
        );
        let want = self.in_dim / 32;
        let got = t.shape().first().copied().unwrap_or(0);
        assert_eq!(
            got, want,
            "Linear::set_gguf_col_index: 表长 {got} ≠ in_dim/32 = {want}(与激活量化核的块数不同口径)"
        );
        self.gguf_col_index = Some(t);
    }

    /// **GGUF 块型定形**(2026-10-19;构造期私有):权重槽 = 原始块字节
    /// (`U8`、形状 `[out, 块行字节]`),不转不复排 —— 块布局 = 内核读序。
    ///
    /// 形态门:最内维必须被块元素数整除(不满足 = 块布局无定义 ⇒ **响亮拒**);
    /// 未移植型由 `entry_for` 在 forward 期拒(此处只管槽形态 —— 声明面不该
    /// 因“内核还没写”而无法构造,否则装载测试也跑不了)。
    fn shape_gguf(&mut self, ty: crate::formats::gguf::GgmlTy) {
        let (be, bb) = ty.block();
        assert!(
            self.in_dim % be == 0,
            "GGUF {}: 最内维 {} 不是块大小 {} 的整数倍(块布局无定义)",
            ty.name(),
            self.in_dim,
            be
        );
        let row_bytes = self.in_dim / be * bb;
        // **零拷贝直读**:同名键、同形状(行主序),只是行宽单位从元素变成字节
        self.w = Weight::new_typed(self.w.key(), vec![self.out_dim, row_bytes], Dtype::U8);
        self.gguf_ty = Some(ty);
    }

    /// W4A16 定形(E3;REQ-PRE-01,构造期私有):forward 换 marlin GEMM
    /// (foreign 通道),装载换三件套(qweight marlin-packed i32 / scales
    /// f16 / workspace 零初始化)。
    fn quantize_g128(&mut self) {
        // 构造期前置(**响亮**,release 也拦 —— 2026-10-17 审查 §五轮-C2):
        // 唯一调用点(`new` 的 match)已用同一谓词门控,这里是私有函数自己的
        // 不变式;原先 `debug_assert` ⇒ release 静默放行(将来新增调用点漏门 =
        // 按非法形态建 plan)。
        assert!(
            crate::formats::util::marlin_eligible(self.out_dim, self.in_dim),
            "g128: marlin 形态不满足(out {}, in {}),构造点漏了 eligible 门",
            self.out_dim,
            self.in_dim
        );
        let g = 128usize;
        let n_pack = marlin_n_pack(self.out_dim); // 直通(eligible 保证 n%256==0)
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        self.sc = Some(Weight::new_typed_scales(
            key,
            vec![self.in_dim / g, n_pack],
        ));
        self.ws = Some(Weight::new_typed_u32(
            key,
            vec![crate::ops::marlin_ws_elems(n_pack)],
        ));
        // c_tmp 独立小块(use_fp32_reduce=false 不触碰,但槽位契约要求
        // 独立指针 —— 不可与 ws alias)
        self.ctmp = Some(Weight::new_typed_u32(key, vec![1]));
        self.quant_group = Some(g);
    }

    /// AWQ g32-asym 定形(2026-10-01;构造期私有):与 g128 同 B 打包,
    /// 增 zeros 槽(pack_marlin_z 产物 [in/g, out/8] U32),GEMM 换
    /// kU4(has_zp)内核;组大小 32(group_blocks = 32/16 = 2 幂次合法)。
    fn quantize_awq32(&mut self) {
        // 构造期前置(**响亮**,release 也拦 —— 2026-10-17 审查 §五轮-C2):
        // 唯一调用点(`new` 的 match)已用同一谓词门控,这里是私有函数自己的
        // 不变式;原先 `debug_assert` ⇒ release 静默放行(将来新增调用点漏门 =
        // 按非法形态建 plan)。
        assert!(
            crate::formats::util::marlin_eligible(self.out_dim, self.in_dim),
            "awq-g32: marlin 形态不满足(out {}, in {}),构造点漏了 eligible 门",
            self.out_dim,
            self.in_dim
        );
        let g = 32usize;
        let n_pack = marlin_n_pack(self.out_dim);
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        self.sc = Some(Weight::new_typed_scales(
            key,
            vec![self.in_dim / g, n_pack],
        ));
        // zeros:pack_marlin_z 产出 (groups, n/8) i32 —— 与 scales 同组轴
        self.zs = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / g, n_pack / 8],
        ));
        self.ws = Some(Weight::new_typed_u32(key, vec![crate::ops::marlin_ws_elems(n_pack)]));
        self.ctmp = Some(Weight::new_typed_u32(key, vec![1]));
        self.quant_group = Some(g);
    }

    /// **W4A8 × AWQ 定形**(S2;构造期私有):与 `quantize_awq32` **同尺寸**
    /// (B/scales/zp 都在同一套 tile 尺寸下换布局 —— 字节数不变),另加
    /// `a8_factor` f32[1]。
    ///
    /// 容量说明:`a_scales`(per-token 尺度)**不在这里** —— 它是**每轮
    /// 按调用形状**的瞬态(`TensorOps::zeros(F32, [m,1])`,引擎按形状分配),
    /// 故层里零容量常量、零预留(装 8192 行也只是一次 32KB memset)。
    fn quantize_awq_a8(&mut self) {
        // 构造期前置(**响亮**,release 也拦 —— 2026-10-17 审查 §五轮-C2):
        // 唯一调用点(`new` 的 match)已用同一谓词门控,这里是私有函数自己的
        // 不变式;原先 `debug_assert` ⇒ release 静默放行(将来新增调用点漏门 =
        // 按非法形态建 plan)。
        assert!(
            crate::formats::util::marlin_eligible(self.out_dim, self.in_dim),
            "w4a8: marlin 形态不满足(out {}, in {}),构造点漏了 eligible 门",
            self.out_dim,
            self.in_dim
        );
        let g = 32usize;
        let n_pack = marlin_n_pack(self.out_dim);
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        // scales_a8 线 dtype = F16 槽(int16 归一位型;与 a16 同尺寸)
        self.sc = Some(Weight::new_typed_scales(
            key,
            vec![self.in_dim / g, n_pack],
        ));
        self.zs = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / g, n_pack / 8],
        ));
        self.ws = Some(Weight::new_typed_u32(key, vec![crate::ops::marlin_ws_elems(n_pack)]));
        self.ctmp = Some(Weight::new_typed_u32(key, vec![1]));
        // 全局因子(装载器填;核侧设备标量块)
        self.a8_factor = Some(Weight::new_typed(key, vec![1], Dtype::F32));
        self.quant_group = Some(g);
    }
}

impl Module for Linear {
    /// f16 臂:y = x @ W^T([.., in] → [.., out];W 原生 [out, in],nt 直读;
    /// 未装载 → 毒值声明)。
    /// W4A16 臂(E3):foreign GEMM_W4A16(6 Block + 4 sz;out 槽 2 ——
    /// sig 逃生舱 O 槽表达),ws 同块双槽(ws + c_tmp,use_fp32_reduce=false
    /// 不触碰 c_tmp)。
    /// AWQ 臂(2026-10-01):同构但换 GEMM_W4A16_AWQ(7 Block = scales 后
    /// 插 zeros 槽;kU4 has_zp 内核,组大小 32)。
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let _ = ctx; // 层**不读 ctx 的 TP 面**(S4 ③':决策在解释器)
        // **输出 dtype 钉定 + GGUF 动态范围护栏**(仅显式声明者)
        //
        // GGUF 块型 GEMM 的输出核写 `half*`(F16 上限 65504),而 BF16 执行域
        // (DFlash2 草稿)的层输出实测可达 ~6e4 ⇒ 直接落 F16 会饱和成 inf。
        // 成对移位:`x·2⁻⁶ →(GEMM)→ f16→bf16 → ·2⁶`。2 的幂在 f16/bf16 上
        // **精确**,相对精度仍由 F16 尾数决定 ⇒ 与"BF16 输出 GEMM"等价。
        // `2⁶` = 64× 上限余量(1.7e7 以下不饱和);F16 次正规临界 6e-8·2⁶
        // 远低于 bf16 自身粒度 ⇒ 无可见损失。
        const GEMM_SCALE: f32 = 64.0;
        let (xs, out_dt) = match self.out_dt {
            Some(dt) if self.gguf_ty.is_some() && dt != Dtype::F16 => {
                (crate::layers::scale16(xs, 1.0 / GEMM_SCALE), Some(dt))
            }
            other => (xs.clone(), other),
        };
        // 小 n 判据的阈值走**配置**(`[kernel] gguf_mmvq_l2_mib`;ctx 里带)
        let y = self.forward_inner(&xs, &ctx.env.gguf);
        let y = match out_dt {
            Some(dt) if self.gguf_ty.is_some() => {
                let b = crate::layers::cast_16(&y, dt);
                crate::layers::scale16(&b, GEMM_SCALE)
            }
            Some(dt) if y.dtype() != dt => crate::layers::cast_16(&y, dt),
            _ => y,
        };
        // ★ **S4 ③':TP 只给标记** —— 构造期已由规则表算好([`Self::tp_role`]),
        //   这里**纯透传**到图节点;解释器在求值尾部按标记统一追加 allreduce。
        //   (用户裁决 2026-10-17:感知下沉到解释器层,原子 layer 只留标记。)
        y.with_tp_role(self.tp_role)
    }
}

impl Linear {
    /// ctx 无关前向(Embedding::lm_head_matmul 复用;臂集由调用方显式传入)。
    pub(crate) fn forward_inner(&self, xs: &TensorOps, arms: &crate::env::GgufEnv) -> TensorOps {
        // ---- **GGUF 块型臂**(2026-10-19):激活 → q8_1 → dequant-in-kernel GEMM ----
        if let Some(ty) = self.gguf_ty {
            return gguf_quant_gemm(
                &self.w,
                ty,
                xs,
                self.out_dim,
                self.in_dim,
                self.gguf_col_index.as_ref(),
                arms,
            );
        }
        // ---- W4A8 × AWQ 臂(S2):激活量化 → a8 GEMM ----
        //
        // 与 a16 臂**同源不同布局**;两处本仓特有的形态:
        // ① `a_scales` = **每轮瞬态**(`TensorOps::zeros`)—— 引擎按调用形状
        //    分配(m 行 ⇒ m 个 f32),故无容量常量/无预留、无跨调用竞争;
        // ② 全局因子 = **设备标量块**(`a8_factor`,装载器填)—— 量化核直接读它,
        //    零 host 回路零额外发射。
        if let (Some(g), Some(qw), Some(sc), Some(zs), Some(ws), Some(ctmp), Some(fac)) = (
            self.quant_group,
            &self.qw,
            &self.sc,
            &self.zs,
            &self.ws,
            &self.ctmp,
            &self.a8_factor,
        ) {
            let m: usize = xs.shape()[..xs.shape().len() - 1].iter().product();
            let n_pack = marlin_n_pack(self.out_dim);
            let a_scales = TensorOps::zeros(Dtype::F32, vec![m, 1]);
            let q = crate::ops::quant_int8_row(xs, &fac.decl(), &a_scales, m, self.in_dim);
            // 槽序(w4a8 表)= a, b, out, scales, zeros, a_scales, ws, c_tmp,
            // m, k, n, groupsize(out 由装配器按 O 槽插入 —— 声明驱动)
            let gemm = TensorOps::call(SemanticKernel::MarlinW4A8Awq)
                .arg(&q)
                .arg(&qw.decl())
                .arg(&sc.decl())
                .arg(&zs.decl())
                .arg(&a_scales)
                .arg(&ws.decl())
                .arg(&ctmp.decl())
                .arg_usize(m)
                .arg_usize(self.in_dim)
                .arg_usize(n_pack)
                .arg_usize(g)
                .with_shape(Dtype::F16, vec![m, n_pack]);
            return narrow_strided(&gemm, m, n_pack, 0, self.out_dim, vec![m, self.out_dim]);
        }
        if let (Some(g), Some(qw), Some(sc), Some(ws), Some(ctmp)) = (
            self.quant_group, &self.qw, &self.sc, &self.ws, &self.ctmp,
        ) {
            let m: usize = xs.shape()[..xs.shape().len() - 1].iter().product();
            // n_pack = 打包档(= out_dim;pad 路线废案)
            let n_pack = marlin_n_pack(self.out_dim);
            // 激活 dtype 决定臂:F16 → GEMM_W4A16;BF16 → GEMM_W4A16_BF16
            // (E5-DF3 同日十二:草稿路径 BF16 化,scales 保持 F16 装载,
            // a/c=BF16 的 marlin 实例;s_type = F16 两族一致)
            // **臂由装载事实决定**(2026-10-16 定谳):尺度槽的声明 dtype = 装载期
            // 按模型 dtype 钉的(bf16 模型 ⇒ BF16,见 `Weight::new_typed_scales`)。
            // 激活若与它不符(f16 探针 / 混合臂),**显式铸**过去 —— 绝不能落进
            // “另一条臂”按错口径读尺度:C 侧 `s_type = c_type`,实测 bf16 尺度被
            // f16 臂按 f16 读 ⇒ 单块幅度 ~15×、端到端残留 7%(立案 §P)。
            let sc_dt = sc.dtype();
            let xs_aligned = if xs.dtype == sc_dt {
                None
            } else {
                Some(crate::layers::dflash2::ensure_dt(xs, sc_dt))
            };
            let xs: &TensorOps = xs_aligned.as_ref().unwrap_or(xs);
            let bf16_act = xs.dtype == Dtype::BF16;
            // marlin 臂(2026-10-12 三形态归一 Call):语义动作声明,
            // 名字路由归 driver(zs 在场 = AWQ 7 父;dt = BF16 → bf16 臂)
            let marlin = match self.zs.as_ref() {
                Some(z) => TensorOps::call(SemanticKernel::MarlinW4A16)
                    .arg(xs)
                    .arg(&qw.decl())
                    .arg(&sc.decl())
                    .arg(&z.decl())
                    .arg(&ws.decl())
                    .arg(&ctmp.decl()),
                None => TensorOps::call(SemanticKernel::MarlinW4A16)
                    .arg(xs)
                    .arg(&qw.decl())
                    .arg(&sc.decl())
                    .arg(&ws.decl())
                    .arg(&ctmp.decl()),
            };
            let marlin = marlin
                .arg_usize(m)
                .arg_usize(self.in_dim)
                .arg_usize(n_pack)
                .arg_usize(g)
                .with_shape(if bf16_act { Dtype::BF16 } else { Dtype::F16 }, vec![m, n_pack]);
            return narrow_strided(
                &marlin,
                m,
                n_pack,
                0,
                self.out_dim,
                vec![m, self.out_dim],
            );
        }
        xs.matmul_nt(&self.w.decl())
    }
}

impl Loadable for Linear {
    /// f16 臂:单 Want(w)。W4A16 臂:三 Want(qweight U32 / scales F16 /
    /// ws U32 零初始化)—— 键 = `{w.key}.qweight/.scales/.ws`,装载源
    /// (w4a16.rs)按同款谓词供给。AWQ 臂增 zeros Want(`{w.key}.zeros`,
    /// pack_marlin_z 产物;源 awq.rs 供给)。
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let ops = self.layout_arm(ctx);
        // **层声明的规则覆盖**(P1.1,2026-10-19):规则单源 = 构造期 `tp_role`
        // (`Linear::with_tp_rule` 是唯一覆盖口;唯一消费者 = 词表切的 lm_head)。
        // 只在**真覆盖**(≠ 键表默认)时打标 ⇒ 其余键零行为变化;
        // 默认规则仍归键表(`rule_of`),段长降级(`rule_of_want`)不受影响
        // —— 覆盖只在"构造者显式声明"时发生(判据见 `Embedding::new_untied_head`)。
        if self.tp_role != crate::plan::rules::rule_of(self.w.key()) {
            ops.with_rule(self.tp_role)
        } else {
            ops
        }
    }
}

impl Linear {
    /// **装载分支臂**(f16 / W4A16 / W4A8 三族;`layout` 的唯一产出点)
    fn layout_arm(&self, ctx: &LoaderCtx) -> LoaderOps {
        // ---- W4A8 键族(S2):与 a16 键族**并列**,同源不同布局 ----
        if let (Some(_g), Some(qw), Some(sc), Some(zs), Some(ws), Some(ctmp), Some(fac)) = (
            self.quant_group,
            &self.qw,
            &self.sc,
            &self.zs,
            &self.ws,
            &self.ctmp,
            &self.a8_factor,
        ) {
            let base = self.w.key();
            let device = ctx.device_repack;
            let ops = if device {
                // 设备重排(生产默认):源 packed 直拷上卡 → `owl_ct_repack_a8_u32`
                // 出 a8 布局 B(与 a16 共用 `load_direct_device_rearrange` 快路)
                qw.layout_as_device_rearrange(
                    format!("{base}.packed_raw"),
                    crate::ops::SemanticKernel::CtRepackA8,
                    self.out_dim,
                    self.in_dim / 8,
                    ctx,
                )
            } else {
                // CPU 懒物化(a8 键族;含因子)
                qw.layout_as(format!("{base}.qweight_a8"), ctx)
            };
            return ops
                .chain(sc.layout_as(format!("{base}.scales_a8"), ctx))
                .chain(zs.layout_as(format!("{base}.zeros_a8"), ctx))
                .chain(ws.layout_as(format!("{base}.marlin_ws"), ctx))
                .chain(ctmp.layout_as(format!("{base}.marlin_ctmp"), ctx))
                .chain(fac.layout_as(format!("{base}.a8_factor"), ctx));
        }
        match (self.quant_group, (&self.qw, &self.sc, &self.ws, &self.ctmp)) {
            (Some(_g), (Some(qw), Some(sc), Some(ws), Some(ctmp))) => {
                let base = self.w.key();
                // qweight 键:设备重排 → 原始 packed,DMA 上卡后 GPU 重排到
                // marlin 布局(AWQ kU4 与 W4A16 对称共用同核同 B 字节布局;
                // 2026-10-13 放开 zs 限制 —— W4A16 无 zeros want,原条件
                // `zs.is_some()` 把草稿钉在 CPU 两步重排 ≈24.6s);否则
                // 经典键(CPU 懒物化)。
                let device = ctx.device_repack;
                let ops = if device {
                    // 设备重排:want 键 = 原始 packed 源键(源零重排直拷),
                    // 布局带核名与原始形状;DMA 上卡后 GPU 重排到 marlin 布局
                    qw.layout_as_device_rearrange(
                        format!("{base}.packed_raw"),
                        crate::ops::SemanticKernel::CtRepack,
                        self.out_dim,
                        self.in_dim / 8,
                        ctx,
                    )
                } else {
                    qw.layout_as(format!("{base}.qweight"), ctx)
                };
                let ops = ops.chain(sc.layout_as(format!("{base}.scales"), ctx));
                let ops = match (&self.zs, device) {
                    (Some(zs), true) => {
                        // 设备重排:scales/zeros 暂留 CPU 懒物化(数据小;
                        // 浮出为瓶颈再上 gather 核)
                        ops.chain(zs.layout_as(format!("{base}.zeros"), ctx))
                    }
                    (Some(zs), false) => ops.chain(zs.layout_as(format!("{base}.zeros"), ctx)),
                    (None, _) => ops,
                };
                ops.chain(ws.layout_as(format!("{base}.marlin_ws"), ctx))
                    .chain(ctmp.layout_as(format!("{base}.marlin_ctmp"), ctx))
            }
            _ => {
                let ops = self.w.layout(ctx);
                // GGUF 内维列序表:与权重**同层声明**(键 = `blk.N.ssm_out.perm`
                // 虚拟键,源侧合成字节;见 `GgufSource::virtual_perm`)
                match &self.gguf_col_index {
                    Some(t) => ops.chain(t.layout(ctx)),
                    None => ops,
                }
            }
        }
    }
}


#[cfg(test)]
mod tp_marking_tests {
    //! S4 ③':TP **标记**面(纯逻辑;插入动作在解释器 —— 真机数值由 G8 验)。
    use super::*;
    use crate::plan::rules::WeightRule;

    /// 构造期标记:行并行(down_proj/o_proj/out_proj)⇒ `UniformRow`;
    /// 列并行(gate_proj/up_proj)⇒ `UniformCol`;小权重 ⇒ `Replicate`
    #[test]
    fn new_marks_tp_role_from_rule_table() {
        let cases: &[(&'static str, WeightRule)] = &[
            ("down_proj", WeightRule::UniformRow),
            ("o_proj", WeightRule::UniformRow),
            ("out_proj", WeightRule::UniformRow),
            ("gate_proj", WeightRule::UniformCol),
            ("up_proj", WeightRule::UniformCol),
            ("q_proj", WeightRule::SegmentedCol(crate::plan::rules::SegKind::QGate)),
            ("in_proj_qkvz", WeightRule::SegmentedCol(crate::plan::rules::SegKind::QKVZ)),
            ("in_proj_a", WeightRule::PerHead),
        ];
        for (key, want) in cases {
            let lin = Linear::new(key, 8, 16, QuantPlan::F16);
            assert_eq!(lin.tp_role, *want, "{key}: 构造期标记");
        }
    }

    /// **forward 纯透传**:图节点的 tp_role == 构造期标记(层零决策)
    #[test]
    fn forward_passes_mark_through() {
        use crate::module::ForwardCtx;
        let row = Linear::new("down_proj", 8, 16, QuantPlan::F16);
        let x = TensorOps::zeros(Dtype::F16, vec![1, 16]);
        let y = row.forward(&x, &ForwardCtx::minimal(1));
        assert_eq!(y.tp_role(), WeightRule::UniformRow, "标记应透传到图节点");
        assert_eq!(y.shape(), &[1, 8], "输出形状不被标记改变");
        let col = Linear::new("gate_proj", 8, 16, QuantPlan::F16);
        let y2 = col.forward(&x, &ForwardCtx::minimal(1));
        assert_eq!(y2.tp_role(), WeightRule::UniformCol);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    use crate::tensor::Dtype;
    use std::collections::HashMap;

    /// **W4A8 验收门(S2;GPU + 真检查点)**:同一条真实 AWQ 层,两条 plan
    /// (`W4A16Awq` ↔ `W4A8Awq`)**同输入对拍** —— 走全链:装载(a8 键族 /
    /// 设备重排 a8 核)→ `Linear::forward`(激活 int8 量化 + kS8×kU4 GEMM)→ 回读。
    ///
    /// 判据 = **噪声/信号 RMS**(a8 是另一档量化:per-token int8 激活 + int16
    /// 权重尺度 ⇒ 不逐位;a16 是 f16 激活 + f16 尺度)。阈值 1e-2 远超理论
    /// 偏差(激活 1/(2·127) ≈ 4e-3 + 尺度 1.2e-4),但足以抓住**布局/接线级**
    /// 错误(拿错 B 表、a_scales 没接上、因子没折——量级都在 10⁻¹ 以上)。
    ///
    /// 跑法:`OWL_TEST_DEVICE=1 ckpt:27b-awq=<cyankiwi 27B> cargo test -p owl-models
    ///       --lib gpu_w4a8_matches_w4a16_on_real_layer -- --nocapture`
    #[tokio::test]
    async fn gpu_w4a8_matches_w4a16_on_real_layer() {
        use crate::module::QuantPlan;
        use crate::testkit::{gpu_client, gpu_enabled};
        if !gpu_enabled() {
            owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
            return;
        }
        let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
            owl_shared::testkit::report::skip("ckpt:27b-awq 未设(cyankiwi 27B 检查点目录)");
            return;
        };
        // 真层:layer0.mlp.gate_proj(17408×5120)——两档都 marlin_eligible。
        // 局部键 = "gate_proj"(层内惯例);全键由 `map_keys` 在**声明期**改写
        // (与整模路同一条 `KeyConvention` 语义 —— 这里手写前缀即可)。
        let (local, prefix) = ("gate_proj", "model.language_model.layers.0.mlp.");
        let (out_dim, in_dim, m) = (17408usize, 5120usize, 8usize);
        struct Prefixed<'a>(&'a Linear, &'a str);
        impl crate::module::Loadable for Prefixed<'_> {
            fn layout(&self, ctx: &LoaderCtx) -> crate::module::LoaderOps {
                let p = self.1.to_string();
                self.0.layout(ctx).map_keys(move |k| format!("{p}{k}"))
            }
        }
        // 确定性输入(r16 对齐:进 f16 域前先舍入,免"装载域 vs 输入域"两套口径)
        let r16 = |x: f32| half::f16::from_f32(x).to_f32();
        let xs: Vec<f32> = (0..m * in_dim).map(|i| r16(((i as f32) * 0.013).sin() * 0.8)).collect();
        let xb: Vec<u8> = xs.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect();

        let mut got: Vec<Vec<f32>> = Vec::new();
        for plan in [QuantPlan::W4A16Awq, QuantPlan::W4A8Awq] {
            let lin = Linear::new(local, out_dim, in_dim, plan);
            let comp = Prefixed(&lin, prefix);
            let mut gpu = gpu_client().await;
            let opts = crate::formats::load::LoadOpts::new(&gpu, Dtype::F16);
            let manifest = crate::formats::load::load_from_dir(
                std::path::Path::new(&dir),
                &mut gpu,
                &comp,
                opts,
            )
            .await
            .unwrap_or_else(|e| panic!("{plan:?} 装载失败: {e}"));
            let keys: Vec<String> = manifest.entries().iter().map(|e| e.key.clone()).collect();
            eprintln!("[{plan:?}] 装载 {} 键", keys.len());
            if matches!(plan, QuantPlan::W4A8Awq) {
                for suf in [".scales_a8", ".zeros_a8", ".a8_factor"] {
                    assert!(
                        keys.iter().any(|k| k.ends_with(suf)),
                        "a8 档缺键 {suf};实得 {keys:?}"
                    );
                }
            }
            let x = TensorOps::from_host(Dtype::F16, vec![m, in_dim], &xb);
            let decl = lin.forward(&x, &crate::module::ForwardCtx::minimal(m));
            let out = crate::testkit::harvest_f16(&mut gpu, &decl).await;
            gpu.close().await.expect("关机");
            got.push(out);
        }
        // ---- 判据:噪声/信号 RMS(a8 vs a16)----
        let (a16, a8) = (&got[0], &got[1]);
        assert_eq!(a16.len(), a8.len());
        let n = a16.len();
        let rms = (a16.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64).sqrt();
        let d: f64 = a16
            .iter()
            .zip(a8)
            .map(|(x, y)| ((x - y) as f64).abs())
            .sum::<f64>()
            / n as f64;
        let noise = d / rms.max(1e-9);
        let nan = a8.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "[w4a8-ab] 真实层 {prefix}{local} m={m}: 噪声/信号 RMS = {noise:.5}(a16 信号 RMS={rms:.4});a8 非有限 {nan}"
        );
        assert_eq!(nan, 0, "a8 输出含非有限值");
        assert!(
            noise < 1e-2,
            "a8 与 a16 偏差过大({noise:.5})—— 布局/接线级错误距此量级 ≫ 10⁻¹"
        );
    }

    /// **W4A8 接线门(零 GPU)**:plan → 声明面(a8 键族 want)→ 前向面
    /// (量化 → a8 GEMM 的节点链)。这条门锁的是"层把 a8 档接对了",
    /// 数值正确性在 kernels 侧已有(§S1d 对拍 + a8 repack 逐位门)。
    #[test]
    fn w4a8_plan_declares_a8_keys_and_emits_quant_then_gemm() {
        use crate::module::{Loadable, LoaderCtx, QuantPlan};
        // marlin_eligible:out%256==0 && k%128==0
        let lin = Linear::new("l.w", 256, 256, QuantPlan::W4A8Awq);

        // ---- ① 声明面:键族与 dtype(设备重排档 = 生产默认) ----
        let ctx = LoaderCtx { device_repack: true, ..Default::default() };
        let ops = lin.layout(&ctx);
        let wants: Vec<(String, Dtype)> = ops
            .wants()
            .iter()
            .map(|w| (w.key.clone(), w.dtype))
            .collect();
        let has = |suf: &str, dt: Dtype| {
            wants.iter().any(|(k, d)| k.ends_with(suf) && *d == dt)
        };
        assert!(
            has(".packed_raw", Dtype::U32),
            "设备重排档:B 走原始 packed 源键(U32 位型;经 a8 repack 核);实得 {wants:?}"
        );
        assert!(has(".scales_a8", Dtype::F16), "scales_a8 = int16 位型塞 F16 槽");
        assert!(has(".zeros_a8", Dtype::U32), "zeros_a8");
        assert!(has(".a8_factor", Dtype::F32), "全局因子(设备标量块)");
        assert!(!has(".qweight_a8", Dtype::U32), "设备档不得同时要 CPU 版 B(白跑全模重排)");
        // CPU 档换键
        let ctx_cpu = LoaderCtx { device_repack: false, ..Default::default() };
        let cpu_keys: Vec<String> = lin.layout(&ctx_cpu).wants().iter().map(|w| w.key.clone()).collect();
        assert!(cpu_keys.iter().any(|k| k.ends_with(".qweight_a8")), "CPU 档要 qweight_a8");

        // ---- ② 前向面:量化(quant_int8_row) → a8 GEMM,7 父 ----
        let xs = TensorOps::zeros(Dtype::F16, vec![8, 256]);
        let out = lin.forward_inner(&xs, &crate::env::GgufEnv::default());
        // 外层是窄视图(与 a16 臂同款的 out_dim 裁切)
        let gemm = out.parents.first().expect("窄视图的父 = GEMM").clone();
        match &gemm.op {
            crate::ops::Op::Call { op, .. } => assert_eq!(
                op.op_id().0,
                "marlin.w4a8_awq",
                "a8 臂必须发 w4a8 语义动作"
            ),
            other => panic!("期望 Call(MarlinW4A8Awq),实得 {other:?}"),
        }
        assert_eq!(gemm.parents.len(), 7, "a8 槽 7 父:q/b/scales/zeros/a_scales/ws/c_tmp");
        // 第一父 = 量化输出(其 op 必须是量化动作)
        let q = gemm.parents[0].clone();
        match &q.op {
            crate::ops::Op::Call { op, .. } => assert_eq!(op.op_id().0, "elems.quant_int8_row_f16"),
            other => panic!("期望量化 Call,实得 {other:?}"),
        }
        assert_eq!(q.dtype, Dtype::I8, "量化出 int8(有符号)");
        // 量化 3 父:x(f16 激活) / mul(设备标量因子) / scales(每轮瞬态)
        assert_eq!(q.parents.len(), 3, "量化槽 3 父:x / mul / scales");
        assert!(
            matches!(q.parents[2].op, crate::ops::Op::Zeros),
            "a_scales = 引擎按形状分配的瞬态(Op::Zeros);实得 {:?}",
            q.parents[2].op
        );
    }

    #[tokio::test]
    async fn native_nt_matmul_matches_host() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 原生 [2,3] 行主序(out=2, in=3)
        let x = vec![0.5, -1.0, 2.0];
        let mut want = vec![0.0f32; 2];
        for (o, wo) in want.iter_mut().enumerate() {
            *wo = (0..3).map(|i| x[i] * w[o * 3 + i]).sum();
        }

        let mut face = owl_cpu::CpuFace::new();
        let lin = Linear::new("w", 2, 3, QuantPlan::F16);
        let src = HashMap::from([("w".to_string(), w)]);
        crate::interpreters::eval_load(&lin, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&x));
        let got = harvest(&mut face, &lin.forward(&xs, &ForwardCtx::minimal(1))).await;
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[tokio::test]
    async fn unloaded_slot_becomes_poison_at_boundary() {
        let mut face = owl_cpu::CpuFace::new();
        let lin = Linear::new("w", 2, 3, QuantPlan::F16);
        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[1.0, 2.0, 3.0]));

        let out = lin.forward(&xs, &ForwardCtx::minimal(1));
        assert!(out.is_poisoned(), "未装载槽的声明应立即带毒(随链流动)");
        let err = crate::interpreters::eval_ops(out.step(), &mut face)
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("未装载"), "{err:?}");

        let empty: Src = HashMap::new();
        let err = crate::interpreters::eval_load(&lin, &mut face, &empty, &Default::default())
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("缺键"), "{err:?}");

        let lin3 = Linear::new("w", 2, 3, QuantPlan::F16);
        let bad_len = HashMap::from([("w".to_string(), vec![1.0; 5])]);
        let err = crate::interpreters::eval_load(&lin3, &mut face, &bad_len, &Default::default())
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("元素"), "{err:?}");
    }

    /// HF parity(F.linear = x@W^T;多行 tokens=2;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, out_dim, in_dim) = (2usize, 2usize, 3usize);
        let x: Vec<f32> = (0..tokens * in_dim).map(|i| ((i as f32 * 0.53) - 1.2).cos()).collect();
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.29) - 0.5).collect();
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, in_dim], &f32b(&x));
        let ctx = ForwardCtx::minimal(tokens);

        let inp = tmp_path("linear_in");
        let outp = tmp_path("linear_out");
        st_write(&inp, &[("x", &x, vec![tokens, in_dim]), ("w", &w, vec![out_dim, in_dim])]);
        let manifest = hf_python("linear.py", &[
            inp.to_str().unwrap(), outp.to_str().unwrap(),
            &format!(r#"{{"out": {out_dim}, "in": {in_dim}}}"#),
        ]);
        eprintln!("[hf manifest] linear: {manifest}");
        let want = st_read(&outp, "y");

        for (face_tag, make) in [("cpu", None), ("gpu", Some(()))] {
            let (_, out) = if make.is_none() {
                let mut face = owl_cpu::CpuFace::new();
                let lin = Linear::new("w", out_dim, in_dim, QuantPlan::F16);
                let src = HashMap::from([("w".to_string(), w.clone())]);
                crate::interpreters::eval_load(&lin, &mut face, &src, &Default::default())
                    .await.expect("eval_load");
                ("cpu", harvest(&mut face, &lin.forward(&xs, &ctx)).await)
            } else {
                let mut gpu = crate::testkit::gpu_client().await;
                let lin = Linear::new("w", out_dim, in_dim, QuantPlan::F16);
                let src = HashMap::from([("w".to_string(), w.clone())]);
                crate::interpreters::eval_load(&lin, &mut gpu, &src, &Default::default())
                    .await.expect("eval_load");
                let o = harvest(&mut gpu, &lin.forward(&xs, &ctx)).await;
                gpu.close().await.expect("server 关机");
                ("gpu", o)
            };
            assert_close(&out, &want, 1e-5, &format!("linear-{face_tag}"));
        }
        owl_shared::file_loader::remove_file(&inp).ok();
        owl_shared::file_loader::remove_file(&outp).ok();
    }

    /// **臂集门**(2026-10-21 env→配置收编):`GgufEnv` 的每一条开关都要
    /// **真的**改变分派 —— 关掉一条 = 落到下一条(纯 host 建树,零设备)。
    ///
    /// 为什么要这条:五个臂原来经 `OWL_GGUF_*` env 读,分布在 five 个
    /// `fn *_arm_enabled()` 里;收编到 `ForwardCtx.env.gguf` 后,若某条
    /// 条件忘了改成新字段,行为会**静默恒开**(A/B 旋钮失效)。
    #[test]
    fn gguf_臂集随配置切换() {
        // 尾核名(DFS 找第一个 Call 节点 —— 根可能是窄视图/包装)
        fn first_call(t: &TensorOps) -> String {
            if let crate::ops::Op::Call { op, .. } = &t.op {
                return op.op_id().0.to_string();
            }
            for p in &t.parents {
                let n = first_call(p);
                if !n.is_empty() {
                    return n;
                }
            }
            String::new()
        }
        let (out_dim, in_dim) = (128usize, 256usize); // Q4_K 块 256 ⇒ 行字节对齐
        let lin = Linear::new("gate_proj", out_dim, in_dim, QuantPlan::Gguf(GgmlTy::Q4K));
        let xs = TensorOps::zeros(Dtype::F16, vec![GGUF_CUBLAS_MIN_M, in_dim]);

        let all = crate::env::GgufEnv::default();
        assert_eq!(first_call(&lin.forward_inner(&xs, &all)), "gguf.mmq2", "缺省第一臂");

        let mut a = all;
        a.mmq2 = false;
        assert_eq!(
            first_call(&lin.forward_inner(&xs, &a)),
            "cublas.gemm_f16_dst",
            "关 mmq2 ⇒ cuBLAS 臂(解量化 + cuBLAS)"
        );

        a.cublas = false;
        assert_eq!(first_call(&lin.forward_inner(&xs, &a)), "gguf.mmq", "关 cuBLAS ⇒ MMQ");

        a.mmq = false;
        assert_eq!(first_call(&lin.forward_inner(&xs, &a)), "gguf.gemm_smem", "关 MMQ ⇒ smem");

        a.smem = false;
        assert_eq!(first_call(&lin.forward_inner(&xs, &a)), "gguf.gemm_tile", "关 smem ⇒ 复用臂");

        a.tile = false;
        assert_eq!(first_call(&lin.forward_inner(&xs, &a)), "gguf.gemm", "全关 ⇒ MMVQ 路(纯 gguf.gemm)");
    }
}
