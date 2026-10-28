//! 层协议:计算声明(Module)+ 装载需求(Loadable/Weight/Want)。
//!
//! 两面一个主题(2026-09-26 重组:原 module.rs + loader.rs 合并)——
//! 层与解释器的全部接口面:
//!
//! ```rust,ignore
//! let mut lin = Linear::new("w", 2, 3, QuantPlan::F16); // new = 准备容器(空包指针,零数据;plan 构造期注入)
//! eval_load(&lin, &mut face, &src, &ctx).await?; // 装载执行(驱动 layout)
//! let out = lin.forward(&xs, &ctx);       // 计算声明(total;毒值随链)
//! let y = eval(&lin, &xs, &mut face, &ctx).await?; // 计算执行
//! ```
//!
//! 三律(与 TensorOps 同构):同步(total)· 纯描述;执行都在解释器。

use crate::contract::{Bytes, DeviceClient, Dtype, ModelError, Shape};
use crate::layers::rope::Rope;
use crate::TensorOps;
use std::sync::{Arc, Mutex};

// ============================================================================
// §1 Module:计算声明协议(+ 每步动态依赖词汇)
// ============================================================================

/// 每步动态依赖(forward 的 ctx;C4 定案 2026-09-26:引用注入取代
/// 步形态(PF1-0 契约,pf1-port-design §二):decode 单 token(原地五核
/// 单发);prefill T-token 块(三族逐 token 展开,其余算子 T 批量单发;
/// PF1b 换批核不动层级)。现有构造器全部 = Decode。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepKind {
    Decode,
    Prefill,
}

/// Copy KernelCtx —— rope 表本来就是设备块,ctx 借用 `&Rope` 零双重
/// 所有权;字段 `TensorOps` 值形态让 runner 自选 from_host(调试)/
/// of_block(图捕获),捕获路径不封)。
///
/// runner 每步构造,层只透传不自取(async-runtime §4.3 grab-bag);
/// 无动态依赖的层(mlp/rmsnorm/…)忽略字段。借用结构,无 Default ——
/// 测试用 [`ForwardCtx::minimal`]。
/// GDN 序列表槽(P-6b,2026-10-15):`cu|coff|idx` 三件同块(布局见
/// `owl_kernels::family::gdn_chunked::{CU_OFF, COFF_OFF, IDX_OFF}`)+ 块数。
///
/// **为什么必须由引擎给**:这三张表决定 chunked 六核的**行区间与递推落点** ——
/// runtime 内部按 `t` 生成时,桶 padding 会把假行算进序列(终态被污染;
/// 探针 `gdn_bucket_pad_probe` 实测 Δ=3.6e-2 → 0)。引擎知道**真长度**
/// (`t_real`),而且表存在**持久槽**里 ⇒ 图(M2b)可直接烘、回放期换内容。
#[derive(Clone, Copy)]
pub struct GdnTableSlot<'a> {
    /// 表块(cu|coff|idx;`t_real` 口径)
    pub table: &'a TensorOps,
    /// `nt = ceil(t_real / BT)`
    pub nt: usize,
}

/// FlashInfer prefill 面(OWL_FLASHINFER=1;engine 每 chunk 构造):
/// K 影子池(kHND)+ 表四件套(i32 设备;契约见 owl_kernels::family::flashinfer)。
#[derive(Clone, Copy)]
pub struct FiPrefillCtx<'a> {
    /// K 影子池**逐注意力层**([nb, page, Hkv, hd] kNHD;层间不可共享 ——
    /// 共享 = 后层 K0-dual 覆写前层前缀,渐进腐败案 2026-10-03)
    pub kcs: &'a [TensorOps],
    /// V 影子池(逐注意力层)
    pub vcs: &'a [TensorOps],
    /// q_cu_seqlens i32 [B+1] = [0, T]
    pub q_cu: &'a TensorOps,
    /// 页表 i32 [nb](物理页 id)
    pub indices: &'a TensorOps,
    /// 页 indptr i32 [B+1] = [0, nb]
    pub indptr: &'a TensorOps,
    /// 末页有效 token 数 i32 [B] = ctx_total % page
    pub last_len: &'a TensorOps,
}

/// VLM 图像 token 注入面(塔输出 + 行号表;`ctx` 显式依赖词表)
#[derive(Clone, Copy)]
pub struct ImageInject<'a> {
    /// 塔输出 merged [N, D](D = 文本 hidden)
    pub merged: &'a TensorOps,
    /// 逐 token 行号 [L] f32:≥ 0 = 取 merged 行;负 = 文本位保留 embed
    pub idx: &'a TensorOps,
}

/// VLM m-RoPE 表注入面([L, half] 行表;pos 通道须为行号 = arange(L))
#[derive(Clone, Copy)]
pub struct RopeTables<'a> {
    pub cos: &'a TensorOps,
    pub sin: &'a TensorOps,
}

/// **FI 断点面**(piecewise verify;`ctx.attn_fi` 在场 = 本树要把 FI 核移出图)。
///
/// 格子 = **宿主标量槽**:FI 的 plan key 含 `ctx_total`(每轮随 ctx 长),
/// 烘死在节点里就只对一个 ctx 有效 ⇒ 引擎逐轮写这个 atomic,
/// 驱动件在**回放期**把它补进核的契约槽(见 `interpreters::BreakDecl`)。
/// 零设备流量、零 D2H。
#[derive(Clone)]
pub struct AttnFiBreak {
    /// 每轮 `ctx_total`(= 本轮 kv 总长);引擎在 step 前写
    pub ctx_total: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

/// **张量并行的运行期上下文**(S4 ③;引擎在装配期构造并注入 `ForwardCtx.tp`)。
///
/// ★ **1-stage allreduce 的对称槽束**(2026-10-18;`TP2-施工设计.md` §35)。
///
/// 与 v1 的 staging(header + slot×2 **同一块**)的关键区别:**槽是独立块** ——
/// 这样上游算子(GEMM)可以**直接写槽**(块是分配接口的最小单位),不需要
/// "块内偏移"这层新接口。代价:每 rank 多 2 次 `peer_alloc`/`import`。
///
/// 分配与握手在**引擎装配期**(捕获窗外)⇒ 图只引用、不分配。
#[derive(Debug)]
pub struct TpArenaSlots {
    /// 本端 payload 槽(**乒乓**;槽选择 = 调用序号 % 2 —— 建图期定死、烘进图)
    pub own_slots: [crate::contract::Bytes; 2],
    /// 对端 payload 槽的跨卡别名
    pub peer_slots: [crate::contract::Bytes; 2],
    /// 本端 header(64B;produced/consumed/err/wait_done/blocks_done)
    pub own_hdr: crate::contract::Bytes,
    /// 对端 header 的别名
    pub peer_hdr: crate::contract::Bytes,
    /// 单槽容量(字节)
    pub slot_bytes: u64,
    /// AR 调用序号(host 侧)。
    ///
    /// ⚠️ **它只在建图期推进**:图回放不重跑 host 代码 ⇒ 图内每个节点烘死的
    /// 槽选择是**自洽**的(同一张图重复回放 ≡ 同一串节点),而设备侧的 credit
    /// 游标是**相对量** ⇒ 二者互不依赖。**因此不需要任何 reset 钩子。**
    pub call: std::sync::atomic::AtomicU64,
}

impl TpArenaSlots {
    /// 取本次 AR 用的 `(own_slot, peer_slot)` —— 乒乓 + 递增序号
    pub fn next_pair(&self) -> (crate::contract::Bytes, crate::contract::Bytes) {
        use std::sync::atomic::Ordering;
        let i = (self.call.fetch_add(1, Ordering::Relaxed) % 2) as usize;
        (self.own_slots[i].clone(), self.peer_slots[i].clone())
    }
}

/// 层结构体**零感知**(用户裁决:切分不进结构体):唯一读它的地方是
/// 解释器求值尾部的 allreduce 注入(节点标记 = [`TensorOps::tp_role`]);
/// 装载期的 `needs_ar` 查询 = 规则表单源。
///
/// `Clone` = 轻句柄克隆(`Bytes` 为小 POD;每个图/每次求值各自持一份)。
#[derive(Clone)]
pub struct TpCtx {
    /// 本 rank 的分片计划(`(slice_id, n_slices)` 确定性分片协议)
    pub plan: crate::plan::tp::TpPlan,
    /// 本端 staging(常驻;**建图前**已完成 VMM 握手)
    pub own: crate::contract::Bytes,
    /// 对端 staging 的跨卡别名(常驻;只读槽)
    pub peer: crate::contract::Bytes,
    /// staging 的 slot 字节数(`comm::ar::StagingLayout::slot_bytes`)
    pub slot_bytes: u64,
    /// ★ **1-stage 对称槽束**(`None` = 走 v1 的 stage+reduce 两段;
    /// `Some` = 融合核直接原地加对端槽 —— 见 `TpArenaSlots` 与 §35)
    pub arena: Option<std::sync::Arc<TpArenaSlots>>,
    /// ★ **push 档**(2026-10-18;§36.6/§36.7):`true` ⇒ allreduce 注入点把
    /// `ArStageCall` / `ArReduceCall` 的 `own`/`peer` **实参对调** ——
    /// stage 写**对端**槽、reduce 读**本端**槽 ⇒ 跨 PCIe 只剩**写**(posted)。
    /// 要求别名块是 RW 导入的(`TpOpts::push` 会在握手期用 `peer_import_rw`)。
    pub push: bool,
    /// ★ **fp8 载荷档**(§47 第二步;2026-10-19):`true` ⇒ AR 两条腿的载荷走
    /// e4m3(1B/元素 + 64 元素/尺度 f16 ⇒ 1.031× 原字节)。两侧必须同档
    /// (rank 同装配 ⇒ 同值)。缺省 false ⇒ 与历史逐字一致。
    pub ar_f8: bool,
    /// ★ **v1 两核融合档**(§11.2;2026-10-19):`true` ⇒ 一核做完两段
    /// (`ArFusedCall`);**只非 fp8 载荷**(fp8 腿仍两核)。缺省 false。
    pub ar_fused: bool,
    /// ★ **barrier-only AR-2**(2026-10-23):小包 graph 专用;与 fused 同线格式,
    /// 复用 v1 produced/consumed 相对游标,单 block 去掉尾部 rendezvous。
    pub ar2: bool,
}

impl TpCtx {
    /// 该权重键所在层的输出是否为**部分和**(需要 allreduce;规则表单源)
    pub fn needs_ar(&self, key: &str) -> bool {
        crate::plan::rules::needs_ar(key)
    }

}

pub struct ForwardCtx<'a> {
    /// 本步 token 数(decode = 1;prefill = 块长)
    pub tokens: usize,
    /// 步形态(prefill 词汇见 kv_slots/kv_lens/gdn_slot)
    pub kind: StepKind,
    /// 位置(attention/GDN 用;decode [1] / prefill [T] 表;
    /// 缺省 = 依赖违约 → 层侧毒值)
    pub pos: Option<&'a TensorOps>,
    /// decode KV 直排引用(attention 用;GDN state 同款随域扩)
    pub kv: Option<&'a KvBuffers>,
    /// 位置编码层(全局一份;表经 eval_load 已是设备块。
    /// 注:类型住在 layers,此处反向引用 —— ctx 是 runner 词汇,
    /// runner 持全局 rope,与旧世界 rotary_emb 注入同一形态)
    pub rope: Option<&'a Rope>,
    /// GDN 常驻状态(conv 三段 + recurrent + slots;类型住 layers::gdn,
    /// rope 反向引用同款先例)
    pub gdn: Option<&'a crate::layers::gdn::GdnBuffers>,
    /// 整模路径(C5;批8):full 层 KV 缓存序列 —— 按 layers 数组同序
    /// 过滤(full 层出现序)。Model::forward 按层派生单层子 ctx,
    /// Attention/Gdn 层零感知(仍读 kv/gdn 单引用)。
    pub kvs: Option<&'a [KvBuffers]>,
    /// 整模路径:gdn 层常驻状态序列(gdn 层出现序)
    pub gdns: Option<&'a [crate::layers::gdn::GdnBuffers]>,
    /// prefill:attention 槽表 [T] = base..base+T-1(因果:token t 占槽
    /// base+t;decode 走 kv.slots [bs])
    pub kv_slots: Option<&'a TensorOps>,
    /// prefill:attention kv_len 表 [T] = past+1..past+T(窗 =
    /// [slot−kv_len+1, slot],前位 token 槽已由前步写好)
    pub kv_lens: Option<&'a TensorOps>,
    /// prefill:GDN 序列状态格 [1](全块恒定;decode 走 gdn.slots [bs])
    pub gdn_slot: Option<&'a TensorOps>,
    /// prefill:chunk 绝对起点(块首 token 的绝对 pos;split attention 的
    /// partition 网格声明需要 max_ctx = ctx_base + tokens;builder 缺省 0)
    pub ctx_base: usize,
    /// **FI plan 的建树期 ctx**(`Some` = 覆盖 `ctx_base + tokens` 作为喂 FI 的
    /// `ctx_total`,**只影响 FI 节点**,不动其它核 arg / 分派决策 / 拓扑)。
    ///
    /// 用途(2026-10-18 图档单图立案):FI 进捕获图时,喂进去的 `ctx_total` 被
    /// **烘死** ⇒ 若沿用占位值(1),烘出的就是"ctx=1 的调度表"(plan 无 split /
    /// chunk 数与运行期不符)。给桶上界 ⇒ 烘出的表按上界枚举工作项,运行期由
    /// `chunk_start/end = min(.., kv_len)` 夹取;**T=1 实测与逐 ctx 表逐位同级**
    /// (`fi_prefill_probe` 的 `FI_PF_PLAN_CTX` 腿)。`None` = 历史行为。
    pub fi_plan_ctx: Option<usize>,
    /// FlashInfer prefill 面(**FI 恒启用**,engine 按池 dtype/paged 注入;
    /// None = 该轮不走 FI(非 paged 池 / 非 f16 / 小 chunk < FI_MIN_TOKENS))
    pub fi: Option<FiPrefillCtx<'a>>,
    /// 本层注意力序号(layer_ctx 派生;fi 影子切片下标)
    pub fi_kvi: usize,
    /// 解释器执行环境(EnvProvider;engine/测试构造定制,构造器取默认)
    pub env: crate::env::EnvProvider,
    /// GDN 状态槽号(host 侧真值;chunked/scalar 臂发射标量用)。
    /// **必须由调用方注入**(`model_prefill` 参数)。2026-10-14 定谳(A13):
    /// 本字段曾硬编码 0 且全仓无赋值 ⇒ `gdn_slots > 1` 时 chunked/scalar
    /// 臂把每会话的 rec 态读写到**槽 0**(别人的态),而 varlen 臂走设备槽
    /// 张量(正确)⇒ 长 prompt 误态塌陷。`None` = 调用方未提供,此时
    /// **需要它的臂一律 poisoned 拒发**(不许静默取 0)。
    pub gdn_slot_host: Option<usize>,
    /// GDN 序列表槽(P-6b;chunked/scalar 递推臂用;None = 未注入 ⇒ 该臂拒发)
    pub gdn_tables: Option<GdnTableSlot<'a>>,
    /// 刀D 取证:时间戳探针缓冲块(OWL_TS_PROBE;None = 关。U32 [4096]
    /// 输入槽,ts 内核逐层 clock64 原地写,回放后 dtoh 读原生时间线)
    pub ts_buf: Option<TensorOps>,
    /// v2 分页 decode scratch(None = v1 既有路径;engine 预分配注入,
    /// per-layer 子 ctx 透传 —— 同 ts_buf 形态)
    pub attn_v2: Option<AttnV2Scratch>,
    /// 序列 cu_seqlens/query_start_len 注入(E5-M4 verify 图:捕获期禁
    /// Htod,层内 from_host 的 [0,T] 表由图输入槽供给 —— GDN cu 与
    /// attention qsl 同值语义双消费;None = 层内自建,eager 路不变)
    pub seq_cu: Option<&'a TensorOps>,
    /// verify v2 伪序列页表([seqs×nb];§三十一 verify 臂;None = 非 verify 图)
    pub v2_bt8: Option<&'a TensorOps>,
    /// XQA verify 面(2026-10-13 D3;None = 不走 XQA 臂)
    pub xqa: Option<XqaVerifyCtx>,
    /// E5-M4 fold 记录面(verify 图专用):GDN 层把 fold 重放所需张量
    /// 依层序推入(每层 8:q/k/v raw → conv 重放;q_n/k_n/v_c/g/beta →
    /// 递推重放)。None = 零开销(生产 decode/prefill 不设)。
    pub gdn_tap: Option<std::rc::Rc<std::cell::RefCell<Vec<TensorOps>>>>,
    /// VLM 图像 token 注入面(prefill 单块整发;`None` = 纯文本路径零开销)。
    /// 与 `ts_buf`/`seq_cu` 同款"动态依赖经 ctx 显式传入"律 —— 塔输出
    /// [N, D] 与行号表 [L](负 = 文本位保留 embed)由调用方给足。
    pub vlm_image: Option<ImageInject<'a>>,
    /// VLM m-RoPE 表注入面(prefill;`None` = 用 rope 层的静态 1D 表)。
    /// 多模态序列表 = [L, half](host 由 `layers::mrope` 算好上传)。
    pub vlm_rope: Option<RopeTables<'a>>,
    /// GDN 分派门控打印(EngineKnobs.probes.trace_gate 注入;显式依赖律,
    /// 原 OWL_TRACE_GATE 层内直读 —— 热路径零 env)
    pub trace_gate: bool,
    /// **注意力核断点采集面**(piecewise 捕获图;形态同 `gdn_tap` ——
    /// 层把**核节点**推进来,不是引擎去认识算子):注意力层在 FI/XQA 那个
    /// `Call` 节点建好后推入。核 = 唯一「每轮按 ctx 重算 plan」的节点,故
    /// piecewise 把它移出捕获图、在段间 eager 重放(设计见
    /// `docs/piecewise-断点图-设计与实现.md`)。None = 非 piecewise
    /// (一次 `is_some()` 判,热路径零开销)。
    /// **FI 断点面**(piecewise verify;None = 本树不换 FI 臂)
    /// **Q 输出槽**(B 轴 KVMem 选择层覆盖率探针;`Some` = 逐全注意力层把
    /// **post-RoPE** 的 Q 节点写进槽 `fi_kvi`)。形态同 `attn_break` ——
    /// 调用方建、调用方读,层只写。
    ///
    /// - Q **本就是声明树里的节点**(decode 融合臂的 `q_out` 返回的即该调用
    ///   节点;prefill 臂 = NormRope/Rope 的返回),故本槽 = 「已有节点多挂一个
    ///   引用」,**不产生新节点、不改核的写目标**;
    /// - 缺省 `None` ⇒ 生产路径逐字不变(臂内一次 `is_some()` 判即返)。
    ///
    /// 用途:探针 `dtoh` 该槽 + host 逆 RoPE(rotate-half = 核内约定)
    /// ⇒ 去 RoPE 的 `q̃` 喂 `score_blocks`;post-RoPE 的 `q` 与回读 K 点积
    /// ⇒ 真实逐块注意力质量 ⇒ `coverage_curve`。设计见
    /// `docs/arch/kvmem-选择层-设计与施工.md` §8.7/§8.8。
    pub q_sink: Option<&'a std::cell::RefCell<Vec<Option<TensorOps>>>>,
}

impl<'a> ForwardCtx<'a> {
    /// 最小 ctx(无动态依赖;mlp/rmsnorm/linear/embedding 测试用)
    pub fn minimal(tokens: usize) -> Self {
        Self {
            tokens,
            kind: StepKind::Decode,
            pos: None,
            kv: None,
            rope: None,
            gdn: None,
            kvs: None,
            gdns: None,
            kv_slots: None,
            kv_lens: None,
            gdn_slot: None,
            ctx_base: 0,
            fi_plan_ctx: None,
            fi: None,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host: None,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// decode 步 ctx(attention 全量动态依赖;gdn 置 None)
    pub fn decode(
        tokens: usize,
        pos: &'a TensorOps,
        kv: &'a KvBuffers,
        rope: &'a Rope,
    ) -> Self {
        Self {
            tokens,
            kind: StepKind::Decode,
            pos: Some(pos),
            kv: Some(kv),
            rope: Some(rope),
            gdn: None,
            kvs: None,
            gdns: None,
            kv_slots: None,
            kv_lens: None,
            gdn_slot: None,
            ctx_base: 0,
            fi_plan_ctx: None,
            fi: None,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host: None,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// GDN decode 步 ctx(gdn 全量;attention 依赖置 None)
    pub fn gdn_decode(tokens: usize, gdn: &'a crate::layers::gdn::GdnBuffers) -> Self {
        Self {
            tokens,
            kind: StepKind::Decode,
            pos: None,
            kv: None,
            rope: None,
            gdn: Some(gdn),
            kvs: None,
            gdns: None,
            kv_slots: None,
            kv_lens: None,
            gdn_slot: None,
            ctx_base: 0,
            fi_plan_ctx: None,
            fi: None,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host: None,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// GDN prefill ctx(PF1a):全块状态格恒 gdn_slot(单序列;T-token 块)
    pub fn gdn_prefill(
        tokens: usize,
        gdn: &'a crate::layers::gdn::GdnBuffers,
        gdn_slot: &'a TensorOps,
    ) -> Self {
        Self {
            tokens,
            kind: StepKind::Prefill,
            pos: None,
            kv: None,
            rope: None,
            gdn: Some(gdn),
            kvs: None,
            gdns: None,
            kv_slots: None,
            kv_lens: None,
            gdn_slot: Some(gdn_slot),
            ctx_base: 0,
            fi_plan_ctx: None,
            fi: None,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host: None,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// attention prefill ctx(PF1a):槽/kv_len 表 [T] + pos [T]
    pub fn attn_prefill(
        tokens: usize,
        pos: &'a TensorOps,
        kv: &'a KvBuffers,
        rope: &'a Rope,
        kv_slots: &'a TensorOps,
        kv_lens: &'a TensorOps,
    ) -> Self {
        Self {
            tokens,
            kind: StepKind::Prefill,
            pos: Some(pos),
            kv: Some(kv),
            rope: Some(rope),
            gdn: None,
            kvs: None,
            gdns: None,
            kv_slots: Some(kv_slots),
            kv_lens: Some(kv_lens),
            gdn_slot: None,
            ctx_base: 0,
            fi_plan_ctx: None,
            fi: None,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host: None,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// 整模 decode 步 ctx(批8):双 mixer 序列 + 共享 pos/rope;
    /// kv/gdn 单引用置 None(由 Model::forward 按层派生)。
    pub fn model_decode(
        tokens: usize,
        pos: &'a TensorOps,
        kvs: &'a [KvBuffers],
        rope: &'a Rope,
        gdns: &'a [crate::layers::gdn::GdnBuffers],
    ) -> Self {
        Self {
            tokens,
            kind: StepKind::Decode,
            pos: Some(pos),
            kv: None,
            rope: Some(rope),
            gdn: None,
            kvs: Some(kvs),
            gdns: Some(gdns),
            kv_slots: None,
            kv_lens: None,
            gdn_slot: None,
            ctx_base: 0,
            fi_plan_ctx: None,
            fi: None,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host: None,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// 整模 prefill ctx(PF1a;批P5):单序列 T-token 块,
    /// 槽表/kv_len 表 [T] + GDN 状态格 [1]。fi = FlashInfer 面(engine
    /// 按 OWL_FLASHINFER 注入;None = 旧分派)。
    #[allow(clippy::too_many_arguments)]
    pub fn model_prefill(
        tokens: usize,
        pos: &'a TensorOps,
        kvs: &'a [KvBuffers],
        rope: &'a Rope,
        gdns: &'a [crate::layers::gdn::GdnBuffers],
        kv_slots: &'a TensorOps,
        kv_lens: &'a TensorOps,
        gdn_slot: &'a TensorOps,
        gdn_slot_host: Option<usize>,
        ctx_base: usize,
        fi: Option<FiPrefillCtx<'a>>,
    ) -> Self {
        Self {
            tokens,
            kind: StepKind::Prefill,
            pos: Some(pos),
            kv: None,
            rope: Some(rope),
            gdn: None,
            kvs: Some(kvs),
            gdns: Some(gdns),
            kv_slots: Some(kv_slots),
            kv_lens: Some(kv_lens),
            gdn_slot: Some(gdn_slot),
            ctx_base,
            fi_plan_ctx: None,
            fi,
            fi_kvi: 0,
            env: crate::env::EnvProvider::default(),
            gdn_slot_host,
            gdn_tables: None,
            ts_buf: None,
            trace_gate: false,
            attn_v2: None,
            seq_cu: None,
            v2_bt8: None,
            xqa: None,
            gdn_tap: None,
            vlm_image: None,
            vlm_rope: None,
            q_sink: None,
        }
    }

    /// 位置通道接管(VLM 行号表:pos[t] = t,真位置住 m-RoPE 表)
    pub fn with_pos(mut self, pos: &'a TensorOps) -> Self {
        self.pos = Some(pos);
        self
    }

    /// Q 输出槽接线(B 轴覆盖率探针;与 `with_pos` 同族 —— 只改自身字段,
    /// 不做 I/O)。调用方持 `RefCell` 并在执行后读;不调 ⇒ 槽恒 `None`,
    /// 层内零写入(生产路径逐字不变)。
    pub fn with_q_sink(
        mut self,
        sink: &'a std::cell::RefCell<Vec<Option<TensorOps>>>,
    ) -> Self {
        self.q_sink = Some(sink);
        self
    }

    /// VLM 注入面(prefill):图像 token 携带塔输出 merged + 行号表;本
    /// 构建器只改自身字段,不做 I/O(自肃)。0.8B/纯文本路径不调 ⇒ 零开销。
    pub fn with_vlm_image(mut self, merged: &'a TensorOps, idx: &'a TensorOps) -> Self {
        self.vlm_image = Some(ImageInject { merged, idx });
        self
    }

    /// VLM 注入面(prefill):m-RoPE 行表取代 rope 层静态 1D 表
    /// (纯文本序列下两者逐元素同 ⇒ 退化零回归)
    pub fn with_vlm_rope(mut self, cos: &'a TensorOps, sin: &'a TensorOps) -> Self {
        self.vlm_rope = Some(RopeTables { cos, sin });
        self
    }
}

/// decode 步 KV 直排缓冲(引用形态:块归 engine,层零所有权)。
///
/// - `k_cache` / `v_cache`:常驻池块引用(`TensorOps::of_block`),
///   [max_slots, Hkv, HD] f32,块只增不减;
/// - `slots` / `kv_lens`:[bs] f32 数值过线(契约 5;核内 cast);
///   **kv_lens 含本步**(kernel 先写 cache 后打分,caller 传 past+1)。
///   (2026-09-26 C4 重组:自 layers/attention 归位协议层 —— 它是 runner
///   与层之间的动态依赖词汇,非层私有。)
///   KV paged 核集策略(REQ-HW-01 表驱动;dtype → 布局/核名几何)。
///   None = 该 KV dtype 暂无 paged 核集 → 层/引擎走 legacy 回退
///   (naive slot 直排 + token-major 池)。新增 dtype = 补表行 +
///   flatten 对应 vendor 模板臂(vendor 核 cache_t 本就参数化)+
///   REGISTRY 登记,调用点零改动(operator-matrix Q0 激活域跟随者律)。
#[derive(Clone, Copy, Debug)]
pub struct KvPagedPolicy {
    /// 池页大小(tokens/block;decode v1/v2 bs32 特化 + prefill 契约交集)
    pub page: usize,
    /// x 向量宽(elements)= 16B / sizeof(cache_t);classic 布局
    /// `[nb, Hkv, hd/x, page, x]` 的最后一维
    pub x: usize,
}

/// 按 KV dtype 取策略(FP8/Q4:REQ-CTX-03 预留——vendor 核模板臂已备,
/// flatten + registry 随 KV 量化立项;bf16 同理)
pub fn kv_paged_policy(dt: Dtype) -> Option<KvPagedPolicy> {
    match dt {
        // f16:16B / 2B = 8 向量宽。页 32 = 生产档定谳(2026-10-01 配对律
        // 结案):页 = decode v1 wrapper × prefill 核的交集 —— prefill 核
        // bs32 特化(vendor 契约 BLOCK∈{32,64}),decode 取 bs32 wrapper
        // (TG=1;bs16/TG=2 曾误判“跨块错值”,实为 wrapper×池页错配,
        // 三轮立案,详见 attention.rs v1_name 注)。页 16 需 bs16 prefill
        // 实例化(越契约)方可启,届时只改此处。
        Dtype::F16 => Some(KvPagedPolicy { page: 32, x: 8 }),
        _ => None,
    }
}

pub struct KvBuffers {
    pub k_cache: TensorOps,
    pub v_cache: TensorOps,
    pub slots: TensorOps,
    pub kv_lens: TensorOps,
    /// 分页块表 [num_seqs, max_blocks](f32 物理块号;classic paged 核与
    /// 后续 FlashInfer 共用)。恒等表 = 单序列连续分配(物理块 = 逻辑块)。
    /// naive 旧核不解引用(测试/回退路径可填哑表)。
    pub block_tables: TensorOps,
}

/// v2 分页 decode scratch(在线 softmax 分块 + LSE 归并;E-decode 主攻
/// 2026-10-04)。引擎捕获前持久块(指针烘焙图安全);None = 层走 v1 既有
/// 路径(全部存量测试/构造点零迁移)。
/// - `exp_sums` / `max_logits`:[1, hq, nparts] f32,v2 主核写、reduce 读;
/// - `tmp_out`:[1, hq·nparts·hd] f16(v2 节点输出写同布局,reduce 读;
///   核内 stride = max_nparts = 声明 nparts);
/// - `nparts` = ceil(nb·page/512),launch grid z = nparts(核从 gridDim.z 读)。
#[derive(Clone)]
pub struct AttnV2Scratch {
    pub exp_sums: TensorOps,
    pub max_logits: TensorOps,
    pub tmp_out: TensorOps,
    pub nparts: usize,
    /// 伪序列数(§三十一:decode=1;verify=depth1,8 token 当 8 伪序列,
    /// grid.y 并行 + partition 维 = flash-decoding 形态;层臂按此防误配)
    pub seqs: usize,
}

impl AttnV2Scratch {
    /// 取前 `seqs` 条伪序列的**视图**(同一底层块,零拷贝)。
    ///
    /// 用途:MTP extend 的 T=m+1 随接受数变化,但复用 target verify 的
    /// 4 序列 scratch(seqs=4)。核只写前 `seqs` 行,底层块更大 ⇒ 安全。
    /// 返回 None = 越界/空(调用方回落 chunked,不静默截断)。
    pub fn prefix(&self, seqs: usize) -> Option<Self> {
        if seqs == 0 || seqs > self.seqs {
            return None;
        }
        if seqs == self.seqs {
            return Some(self.clone());
        }
        let es_row = *self.exp_sums.shape().last()?;
        let ml_row = *self.max_logits.shape().last()?;
        let to_row = *self.tmp_out.shape().last()?;
        Some(Self {
            exp_sums: self.exp_sums.slice_view(0, vec![seqs, es_row]),
            max_logits: self.max_logits.slice_view(0, vec![seqs, ml_row]),
            tmp_out: self.tmp_out.slice_view(0, vec![seqs, to_row]),
            nparts: self.nparts,
            seqs,
        })
    }
}

/// XQA verify/decode 面(2026-10-13 D3 立案;kNHD fp8 池 + spec-dec q=1+K)。
/// 张量全由 engine/测试注入(层内零自建 —— 捕获期禁 Htod 同 FI 先例):
/// - `page_table` **i32** `[1, nb]`(owl 现有 bt 是 f32 → 需 i32 孪生)
/// - `seq_lens` **u32** `[1]` = **整请求总 KV 长**(不是逐行前缀长)
/// - `mask` **u32** `[1, q_seq_len, ceil(q/32)]`:行 i = 块内第 i 个 token,
///   bit k = 允许看块内第 k 个 key,因果 = `(1<<(i+1))-1`
/// - `semaphores` **u32**(KV-split 归约计数;**每次发射前必须为零**)
/// - `scratch` 工作区(持久;参考实现 256MiB)
#[derive(Clone)]
pub struct XqaVerifyCtx {
    pub page_table: TensorOps,
    pub seq_lens: TensorOps,
    pub mask: TensorOps,
    pub semaphores: TensorOps,
    pub scratch: TensorOps,
    pub q_seq_len: usize,
    pub max_seq_len: usize,
    pub sm_count: usize,
}

/// KV 动态上下文:每步由 runner 构造。(注:decode 直排路径暂走
/// [`KvBuffers`];本结构为 paged 路线预留。)
#[derive(Clone, Debug)]
pub struct KvCtx {
    pub step: u64,
    pub slots: Vec<u32>,
}

/// 模型层统一接口
pub trait Module {
    /// 计算声明:xs → y(同步 · total · 纯描述;ctx = 动态依赖,
    /// 缺依赖 → 层侧毒值声明,eval 边界收割 —— 与未装载槽同构)
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps;
}

// ============================================================================
// §2 数据源:执行器的取数对象(测试 HashMap / 将来 safetensors / 层内表)
// ============================================================================

/// **取值三态**(2026-10-17 审查 §③:错误面升 `Result`)。
///
/// 为什么不能只用 `Option`:`None` 要同时表达"**正常语义**"与"**真错误**",
/// 结果把 mmap/打包/索引失败全压成"数据源缺键"(假归因) ——
/// 现场:`AwqSource::build_linear` 的 `Result` 被 `.ok()?` 抹平,
/// 运维看到"缺键"去查键名,真因可能是坏分片。
///
/// - `Ok(Some(v))` —— 拿到了;
/// - `Ok(None)` —— **正常语义**:键不归我管 / 区间越界 / 该口未实现
///   (调用方收割为缺键/形状错,**不许 panic**,见各口 doc);
/// - `Err(e)` —— **真错误**:读失败 / 打包失败 / 索引损坏 / 锁中毒 ⇒
///   响亮上抛到装载入口(全链顶层恒为 `Result<_, ModelError>`)。
pub type Taken<T> = Result<Option<T>, ModelError>;

/// 权重数据源(流式,2026-09-26 M-e:查键即**取走所有权** ——
/// 数据经变换/上传后随宿主消亡,源驻留随装载单调下降,主机上永远
/// 只有"在途"的一份;同键多 Want(tied 双槽)由装载域按键分组,
/// 一次 take 供全组)。
pub trait WeightSource {
    /// **源是否自己承担分片**(默认 `false` = 由 `formats::shard` 的 span
    /// 包装器翻译"目标区间 → 源区间")。
    ///
    /// GGUF 臂为 `true`:量化块的列切是**块算术**,且 `ssm_out` 的列要按
    /// v-head **grouped 序聚合** —— 通用包装器的"连续源区间"模型表达不了
    /// (见 `formats::gguf::shard`)。此时 `eval_load` **跳过**分片 pass,
    /// 由源自己按 `(slice_id, world)` 交付本 rank 的字节;`elem_len` 也必须
    /// 报分片口径(装载面按 `shape.product` 对账,不符即拒)。
    fn owns_shard(&self) -> bool {
        false
    }

    /// **装载前预取**(可选;默认 no-op)。`keys` = 本档将消费的键,顺序
    /// 与装载循环一致。mmap 源可在后台把底层区间读入页缓存,让冷盘 I/O
    /// 与 CPU 物化重叠;实现必须廉价、只读、无副作用(不得 resolve/物化)。
    fn prefetch(&self, _keys: &[&str]) {}

    /// 等 [`Self::prefetch`] 的后台 I/O 收口(默认 no-op)。
    fn prefetch_finish(&self) {}

    fn take(&self, key: &str) -> Taken<Vec<f32>>;

    /// 区间取数(流式大张量:分块转换上传,主机在途 = 单块)。
    /// 默认 = 整取切片;流式源(mmap)覆写为按需转换区间,且
    /// **不移除条目**(同一键的多块重复取)。
    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Taken<Vec<f32>> {
        // 越界 = `Ok(None)`(调用方收割为缺键/形状错;禁 panic)
        let v = self.take(key)?;
        Ok(v.and_then(|v| v.get(offset_elems..offset_elems + len).map(<[f32]>::to_vec)))
    }

    /// 键的元素数(错误归因用:缺键 vs 长度不符;默认 `Ok(None)` = 未知)
    fn elem_len(&self, _key: &str) -> Taken<usize> {
        Ok(None)
    }

    /// **源天然元素 dtype**(原始字节通道可用时;`Ok(None)` = 无原始通道 ⇒
    /// 走元素转换路径)。供"原始字节上卡 + 设备端转换"臂判定
    /// (2026-10-16 装载提速:CPU 只 memcpy,转换挪到 GPU)。
    ///
    /// ⚠️ **当前消费集 = 仅 `Some(Dtype::BF16)`**(`load.rs` 两处:暂存槽 sizing 与
    /// 设备 cast 臂门)—— 实现里返回的 F32/F16 是**源的事实陈述**,不是已接线的能力;
    /// 谁要用这两档,必须连带把消费侧(臂条件)一起改,并补"两臂逐位一致"的门。
    /// 2026-10-17 审查 §55:接口宽于消费集 ⇒ 此处写实,防后来者把它当"已实现"。
    fn native_dtype(&self, _key: &str) -> Taken<Dtype> {
        Ok(None)
    }

    /// 原始字节区间直拷(零元素转换;`dst.len() ≥ len × 天然宽度`)。
    /// 上卡后由设备 cast 核完成 dtype 转换 —— 本口只保证"字节原样"。
    /// `Ok(None)` = 不支持(调用方回退到 [`Self::convert_chunk_into_bytes`])。
    fn copy_native_into(
        &self,
        _key: &str,
        _offset_elems: usize,
        _len: usize,
        _dst: &mut [u8],
    ) -> Taken<()> {
        Ok(None)
    }

    /// 分块转换**直写**目标缓冲(2026-09-26 二拷贝预算:转换这 1 次
    /// 直达 pinned 租约,不再有中间 Vec;流式源覆写为 mmap 直解码)。
    /// f16 基线尾批:字节口径 + dtype 参数化 —— F32 写 f32 LE、F16 写
    /// half LE;流式源(BF16 条目)直写,免 f32 中间 Vec。
    ///
    /// `dst` 太小 = **调用方违约**,响亮拒(旧版返回 `None` ⇒ 被上层
    /// 收成"缺键",假归因);`Ok(None)` 只留给"目标 dtype 不支持"。
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let Some(v) = self.take_range(key, offset_elems, len)? else {
            return Ok(None);
        };
        let need = match dtype {
            Dtype::F32 => len * 4,
            Dtype::F16 | Dtype::BF16 => len * 2,
            _ => return Ok(None),
        };
        if dst.len() < need {
            return Err(ModelError::Msg(format!(
                "convert_chunk_into_bytes: dst {}B < 需 {need}B({dtype:?}, len {len})",
                dst.len()
            )));
        }
        for (i, f) in v.iter().enumerate() {
            match dtype {
                Dtype::F32 => dst[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes()),
                Dtype::F16 => {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(*f).to_le_bytes())
                }
                Dtype::BF16 => {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(*f).to_le_bytes())
                }
                _ => unreachable!("上面已筛过 dtype"),
            }
        }
        Ok(Some(()))
    }
}

/// Arc 转发(F5-3:并行装载所有权入任务;全方法直通内层)
impl<S: WeightSource + ?Sized> WeightSource for std::sync::Arc<S> {
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        (**self).take(key)
    }
    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Taken<Vec<f32>> {
        (**self).take_range(key, offset_elems, len)
    }
    fn elem_len(&self, key: &str) -> Taken<usize> {
        (**self).elem_len(key)
    }
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        (**self).convert_chunk_into_bytes(key, offset_elems, len, dst, dtype)
    }
}

impl WeightSource for std::collections::HashMap<String, Vec<f32>> {
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        Ok(self.get(key).cloned())
    }
    fn elem_len(&self, key: &str) -> Taken<usize> {
        Ok(self.get(key).map(|v| v.len()))
    }
}

/// 容器内表源(rope cos/sin 等纯计算产物;借用层内 Vec,生命周期随层)
pub struct TableSource<'a> {
    entries: Vec<(&'static str, &'a [f32])>,
}

impl<'a> TableSource<'a> {
    pub fn new(entries: Vec<(&'static str, &'a [f32])>) -> Self {
        Self { entries }
    }
}

impl WeightSource for TableSource<'_> {
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        Ok(self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_vec()))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        Ok(self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| v.len()))
    }

    /// 表即 host f32:直接切片转写字节 —— F16 走 crate::f16c(F16C 单
    /// pass;默认逐元素 from_f32 实现是 debug 档标量税,rope 表 67MB
    /// 曾付 ~0.5s,F5-4 收口时实测定谳)
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let Some(v) = self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| *v) else {
            return Ok(None);
        };
        let Some(src) = v.get(offset_elems..offset_elems + len) else {
            return Ok(None);
        };
        match dtype {
            Dtype::F16 => {
                if dst.len() < len * 2 {
                    return Err(ModelError::Msg(format!(
                        "convert_chunk_into_bytes(表源): dst {}B < 需 {}B",
                        dst.len(),
                        len * 2
                    )));
                }
                crate::f16c::f32_slice_to_f16_bytes(src, &mut dst[..len * 2]);
                Ok(Some(()))
            }
            Dtype::F32 => {
                if dst.len() < len * 4 {
                    return Err(ModelError::Msg(format!(
                        "convert_chunk_into_bytes(表源): dst {}B < 需 {}B",
                        dst.len(),
                        len * 4
                    )));
                }
                // LE 主机:f32 位型即 LE 字节 —— 整块 memcpy(逐元素
                // to_le_bytes 循环在 debug 档 ~25ns/elem,曾占 211ms)
                let bytes =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, len * 4) };
                dst[..len * 4].copy_from_slice(bytes);
                Ok(Some(()))
            }
            _ => Ok(None),
        }
    }
}

/// C10(2026-09-26,批 8;M-e 可插拔化):检查点键名约定 ——
/// 层内短键 → checkpoint 全键的改写规则。Model 在 `layout()` 期经
/// `LoaderOps::map_keys` 应用约定(声明期改写,Want 清单里已是最终键),
/// 解释器零约定感知:
/// - [`LlamaFamily`]:HF/Llama 系默认(`{base}.embed_tokens.weight` /
///   `{base}.layers.{i}.{local}.weight` / `{base}.{local}.weight`);
/// - 模型特有约定住 `specs/<model>.rs`(拆分律 §四 23),如 Qwen3.5
///   的 `linear_attn.`/`self_attn.` 子前缀与 `A_log`/`dt_bias` 裸键。
pub trait KeyConvention {
    /// tied embedding:局部键(恒 "weight")→ checkpoint 全键
    fn embed_key(&self, local: &str) -> String;
    /// 第 i 层:局部键 → checkpoint 全键
    fn layer_key(&self, i: usize, local: &str) -> String;
    /// final norm:局部键(恒 "norm")→ checkpoint 全键
    fn norm_key(&self, local: &str) -> String;

    /// **untied lm_head**(第 4 类键;2026-10-19 GGUF 轮新增)。
    ///
    /// 局部键 = `lm_head`(裸)或带量化后缀(`lm_head.qweight` …);
    /// 默认实现 = 既行为(`lm_head` → `lm_head.weight`,后缀族原样)。
    /// GGUF 覆写:`lm_head` → **`output.weight`**(源的命名事实)。
    fn head_key(&self, local: &str) -> String {
        if local == "lm_head" {
            "lm_head.weight".to_string()
        } else {
            local.to_string()
        }
    }
}

/// HF/Llama 系默认约定(纯前缀 + `.weight` 叶;无子前缀无裸键)
pub struct LlamaFamily {
    base: String,
}

impl LlamaFamily {
    /// base = 检查点基座(多模态仓 `model.language_model`,纯文本仓 `model`)
    pub fn new(base: impl Into<String>) -> Self {
        Self { base: base.into() }
    }
}

impl KeyConvention for LlamaFamily {
    fn embed_key(&self, local: &str) -> String {
        format!("{}.embed_tokens.{local}", self.base)
    }
    fn layer_key(&self, i: usize, local: &str) -> String {
        format!("{}.layers.{i}.{local}.weight", self.base)
    }
    fn norm_key(&self, local: &str) -> String {
        format!("{}.{local}.weight", self.base)
    }
}


/// f32 → LE 字节(host 装载辅助;装载域唯一权威)
pub(crate) fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// §3 空包指针:块句柄格(装载产物写入处;decl 读取处)
// ============================================================================

/// 共享块句柄格(空包):装载前 None,执行器物化后 Some(块句柄)
#[derive(Clone, Default)]
pub struct SlotCell(Arc<Mutex<Option<Bytes>>>);

impl SlotCell {
    /// **锁取用口**(2026-10-17 审查 §六轮,按判例 §52 两分法定档)。
    ///
    /// 载荷 = 单 `Option<Bytes>`(空包 → 装载产物句柄),**写入是单次 move**
    /// ⇒ 持锁线程 panic 不会留下"半写"状态;且中毒**粘性**(`lock()` 之后恒
    /// Err)⇒ 取值 = 就地取回 + **复位毒标** + 告警(恢复可见、且不每取一次
    /// 走一遍恢复分支)。档位 = **可安全恢复**(非消费式状态,与 safetensors
    /// 的索引相反 —— 那里必须响亮拒)。
    fn guard(&self) -> std::sync::MutexGuard<'_, Option<Bytes>> {
        match self.0.lock() {
            Ok(g) => g,
            Err(p) => {
                let g = p.into_inner();
                owl_shared::log_warn!(
                    "load",
                    "SlotCell 锁中毒:就地取回(单 Option 载荷,无半写风险)并复位毒标"
                );
                self.0.clear_poison();
                g
            }
        }
    }

    fn deliver(&self, b: Bytes) {
        *self.guard() = Some(b);
    }

    fn loaded(&self) -> Option<Bytes> {
        self.guard().clone()
    }
}

// ============================================================================
// §4 LoaderOps:数据需求清单(纯元数据;零数据;total)
// ============================================================================

/// 数据存在形式(布局)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// 源即装载布局(行主序直给)
    Direct,
    /// 源为 [shape[1], shape[0]] 行主序,装载时 host 转置
    /// (Linear 惯例:权重 safetensors 原生 [out,in],装载 [in,out])
    Transposed,
    /// 设备重排(2026-10-01 AWQ 装载提速):源供**原始 packed** 字节
    /// (线性 raw_elems,U32),DMA 上卡后 launch kernel 就地重排到
    /// want.shape(marlin 布局),原始块随即回收 —— CPU 重量排出热路径
    /// (141.7→37.0s 的 mat.ct 33s → GPU ~0.5s)
    DeviceRearrange {
        /// 语义算子(动作词表;实现名执行期 resolve)
        op: crate::ops::SemanticKernel,
        /// 原始 packed 行数(= out)
        rows: usize,
        /// 原始 packed u32 列数(= in/8)
        cols: usize,
    },
    /// **NVFP4 权重打包**(2026-10-28 装载管线 P2):源 = f16/bf16 全精度权重
    /// (`[rows, cols]` 行主序),装载期在 **host** 量化为
    /// `q`(`[rows, cols/2]` 的 e2m1 对)+ `sf`(`rows*cols/16` 字节的
    /// CUTLASS 交错块缩放)+ 回填 host 标量 `gb`。
    ///
    /// 与其余三臂的区别:那三条是**搬运/换序**(字节进、字节出),这条是
    /// **真格式变换**(精度从 16 位降到 4 位 + 块缩放)—— 实现在
    /// [`crate::formats::transform::NvFp4Pack`],本变体只携带**事实**
    /// (源的行/列数;目标形状由变换自己算)。
    ///
    /// **一源二槽**:主槽 = `q`(`Want.sink`),`sf` 走 `Want.extra`。
    NvFp4Weight {
        /// 源行数(= out_dim;= 目标 N)
        rows: usize,
        /// 源列数(= in_dim;= 目标 K)
        cols: usize,
    },
}

/// 空包回填口(执行器写入;使用者零感知)
#[derive(Clone)]
pub struct Sink(SlotCell);

/// **host 标量回填口**(一源**一个**标量;2026-10-28 NVFP4 的权重全局缩放 `gb`)。
///
/// 为什么另开一个口而不复用 [`Sink`]:Sink 交付的是**设备块**(`Bytes` 句柄),
/// 装载面按它记 manifest / 做回读校验;而 `gb` 是 4 字节**主机常量**,
/// 发射时直接作为 kernel 标量参数(`arg_f32`),不上卡、不进 manifest。
/// 两个东西的生命周期与校验面都不同,合成一个类型只会让两者互相污染。
#[derive(Clone)]
pub struct ScalarSink(std::sync::Arc<std::sync::atomic::AtomicU32>);

impl Default for ScalarSink {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ScalarSink {
    /// 建口(初值 = f32 位型;缺省 1.0 = “尚无缩放”,乘法中性元)
    #[must_use]
    pub fn new(v: f32) -> Self {
        Self(std::sync::Arc::new(std::sync::atomic::AtomicU32::new(
            v.to_bits(),
        )))
    }

    /// 装载期回填(f32 位型走 `AtomicU32`:`Model` 要 `Sync`,写入晚于构造)
    pub fn store(&self, v: f32) {
        self.0.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// 取值(发射面读)
    #[must_use]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// **附加槽**(一源多槽里的第 2..N 件;2026-10-28 装载管线 P1)。
///
/// 主槽仍是 [`Want`] 的 `dtype`/`shape`/`sink`(**旧三臂与 manifest 零变化**),
/// `extra` 只装同一源键的其余产物(NVFP4 的 `sf`)。
/// 为什么不一步改成 `slots: Vec<..>`:那要同时改分片 pass(按 `shape` 改写
/// 分片形状)、manifest 与所有声明点 —— 在只有一种 1:N 变换时不值得;
/// 设计 §8 的 P3 再把主槽并进 `slots`(那时分片规则也一并由变换声张)。
#[derive(Clone)]
pub struct SlotWant {
    /// 诊断名(`q`/`sf`/…)—— 日志与 manifest 用;判决不按名字,按 `dtype × shape`
    pub name: &'static str,
    pub dtype: Dtype,
    pub shape: Shape,
    pub sink: Sink,
}

impl Sink {
    pub(crate) fn deliver(&self, b: Bytes) {
        self.0.deliver(b);
    }

    /// 测试专用空口(S4 ② 分片装载的单测要裸造 Want,不挂真实槽)
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self(SlotCell(Arc::new(Mutex::new(None))))
    }
}

/// 单条数据需求(纯元数据 + 回填口)。key 为 owned:容器聚合
    /// (Model::layout)经 map_keys 改写为 checkpoint 全键 —— 键名
    /// 约定在声明期应用,Want 清单里已是最终键。
#[derive(Clone)]
pub struct Want {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Shape,
    pub layout: Layout,
    /// **融合权重的段长**(源坐标系;S4 ②)。`Some` = 本张量是拼接而来
    /// (q⊕gate / q⊕k⊕v / conv 通道),分段切按它展开;`None` = 不分段。
    ///
    /// 为什么挂在 Want 上:段长是**权重的事实**(这个张量怎么拼的),由它的
    /// 构造者(Attention/GDN)用 [`LoaderOps::with_segments`] 声明;
    /// **怎么切**仍归装载面(规则表 `plan::rules::rule_of`)—— 层不感知切分。
    pub segments: Option<Vec<usize>>,
    /// **层声明的规则覆盖**(P1.1,2026-10-19)。`Some` = 本张量的切分规则
    /// **不由键名决定**,由它的构造者声明(唯一消费者 = lm_head 的词表切:
    /// `Linear::with_tp_rule(UniformCol)`);`None` = 走键表默认
    /// ([`crate::plan::rules::rule_of_want`])。
    ///
    /// 为什么需要:键表按**键名**推规则(`lm_head` = `Replicate`),而
    /// "这个头切不切"是**装配期事实**(`ModelSpec::head_split`,会随档位变)
    /// —— 键名里没有这个信息。层是规则的声明者,装载面是消费者:
    /// `build_shard_pass` 优先读本字段,缺省回落键表 ⇒ 只有**真覆盖**的键
    /// (现仅词表切档的 lm_head)行为改变。
    pub rule: Option<crate::plan::rules::WeightRule>,
    /// **附加槽**(1:N 面):同一源键的其余产物(2026-10-28 NVFP4 的 `sf`)。
    /// 空 = 单槽(旧路径全部如此,零行为变化)。
    pub extra: Vec<SlotWant>,
    /// **host 标量回填口**(与槽无关的旁路产物;NVFP4 的 `gb`)。
    pub scalar: Option<ScalarSink>,
    pub(crate) sink: Sink,
}

/// 需求清单(封闭有序;组合层 = 子层清单 chain 聚合)
#[derive(Clone, Default)]
pub struct LoaderOps {
    wants: Vec<Want>,
}

impl LoaderOps {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn want(w: Want) -> Self {
        Self { wants: vec![w] }
    }

    /// 聚合(组合层;清单顺序 = 装填顺序)
    pub fn chain(mut self, other: LoaderOps) -> Self {
        self.wants.extend(other.wants);
        self
    }

    /// 键改写组合子(旧世界 VarBuilder.pp 的声明式对应物):容器
    /// layout 聚合子层清单时,把局部键改写为 checkpoint 全键。
    /// 改写后键可重复(如 tied 双 Want 同键)—— 查键取同一源条目。
    pub fn map_keys(mut self, mut f: impl FnMut(&str) -> String) -> Self {
        for w in &mut self.wants {
            w.key = f(&w.key);
        }
        self
    }

    pub fn wants(&self) -> &[Want] {
        &self.wants
    }

    /// **可变访问**(S4 ②:装载面的分片化 pass 就地改写 shape / 布局)
    pub fn wants_mut(&mut self) -> &mut [Want] {
        &mut self.wants
    }

    /// **声明融合权重段长**(S4 ②;仅对**单 Want** 的链调用 —— 如
    /// `qw.layout_as_device_rearrange(..).with_segments(vec![nk*hk, nk*hk, nv*hv])`)。
    ///
    /// 语义:本 Want 的源张量是 `Σsegs` 行拼接而成(源坐标系),分段切时
    /// 每段独立等分([`plan::span::plan_span`])。
    pub fn with_segments(mut self, segs: Vec<usize>) -> Self {
        for w in &mut self.wants {
            w.segments = Some(segs.clone());
        }
        self
    }

    /// 便捷:单 Want 的链(debug 断言:段长只对单 Want 有意义)
    pub fn is_single(&self) -> bool {
        self.wants.len() == 1
    }

    /// **定向标注段长**(S4 ②;推荐形态):只对键尾匹配的 want 生效
    /// (`in_proj_qkvz` / `conv1d` / `q_proj` 这类融合权重,由它们的构造者声明)。
    ///
    /// **匹配面 = 语义基键**(2026-10-17):量化档的 want 键是派生键
    /// (`q_proj.packed_raw` / `q_proj.scales` / …),只用 `tail` 会**漏标**
    /// ⇒ 段切键在 `plan_span` 处"段长为空"拒启。段长声明 = **语义单位**
    /// (q_gate / k_dim / v_dim);派生键的 n 侧若是打包单位(scales: n_pack、
    /// zeros: n_pack/8),由 `build_shard_pass` 按形状等比换算(非整除则拒)。
    pub fn with_segments_for(mut self, key_tail: &str, segs: Vec<usize>) -> Self {
        for w in &mut self.wants {
            let bk = crate::plan::rules::base_key(&w.key);
            if crate::plan::rules::tail(bk, key_tail) {
                w.segments = Some(segs.clone());
            }
        }
        self
    }

    /// **声明规则覆盖**(P1.1,2026-10-19):层声明"本清单各 Want 的切分规则"
    /// —— 与 [`LoaderOps::with_segments`] 同律(声明者 = 构造者,
    /// 消费者 = `build_shard_pass`)。**只在真覆盖时调用**
    /// (判据见 `Linear::layout`:声明规则 ≠ 键表默认)。
    pub fn with_rule(mut self, rule: crate::plan::rules::WeightRule) -> Self {
        for w in &mut self.wants {
            w.rule = Some(rule);
        }
        self
    }
}

// ============================================================================
// §4.5 LoadManifest:装载清单(eval_load 的对账面;校验/审计 vocabulary)
// ============================================================================

/// 单条装载记录:checkpoint 全键 + 声明形状 + 布局 + VRAM 块句柄。
/// 校验测试据此 dtoh 读回显存,与独立锚逐位比较。
#[derive(Clone, Debug)]
pub struct LoadEntry {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Shape,
    pub layout: Layout,
    pub block: Bytes,
    /// staged 字节 FNV-1a 校验和(E5-DF4 装载校验;0 = 未校验:
    /// DeviceRearrange 臂 staged = raw 而块 = 重排后,回读不可比)
    pub csum: u64,
}

/// 装载清单(顺序 = Want 清单顺序)
#[derive(Clone, Debug, Default)]
pub struct LoadManifest(pub Vec<LoadEntry>);

impl LoadManifest {
    pub fn entries(&self) -> &[LoadEntry] {
        &self.0
    }
    pub fn push(&mut self, e: LoadEntry) {
        self.0.push(e);
    }
}

impl FromIterator<LoadEntry> for LoadManifest {
    fn from_iter<I: IntoIterator<Item = LoadEntry>>(iter: I) -> Self {
        LoadManifest(iter.into_iter().collect())
    }
}

// ============================================================================
// §5 LoaderCtx:装载语境(layout 的 ctx;形状切分/dtype 决策的词汇)
// ============================================================================

/// 量化计划(构造期定形,REQ-PRE-01):模型需要什么形态,在构造时
/// 就传入 —— 禁 `enable_*` 可变后置范式(声明期定形,层容器零突变面)。
/// 尺寸门控(marlin_eligible)在 Linear 构造期定形:不安全 n 自动落 f16
/// (装载源同谓词,单一来源)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantPlan {
    /// f16 直读(基线;默认)
    #[default]
    F16,
    /// W4A16 marlin(g128-sym;ct pack-quantized U4B8)
    W4A16,
    /// W4A16 marlin **AWQ 臂**(g32-asym + zp;ct pack-quantized kU4,
    /// 2026-10-01 cyankiwi/Qwen3.8-27B-AWQ-INT4 装载线):GEMM 走
    /// kU4(has_zp)内核,B 打包与 U4B8 同套,zp 单独烘焙 marlin 布局
    W4A16Awq,
    /// **W4A8 × AWQ**(S2;2026-10-17):权重=**a8 布局**三件套
    /// (`qweight_a8`/`scales_a8`[int16 位型]/`zeros_a8`)+ 全局因子
    /// (`a8_factor` f32[1]);激活走 `quant_int8_row`(per-token int8)+
    /// `marlin_gemm_w4a8_awq`(kS8×kU4,int8 张量核 ⇒ prefill 2.29×)。
    ///
    /// 与 `W4A16Awq` 的差别**全在布局与激活面**:权重同源、模型同构 ⇒
    /// 换 plan 即 A/B(端口/配置不动)。
    W4A8Awq,
    /// **NVFP4 块缩放**(2026-10-28):权重装载期 RTN 量化成
    /// `q`(`u8 [out, in/2]`,每字节两个 `e2m1`)+ `sf`(`u8`,`out*in/16` 字节,
    /// CUTLASS 交错布局),另带 per-tensor 全局缩放 `gb`(host 标量)。
    ///
    /// 前向 = **一次 `nvfp4_gemm`**(家族内自己做设备侧激活量化 + CUTLASS
    /// `m16n8k64` 块缩放 MMA)。**统一臂**:不按 m 分流(decode m=1 也走这条)。
    NvFp4,
    /// **GGUF 块型**(2026-10-19 gguf 轮):权重以 **GGUF 块字节原样驻留**
    /// (槽 dtype = `U8`、形状 `[out, row_bytes]`),forward = **两发**:
    /// `GgufQuantAct{_F16,_Bf16}`(激活 → q8_1) + `GgufGemm`(dequant-in-kernel)。
    ///
    /// **逐张量型**:26B/4B 等混型 GGUF 里同一个模型不同张量型不同
    /// ⇒ 由 [`PlanSource`] 逐键给(设计刀 0);未移植型 = 构造期**响亮拒**。
    Gguf(crate::formats::gguf::GgmlTy),
    /// **int8 per-row 头**(P1.5-a,2026-10-20):只对 **lm_head** 有意义
    /// (装载期现算:`s_o = absmax/127`,`q = RN-even(w/s)`;头走
    /// `owl_gemv_i8w_{f16,bf16}`(m=1)或 2026-10-22 起的
    /// `owl_gemv_i8w_mt{,7}_{f16,bf16}`(spec m=7/8,权重只读一遍))。
    ///
    /// **不走 `Linear`**:`Linear` 已有四条臂(F16/W4A16/W4A16Awq/W4A8Awq),
    /// 再插一条会变成"万能类型";int8 头只有 lm_head 一个消费者 ⇒
    /// 由 `Embedding::head_i8` 单槽承载(回退 = 删槽,bf16 原路不动)。
    ///
    /// **表计划位也复用它**(P8,2026-10-22):`Model::new_with_embed` 传
    /// `Some(I8W)` ⇒ `Embedding::set_table_plan` 建 int8 表槽(按调用位置
    /// 区分,不与 `head_i8` 同槽);仅 untied。
    I8W,
    /// **W8A8 头**(2026-10-22):权重同 `I8W`(per-row int8),前向额外做
    /// per-token int8 激活量化 + `__dp4a`。只对 lm_head 有意义;
    /// 仅用于 spec m>1,prefill/MTP 的 m=1 仍走 `I8W`。
    I8W8,
}

/// **逐张量量化计划表**(混型源的计划来源;2026-10-19 GGUF 轮)。
///
/// 为什么需要:量化型在 GGUF 里是**逐张量**事实 —— `Qwen3.8-27B-Q4_K_M`
/// 同一模型混存 Q4_K(193 张)/Q8_0(288)/Q6_K(17)/F32(353),
/// `Holo-3.1-9B-Q5_K_M` 更是**同一角色跨层不同型**(`attn_qkv` Q5_K 12 层
/// + Q6_K 12 层)⇒ 单一 [`QuantPlan`] 值表达不了。
///
/// 纪律**不变**:计划仍在**构造期**定形(禁 `enable_*` 可变后置范式)——
/// 变的只是定形依据,从"全局一个值"变成"检查点全键 → 型"的表。
/// 键算全复用 [`KeyConvention`](与 `Model::layout` 的 `map_keys` 同源;
/// 不存在第二套键规则)。
///
/// **总量性与响亮拒**:表的覆盖面由源保证(每个 GGUF 条目都有 ggml 型);
/// 模型问出表外的键(缺键 / 形态不合法)⇒ 记进 [`PlanTable::take_unresolved`],
/// 由 `Model::new` **响亮拒** —— 禁静默落 F16(那会让显存账静默失真)。
pub trait PlanTable {
    /// 检查点全键 → 计划(`out_dim`/`in_dim` = 目标形状,供形态门控用)
    fn plan_at(&self, full_key: &str, out_dim: usize, in_dim: usize) -> QuantPlan;
    /// 取走"模型问过但表里没有"的键清单(**取走即清**;构造期尾声消费)
    fn take_unresolved(&self) -> Vec<String>;
}

/// **源的事实**(构造期常量;与 plan 正交 —— plan 答"这个键什么精度",
/// facts 答"这个源把什么烘焙进了数值/键面")。
///
/// 为什么挂在 [`PlanSource`] 上而不是单开一路参数:facts 的**消费者**
/// (RmsNorm / GDN 分段)正是拿 plan 的那批构造器 ⇒ 同一注入通道,层构造器
/// 签名不用再加一路(`impl Into<PlanSource>` 形态对既有调用点零改动)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceFacts {
    /// **norm 的 gamma 已烘焙 `+1`**(GGUF 转换器行为:除 `linear_attn.norm`
    /// 外一律 `w+1`)⇒ 该档 RmsNorm 用 `×w`(零变换);
    /// `false`(HF 一切档位)= 零中心 `×(1+w)`(`RmsNorm::new_add_one`)。
    /// 详见设计 §11.3。
    pub norm_baked_plus_one: bool,
    /// **融合键的分段可异型**:GGUF 逐张量定精度 ⇒ `attn_qkv ⊕ attn_gate`
    /// 两段型常常不同(4B 实测 24/24 层 Q6_K vs Q4_K)—— 此时**不可**合成
    /// 单一同型张量,GDN 必须声明两个 Linear(设计 §11.5 ①)。
    pub merged_segments_may_differ: bool,
    /// **v-head 列序已被转换器改成 tiled 序**(GGUF 写侧 `_reorder_v_heads`;
    /// 仅 `nk != nv` 时发生)⇒ `ssm_out` 权重的**内维**是 ggml 序,而 GDN 核
    /// 产出 grouped 序 ⇒ 消费者必须先在**激活侧**做等价置换
    /// (置换跨量化块 ⇒ 权重字节不可逆;设计 §11.4)。
    ///
    /// 消费者 = GDN 层:声明 `ssm_out_perm` 虚拟键(源生成 32 元素块序表)
    /// + `out_proj` 前插一发 `owl_gguf_col_gather`。
    pub vhead_tiled_columns: bool,
}

impl SourceFacts {
    /// 既有一切档位(HF / safetensors / ct / awq):零中心 norm、融合键同型、HF v-head 列序
    pub const HF: SourceFacts = SourceFacts {
        norm_baked_plus_one: false,
        merged_segments_may_differ: false,
        vhead_tiled_columns: false,
    };
    /// GGUF 起点族(norm 已 +1;逐张量型 ⇒ 融合键分段独立;tiled v-head 列序)
    pub const GGUF: SourceFacts = SourceFacts {
        norm_baked_plus_one: true,
        merged_segments_may_differ: true,
        vhead_tiled_columns: true,
    };
    /// **GGUF + TP 分片档**(W-1):装载期已按 v-head **grouped 序聚合**
    /// (`GgufSource` 自担分片)⇒ 本 rank 的权重内维就是 grouped 序,
    /// **不需要** A2 激活侧列序表(`vhead_tiled_columns = false`)。
    pub const GGUF_TP: SourceFacts = SourceFacts {
        norm_baked_plus_one: true,
        merged_segments_may_differ: true,
        vhead_tiled_columns: false,
    };

    /// norm 的 I/O 语义:GGUF 用 `×w`(`w_off = false`),HF 用 `×(1+w)`
    pub fn rmsnorm_w_off(self) -> bool {
        !self.norm_baked_plus_one
    }
}

/// **计划来源**(构造期注入;值语义 `Copy`)—— 一切 plan 与源事实的唯一入口。
///
/// - [`PlanKind::Uniform`]:全模型统一计划(既有一切档位;零开销臂);
/// - [`PlanKind::Keyed`]:逐键查表(混型源,如 GGUF)。
///
/// `QuantPlan` 经 `From` 直落 Uniform 臂 ⇒ 既有调用点写字面量即可 ——
/// **单接口,非旁路**:Uniform 本身就是一种 [`PlanSource`]。
#[derive(Clone, Copy)]
pub struct PlanSource<'a> {
    kind: PlanKind<'a>,
    facts: SourceFacts,
}

/// 计划的两种来源(内层;外部一律经 [`PlanSource`] 构造/消费)
#[derive(Clone, Copy)]
enum PlanKind<'a> {
    Uniform(QuantPlan),
    Keyed {
        table: &'a dyn PlanTable,
        keys: &'a dyn KeyConvention,
        /// 本层层号(层内局部键算全用);非层键走 [`PlanSource::at`]
        layer: usize,
    },
}

impl<'a> From<QuantPlan> for PlanSource<'a> {
    fn from(p: QuantPlan) -> Self {
        Self { kind: PlanKind::Uniform(p), facts: SourceFacts::HF }
    }
}

impl<'a> PlanSource<'a> {
    /// **逐键查表来源**(混型源接线点;`facts` = 源族事实)
    pub fn keyed(
        table: &'a dyn PlanTable,
        keys: &'a dyn KeyConvention,
        facts: SourceFacts,
    ) -> Self {
        Self { kind: PlanKind::Keyed { table, keys, layer: 0 }, facts }
    }

    /// 覆写源事实(既有牌位一律 [`SourceFacts::HF`],无需调用)
    pub fn with_facts(mut self, facts: SourceFacts) -> Self {
        self.facts = facts;
        self
    }

    /// 源事实(norm 语义 / 融合键分段)
    pub fn facts(&self) -> SourceFacts {
        self.facts
    }

    /// Uniform 臂的计划(Keyed 臂 = None)。层构造器据此决定是否使用
    /// 合并虚拟键(如 MLP gate_up_proj 仅 W4A16/AWQ 目标档启用)。
    pub fn uniform_plan(&self) -> Option<QuantPlan> {
        match self.kind {
            PlanKind::Uniform(p) => Some(p),
            PlanKind::Keyed { .. } => None,
        }
    }

    /// 绑定层号(`Model::new` 每层造一份;Uniform 臂原样返回)
    pub fn at_layer(self, i: usize) -> Self {
        match self.kind {
            PlanKind::Keyed { table, keys, .. } => {
                Self { kind: PlanKind::Keyed { table, keys, layer: i }, facts: self.facts }
            }
            _ => self,
        }
    }

    /// **层内局部键 → 计划**(层构造器唯一入口;Uniform 臂忽略键)
    pub fn of(&self, local: &str, out_dim: usize, in_dim: usize) -> QuantPlan {
        match self.kind {
            PlanKind::Uniform(p) => p,
            PlanKind::Keyed { table, keys, layer } => {
                table.plan_at(&keys.layer_key(layer, local), out_dim, in_dim)
            }
        }
    }

    /// **可选组件**的计划(conv 投影 / selector 投影等「两边不同档」的件)。
    ///
    /// Uniform 臂(既有一切档位)取 `default` —— 这些件在 HF 检查点里恒
    /// 原生精度(`GroupedConv::proj` / `CandidateSelector::proj` 构造即 F16),
    /// **不随主干 plan 变**(改它会静默改变 W4A16/BF16 草稿的数值);
    /// Keyed 臂(GGUF 混型)逐键查表 —— 混型文件里它们**可能就是量化块**。
    pub fn or_default(
        &self,
        local: &str,
        out_dim: usize,
        in_dim: usize,
        default: QuantPlan,
    ) -> QuantPlan {
        match self.kind {
            PlanKind::Uniform(_) => default,
            PlanKind::Keyed { .. } => self.of(local, out_dim, in_dim),
        }
    }

    /// **检查点全键 → 计划**(embed / final norm 等非层键;Uniform 臂忽略键)
    pub fn at(&self, full_key: &str, out_dim: usize, in_dim: usize) -> QuantPlan {
        match self.kind {
            PlanKind::Uniform(p) => p,
            PlanKind::Keyed { table, .. } => table.plan_at(full_key, out_dim, in_dim),
        }
    }

    /// untied `lm_head` 计划(键名属**源**事实;Keyed 臂向**约定**索取 ——
    /// 与 want 侧 [`KeyConvention::head_key`] 同一实现,单源)
    pub fn head(&self, out_dim: usize, in_dim: usize) -> QuantPlan {
        match self.kind {
            PlanKind::Uniform(p) => p,
            PlanKind::Keyed { table, keys, .. } => {
                table.plan_at(&keys.head_key("lm_head"), out_dim, in_dim)
            }
        }
    }

    /// 未解析键清单(Uniform 臂恒空;Keyed 臂取走即清)
    pub fn take_unresolved(&self) -> Vec<String> {
        match self.kind {
            PlanKind::Uniform(_) => Vec::new(),
            PlanKind::Keyed { table, .. } => table.take_unresolved(),
        }
    }
}

/// 装载语境(layout 钩子的 ctx;按值传递,Copy)
/// 字段随域扩:并行切分 / dtype 决策 / 容量约束……起步只带两维。
#[derive(Clone, Copy, Debug)]
pub struct LoaderCtx {
    /// 目标 dtype(权重按语境声明需求;F32 起步)
    pub dtype: Dtype,
    /// 并行度(shard 形状切分的词汇;1 = 单卡)
    pub shard: usize,
    /// **本切片 id**(S4 ②;`0..shard`;`shard = 1` 时忽略)。
    ///
    /// `(slice_id, shard)` = **确定性分片协议**的入参:每个 rank 独立算出
    /// 自己该读源的哪一块 —— 零协调、零通信、可并行装载(用户裁决
    /// 2026-10-16;TP/PP/EP 同一函数)。
    pub slice_id: usize,
    /// 设备重排装载(AWQ 线;**已裁决结果,唯一产地 =
    /// [`RepackPath::resolve`]** —— 能力否决/显式参数/默认 GPU;
    /// 本字段不再承载决策,消费点 = linear layout + awq source 物化)
    pub device_repack: bool,
    /// 装载校验(整块回读 vs staged 校验和;原 OWL_LOAD_VERIFY env 直读
    /// —— 显式依赖律:入口/测试经本字段传,缺省关)
    pub verify: bool,
    /// 装载观测 tap(stderr 逐键打印;原 OWL_LOAD_DEBUG env 直读)
    pub debug_tap: bool,
    /// **装载暂存槽上限 MiB**(原 `OWL_LOAD_STAGING_MB`;从配置来 —— 用户裁决 2026-10-19)
    pub staging_mb: usize,
    /// 逐键字节 dump(原 `OWL_LOAD_DUMP`;诊断面)
    pub dump: bool,

}

/// repack 路径(装载域**唯一收口**;2026-10-01 用户裁决:默认 GPU 优先,
/// 无 GPU 能力回退 CPU)—— CPU/GPU 两条 repack 臂的决策唯一产地。
/// 算法本体互为镜像:host = kernels/src/marlin/repack.rs(pack_marlin_b_fused
/// 等,rayon);device = kernels/cu/marlin/marlin_repack_ct.cu(owl_ct_repack_u32,
/// nvrtc);对拍锚 = cuda/tests/marlin_parity.rs(逐 u32 一致)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepackPath {
    /// GPU kernel 重排(量化大键;CPU 仅产 scales/zeros 轻路径)
    Device,
    /// CPU rayon 打包(回退;host face / env 强制)
    Host,
}

impl RepackPath {
    /// 唯一裁决点。裁决序(**高 → 低**):
    /// 1. **能力否决**:`device_capable = false`(CpuFace 等纯 host face)
    ///    → Host,其余参数不看 —— GPU 不在场,自动回退;
    /// 2. **显式参数**:`Some(false)` → Host(运维 A/B 走入口显式传参;
    ///    原 OWL_LOAD_CPU_REPACK env 强制臂已随显式依赖律移除 ——
    ///    调用方自 knobs 读后传入);
    /// 3. **默认**:Device(GPU 优先)。
    pub fn resolve(explicit: impl Into<Option<bool>>, device_capable: bool) -> Self {
        if !device_capable {
            return Self::Host;
        }
        match explicit.into() {
            Some(false) => Self::Host,
            _ => Self::Device,
        }
    }

    /// Device 臂判定(linear layout / source 物化的唯一谓词)
    pub fn is_device(self) -> bool {
        self == Self::Device
    }
}

impl Default for LoaderCtx {
    fn default() -> Self {
        Self { dtype: Dtype::F32, shard: 1, slice_id: 0, device_repack: false, verify: false, debug_tap: false, staging_mb: 128, dump: false }
    }
}

// ============================================================================
// §6 Loadable:容器装载契约(唯一生命周期钩子 layout)
// ============================================================================

/// 可装载容器:装载生命周期只有一个钩子 ——
/// `layout(&self, ctx) -> LoaderOps`(声明每个空包的排布:同步 · 零 src ·
/// 零 Result)。执行(`eval_load`)与回填(sink)都是解释器/执行器的
/// 内部动作,使用者零感知。
pub trait Loadable {
    /// 装载生命周期:按语境导出数据需求清单(布局声明)
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps;
}

// ============================================================================
// §7 Weight:跨设备权重(空包指针 + 布局标注)
// ============================================================================

/// `Weight.dtype` 的原子编码（2026-10-24;TP2 并行装载要求 `Model: Sync`）。
/// 只做 in-process 值域往返,不跨进程序列化。
fn dtype_code(dt: Dtype) -> usize {
    match dt {
        Dtype::F32 => 0,
        Dtype::BF16 => 1,
        Dtype::F16 => 2,
        Dtype::F8E4M3 => 3,
        Dtype::F8E5M2 => 4,
        Dtype::I8 => 5,
        Dtype::U32 => 6,
        Dtype::U8 => 7,
    }
}

fn dtype_of(code: usize) -> Dtype {
    match code {
        0 => Dtype::F32,
        1 => Dtype::BF16,
        2 => Dtype::F16,
        3 => Dtype::F8E4M3,
        4 => Dtype::F8E5M2,
        5 => Dtype::I8,
        6 => Dtype::U32,
        _ => Dtype::U8,
    }
}

/// 跨设备权重:布局标注 + 空包指针。未装载时 decl() → 毒值声明
/// (forward 全程 total,零 panic 零 expect)。
///
/// 与 Tensor<D> 同构(2026-09-25 拍板:句柄落线 A)—— **容器只持句柄,
/// 存放归设备**:CpuFace 解析为 host 值块,GpuClient 解析为显存池块,
/// 层代码零设备感知。
pub struct Weight {
    key: &'static str,
    shape: Shape,
    transposed: bool,
    cell: SlotCell,
    /// **尺度槽随激活臂**(2026-10-16 定谳):true ⇒ 装载声明由 `LoaderCtx.dtype`
    /// 决定(BF16 模型 ⇒ BF16,否则 F16)。为什么必须随臂:C 侧 marlin
    /// `s_type = c_type`(激活类型)⇒ bf16 臂**吃 bf16 值的尺度**,
    /// 而 f16 臂吃 f16 —— 声明必须等于事实(不许"声明 F16、内容 bf16")。
    scales_arm: bool,
    /// 装载 dtype(F5 整模切换):layout 期从 LoaderCtx 写入,decl 按此
    /// 出声明 —— 默认 F32(CPU 单元面);f16 模型经 LoaderCtx 自动跟随,
    /// 层构造器零涟漪(原子内部可变;Weight 使用面恒 &self)。
    ///
    /// **2026-10-24 TP2 并行装载**:原 `Cell<Dtype>` 让 `Model` 非 `Sync`
    /// ⇒ `LoadedModel` 不能跨线程返回,双 rank 只能串行。此处只改声明
    /// 存储层(原子 Relaxed),语义不变,换来 `Model: Send + Sync`。
    dtype: std::sync::atomic::AtomicUsize,
    /// dtype 覆盖(E3 量化臂:qweight U32 / scales F16 不随 LoaderCtx
    /// 整模 dtype 漂移;Some = layout 期钉死)
    dtype_override: Option<Dtype>,
}

impl Clone for Weight {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            shape: self.shape.clone(),
            transposed: self.transposed,
            cell: self.cell.clone(),
            scales_arm: self.scales_arm,
            dtype: std::sync::atomic::AtomicUsize::new(
                self.dtype.load(std::sync::atomic::Ordering::Relaxed),
            ),
            dtype_override: self.dtype_override,
        }
    }
}

impl Weight {
    /// 登记权重(new 期;零副作用;Direct 布局)
    pub fn new(key: &'static str, shape: Shape) -> Self {
        Self {
            key,
            shape,
            transposed: false,
            cell: SlotCell::default(),
            dtype: std::sync::atomic::AtomicUsize::new(dtype_code(Dtype::F32)),
            dtype_override: None,
            scales_arm: false,
        }
    }

    /// 定 dtype 的权重(E3 量化三件套:qweight U32 / scales F16 /
    /// marlin ws U32 —— dtype 不随 LoaderCtx 整模切换漂移)
    pub fn new_typed(key: &'static str, shape: Shape, dtype: Dtype) -> Self {
        Self {
            key,
            shape,
            transposed: false,
            cell: SlotCell::default(),
            dtype: std::sync::atomic::AtomicUsize::new(dtype_code(dtype)),
            dtype_override: Some(dtype),
            scales_arm: false,
        }
    }

    /// U32 定 dtype 简写(marlin qweight / workspace)
    pub fn new_typed_u32(key: &'static str, shape: Shape) -> Self {
        Self::new_typed(key, shape, Dtype::U32)
    }

    /// F16 定 dtype 简写(marlin scales)
    pub fn new_typed_f16(key: &'static str, shape: Shape) -> Self {
        Self::new_typed(key, shape, Dtype::F16)
    }

    /// **量化尺度槽**:声明**随激活臂**(bf16 模型 ⇒ BF16;否则 F16)。
    ///
    /// 判据(C 侧 `marlin_host.cu`: `s_type = c_type`):bf16 激活 ⇒ bf16 值的
    /// 尺度;f16 激活 ⇒ f16 值的尺度。声明与实际不符就是"影子口径",
    /// 任何按声明解字节的读者都会静默错值(2026-10-16 立案)。
    pub fn new_typed_scales(key: &'static str, shape: Shape) -> Self {
        Self {
            key,
            shape,
            transposed: false,
            cell: SlotCell::default(),
            dtype: std::sync::atomic::AtomicUsize::new(dtype_code(Dtype::F16)),
            dtype_override: None,
            scales_arm: true,
        }
    }

    /// 转置权重:源 [rows, cols] 行主序 → 装载声明 [cols, rows]
    /// (Linear 惯例:forward 直 matmul,decode 零转置)
    pub fn new_transposed(key: &'static str, rows: usize, cols: usize) -> Self {
        Self { key, shape: vec![cols, rows], transposed: true, cell: SlotCell::default(), dtype: std::sync::atomic::AtomicUsize::new(dtype_code(Dtype::F32)), dtype_override: None, scales_arm: false }
    }

    /// 装载生命周期:按语境声明需求(元数据直出 + 回填口;零 src 零数据)
    pub fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.layout_as(self.key, ctx)
    }

    /// 变键布局(tied 聚合用:lm_head 转置槽与 w 同源键直读;
    /// Want 键不需唯一 —— 各带各的 sink,eval_want 逐条取数回填)
    pub(crate) fn layout_as(&self, key: impl Into<String>, ctx: &LoaderCtx) -> LoaderOps {
        // decl 与装载同源(F5 整模切换);dtype_override = 量化臂钉死
        // (E3:qweight U32 / scales F16 不随整模 dtype 漂移)
        let dtype = if self.scales_arm {
            // 尺度槽:bf16 臂 ⇒ BF16、f16 臂 ⇒ F16(其余含 F32 单元面落 F16)
            if ctx.dtype == Dtype::BF16 { Dtype::BF16 } else { Dtype::F16 }
        } else {
            self.dtype_override.unwrap_or(ctx.dtype)
        };
        self.dtype.store(dtype_code(dtype), std::sync::atomic::Ordering::Relaxed);
        LoaderOps::want(Want {
            key: key.into(),
            dtype,
            shape: self.shape.clone(),
            layout: if self.transposed { Layout::Transposed } else { Layout::Direct },
            segments: None,
            rule: None,
            extra: Vec::new(),
            scalar: None,
            sink: Sink(self.cell.clone()),
        })
    }

    /// 设备重排布局(2026-10-01 AWQ 装载线):want 键/形状/dtype 自定
    /// (键 = 原始 packed 源键;shape = 原始 packed 形状;dtype 钉 U32),
    /// Layout = DeviceRearrange;sink = 本槽(最终交付重排后 marlin 块)。
    /// 调用方须已 new_typed_u32(槽 dtype = U32)。
    pub(crate) fn layout_as_device_rearrange(
        &self,
        key: impl Into<String>,
        op: crate::ops::SemanticKernel,
        rows: usize,
        cols: usize,
        ctx: &LoaderCtx,
    ) -> LoaderOps {
        let dtype = self.dtype_override.unwrap_or(ctx.dtype);
        self.dtype.store(dtype_code(dtype), std::sync::atomic::Ordering::Relaxed);
        LoaderOps::want(Want {
            key: key.into(),
            dtype,
            shape: vec![rows, cols],
            layout: Layout::DeviceRearrange { op, rows, cols },
            segments: None,
            rule: None,
            extra: Vec::new(),
            scalar: None,
            sink: Sink(self.cell.clone()),
        })
    }

    /// **NVFP4 打包布局**(2026-10-28 装载管线 P2):源 = 全精度权重
    /// (`[rows, cols]` 行主序),装载期 host 量化成
    /// `q`(本槽) + `sf`(附加槽) + `gb`(host 标量)。
    ///
    /// 调用方须已 [`Weight::new_typed`] 把本槽定成 `U8`、形状 `[rows, cols/2]`
    /// (见 `Linear::quantize_nvfp4`);`sf` 槽同理(`U8`、`[rows*cols/16]`)。
    /// **两槽形状不在此计算**:声明面只搬事实,目标形状归变换
    /// (`formats::transform::NvFp4Pack::slots`),两边不一致就在这里响。
    pub(crate) fn layout_as_nvfp4(
        &self,
        key: impl Into<String>,
        rows: usize,
        cols: usize,
        sf: &Weight,
        gb: &ScalarSink,
    ) -> LoaderOps {
        assert_eq!(
            self.shape,
            vec![rows, cols / 2],
            "NVFP4 q 槽形状 {:?} 与源 {rows}x{cols} 的口径不符(需 [rows, cols/2])",
            self.shape
        );
        assert_eq!(
            sf.shape,
            vec![rows * cols / 16],
            "NVFP4 sf 槽形状 {:?} 与源 {rows}x{cols} 的口径不符(需 {} 字节)",
            sf.shape,
            rows * cols / 16
        );
        self.dtype
            .store(dtype_code(Dtype::U8), std::sync::atomic::Ordering::Relaxed);
        sf.dtype
            .store(dtype_code(Dtype::U8), std::sync::atomic::Ordering::Relaxed);
        LoaderOps::want(Want {
            key: key.into(),
            dtype: Dtype::U8,
            shape: self.shape.clone(),
            layout: Layout::NvFp4Weight { rows, cols },
            segments: None,
            rule: None,
            extra: vec![SlotWant {
                name: "sf",
                dtype: Dtype::U8,
                shape: sf.shape.clone(),
                sink: Sink(sf.cell.clone()),
            }],
            scalar: Some(gb.clone()),
            sink: Sink(self.cell.clone()),
        })
    }

    /// 权重声明(未装载 → 毒值声明;of_block:eval 时 Block 叶子零操作)
    pub fn decl(&self) -> TensorOps {
        let dt = dtype_of(self.dtype.load(std::sync::atomic::Ordering::Relaxed));
        match self.cell.loaded() {
            Some(b) => TensorOps::of_block(b.id, dt, self.shape.clone()),
            None => TensorOps::poisoned(
                dt,
                self.shape.clone(),
                format!("Weight '{}': 未装载(eval_load 前禁止执行)", self.key),
            ),
        }
    }

    /// 装载 dtype 访问器(核名路由用;layout 前 = 默认 F32)
    pub fn dtype(&self) -> Dtype {
        dtype_of(self.dtype.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn is_loaded(&self) -> bool {
        self.cell.loaded().is_some()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn key(&self) -> &'static str {
        self.key
    }
}

// ============================================================================
// §8 顶层装载函数(最底层基本件:单权重直通,完整自足)
// ============================================================================

/// 单权重装载(最底层基本函数,完整自足):给它权重容器 + 设备 + 数据源,
/// 它把权重物化进设备块并装进空包。取数 → 长度校验 → 转置 → htod →
/// 填空包,五步全在本函数 —— 不经 LoaderOps / eval_load(那是多层
/// 聚合路径)。
pub async fn load_weight<D: DeviceClient, S: WeightSource + ?Sized>(
    weight: &mut Weight,
    face: &mut D,
    src: &S,
) -> Result<(), ModelError> {
    let key = weight.key;

    // 1. 取数:容器内表没有(表走 layout 清单),数据槽直查源(取走所有权)
    // 三态(2026-10-17 审查 §③):`Err` = 源的真错误,**先上抛**(不再被收成"缺键");
    // `Ok(None)` 才是"缺键"正常语义
    let data = src
        .take(key)?
        .ok_or_else(|| ModelError::Msg(format!("Weight '{key}': 数据源缺键")))?;

    // 2. 长度校验 + 布局变换 → (装载形状, LE 字节)
    let n: usize = weight.shape.iter().product();
    if data.len() != n {
        return Err(ModelError::Msg(format!(
            "Weight '{key}': 元素 {} != shape {:?}({n})",
            data.len(),
            weight.shape
        )));
    }
    let (shape, bytes) = if weight.transposed {
        // 源 [shape[1], shape[0]] → 转置装载 [shape[0], shape[1]]
        let (cols, rows) = (weight.shape[0], weight.shape[1]);
        let mut t = vec![0.0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = data[r * cols + c];
            }
        }
        (weight.shape.clone(), f32b(&t))
    } else {
        (weight.shape.clone(), f32b(&data))
    };

    // 3. 物化(解释层原语:htod → 块句柄)。记名:`wt.<键名>`(装载域统一命名)。
    let b = face
        .htod(Dtype::F32, &shape, &bytes, &format!("wt.{key}"))
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{key}': {e}")))?;

    // 4. 填空包
    weight.cell.deliver(b);
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::testkit::{f32b, f32_of, Src};

    /// TP2 并行装载的前提:`Model` 必须能跨线程移动/共享(Weight.dtype 原子化)。
    #[test]
    fn model_is_send_sync_for_tp_parallel_load() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::model::Model>();
    }

    /// RepackPath 裁决序单测(收口唯一性;env 支不入进程内单测 ——
    /// env 是全局态,并行测试互扰;该支由 specs 装载路径 + 运维 A/B 人工覆盖)
    #[test]
    fn repack_path_resolve_order() {
        use RepackPath::{Device, Host};
        // 1. 能力否决(最高):host face 无论显式参数一律 Host(自动回退)
        assert_eq!(RepackPath::resolve(None, false), Host);
        assert_eq!(RepackPath::resolve(Some(true), false), Host);
        // 3. 显式参数:false → Host;true/None → 默认
        assert_eq!(RepackPath::resolve(Some(false), true), Host);
        // 4. 默认 GPU 优先
        assert_eq!(RepackPath::resolve(None, true), Device);
        assert_eq!(RepackPath::resolve(Some(true), true), Device);
        assert!(Device.is_device() && !Host.is_device());
    }

    #[tokio::test]
    async fn load_weight_basic() {
        let mut face = owl_cpu::CpuFace::new();
        let mut w = Linear::new("w", 2, 3, QuantPlan::F16).into_weight();
        assert!(!w.is_loaded(), "初始未装载");

        let src = Src::from([("w".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        load_weight(&mut w, &mut face, &src).await.expect("load_weight");
        assert!(w.is_loaded(), "装载后应有块句柄");

        let xs = crate::TensorOps::from_host(crate::contract::Dtype::F32, vec![1, 3], &f32b(&[0.5, -1.0, 2.0]));
        let got = {
            // F5-2:Linear 原生 [out,in],forward = matmul_nt
            let bytes = crate::interpreters::eval_ops(xs.matmul_nt(&w.decl()).step(), &mut face)
                .await
                .expect("eval");
            let mut buf = vec![0u8; 8];
            face.dtoh(&bytes, &mut buf).await.expect("dtoh");
            f32_of(&buf)
        };
        assert!((got[0] - 4.5).abs() < 1e-6 && (got[1] - 9.0).abs() < 1e-6);

        let mut w2 = Linear::new("w", 2, 3, QuantPlan::F16).into_weight();
        let empty = Src::new();
        assert!(load_weight(&mut w2, &mut face, &empty).await.is_err(), "缺键应 Err");
        assert!(!w2.is_loaded(), "失败装载不应污染容器");
    }

    fn run_through<M: Module>(m: &M, x: &crate::TensorOps, ctx: &ForwardCtx) -> crate::TensorOps {
        m.forward(x, ctx)
    }

    #[tokio::test]
    async fn module_trait_polymorphism() {
        let mlp = crate::layers::mlp::Mlp::new(4, 6, QuantPlan::F16);
        let src = Src::from([
            ("gate_proj".to_string(), vec![0.1; 24]),
            ("up_proj".to_string(), vec![0.2; 24]),
            ("down_proj".to_string(), vec![0.3; 24]),
        ]);
        let mut face = owl_cpu::CpuFace::new();
        crate::interpreters::eval_load(&mlp, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let x = crate::TensorOps::from_host(crate::contract::Dtype::F32, vec![1, 4], &f32b(&[0.5; 4]));
        let ctx = ForwardCtx::minimal(1);
        let a = crate::testkit::harvest(&mut face, &run_through(&mlp, &x, &ctx)).await;
        let layers: Vec<&dyn Module> = vec![&mlp];
        let b = crate::testkit::harvest(&mut face, &layers[0].forward(&x, &ctx)).await;
        assert_eq!(a, b, "静态/动态分发同链同果");
        assert_eq!(a.len(), 4);
    }
}
