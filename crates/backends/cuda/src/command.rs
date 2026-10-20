//! 客户端 → server 的命令信封 + 回执桥(Ack/Waiter)。
//!
//! 信封线格式封闭(枚举);泛型只在回执桥内部 —— 契约与 async-runtime.md §四一致:
//! server 只认信封,不感知 client 的 future/waker。
//!
//! 本模块公开:手工组装者需用它创建管道(`mpsc::channel<Command>`)
//! 并分别传入 [`GpuServer::new`](crate::GpuServer) 与
//! [`GpuClient::new`](crate::GpuClient)。

use owl_iface::contract::{Bytes, GraphId, LaunchMsg};
use owl_iface::contract::{Connector, ModelError, SwapDir, SwapPiece};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// 命令信封(server dispatch 的键;新增能力 = 新变体 + actor 一臂)。
/// 除 NewStream 外均携带目标流 id(流内保序;流间并发)。
/// 命令信封(server dispatch 的键;新增能力 = 新变体 + actor 一臂)。
/// 无流字段:流路由是 server 内部策略(htod→H2D,launch/alloc/graph→
/// COMPUTE,dtoh→D2H,sync 排空全部)。
pub enum Command {
    /// 图捕获开始(护栏见 state.rs graph_begin;捕获期仅 Launch/Alloc 放行)
    GraphBegin {
        ack: Ack<Result<(), ModelError>>,
    },
    /// 图捕获结束(实例化 + 登记;空捕获窗拒绝)
    GraphEnd {
        ack: Ack<Result<GraphId, ModelError>>,
    },
    /// 图重放(流序异步提交;捕获期间拒绝)
    GraphLaunch {
        graph: GraphId,
        ack: Ack<Result<(), ModelError>>,
    },
    /// **FI plan 表同步**(W3):让 FI 族把 `spec.ctx_total` 对应的调度表刷进
    /// int_ws。用于 "FI 核在捕获图内" 的档 —— 图回放不再经过宿主处理器,
    /// 表必须由图外一步显式刷新(见 `DeviceClient::fi_plan_sync`)。
    FiPlanSync {
        spec: owl_iface::contract::FiPlanSpec,
        ack: Ack<Result<(), ModelError>>,
    },
    /// **捕获窗作废**(窗内出错/空窗的收尾;捕获期间**允许** —— 它就是窗内
    /// 收尾手段)。语义 = "这一窗从未存在":退出捕获态 + 销毁半成品图 +
    /// **窗内自产块整批移出账房**(⇒ slab 释放)。不做它则失败捕获会把整条
    /// slab 钉到进程结束(state.rs `graph_abort`)
    GraphAbort {
        ack: Ack<Result<(), ModelError>>,
    },
    /// ★ **捕获期的发射路由切换**(图内 fork/join,2026-10-18;B1 门)。
    ///
    /// `sid` = 目标流(0=H2D / 1=COMPUTE / 2=D2H / 3=COMPUTE_AUX;
    /// 数值与 `state::STREAM_*` 同源 —— 线上只有 u8,防枚举漂移)。
    /// 语义三条(实现见 `state.rs::capture_route_to`):
    /// 1. 在**当前**路由流记一个 `DISABLE_TIMING` 事件,目标流 `wait` 它
    ///    —— 这条依赖边**被捕获进图** ⇒ 目标流被卷入同一张图;
    /// 2. 之后本窗的 `Launch`/`CopyBlock`/`Memset` **路由到目标流**;
    /// 3. `graph_end` 仍在 **origin**(发起捕获那条)上收窗。
    ///
    /// 为何是**独立命令**而不是给 `Launch` 加流字段:§3.2 的立场是**流不暴露给
    /// 客户端**(序由意图复合、策略由 server 独占) —— 这条同样落在 server 侧,
    /// 且它只在**捕获期**合法(回放期的流由图自己记)。
    CaptureRoute {
        sid: u8,
        ack: Ack<Result<(), ModelError>>,
    },
    /// ★ **捕获期的 join**(主链等支链;2026-10-18)。
    ///
    /// 在当前路由流记一个 `DISABLE_TIMING` 事件,**origin 等它**,路由切回 origin。
    /// 只在**有人真的要消费支链结果**时用 —— 拿它当"切流程"用会把两条链拴成
    /// 串行(`capture_route_to` 的 M5 教训同源)。
    CaptureJoin {
        ack: Ack<Result<(), ModelError>>,
    },
    /// **销毁已登记图**(单图退休:LRU 驱逐 / plan `close()` 底层)。三件事:
    /// 排空回放 → 解除本图块租约(归零即解钉)→ 本图 slab 块整批移出账房
    GraphDestroy {
        graph: GraphId,
        ack: Ack<Result<(), ModelError>>,
    },
    /// 清零分配(设备侧 memset,流序非阻塞;捕获期自动切 slab)。
    /// n_bytes = 清零/切块字节数;elems = Bytes 账长(元素口径,f16 ≠ bytes/4)
    /// zero = false 时免 memset(S1:scratch 输出 kernel 全量覆写;
    /// 捕获窗免烙 memset 节点 —— 1207 节点双税定谳)
    Alloc {
        n_bytes: usize,
        elems: usize,
        zero: bool,
        /// **分配点点名**(全量记名;2026-10-16):核名/层名/装载键名。
        /// 必填(空串 = 调用点违约,账房侧计 `未命名!`)。
        name: String,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// host → device(字节传输,f16 基线;Staging 字节码头 + 异步 memcpy)
    Htod {
        data: Vec<u8>,
        /// 上传落地块的点名(同 [`Command::Alloc`];必填)
        name: String,
        /// 落点流(0 = H2D 批量面;1 = COMPUTE 参数面 —— 见 `UpStream`)
        stream: u8,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// **PeerShared 分配**(TP2 S1;A2.6 修正后的唯一跨卡分配口)。
    ///
    /// 语义:VMM(`cuMemCreate` 取设备最小粒度,向上取整)拼块 + 本地 RW 授予,
    /// 返回块句柄(同 `Alloc` 的 `Bytes`),并登记对端可导入的凭据。
    ///
    /// 纪律(设计出处 `docs/arch/TP2-施工设计.md` §2):
    /// - 单笔 ≤ `per_alloc_limit`(Window 档 = BAR1 − 64MiB 预留);**超限 = 响亮拒**;
    /// - **捕获期拒绝**(peer 授予必须发生在建图之前);
    /// - 全程**不调** `cudaDeviceEnablePeerAccess`(3080 档进程级急切映射 = 预算墙)。
    ///
    /// `credential` = 驱动可共享凭据的序列化(当前口径 POSIX fd / shareable handle;
    /// S1 定形)。空 = 分配侧尚未实现时的桩路径(当前壳子即此)。
    PeerAlloc {
        bytes: usize,
        name: String,
        ack: Ack<Result<(Bytes, Vec<u8>), ModelError>>,
    },
    /// **对端映射**(TP2 S1):消费对端 `PeerAlloc` 的凭据,导入为本卡 VA 别名,
    /// 返回别名的块句柄(供 AR 核读对端 staging)。
    PeerImport {
        credential: Vec<u8>,
        bytes: usize,
        peer_uuid: String,
        name: String,
        /// **导入侧访问级别**(2026-10-18;TP2 施工设计 §36.6):
        /// `false` = `PROT_READ`(v1 只读对端,现役行为);
        /// `true` = `PROT_READWRITE`(**push 协议前置** —— 本端要写对端槽)。
        /// 授权对象始终是**本端设备**(`set_access_self`),不是 peer。
        rw: bool,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// **peer 资源释放**(TP2 S1;D2 裁决:显式释放 + `Drop` 兜底)。
    /// `ids` 同时接受分配块与别名块的 id(与 `Free` 的块账语义一致)。
    PeerRelease {
        ids: Vec<u64>,
        ack: Ack<Result<(), ModelError>>,
    },
    /// 块清零(设备侧 memset_d8;引擎 turn-open GDN 状态重置专用;
    /// E2b 增 offset_bytes = 按格重置,多会话各清各格)
    MemsetZero {
        block: u64,
        offset_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    },
    /// host → device **分块写入**(write_block 路径):向已 alloc 的块在
    /// offset_bytes 处写字节(pinned 码头 + 异步 memcpy 原位)
    HtodChunk {
        block: u64,
        offset_bytes: usize,
        data: Vec<u8>,
        /// 落点流(0 = H2D 批量面;1 = COMPUTE 参数面 —— 见 `UpStream`)
        stream: u8,
        ack: Ack<Result<(), ModelError>>,
    },
    /// host → device **批量分块写入**(S1):decode 每步多槽装填合一次
    /// 命令/一次往返 —— 保序协议不变(单 ack = 全部写完,launch 才发);
    /// server 逐写池化码头 + H2D 流序,末尾单 notify
    HtodChunks {
        writes: Vec<(u64, usize, Vec<u8>)>,
        /// 落点流(0 = H2D 批量面;1 = COMPUTE 参数面 —— 见 `UpStream`)
        stream: u8,
        ack: Ack<Result<(), ModelError>>,
    },
    /// 分配 pinned 租约(流式装载 DMA 源;池优先,miss 才 cudaHostAlloc)
    AllocPinned {
        bytes: usize,
        /// 租约点名(全量记名;pinned 泄漏也看得见主人)
        name: String,
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    },
    /// **块账按名清单**(诊断查询面;2026-10-16 全量记名)。
    ///
    /// 纯读:不改块账/不碰设备。OOM dump 与排障共用同一渲染
    /// ([`crate::state::render_block_report`])—— 口径单源。
    /// 捕获期**放行**(读操作不破图);`top` = 列多少类名字。
    LedgerReport {
        top: usize,
        ack: Ack<Result<String, ModelError>>,
    },
    /// **显存现场 JSON**(块账 + 家族 scratch;同 [`Command::LedgerReport`] 的
    /// 结构化姊妹面)。OOM 落盘与机器消费共用同一序列化器 —— 有它才谈得上
    /// "落盘的内容被验证过"。
    LedgerJson {
        ack: Ack<Result<String, ModelError>>,
    },
    /// 上传租约:buf 所有权移入,DMA 到 dst+offset_bytes(字节口径);
    /// 完成后 buf 回池
    UploadPinned {
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        /// 字节长度(池租约容量 ≥ 请求;只传前 len_bytes)
        len_bytes: usize,
        /// 落点流(0 = H2D 批量面;1 = COMPUTE 参数面 —— 见 `UpStream`)
        stream: u8,
        ack: Ack<Result<(), ModelError>>,
    },
    /// device → host(异步 memcpy 到 pinned 码头;want_bytes 字节口径)
    Dtoh {
        id: u64,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    },
    /// **带偏移的 device → host**(状态存储搬出面,2026-10-16)。
    ///
    /// 与 [`Command::Dtoh`] **同一条执行路径**(老命令 = `off_bytes = 0` 的特例),
    /// 只是起点可偏移 —— "逻辑块 = 层缓冲内的一个页"必须按槽偏移读
    /// (设计卷 §10.3;`docs/arch/store-状态存储-设计与迁移.md`)。
    DtohAt {
        id: u64,
        /// 块内字节偏移(= 槽 × 每页字节)
        off_bytes: usize,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    },
    /// **批量分块 device → host**(状态存储"整块搬出",2026-10-16)。
    ///
    /// 一次命令搬一个逻辑块的全部层片段(0.8B 12 段 / 27B 32 段)⇒ **一次往返**;
    /// 回执 = 各片段按**入参顺序**拼接后的字节(逐段等价于 `DtohAt`)。
    /// 为什么必须批:不批则每个块 12/32 次 actor 往返,搬出去的收益会被往返吃光
    /// (设计卷 §10.3 推论 B)。
    DtohBatch {
        /// (块 id, 偏移字节, 长度字节)
        pieces: Vec<(u64, usize, usize)>,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    },
    /// 中间块回收(E2a,2026-09-27):调用方声明这批块此后不再引用 ——
    /// 账房移除 Owned 块(CudaSlice drop → 流序 free_async 归还设备池);
    /// Carved 块(slab 切片,图/scratch 生命周期)跳过不删。安全前提:
    /// 调用方已在回收前收割所需数据(dtoh 回执即 COMPUTE 已排空)。
    Free {
        ids: Vec<u64>,
        ack: Ack<Result<(), ModelError>>,
    },
    /// **swap 一条命令**(设计卷 §15;2026-10-16):N 片一次提交、一次 ack。
    ///
    /// 语义:
    /// - `Out` = 设备 → 连接器(D2H 流);`In` = 连接器 → 设备(H2D 流);
    /// - 一次 ack = 全部片段完成(**回执即落地**);
    /// - `conn` 随命令进出(回执**原箱奉还**,单主零锁);
    /// - **层级无关**:server 只按 `Connector` 三件读写,不关心另一端是 RAM 或盘;
    /// - 捕获窗内 = **结构化拒**(搬运只在段边界,时序归 engine —— §10.3);
    /// - 中途失败 = 整条失败(不留半态;调用方回滚账)。
    Swap {
        dir: SwapDir,
        pieces: Vec<SwapPiece>,
        conn: Box<dyn Connector + Send>,
        ack: Ack<Result<Box<dyn Connector + Send>, ModelError>>,
    },
    /// 设备内块→块拷贝(E2c 快照通道;流序非阻塞)
    CopyBlock {
        src: u64,
        src_off_bytes: usize,
        dst: u64,
        dst_off_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    },
    /// 设备内批量块→块拷贝(E5-M4):spec 快照 capture/restore 的
    /// 192 次 copy_block_at actor 往返收敛为一次命令(单 ack = 全部
    /// 拷贝已入 COMPUTE 流;流序保序语义与逐条 CopyBlock 一致)
    CopyBatch {
        copies: Vec<(u64, usize, u64, usize, usize)>,
        ack: Ack<Result<(), ModelError>>,
    },
    /// **设备池回收**(内存面):池内已 free 的缓存块还 OS;
    /// ack 回**释放字节数**(`RESERVED_MEM_CURRENT` 差值;0 = 无可回收)。
    /// 用法 = 装载收尾(暂存槽 free 之后)—— 见 `DeviceClient::mem_pool_trim`。
    MemPoolTrim {
        ack: Ack<Result<u64, ModelError>>,
    },
    /// **设备池账**(内存面;只读):(used, used_high, reserved, reserved_high)
    /// —— `/debug/vram` 的 per-rank 行(物理 − live = 池缓存,见 §23.9)。
    MemPoolStats {
        ack: Ack<Result<(u64, u64, u64, u64), ModelError>>,
    },
    /// **设备 SM 数查询**(宿主面只读;`cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)`)。
    /// 与 `BlockLedgerDump` 同族:无 GPU 活、无流、捕获期可放行(恒定值,不碰设备状态)。
    DeviceSmCount {
        ack: Ack<Result<usize, ModelError>>,
    },
    /// **块账本投影**(调试面;`GET /debug/blocks`):全活块 (id, 字节, dtype, 地址, 账本)
    /// —— 判「同 id 两地址」与「两活块区间相交」的唯一权威快照。
    BlockLedgerDump {
        ack: Ack<Result<Vec<owl_iface::contract::BlockLedgerRow>, ModelError>>,
    },
    /// **UploadAndLaunch**(E2 复合;闭集 1:1,设计 §3.2):一次 pinned 搬运
    /// + 一次发射。**客户端不出现流/事件/槽** —— 序由 server 内部保证
    /// (E2a:线格式/校验/负门;**E2b 接线** = 完成回调链 → 设备端事件序)。
    /// 失败带**结构化阶段**(transfer / launch)。
    UploadAndLaunch {
        dst: Bytes,
        /// 分块搬运(装载按 CHUNK 切;序罩住**全部**块 —— 见 server 侧前置等待)
        chunks: Vec<owl_iface::contract::UpChunk>,
        msg: LaunchMsg,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// **LaunchAndHarvest**(E2 复合;闭集 1:1):一次发射 + 一次回读。
    /// 校验:`id` 必须是本 launch 的**输出块**(槽序末位 Block);
    /// 契约:harvest 走 D2H 流异步,**禁退化成隐式全排空**。
    LaunchAndHarvest {
        msg: LaunchMsg,
        id: u64,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    },
    /// kernel 发射(非阻塞;fire-and-forget)
    Launch {
        msg: LaunchMsg,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// 排空点(三条流全部 synchronize)
    Sync {
        ack: Ack<Result<(), ModelError>>,
    },
    /// 优雅关机:回执后进入 Closing —— 排空积压命令(结构化拒绝),
    /// 排空在飞完成回调,设备栅栏,线程退出
    Close {
        ack: Ack<Result<(), ModelError>>,
    },
}

impl Command {
    /// 命令类名(分相 tag 维度;**有界词表**,与各变体一一对应)
    pub fn kind(&self) -> &'static str {
        match self {
            Self::GraphBegin { .. } => "graph_begin",
            Self::GraphEnd { .. } => "graph_end",
            Self::GraphLaunch { .. } => "graph_launch",
            Self::GraphAbort { .. } => "graph_abort",
            Self::CaptureRoute { .. } => "capture_route",
            Self::CaptureJoin { .. } => "capture_join",
            Self::GraphDestroy { .. } => "graph_destroy",
            Self::FiPlanSync { .. } => "fi_plan_sync",
            Self::Alloc { .. } => "alloc",
            Self::Htod { .. } => "htod",
            Self::MemsetZero { .. } => "memset_zero",
            Self::DeviceSmCount { .. } => "device_sm_count",
            Self::BlockLedgerDump { .. } => "block_ledger_dump",
            Self::MemPoolTrim { .. } => "mem_pool_trim",
            Self::MemPoolStats { .. } => "mem_pool_stats",
            Self::UploadAndLaunch { .. } => "upload_and_launch",
            Self::LaunchAndHarvest { .. } => "launch_and_harvest",
            Self::HtodChunk { .. } => "htod_chunk",
            Self::HtodChunks { .. } => "htod_chunks",
            Self::AllocPinned { .. } => "alloc_pinned",
            Self::LedgerReport { .. } => "ledger_report",
            Self::LedgerJson { .. } => "ledger_json",
            Self::UploadPinned { .. } => "upload_pinned",
            Self::Dtoh { .. } => "dtoh",
            Self::DtohAt { .. } => "dtoh_at",
            Self::DtohBatch { .. } => "dtoh_batch",
            Self::Swap { dir, .. } => match dir {
                owl_iface::contract::SwapDir::Out => "swap_out",
                owl_iface::contract::SwapDir::In => "swap_in",
            },
            Self::Free { .. } => "free",
            Self::CopyBlock { .. } => "copy_block",
            Self::CopyBatch { .. } => "copy_batch",
            Self::Launch { .. } => "launch",
            Self::Sync { .. } => "sync",
            Self::PeerAlloc { .. } => "peer_alloc",
            Self::PeerImport { .. } => "peer_import",
            Self::PeerRelease { .. } => "peer_release",
            Self::Close { .. } => "close",
        }
    }

    /// 命令创建时刻(客户端 Ack::pair 盖;**分相探针**:server 出队读)
    pub fn created_at(&self) -> std::time::Instant {
        match self {
            Self::GraphBegin { ack } => ack.created_at(),
            Self::GraphEnd { ack } => ack.created_at(),
            Self::GraphLaunch { ack, .. } => ack.created_at(),
            Self::GraphAbort { ack } => ack.created_at(),
            Self::CaptureRoute { ack, .. } => ack.created_at(),
            Self::CaptureJoin { ack } => ack.created_at(),
            Self::GraphDestroy { ack, .. } => ack.created_at(),
            Self::FiPlanSync { ack, .. } => ack.created_at(),
            Self::Alloc { ack, .. } => ack.created_at(),
            Self::Htod { ack, .. } => ack.created_at(),
            Self::DeviceSmCount { ack } => ack.created_at(),
            Self::BlockLedgerDump { ack } => ack.created_at(),
            Self::MemPoolStats { ack } => ack.created_at(),
            Self::MemPoolTrim { ack } => ack.created_at(),
            Self::MemPoolTrim { ack } => ack.created_at(),
            Self::UploadAndLaunch { ack, .. } => ack.created_at(),
            Self::LaunchAndHarvest { ack, .. } => ack.created_at(),
            Self::MemsetZero { ack, .. } => ack.created_at(),
            Self::HtodChunk { ack, .. } => ack.created_at(),
            Self::HtodChunks { ack, .. } => ack.created_at(),
            Self::AllocPinned { ack, .. } => ack.created_at(),
            Self::LedgerReport { ack, .. } => ack.created_at(),
            Self::LedgerJson { ack, .. } => ack.created_at(),
            Self::UploadPinned { ack, .. } => ack.created_at(),
            Self::Dtoh { ack, .. } => ack.created_at(),
            Self::DtohAt { ack, .. } => ack.created_at(),
            Self::DtohBatch { ack, .. } => ack.created_at(),
            Self::Swap { ack, .. } => ack.created_at(),
            Self::Free { ack, .. } => ack.created_at(),
            Self::CopyBlock { ack, .. } => ack.created_at(),
            Self::CopyBatch { ack, .. } => ack.created_at(),
            Self::Launch { ack, .. } => ack.created_at(),
            Self::Sync { ack, .. } => ack.created_at(),
            Self::PeerAlloc { ack, .. } => ack.created_at(),
            Self::PeerImport { ack, .. } => ack.created_at(),
            Self::PeerRelease { ack, .. } => ack.created_at(),
            Self::Close { ack, .. } => ack.created_at(),
        }
    }
}

// ============================================================================
// 回执桥(waker oneshot;server 侧 send 唤醒,client 侧 Future 真异步 pending)
// ============================================================================
// 语义:htod/dtoh/sync 的 Future pending 至 GPU 真完成(host 回调/栅栏触发
// send);launch/alloc issue 即 send → Future 立即 resolve。
// 不占线程:pending 期间执行器自由调度其他任务(非 Condvar 死等)。

struct Once<T> {
    value: Mutex<Option<T>>,
    waker: Mutex<Option<Waker>>,
    /// 命令创建时刻(客户端 [`Ack::pair`] 盖;server 出队读 ⇒ 队列等待可测)
    created: std::time::Instant,
}

pub struct Ack<T>(Arc<Once<T>>);

impl<T> Ack<T> {
    pub fn pair() -> (Self, Waiter<T>) {
        let inner = Arc::new(Once {
            value: Mutex::new(None),
            waker: Mutex::new(None),
            created: std::time::Instant::now(),
        });
        (Self(inner.clone()), Waiter(inner))
    }

    /// 命令创建时刻(分相探针:server 出队时减它 = 入队→出队的唤醒延迟)
    pub fn created_at(&self) -> std::time::Instant {
        self.0.created
    }

    /// server 侧回执:落值 + 唤醒 client Future
    pub fn send(self, v: T) {
        *self.0.value.lock().unwrap() = Some(v);
        if let Some(w) = self.0.waker.lock().unwrap().take() {
            w.wake();
        }
    }
}

pub struct Waiter<T>(Arc<Once<T>>);

impl<T: Send + 'static> Future for Waiter<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        if let Some(v) = self.0.value.lock().unwrap().take() {
            return Poll::Ready(v);
        }
        *self.0.waker.lock().unwrap() = Some(cx.waker().clone());
        // 双检:send 可能发生在注册 waker 之前后夹缝
        if let Some(v) = self.0.value.lock().unwrap().take() {
            return Poll::Ready(v);
        }
        Poll::Pending
    }
}
