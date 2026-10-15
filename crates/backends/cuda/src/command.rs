//! 客户端 → server 的命令信封 + 回执桥(Ack/Waiter)。
//!
//! 信封线格式封闭(枚举);泛型只在回执桥内部 —— 契约与 async-runtime.md §四一致:
//! server 只认信封,不感知 client 的 future/waker。
//!
//! 本模块公开:手工组装者需用它创建管道(`mpsc::channel<Command>`)
//! 并分别传入 [`GpuServer::new`](crate::GpuServer) 与
//! [`GpuClient::new`](crate::GpuClient)。

use owl_iface::contract::{Bytes, GraphId, LaunchMsg};
use owl_iface::contract::ModelError;
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
    /// **捕获窗作废**(窗内出错/空窗的收尾;捕获期间**允许** —— 它就是窗内
    /// 收尾手段)。语义 = "这一窗从未存在":退出捕获态 + 销毁半成品图 +
    /// **窗内自产块整批移出账房**(⇒ slab 释放)。不做它则失败捕获会把整条
    /// slab 钉到进程结束(state.rs `graph_abort`)
    GraphAbort {
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
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// host → device(字节传输,f16 基线;Staging 字节码头 + 异步 memcpy)
    Htod {
        data: Vec<u8>,
        ack: Ack<Result<Bytes, ModelError>>,
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
        ack: Ack<Result<(), ModelError>>,
    },
    /// host → device **批量分块写入**(S1):decode 每步多槽装填合一次
    /// 命令/一次往返 —— 保序协议不变(单 ack = 全部写完,launch 才发);
    /// server 逐写池化码头 + H2D 流序,末尾单 notify
    HtodChunks {
        writes: Vec<(u64, usize, Vec<u8>)>,
        ack: Ack<Result<(), ModelError>>,
    },
    /// 分配 pinned 租约(流式装载 DMA 源;池优先,miss 才 cudaHostAlloc)
    AllocPinned {
        bytes: usize,
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    },
    /// 上传租约:buf 所有权移入,DMA 到 dst+offset_bytes(字节口径);
    /// 完成后 buf 回池
    UploadPinned {
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        /// 字节长度(池租约容量 ≥ 请求;只传前 len_bytes)
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    },
    /// device → host(异步 memcpy 到 pinned 码头;want_bytes 字节口径)
    Dtoh {
        id: u64,
        want_bytes: usize,
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
            Self::GraphDestroy { .. } => "graph_destroy",
            Self::Alloc { .. } => "alloc",
            Self::Htod { .. } => "htod",
            Self::MemsetZero { .. } => "memset_zero",
            Self::UploadAndLaunch { .. } => "upload_and_launch",
            Self::LaunchAndHarvest { .. } => "launch_and_harvest",
            Self::HtodChunk { .. } => "htod_chunk",
            Self::HtodChunks { .. } => "htod_chunks",
            Self::AllocPinned { .. } => "alloc_pinned",
            Self::UploadPinned { .. } => "upload_pinned",
            Self::Dtoh { .. } => "dtoh",
            Self::Free { .. } => "free",
            Self::CopyBlock { .. } => "copy_block",
            Self::CopyBatch { .. } => "copy_batch",
            Self::Launch { .. } => "launch",
            Self::Sync { .. } => "sync",
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
            Self::GraphDestroy { ack, .. } => ack.created_at(),
            Self::Alloc { ack, .. } => ack.created_at(),
            Self::Htod { ack, .. } => ack.created_at(),
            Self::UploadAndLaunch { ack, .. } => ack.created_at(),
            Self::LaunchAndHarvest { ack, .. } => ack.created_at(),
            Self::MemsetZero { ack, .. } => ack.created_at(),
            Self::HtodChunk { ack, .. } => ack.created_at(),
            Self::HtodChunks { ack, .. } => ack.created_at(),
            Self::AllocPinned { ack, .. } => ack.created_at(),
            Self::UploadPinned { ack, .. } => ack.created_at(),
            Self::Dtoh { ack, .. } => ack.created_at(),
            Self::Free { ack, .. } => ack.created_at(),
            Self::CopyBlock { ack, .. } => ack.created_at(),
            Self::CopyBatch { ack, .. } => ack.created_at(),
            Self::Launch { ack, .. } => ack.created_at(),
            Self::Sync { ack, .. } => ack.created_at(),
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
