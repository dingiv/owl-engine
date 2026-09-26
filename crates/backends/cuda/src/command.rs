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
    /// 清零分配(设备侧 memset,流序非阻塞;捕获期自动切 slab)
    Alloc {
        n_elems: usize,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// host → device(pinned 码头 + 异步 memcpy;完成经 host 回调回执)
    Htod {
        data: Vec<f32>,
        ack: Ack<Result<Bytes, ModelError>>,
    },
    /// host → device **分块写入**(流式装载):向已 alloc 的块在
    /// offset_elems 处写 data(pinned 码头 + 异步 memcpy 原位)
    HtodChunk {
        block: u64,
        offset_elems: usize,
        data: Vec<f32>,
        ack: Ack<Result<(), ModelError>>,
    },
    /// 分配 pinned 租约(流式装载 DMA 源;池优先,miss 才 cudaHostAlloc)
    AllocPinned {
        elems: usize,
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    },
    /// 上传租约:buf 所有权移入,DMA 到 dst+offset;完成后 buf 回池
    UploadPinned {
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_elems: usize,
        /// 逻辑长度(池租约容量 ≥ 请求;只传前 elems)
        elems: usize,
        ack: Ack<Result<(), ModelError>>,
    },
    /// device → host(异步 memcpy 到 pinned 码头;完成经 host 回调回执并转换字节)
    Dtoh {
        id: u64,
        want_elems: usize,
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

// ============================================================================
// 回执桥(waker oneshot;server 侧 send 唤醒,client 侧 Future 真异步 pending)
// ============================================================================
// 语义:htod/dtoh/sync 的 Future pending 至 GPU 真完成(host 回调/栅栏触发
// send);launch/alloc issue 即 send → Future 立即 resolve。
// 不占线程:pending 期间执行器自由调度其他任务(非 Condvar 死等)。

struct Once<T> {
    value: Mutex<Option<T>>,
    waker: Mutex<Option<Waker>>,
}

pub struct Ack<T>(Arc<Once<T>>);

impl<T> Ack<T> {
    pub fn pair() -> (Self, Waiter<T>) {
        let inner = Arc::new(Once { value: Mutex::new(None), waker: Mutex::new(None) });
        (Self(inner.clone()), Waiter(inner))
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
