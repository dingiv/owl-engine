//! signal —— 后端无关的 Signal 式依赖追踪(设计稿 §二·一;2026-09-23 自 owl-shared 并入,词汇 Token 归本包)。
//!
//! Vue 语义映射:`enter(sink)` = 打开 effect 作用域;`emit(token)` =
//! effect 内读信号(自动登记依赖);栈空 = untracked(零开销直通)。
//!
//! 本 crate **零后端依赖**(不认识 CUDA/ROCm):它只负责
//! "把 token 递给当前作用域的 sink"这一件事。物理资源语义
//! (显存/流/相位门禁)归各后端;词汇类型归 owl-iface。
//!
//! 线程模型:A3(每卡一线程)下追踪栈天然线程隔离;本 crate 的
//! thread-local 栈与单进程多实例拓扑(A2.6)兼容。

pub use super::Token;
use std::cell::RefCell;
use std::sync::Arc;

/// 被追踪资源的身份:(id, generation)。
///
/// 与 owl-iface 的 `BufToken` 同构(iface 侧做类型别名);signal 层
/// 不定义任何 CUDA 语义,只做身份的传递与登记。


/// 依赖接收器:token 被当前作用域订阅时调用。
pub type Sink = Arc<dyn Fn(Token) + Send + Sync>;

/// 追踪作用域 id(进程内唯一,单调递增)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(u64);

struct ScopeFrame {
    id: ScopeId,
    sink: Sink,
}

thread_local! {
    static STACK: RefCell<Vec<Arc<ScopeFrame>>> = const { RefCell::new(Vec::new()) };
    static NEXT_SCOPE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// 追踪句柄:**可 Clone**(同一作用域的多份引用,如 KernelCtx 与其
/// 持有的 sink 绑定共享同一 scope)。Drop 时仅当自己是栈顶才弹出
/// (克隆句柄先死不破坏栈序)。
#[derive(Clone)]
pub struct TrackingHandle {
    frame: Arc<ScopeFrame>,
}

impl Drop for TrackingHandle {
    fn drop(&mut self) {
        // try_with:线程析构期访问 TLS 会 AccessError——此时追踪已无意义,
        // 静默放弃(2026-09-22 dry-run 线程退出 abort 事故的修复)
        let _ = STACK.try_with(|s| {
            let mut stack = s.borrow_mut();
            if stack.last().map(|f| f.id) == Some(self.frame.id) {
                stack.pop();
            }
        });
    }
}

/// 打开一个追踪作用域。作用域内所有 [`emit`] 都会路由到 `sink`。
pub fn enter(sink: Sink) -> TrackingHandle {
    NEXT_SCOPE.with(|n| {
        let id = ScopeId(n.get());
        n.set(n.get().wrapping_add(1));
        let frame = Arc::new(ScopeFrame { id, sink });
        STACK.with(|s| s.borrow_mut().push(Arc::clone(&frame)));
        TrackingHandle { frame }
    })
}

/// 发射一个 token:栈非空 → 路由到最内层作用域的 sink;
/// 栈空(untracked,如 E 阶段)→ **no-op**,零分配零锁。
pub fn emit(t: Token) {
    // try_with:同 Drop——线程析构期 emit = 追踪窗口已关,no-op
    let _ = STACK.try_with(|s| {
        if let Some(frame) = s.borrow().last() {
            (Arc::clone(&frame.sink))(t);
        }
    });
}

/// 当前是否处于追踪作用域内(E 阶段 = false,热路径判读一次)
pub fn tracked() -> bool {
    STACK.with(|s| !s.borrow().is_empty())
}

/// 当前作用域 id(调试/断言用)
pub fn current_scope() -> Option<ScopeId> {
    STACK.with(|s| s.borrow().last().map(|f| f.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn sink_vec(v: Arc<Mutex<Vec<Token>>>) -> Sink {
        Arc::new(move |t: Token| v.lock().unwrap().push(t))
    }

    /// 基础:作用域内 emit 被接收
    #[test]
    fn emit_routes_to_top_scope() {
        let got = Arc::new(Mutex::new(Vec::new()));
        let g = enter(sink_vec(Arc::clone(&got)));
        let t = Token { id: 1, gen: 1 };
        emit(t);
        assert_eq!(*got.lock().unwrap(), vec![t]);
        assert!(tracked());
        drop(g);
        // untracked:emit 为 no-op(E 阶段零开销)
        emit(t);
        assert_eq!(got.lock().unwrap().len(), 1);
        assert!(!tracked());
    }

    /// 嵌套:emit 路由到最内层;guard 逆序弹出
    #[test]
    fn nested_scopes_route_innermost() {
        let outer = Arc::new(Mutex::new(Vec::new()));
        let inner = Arc::new(Mutex::new(Vec::new()));
        let _og = enter(sink_vec(Arc::clone(&outer)));
        {
            let _ig = enter(sink_vec(Arc::clone(&inner)));
            emit(Token { id: 1, gen: 1 });
        }
        emit(Token { id: 2, gen: 1 });
        assert_eq!(inner.lock().unwrap().len(), 1);
        assert_eq!(outer.lock().unwrap().len(), 1);
        assert_eq!(outer.lock().unwrap()[0].id, 2);
    }

    /// 多 sink:同一 token 只到最内层(不穿透)
    #[test]
    fn emit_does_not_penetrate() {
        let a = Arc::new(Mutex::new(Vec::new()));
        let b = Arc::new(Mutex::new(Vec::new()));
        let _ga = enter(sink_vec(Arc::clone(&a)));
        let _gb = enter(sink_vec(Arc::clone(&b)));
        emit(Token { id: 7, gen: 3 });
        assert_eq!(b.lock().unwrap().len(), 1);
        assert_eq!(a.lock().unwrap().len(), 0);
    }

    /// scope id 单调
    #[test]
    fn scope_ids_monotonic() {
        let a = current_scope();
        let _g = enter(sink_vec(Arc::new(Mutex::new(Vec::new()))));
        let b = current_scope();
        let _g2 = enter(sink_vec(Arc::new(Mutex::new(Vec::new()))));
        let c = current_scope();
        if let (Some(a), Some(b), Some(c)) = (a, b, c) {
            assert!(b.0 > a.0 && c.0 > b.0);
        }
    }
}
