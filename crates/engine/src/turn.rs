//! 请求面词汇:turn(用户请求)与 step(单 token 前进)。
//!
//! 语义对齐(session-plan.md §一 + 前端 agent 惯例):前端一个 **turn**
//! ⇋ 引擎一个调度单元 —— 提交后入队,引擎逐步推进(teacher-forcing 过
//! prompt + 逐 token 采样),事件经 [`TurnEvent`] 回吐。**一个 turn 包含
//! 多个 step**,step 的治理(槽装填/状态推进/停机判定)归引擎,不上层。
//!
//! 并发形态:M0.5 = 单槽串行(一次一个 turn 占槽;submit 非阻塞入队,
//! pump 按事件点推进);真并发(多 turn 共 batch)随 M2 continuous
//! batching 换入 —— 词汇与 API 形状不变,只换引擎内部。

/// 提交面:一个用户请求(前端一个 turn)
#[derive(Clone, Debug)]
pub struct TurnSpec {
    pub prompt: String,
    /// 生成 token 上限(不含 prompt)
    pub max_new: usize,
}

/// 步级事件(pump 回吐;前端按此流式渲染)。**每 pump 必得一事件**
/// (Idle = 队列排空;Prefill = prompt 中段状态步),调用方循环 pump
/// 至 Idle 即引擎排空。
#[derive(Clone, Debug)]
pub enum TurnEvent {
    /// 引擎空闲(队列空且无活跃 turn;调用方可安全等待新提交)
    Idle,
    /// prompt 中段(teacher-forcing 状态步;无文本产出)
    Prefill { turn: u64, fed: usize, total: usize },
    /// 采样步产出一个 token(delta = 解码后文本增量;多字节字符跨
    /// token 由全量重解差分兜住)
    Token { turn: u64, delta: String },
    /// turn 终了(eos 命中或预算尽;text = 生成全文,eos 不入文)
    Completed { turn: u64, text: String },
    /// turn 失败(槽/执行错误;槽已释放,队列继续)
    Failed { turn: u64, err: String },
}
