//! 请求面词汇:Session / Turn / Step 三层(2026-09-26 用户裁定)。
//!
//! - **Session**([`crate::session`]):agent 连续会话(多轮 prompt);
//!   引擎侧感知 = KV cache 区域 + 会话 token 账;
//! - **Turn**(本模块):用户发来一条新消息(全量 prompt)→ 一个推理
//!   生命周期 —— 提交后入队,引擎推进至产出文本或工具调用;
//! - **Step**:turn 内一次**推理段** —— 模型推理出工具调用 → 客户端
//!   执行 → 结果回传 → 继续推理 = 下一个 step(工具循环轮次)。
//!
//! 引擎内部还有更细的推进单位 —— **调度步**(pump 一次 = 一个采样步
//! 或一个 prefill 块),属实现细节不上层词汇。层级:Session ⊃ Turn ⊃
//! Step ⊃ 调度步。
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
    /// prompt 段推进中(块式喂入后 = 一个 chunk 落定;无文本产出)
    Prefill { turn: u64, fed: usize, total: usize },
    /// 采样步产出一个 token(delta = 解码后文本增量;多字节字符跨
    /// token 由全量重解差分兜住)
    Token { turn: u64, delta: String },
    /// turn 终了(eos 命中或预算尽;text = 生成全文,eos 不入文)
    Completed { turn: u64, text: String },
    /// turn 失败(槽/执行错误;槽已释放,队列继续)
    Failed { turn: u64, err: String },
}
