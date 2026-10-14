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
    /// OpenAI 兼容 `ignore_eos`:true = 忽略 EOS 强制跑满 `max_new`
    /// (2026-10-13 落地)。**用途 = 测速口径**:decode 吞吐必须在不被
    /// "模型想不想说话"干扰的足量生成上测(词汤 OOD 输入下模型常吐
    /// 2 token 就 EOS → 旧口径读数是伪影,见坑册 §23/§24)
    pub ignore_eos: bool,
}

/// per-turn 采样**覆写**(部分字段;`None` 字段 = 继承引擎 `knobs.sampler`)。
/// 为什么不直接传整块 `SamplerCfg`:请求只带 `temperature/top_p/top_k`,
/// 而 `rep_penalty` 等档位旋钮住引擎侧 —— 整块传会**静默丢掉**后者。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TurnSampler {
    /// OpenAI `temperature`;0 = 本 turn 贪心
    pub temp: Option<f32>,
    /// OpenAI `top_k`
    pub topk: Option<usize>,
    /// OpenAI `top_p`
    pub topp: Option<f32>,
}

impl TurnSampler {
    /// 三字段全缺 = 请求没表达采样意图 ⇒ 完全走引擎缺省
    pub fn is_empty(&self) -> bool {
        self.temp.is_none() && self.topk.is_none() && self.topp.is_none()
    }
}

/// 提交面可选项(per-turn;AGT-06:**OpenAI 协议的采样意图逐请求生效**)。
/// 历史教训:这些参数曾被服务面"收下 + 告警但未生效" —— 客户端发来的
/// 采样意图被全局单源静默覆盖,是"模型看起来不听话"的头号制造机。
#[derive(Clone, Debug, Default)]
pub struct TurnOpts {
    /// 忽略 EOS 跑满预算([`TurnSpec::ignore_eos`] 同义;测速口径)
    pub ignore_eos: bool,
    /// 采样覆写(见 [`TurnSampler`])
    pub sampler: Option<TurnSampler>,
    /// 停止串(命中即停,输出截断于该串之前;空 = 不启用)
    pub stop: Vec<String>,
    /// RNG 种子覆写(`None` = turn id 派生;同 turn 重放仍确定)
    pub seed: Option<u64>,
}

/// turn 终了原因(OpenAI `finish_reason` 真源)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// 模型自吐 EOS(自然停)
    Stop,
    /// 触到 max_new 预算(含 ignore_eos 强制跑满)
    Length,
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
    /// turn 终了(text = 生成全文,eos 不入文;finish = 终了原因)
    Completed { turn: u64, text: String, finish: FinishReason },
    /// turn 失败(槽/执行错误;槽已释放,队列继续)
    Failed { turn: u64, err: String },
}
