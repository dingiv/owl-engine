//! **跳轮重复探测**(2026-10-31 立案;第一形态 = **只可见,不改生成**)。
//!
//! # 为什么存在
//!
//! 生产实录(`/debug/agent` seq29–45,3080 TP2 档):模型连续 **17 轮**发出**逐字相同**
//! 的 tool_call(`bash`,87B 参数),每轮 `completion=89 / reasoning=29`,prompt 每轮
//! +1441 token —— 工具回同一份内容 ⇒ 上下文尾部同形 ⇒ 输出逐字复现,永远不收敛。
//!
//! 关键事实:**那台机器已经是采样档(temp=0.3)** ⇒ 温度/重复惩罚**治不了**这种
//! 跨轮固定点(惩罚只能压单词,压不住"整轮重发";且 greedy 档下 rep 槽填哨兵,
//! `rep_penalty` 本就不生效)。所以本模块只在**请求入口**看一眼历史,把"发生了
//! 跳轮重复"这件事**记账 + WARN**,由人决定要不要升级成提示注入/硬截断。
//!
//! # 判据(显式,不猜)
//!
//! 1. **assistant 轮签名** = `hash(tool_calls[].name + 规范化参数 JSON)`;无 tool_call
//!    的 assistant 轮 = `hash("content\0" + content)`(content 空则跳过);
//! 2. `same_call` = 全窗口内**同一签名出现次数**的最大值;
//! 3. `trailing_same` = assistant 签名序列**尾部连续相同**的长度;
//! 4. 触发 = `same_call >= 3`(`REPEAT_MIN`)—— 3 次同一调用在健康 loop 里极少见,
//!    而在实测病灶里是 17。
//!
//! 规范化参数:对象键递归排序后 `serde_json::to_string` —— 客户端键序不稳定时
//! 签名仍然一致(否则同一调用每次哈希都不同,探测形同虚设)。

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use serde_json::Value;

/// 同一签名出现这么多次 ⇒ 判为跳轮重复。
pub const REPEAT_MIN: usize = 3;

/// 探测结论(**只读事实**;不含任何"该怎么处置"的策略)。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RepeatFacts {
    /// 全窗口内同一签名出现次数最大值(含工具调用轮与控制内容的 assistant 轮)
    pub same_call: usize,
    /// assistant 签名序列尾部连续相同的长度
    pub trailing_same: usize,
    /// 命中最多的那个签名(十六进制;与日志同源,便于跨轮比对)
    pub sig: Option<String>,
    /// 该签名的工具名(纯文本轮 = `None`)
    pub tool: Option<String>,
}

impl RepeatFacts {
    /// 是否达触发线(今天只用于 `WARN`;不改生成)
    #[must_use]
    pub fn triggered(&self) -> bool {
        self.same_call >= REPEAT_MIN
    }

    /// 一行摘要(日志/诊断面用;不含参数原文 —— 可能带本地路径)
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "same_call={} trailing_same={} tool={} sig={}",
            self.same_call,
            self.trailing_same,
            self.tool.as_deref().unwrap_or("-"),
            self.sig.as_deref().unwrap_or("-"),
        )
    }
}

/// 扫描一遍历史消息(纯函数:零 I/O、零状态)。
///
/// 空消息表/无 assistant 轮 ⇒ `None`(**不是** "0 次重复":没有观测对象和
/// "观测到 0 次"是两件事,调用方据此决定要不要记账)。
#[must_use]
pub fn scan(messages: &[Value]) -> Option<RepeatFacts> {
    let mut sigs: Vec<(u64, Option<String>)> = Vec::new();
    for m in messages {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if let Some(sig) = assistant_signature(m) {
            sigs.push(sig);
        }
    }
    if sigs.is_empty() {
        return None;
    }

    // 全窗口计数(顺序无关:BTreeMap 只为稳定输出)
    let mut counts: BTreeMap<u64, (usize, Option<String>)> = BTreeMap::new();
    for (sig, tool) in &sigs {
        let e = counts.entry(*sig).or_insert((0, tool.clone()));
        e.0 += 1;
        if e.1.is_none() {
            e.1 = tool.clone();
        }
    }
    let (best_sig, (best_n, best_tool)) = counts
        .iter()
        .max_by_key(|(_, (n, _))| *n)
        .map(|(s, (n, t))| (*s, (*n, t.clone())))
        .expect("sigs 非空 ⇒ counts 非空");

    // 尾部连续相同
    let mut trailing = 1usize;
    for w in sigs.windows(2).rev() {
        if w[0].0 == w[1].0 {
            trailing += 1;
        } else {
            break;
        }
    }

    Some(RepeatFacts {
        same_call: best_n,
        trailing_same: trailing,
        sig: Some(format!("{best_sig:016x}")),
        tool: best_tool,
    })
}

/// 一轮 assistant 的签名:(哈希, 工具名)。
///
/// 有 `tool_calls` ⇒ 按 **name + 规范化 args** 哈希(多调用按顺序拼);
/// 否则按 `content` 哈希(空内容 = 该轮不可判,返回 `None`)。
fn assistant_signature(m: &Value) -> Option<(u64, Option<String>)> {
    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
        if !calls.is_empty() {
            let mut h = DefaultHasher::new();
            let mut first_name: Option<String> = None;
            for c in calls {
                let f = c.get("function").unwrap_or(c);
                let name = f.get("name").and_then(Value::as_str).unwrap_or("");
                let args = f.get("arguments").cloned().unwrap_or(Value::Null);
                name.hash(&mut h);
                canonical(&args).hash(&mut h);
                if first_name.is_none() && !name.is_empty() {
                    first_name = Some(name.to_string());
                }
            }
            return Some((h.finish(), first_name));
        }
    }
    let content = m.get("content").and_then(Value::as_str).unwrap_or("");
    if content.is_empty() {
        return None;
    }
    let mut h = DefaultHasher::new();
    "content\0".hash(&mut h);
    content.hash(&mut h);
    Some((h.finish(), None))
}

/// 规范化 JSON:对象键递归排序(`serde_json` 的 `Value` 是 BTreeMap 序,但客户端
/// 传的是 `Map` 序 —— 这里显式重排,保证同参不同键序得到同一签名)。
fn canonical(v: &Value) -> String {
    match v {
        Value::Object(o) => {
            let sorted: BTreeMap<&str, String> =
                o.iter().map(|(k, v)| (k.as_str(), canonical(v))).collect();
            let body: Vec<String> = sorted
                .into_iter()
                .map(|(k, v)| format!("{}:{}", Value::String(k.to_string()), v))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(a) => {
            let body: Vec<String> = a.iter().map(canonical).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: Value) -> Value {
        json!({"role":"assistant","content":"","tool_calls":[
            {"id":"x","type":"function","function":{"name":name,"arguments":args}}
        ]})
    }

    fn tool_result() -> Value {
        json!({"role":"tool","content":"same body"})
    }

    /// 病灶复刻:17 轮同一调用 ⇒ 触发;尾部连续 = 17
    #[test]
    fn identical_tool_call_rounds_trigger() {
        let mut msgs = vec![json!({"role":"user","content":"go"})];
        for _ in 0..17 {
            msgs.push(call("bash", json!({"command":"sed -n '1,40p' a.ts"})));
            msgs.push(tool_result());
        }
        let f = scan(&msgs).expect("有 assistant 轮");
        assert_eq!(f.same_call, 17);
        assert_eq!(f.trailing_same, 17);
        assert_eq!(f.tool.as_deref(), Some("bash"));
        assert!(f.triggered());
    }

    /// 键序不同 = 同一调用(规范化生效);参数不同 = 不同调用
    #[test]
    fn arg_key_order_does_not_change_signature() {
        let a = scan(&[
            call("bash", json!({"command":"ls","cwd":"/tmp"})),
            call("bash", json!({"cwd":"/tmp","command":"ls"})),
        ])
        .expect("有 assistant 轮");
        assert_eq!(a.same_call, 2, "键序不应改变签名");
        assert!(!a.triggered());

        let b = scan(&[
            call("bash", json!({"command":"ls"})),
            call("bash", json!({"command":"ls -l"})),
        ])
        .expect("有 assistant 轮");
        assert_eq!(b.same_call, 1, "不同参数 = 不同签名");
    }

    /// 纯文本轮重复也算(无 tool_calls 的 assistant)
    #[test]
    fn identical_text_rounds_trigger() {
        let txt = "x".repeat(200);
        let msgs: Vec<Value> = (0..4)
            .map(|_| json!({"role":"assistant","content":txt}))
            .collect();
        let f = scan(&msgs).expect("有 assistant 轮");
        assert_eq!(f.same_call, 4);
        assert_eq!(f.tool, None);
        assert!(f.triggered());
    }

    /// 健康多轮(调用各不相同)⇒ 不触发
    #[test]
    fn distinct_calls_do_not_trigger() {
        let msgs = vec![
            call("read", json!({"path":"a"})),
            call("read", json!({"path":"b"})),
            call("bash", json!({"command":"rg foo"})),
        ];
        let f = scan(&msgs).expect("有 assistant 轮");
        assert_eq!(f.same_call, 1);
        assert_eq!(f.trailing_same, 1);
        assert!(!f.triggered());
    }

    /// 无 assistant 轮 = None(与"0 次重复"区分)
    #[test]
    fn no_assistant_turns_is_none() {
        assert_eq!(scan(&[json!({"role":"user","content":"hi"})]), None);
        assert_eq!(scan(&[]), None);
    }
}
