//! Tokenizer(通用能力;`tokenizers` crate 直载 tokenizer.json)。
//!
//! **拆分律适用(§四 23)**:本模块只含**全模型共用机制**——编解码、
//! eos 判定、chat 包装的执行形态;**家族特有事实**(eos 特殊 token 名、
//! chat 格式串)由各 specs/<family>.rs 以参数注入(见
//! `specs::qwen35::load_tokenizer`)。tokenizer.json 本身是每份模型快照
//! 自带资产,与权重同目录加载。
//!
//! 挂账:serving 层读快照内 chat_template.jinja + minijinja 渲染,才是
//! 模板真通用;ChatFormat 前后缀对是纯文本单轮路径的通用表达。

use crate::contract::ModelError;
use std::path::Path;

/// chat 文本路径格式(前后缀对 + 多轮模板;数据而非代码 —— 任何家族可表达)。
/// jinja 视觉宏归 serving 层;多轮渲染经 [`ChatFormat::render_chat`]。
#[derive(Clone, Debug)]
pub struct ChatFormat {
    /// 用户内容前缀(如 Qwen:`<|im_start|>user\n`)
    pub prefix: String,
    /// 用户内容后缀 + 助手起手(如 Qwen:`<|im_end|>\n<|im_start|>assistant\n`)
    pub suffix: String,
    /// 多轮:消息开启(如 Qwen:`<|im_start|>`;role + 换行由渲染拼接)
    pub im_open: String,
    /// 多轮:消息收尾(如 Qwen:`<|im_end|>\n`)
    pub im_close: String,
    /// 多轮:assistant 起手(如 Qwen:`<|im_start|>assistant\n`)
    pub assistant_open: String,
    /// 非思考模式的空 think 预填(如 Qwen:`<think>\n\n</think>\n\n`;
    /// §三十二/三十三:格式约束任务缺它 = N/A/答案段损坏;"" = 思考模式)
    pub think_prefill: String,
}

impl ChatFormat {
    /// 多轮渲染:(role, text) 序列 → 家族模板;thinking = false 时尾填
    /// 空 think 块(Qwen3 非思考模式;缺它格式约束任务答 N/A/空)。
    pub fn render_chat(&self, msgs: &[(&str, &str)], thinking: bool) -> String {
        let mut out = String::new();
        for (role, text) in msgs {
            out += &format!(
                "{}{}\n{text}{}\n",
                self.im_open, role, self.im_close
            );
        }
        out += &self.assistant_open;
        if !thinking {
            out += &self.think_prefill;
        }
        out
    }
}

/// 分词器声明(家族特有事实的**数据形态**;住 `ModelSpec.tokenizer`,
/// 与 dims/层型表同族 —— 模型档位用什么分词器,是声明不是函数)
#[derive(Clone, Debug)]
pub struct TokenizerSpec {
    /// 终止 token 名族(加载期经词表解析为 id;如 Qwen: im_end/endoftext)
    pub eos_tokens: Vec<&'static str>,
    /// chat 文本路径格式(单轮;多轮/jinja 挂账 serving)
    pub chat: ChatFormat,
}

/// Tokenizer(编解码 + eos 判定 + chat 包装;家族事实经声明注入)
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// 终止 token 族(由注入的特殊 token 名解析;生成循环遇任一即停)
    eos_ids: Vec<u32>,
    fmt: ChatFormat,
}

impl Tokenizer {
    /// 从模型快照目录加载(tokenizer.json 必在)+ 按 spec 注入家族事实
    pub fn from_spec(dir: &Path, spec: &TokenizerSpec) -> Result<Self, ModelError> {
        let p = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| ModelError::Msg(format!("tokenizer 加载 {p:?}: {e}")))?;
        let eos_ids = spec
            .eos_tokens
            .iter()
            .filter_map(|t| inner.token_to_id(t))
            .collect();
        Ok(Self { inner, eos_ids, fmt: spec.chat.clone() })
    }

    /// 文本 → token id(不自动加特殊 token;调用方经 chat_wrap 拼模板)
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// token id → 文本(skip special;增量解码 = 全量解码后取后缀差分,
    /// byte-level BPE 的多字节字符跨 token 场景由全量重解天然兜住)
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

    /// 终止 token 判定(生成循环停机条件)
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }

    /// 已解析的终止 id 族(GenSpec.eos_ids 的原料)
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// 纯文本单轮 chat 包装(执行形态通用;格式串是注入的家族事实)
    pub fn chat_wrap(&self, user_content: &str) -> String {
        format!("{}{user_content}{}", self.fmt.prefix, self.fmt.suffix)
    }
}
