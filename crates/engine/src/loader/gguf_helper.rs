//! GGUF 元数据查询 + tokenizer 组装(T2-三 搬运)。
//!
//! port 出处:xinfer `utils/gguf_helper.rs`(rev 当前 main;akin 宏展开为
//! 显式 impl;anyhow → [`crate::error::Error`];两处 XxHash 场景不涉及)。
//!
//! xinfer 的 `undo_tiled_v_heads_*`/`restore_qwen35_*` 张量重排函数不搬:
//! 它们是 candle Tensor 方法面,归宿在 models/layers(T3 随模型搬运),
//! 避免双源。

use super::gguf::{Content, Value};
use crate::error::{Error, Result};
use std::collections::HashMap;

use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::{self, byte_level::ByteLevel, fuse::Fuse, strip::Strip};
use tokenizers::models::bpe::BpeBuilder;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::{self, Prepend, Replace};
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::{
    processors, AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};

pub fn parse_gguf_value(value: &Value) -> String {
    match value {
        Value::Array(vs) => vs
            .iter()
            .map(parse_gguf_value)
            .collect::<Vec<String>>()
            .join(", "),
        Value::Bool(b) => b.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::String(x) => x.clone(),
        Value::U8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
    }
}

// ---- TryFromValue 机制(akin 宏的显式展开) ----

pub trait TryFromValue: Sized {
    fn try_from_value(value: Value) -> Result<Self>;
}

macro_rules! impl_try_from_value {
    ($($t:ty => $get:ident),* $(,)?) => {
        $(
            impl TryFromValue for $t {
                fn try_from_value(value: Value) -> Result<Self> {
                    value.$get().map_err(|e| Error::Msg(format!("value is not a `{}`: {e}", stringify!($t))))
                }
            }
        )*
    };
}

impl_try_from_value!(
    String => to_string_owned,
    bool => to_bool,
    f32 => to_f32,
    f64 => to_f64,
    i8 => to_i8,
    i16 => to_i16,
    i32 => to_i32,
    i64 => to_i64,
    u8 => to_u8,
    u16 => to_u16,
    u32 => to_u32,
    u64 => to_u64,
);

// String 的 to_string 返回 &String,单独适配
pub(crate) trait ToStringOwned {
    fn to_string_owned(&self) -> Result<String>;
}
impl ToStringOwned for Value {
    fn to_string_owned(&self) -> Result<String> {
        match self {
            Value::String(s) => Ok(s.clone()),
            v => Err(Error::Msg(format!("not a string {v:?}"))),
        }
    }
}

impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: Value) -> Result<Self> {
        let arr = value_vec
            .to_vec()
            .map_err(|_| Error::Msg("value is not a `Vec`".into()))?;
        arr.clone().into_iter().map(T::try_from_value).collect()
    }
}

pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T>;
}

impl<T: TryFromValue> TryValueInto<T> for Value {
    fn try_value_into(self) -> Result<T> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<Value> {
    fn try_value_into(self) -> Result<T> {
        match self {
            Some(value) => value.try_value_into(),
            None => Err(Error::Msg(
                "Expected `Option<Value>` to contain a value".into(),
            )),
        }
    }
}

// ---- ContentMetadata(多分片元数据合并视图) ----

pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, Value>,
}

impl ContentMetadata<'_> {
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();
        value
            .try_value_into()
            .map_err(|e| Error::Msg(format!("`{prop_key}` `{e}`")))
    }

    pub fn get_option_value<T: TryFromValue>(&self, field_name: &str) -> Result<Option<T>> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();
        value
            .map(|v| v.try_value_into().map_err(|e| Error::Msg(format!("`{prop_key}` `{e}`"))))
            .map_or(Ok(None), |res: Result<T>| res.map(Some))
    }

    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_present = true;
        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
            if !self.metadata.contains_key(&prop_key) {
                all_present = false;
                tracing::info!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }
        crate::ensure!(all_present, "Tokenizer is missing required props");
        Ok(())
    }
}

/// 多分片 Content(读 N 个文件合并 all_metadata;= xinfer gguf_helper::Content)
pub struct MultiContent {
    pub contents: Vec<Content>,
    pub all_metadata: HashMap<String, Value>,
}

impl MultiContent {
    pub fn from_paths(paths: &[std::path::PathBuf]) -> Result<Self> {
        let mut contents = Vec::new();
        for p in paths {
            let mut f = std::fs::File::open(p)
                .map_err(|e| Error::Msg(format!("gguf: open {}: {e}", p.display())))?;
            contents.push(Content::read(&mut f)?);
        }
        let n_splits: Vec<u64> = contents
            .iter()
            .filter_map(|c| c.metadata.get("split.count").and_then(|v| v.to_u64().ok()))
            .fold(Vec::new(), |mut acc, x| {
                if !acc.contains(&x) {
                    acc.push(x);
                }
                acc
            });
        if !n_splits.is_empty() && n_splits[0] > 0 && paths.len() != n_splits[0] as usize {
            return Err(Error::Msg(format!(
                "Number of GGUF files does not match the number of splits, expected {} files.",
                n_splits[0]
            )));
        } else if n_splits.len() == 1 {
            tracing::info!("GGUF file has been split into {} shards", n_splits[0]);
        }
        let mut all_metadata = HashMap::new();
        for c in &contents {
            all_metadata.extend(c.metadata.clone());
        }
        Ok(Self { contents, all_metadata })
    }

    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.all_metadata
    }
}

// ---- chat template ----

pub fn get_gguf_chat_template(metadata: &HashMap<String, Value>) -> Result<Option<String>> {
    let cm = ContentMetadata {
        path_prefix: "tokenizer",
        metadata,
    };
    cm.get_option_value("chat_template")
}

// ---- tokenizer 组装 ----

#[derive(Debug)]
enum TokenizerKind {
    Unigram,
    Bpe,
}

pub struct GGUFInfo {
    pub tokenizer: Tokenizer,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub unk: Option<String>,
    pub pad_token: Option<String>,
    pub context_length: Option<usize>,
    pub chat_template: Option<String>,
}

struct PropsGGUF {
    model: String,
    pre: Option<String>,
    tokens: Vec<String>,
    added_tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
    unk: Option<u32>,
    eos: Option<u32>,
    bos: Option<u32>,
    pad: Option<u32>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        c.has_required_keys(&["model", "tokens", "eos_token_id"])?;
        Ok(Self {
            model: c.get_value("model")?,
            pre: c.get_value("pre").ok(),
            tokens: c.get_value("tokens")?,
            added_tokens: c.get_value("added_tokens").ok(),
            scores: c.get_value("scores").ok(),
            merges: c.get_value("merges").ok(),
            unk: c.get_value("unknown_token_id").ok(),
            eos: c.get_value("eos_token_id").ok(),
            bos: c.get_value("bos_token_id").ok(),
            pad: c.get_value("pad_token_id").ok(),
        })
    }
}

/// 从合并元数据组装 tokenizer(= xinfer get_gguf_info 的元数据消费段)
pub fn get_gguf_info(metadata: &HashMap<String, Value>) -> Result<GGUFInfo> {
    let chat_template = get_gguf_chat_template(metadata).ok().flatten();

    let cm = ContentMetadata {
        path_prefix: "tokenizer.ggml",
        metadata,
    };
    let md_get = |s: &str| {
        metadata
            .get(s)
            .ok_or_else(|| Error::Msg(format!("cannot find {s} in metadata")))
    };

    let mut context_length = 4096u32;
    let mut token_types = Vec::<i32>::new();
    for key in metadata.keys() {
        if key.contains(".context_length") {
            context_length = md_get(key)?.to_u32()?;
        }
        if key.contains("tokenizer.ggml.token_type") {
            let vtypes = md_get(key)?.to_vec()?;
            token_types.extend(vtypes.iter().filter_map(|v| v.to_i32().ok()));
        }
    }
    let props = PropsGGUF::try_from(cm)?;

    let (mut tokenizer, _kind) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        "gpt2" => bpe_tokenizer(&props)?,
        other => return Err(Error::Msg(format!("Tokenizer model `{other}` not supported."))),
    };

    let mut num_special_tokens = 0;
    if token_types.len() == props.tokens.len() {
        for (i, tk) in props.tokens.iter().enumerate() {
            if token_types[i] != 1 {
                tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
                num_special_tokens += 1;
            }
        }
    }
    tracing::info!(
        "GGUF tokenizer model `{}`: vocab {} special {} added {} merges {} scores {}",
        props.model,
        tokenizer.get_vocab_size(true),
        num_special_tokens,
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
    );

    let pick = |id: Option<u32>| id.map(|u| props.tokens[u as usize].clone());
    Ok(GGUFInfo {
        tokenizer,
        bos: pick(props.bos),
        eos: pick(props.eos),
        unk: pick(props.unk),
        pad_token: pick(props.pad),
        context_length: Some(context_length as usize),
        chat_template,
    })
}

/// 从文件组直接出 GGUFInfo(= load_gguf_info_from_files)
pub fn load_gguf_info_from_files(paths: &[std::path::PathBuf]) -> Result<GGUFInfo> {
    let content = MultiContent::from_paths(paths)?;
    get_gguf_info(content.get_metadata())
}

fn bpe_pre_tokenizer_regex(pre: Option<&str>) -> &'static str {
    match pre {
        Some("qwen35") => "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        Some("qwen2") | Some("deepseek-r1-qwen") | Some("kormo") => "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        _ => "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    }
}

/// BPE merges 解析(空格分隔对;ahash 解封后由 bpe_tokenizer 调用)
#[allow(dead_code)]
fn build_bpe_merges(p: &PropsGGUF) -> Vec<(String, String)> {
    p.merges
        .as_ref()
        .unwrap_or(&Vec::new())
        .iter()
        .map(|merge| {
            let mut it = merge.splitn(2, ' ');
            let a = it.next().unwrap_or_default();
            let b = it.next().unwrap_or_default();
            (a.to_string(), b.to_string())
        })
        .collect()
}

fn bpe_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let merges = build_bpe_merges(p);

    let mut vocab = HashMap::new();
    for (i, token) in p.tokens.iter().enumerate() {
        vocab.insert(token.clone(), i as u32);
    }
    let mut vocab_vec: Vec<(String, u32)> = vocab.into_iter().collect();
    vocab_vec.sort(); // 保证可重现

    // ahash 已裁决入依赖(tokenizers 0.21 传递依赖,显式声明;零边际成本)
    let vocab_map: ahash::AHashMap<String, u32> = vocab_vec.into_iter().collect();
    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab_map, merges)
        .build()
        .map_err(|e| Error::Msg(format!("bpe build: {e}")))?;

    let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));
    let split = Split::new(
        SplitPattern::Regex(bpe_pre_tokenizer_regex(p.pre.as_deref()).to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| Error::Msg(format!("split: {e}")))?;
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    let bl: DecoderWrapper = decoders::byte_level::ByteLevel::new(false, false, false).into();
    tokenizer.with_decoder(Some(bl));
    tokenizer.with_post_processor(Some(processors::byte_level::ByteLevel::new(
        false, false, false,
    )));

    for i in [p.bos, p.eos, p.unk] {
        if let Some(u) = i {
            let tk = p.tokens[u as usize].clone();
            tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
        }
    }
    Ok((tokenizer, TokenizerKind::Bpe))
}

fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let unk = p.unk.unwrap_or(0); // SentencePiece 默认 UNK = 0
    let model = {
        let Some(scores) = p.scores.as_ref() else {
            return Err(Error::Msg(
                "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
                    .into(),
            ));
        };
        let vocab: Vec<(String, f64)> = p
            .tokens
            .iter()
            .cloned()
            .zip(scores.iter().map(|f| *f as f64))
            .collect();
        Unigram::from(vocab, Some(unk as usize), true)
            .map_err(|e| Error::Msg(format!("unigram: {e}")))?
    };

    let decoders: Vec<DecoderWrapper> = vec![
        ByteFallback::default().into(),
        Fuse::default().into(),
        Strip::new(' ', 1, 0).into(),
        Replace::new("▁", " ")
            .map_err(|e| Error::Msg(format!("replace: {e}")))?
            .into(),
    ];
    let normalizers_seq: Vec<NormalizerWrapper> = vec![
        Prepend::new("▁".to_owned()).into(),
        Replace::new(" ", "▁")
            .map_err(|e| Error::Msg(format!("replace: {e}")))?
            .into(),
    ];

    let mut tokenizer = Tokenizer::new(ModelWrapper::Unigram(model));
    let bpe_decoder: DecoderWrapper =
        tokenizers::decoders::sequence::Sequence::new(decoders).into();
    tokenizer.with_decoder(Some(bpe_decoder));
    tokenizer.with_normalizer(Some(NormalizerWrapper::Sequence(normalizers::Sequence::new(normalizers_seq))));

    for i in [p.bos, p.eos, Some(unk)] {
        if let Some(u) = i {
            let tk = p.tokens[u as usize].clone();
            tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
        }
    }
    Ok((tokenizer, TokenizerKind::Unigram))
}

#[cfg(test)]
mod tests {
    use super::bpe_pre_tokenizer_regex;

    #[test]
    fn qwen35_gguf_bpe_uses_qwen35_regex() {
        let regex = bpe_pre_tokenizer_regex(Some("qwen35"));
        assert!(regex.contains("\\p{M}"));
        assert!(regex.contains("(?:'[sS]"));
    }

    #[test]
    fn default_gguf_bpe_regex_stays_generic() {
        let regex = bpe_pre_tokenizer_regex(None);
        assert!(regex.contains("(?i:'s|'t|'re|'ve|'m|'ll|'d)"));
    }
}
