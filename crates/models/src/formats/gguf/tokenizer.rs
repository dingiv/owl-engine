//! **GGUF 内嵌词表 / 模板 → 可运行的 `tokenizers::Tokenizer`**(2026-10-19 刀 2)。
//!
//! 设计:`docs/arch/gguf-格式支持-设计与施工.md` REQ-GGUF-02 —— **单文件自给**:
//! 词表不再要求同目录 `tokenizer.json`;特殊 token / 模板也从 `.gguf` 读。
//!
//! # 与 HF 路径的关系(为什么不直接复用 `tokenizer.json`)
//!
//! GGUF 把 HF `tokenizer.json` **拆成扁平数组**(`tokens` / `merges` /
//! `token_type`)—— 结构没了,规则得重建。重建规则**逐条对齐 llama.cpp**
//! (`src/llama-vocab.cpp`),因为数值锚 = `llama-tokenize`(同文件、同实现):
//!
//! | GGUF 键 | 用途 | 权威 |
//! |---|---|---|
//! | `tokenizer.ggml.model` | 词表模型(`gpt2` = BPE;`llama`/`t5` = Unigram) | llama.cpp `tokenizer_model` |
//! | `tokenizer.ggml.pre` | 预分词正则名(`qwen35` / `qwen2` / …) | llama.cpp `LLAMA_VOCAB_PRE_TYPE_*` 正则表 |
//! | `tokenizer.ggml.tokens` | 词表面(id = 下标) | —— |
//! | `tokenizer.ggml.merges` | BPE 合并对(空格分隔,`splitn(2, ' ')`) | HF `merges.txt` 同形 |
//! | `tokenizer.ggml.token_type` | 1=normal 2=unk 3=control 4=user_defined 5=unused 6=byte | GGUF 规范 |
//! | `tokenizer.ggml.{bos,eos,unknown,padding}_token_id` | 特殊 token **id** | —— |
//! | `tokenizer.chat_template` | Jinja 模板原文(**由 serving 面渲染**,见下) | HF 同源 |
//!
//! # 模板为什么只**读出**、不在此渲染
//!
//! 仓内已有 HF 级 minijinja 渲染器(`apps/server/src/chat_tpl.rs`,含
//! `trim_blocks/lstrip_blocks` + pycompat + HF `tojson` 四项兼容层,§3.2/§7.1
//! 已对 HF 逐字节验过)。**再放第二套 Jinja 进 models = 接口不唯一** ⇒ 本模块
//! 只把模板原文交出去(它是 GGUF 的事实),渲染归 serving 面同一实现。
//!
//! # 编码语义(与 HF/llama.cpp 对齐的三处)
//!
//! 1. 预分词 = `Split(regex, Isolated) ⊕ ByteLevel(add_prefix_space=false,
//!    trim_offsets=false, use_regex=false)`(HF `tokenizer.json` 实测形态);
//! 2. **decoder / post_processor 与 pre_tokenizer 的 ByteLevel 是同一个 struct**
//!    (tokenizers 0.21:`decoders::DecoderWrapper::ByteLevel(pre_tokenizers::byte_level::ByteLevel)`);
//! 3. 特殊 token 以 `AddedToken{special:true}` 注入(control/user_defined 两类),
//!    `add_special_tokens` 后 `encode(text, false)` 即可(不加 bos —— GGUF 里
//!    `add_bos_token=false`,与 owl `try_encode` 同律)。

use crate::contract::ModelError;

use super::reader::Gguf;

/// token_type 码(GGUF 规范;`tokenizer.ggml.token_type`)
pub const TT_NORMAL: u8 = 1;
pub const TT_UNKNOWN: u8 = 2;
pub const TT_CONTROL: u8 = 3;
pub const TT_USER_DEFINED: u8 = 4;
pub const TT_UNUSED: u8 = 5;
pub const TT_BYTE: u8 = 6;

/// 词表模型族(GGUF `tokenizer.ggml.model`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VocabModel {
    /// `gpt2` —— BPE(byte-level;Qwen 全系走这支)
    Bpe,
    /// `llama` / `t5` —— SentencePiece Unigram(需要 `scores`)
    Unigram,
    /// 未覆盖(响亮拒)
    Other(String),
}

impl VocabModel {
    fn parse(s: &str) -> Self {
        match s {
            "gpt2" => VocabModel::Bpe,
            "llama" | "t5" => VocabModel::Unigram,
            other => VocabModel::Other(other.to_string()),
        }
    }
}

/// 特殊 token id 四件套(GGUF 给 **id**,不是名)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecialIds {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unk: Option<u32>,
    pub pad: Option<u32>,
}

/// GGUF 词表面(本刀的可验主体)
pub struct GgufVocab {
    /// 可直接用于编解码的 tokenizer(预分词/decoder/post_processor 已装)
    pub inner: tokenizers::Tokenizer,
    /// 词表模型(gpt2/llama/…)
    pub model: VocabModel,
    /// 预分词规则名(`tokenizer.ggml.pre` 原文;诊断用)
    pub pre: Option<String>,
    /// 特殊 token id
    pub specials: SpecialIds,
    /// `add_bos_token`(GGUF 缺省 false)
    pub add_bos: bool,
    /// 词表大小(含 unused/control 占位)
    pub n_tokens: usize,
    /// **chat 模板原文**(`tokenizer.chat_template`;渲染归 serving 面)
    pub template: Option<String>,
}

impl GgufVocab {
    /// eos id 列表(owl `Tokenizer::eos_ids` 的原料;`eos_token_id` 在场即一项)
    pub fn eos_ids(&self) -> Vec<u32> {
        self.specials.eos.into_iter().collect()
    }

    /// 词面(id → 原文;诊断/名解析用)
    pub fn token_str(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

/// 预分词正则(llama.cpp `LLAMA_VOCAB_PRE_TYPE_*` 表;**逐字对照**
/// `src/llama-vocab.cpp` 2026-10-19 本机树)。
///
/// 已覆盖:`qwen35`(Qwen3.5/3.8 全系)、`qwen2`(含 stablelm2/hunyuan/
/// solar-open 同表)。**未知 `pre` = 响亮拒** —— 选错预分词规则 = 静默分词错位
/// (设计 R7),宁可拒启。
fn pre_regex(pre: Option<&str>) -> Result<&'static str, ModelError> {
    const DEFAULT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const QWEN35: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const QWEN2: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    match pre {
        None => Ok(DEFAULT),
        Some("qwen35") => Ok(QWEN35),
        Some("qwen2" | "stablelm2" | "hunyuan" | "solar-open" | "deepseek-r1-qwen" | "kormo") => Ok(QWEN2),
        Some(other) => Err(ModelError::Msg(format!(
            "gguf tokenizer: 预分词规则 `{other}` 未覆盖(llama.cpp LLAMA_VOCAB_PRE_TYPE 表里没登记\
             ⇒ 拒建,禁拿默认正则猜;设计 R7)"
        ))),
    }
}

/// **从 GGUF 造 tokenizer**(唯一入口;零文件依赖)。
pub fn build_vocab(g: &Gguf) -> Result<GgufVocab, ModelError> {
    let path = &g.path;
    let bad = |m: String| ModelError::Msg(format!("gguf tokenizer: {}: {m}", path.display()));
    let kv = &g.kv;
    let map = |e: String| bad(e);

    let t0 = std::time::Instant::now();
    let model_s = kv.str("tokenizer.ggml.model").map_err(map)?;
    let model = VocabModel::parse(model_s);
    let tokens = kv
        .opt_strs("tokenizer.ggml.tokens")
        .map_err(map)?
        .ok_or_else(|| bad("缺 `tokenizer.ggml.tokens`".into()))?;
    let t_tokens = t0.elapsed();
    if tokens.is_empty() {
        return Err(bad("`tokenizer.ggml.tokens` 为空".into()));
    }
    let pre = kv.opt_str("tokenizer.ggml.pre").map_err(map)?.map(|s| s.to_string());
    let token_type = kv.opt_u32s("tokenizer.ggml.token_type").map_err(map)?;
    if let Some(tt) = &token_type {
        if tt.len() != tokens.len() {
            return Err(bad(format!(
                "`tokenizer.ggml.token_type` 长度 {} ≠ tokens {}",
                tt.len(),
                tokens.len()
            )));
        }
    }
    let specials = SpecialIds {
        bos: kv.opt_u32("tokenizer.ggml.bos_token_id").map_err(map)?,
        eos: kv.opt_u32("tokenizer.ggml.eos_token_id").map_err(map)?,
        unk: kv.opt_u32("tokenizer.ggml.unknown_token_id").map_err(map)?,
        pad: kv.opt_u32("tokenizer.ggml.padding_token_id").map_err(map)?,
    };
    for (name, id) in [("bos", specials.bos), ("eos", specials.eos), ("unk", specials.unk), ("pad", specials.pad)] {
        if let Some(i) = id {
            if i as usize >= tokens.len() {
                return Err(bad(format!("`{name}_token_id` = {i} 越界(词表 {} 项)", tokens.len())));
            }
        }
    }
    let add_bos = kv.opt_bool("tokenizer.ggml.add_bos_token").map_err(map)?.unwrap_or(false);
    let template = kv.opt_str("tokenizer.chat_template").map_err(map)?.map(|s| s.to_string());

    let t_kv = t0.elapsed();
    let inner = match model {
        VocabModel::Bpe => build_bpe(&tokens, kv, &specials, pre.as_deref(), &bad)?,
        VocabModel::Unigram => build_unigram(&tokens, kv, &specials, &bad)?,
        VocabModel::Other(ref o) => {
            return Err(bad(format!("词表模型 `{o}` 未覆盖(本仓支持 gpt2/llama/t5)")))
        }
    };
    // 分相上屏(启动时间是生产指标;`tokenizers` 建 BPE 是已知重相)
    eprintln!(
        "[gguf] 词表分相:读 tokens {:.2}s + 读其余 KV {:.2}s + 建 tokenizer {:.2}s",
        t_tokens.as_secs_f32(),
        (t_kv - t_tokens).as_secs_f32(),
        (t0.elapsed() - t_kv).as_secs_f32()
    );

    Ok(GgufVocab {
        inner,
        model,
        pre,
        specials,
        add_bos,
        n_tokens: tokens.len(),
        template,
    })
}

/// BPE 臂(gpt2;唯一实测族)
fn build_bpe(
    tokens: &[String],
    kv: &super::reader::KvTable,
    specials: &SpecialIds,
    pre: Option<&str>,
    bad: &dyn Fn(String) -> ModelError,
) -> Result<tokenizers::Tokenizer, ModelError> {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::models::bpe::BpeBuilder;
    use tokenizers::tokenizer::SplitDelimiterBehavior;
    use tokenizers::AddedToken;

    let merges_raw = kv
        .opt_strs("tokenizer.ggml.merges")
        .map_err(&bad)?
        .ok_or_else(|| bad("BPE 词表缺 `tokenizer.ggml.merges`".into()))?;
    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_raw.len());
    for m in &merges_raw {
        // HF merges.txt / llama.cpp 同形:空白分隔的**一对**(其余空白属词面)
        let (a, b) = m
            .split_once(' ')
            .ok_or_else(|| bad(format!("merge `{m}` 不是 `A B` 形态")))?;
        merges.push((a.to_string(), b.to_string()));
    }

    let tb0 = std::time::Instant::now();
    // 词表 map = tokenizers 的 `AHashMap`(ahash 快哈希;唯一构造入口)
    let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }
    let tb1 = tb0.elapsed();

    let mut builder = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = specials.unk {
        builder = builder.unk_token(tokens[unk as usize].clone());
    }
    let bpe = builder.build().map_err(|e| bad(format!("BPE 构造失败: {e}")))?;
    let tb2 = tb0.elapsed();

    let byte_level = ByteLevel::new(false, false, false);
    let mut tok = tokenizers::Tokenizer::new(bpe);
    let regex = pre_regex(pre)?;
    let split = Split::new(
        SplitPattern::Regex(regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| bad(format!("预分词正则编译失败({regex}): {e}")))?;
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level.clone()),
    ])));
    tok.with_decoder(Some(byte_level.clone()));
    tok.with_post_processor(Some(byte_level));

    // 特殊 token = control(3) / user_defined(4):以 AddedToken 注入
    // (spec=1 ⇒ 不切分、不参与 BPE;HF added_tokens 的对应物)
    //
    // ⚠️ **逐 token 读 KV = O(n²) 陷阱**(2026-10-19 实测 67s → 0.3s):
    // `KvTable::opt_u32s` 每次都**重解整个数组**(返新 `Vec`),而本循环要走
    // 248k 个 token ⇒ 先把 token_type 一次读完再索引(同 `tokens` 本身的做法)。
    let tts = kv.opt_u32s("tokenizer.ggml.token_type").map_err(&bad)?;
    if let Some(tt) = &tts {
        if tt.len() != tokens.len() {
            return Err(bad(format!(
                "`tokenizer.ggml.token_type` 长度 {} ≠ tokens {}",
                tt.len(),
                tokens.len()
            )));
        }
    }
    let mut added: Vec<AddedToken> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let tt = tts.as_ref().and_then(|v| v.get(i).copied()).map(|v| v as u8);
        if matches!(tt, Some(TT_CONTROL) | Some(TT_USER_DEFINED)) || tt == Some(TT_UNKNOWN) {
            added.push(AddedToken::from(t.clone(), true));
        }
    }
    if !added.is_empty() {
        // `add_special_tokens` 返回实际新增数(0 合法:词面重复)
        let _ = tok.add_special_tokens(&added);
    }
    eprintln!(
        "[gguf] BPE 分相:词表图 {:.2}s + build {:.2}s + 装配/特殊 {:.2}s(merges {} / added {})",
        tb1.as_secs_f32(),
        (tb2 - tb1).as_secs_f32(),
        (tb0.elapsed() - tb2).as_secs_f32(),
        merges_raw.len(),
        added.len()
    );
    Ok(tok)
}

/// Unigram 臂(SentencePiece)—— **本仓未接线(挂账)**。
///
/// 为什么留拒而不留半成品 tokenizers 0.21 的 `Unigram` 需要
/// `▁` 归一化四件套(`Prepend`/`Replace`/`ByteFallback`/`Fuse`/`Strip`,
/// 见 xinfer `gguf_helper.rs::unigram_tokenizer`)—— 而**本机零 Unigram 样本**
/// (两靶位皆 `gpt2`/BPE)⇒ 写了也无门可验。本仓纪律:**无法验证 = 不宣称支持**。
/// 样本到位(如 Llama 系 GGUF)时按 xinfer 配方接线,门 = 同款逐 token 对拍。
fn build_unigram(
    _tokens: &[String],
    _kv: &super::reader::KvTable,
    _specials: &SpecialIds,
    bad: &dyn Fn(String) -> ModelError,
) -> Result<tokenizers::Tokenizer, ModelError> {
    Err(bad(
        "Unigram/SentencePiece 词表(`tokenizer.ggml.model = llama/t5`)本仓未接线\
         (零样本 ⇒ 无门可验;挂账在设计 §八)。请用 gpt2/BPE 族 GGUF"
            .into(),
    ))
}

/// 取 token_type(缺表 = None)
///
/// ⚠️ **单次调用 = 全数组重解** —— 抛进循环就是 O(n²)(见 `build_bpe` 的注);
/// 批量场景请自先读一次。
#[allow(dead_code)] // 小词表/诊断面用;热路径已改为一次读完
fn tt_of(kv: &super::reader::KvTable, i: usize) -> Result<Option<u8>, ModelError> {
    let tt = kv.opt_u32s("tokenizer.ggml.token_type").map_err(|e| {
        ModelError::Msg(format!("gguf tokenizer: 读 `tokenizer.ggml.token_type`: {e}"))
    })?;
    Ok(tt.map(|v| v[i] as u8))
}

/// 词表 dtype 事实占位(词表不参与 GEMM;保留导出面供诊断对齐)
#[allow(dead_code)]
pub(crate) const VOCAB_NOT_QUANTIZED: () = ();

#[cfg(test)]
mod tests {
    use super::*;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;
    use std::process::Command;

    fn res(key: &str) -> Option<PathBuf> {
        resource_ok(key).then(|| PathBuf::from(resource(key).expect("列了就有")))
    }

    /// **tokenize 对拍门**:与 `llama-tokenize`(llama.cpp 自家实现,同一 GGUF)
    /// 逐 token 对 id。
    ///
    /// 为什么这个锚最有说服力:同一份文件、同一个 `pre` 规则表、**独立实现**
    /// (C++ vs 本仓 Rust + `tokenizers` crate)⇒ 一致就排除了"词表数组读错 /
    /// merges 拆错 / 正则抄错 / 特殊 token 注入错"整族问题。
    #[test]
    fn tokenizes_like_llama_cpp() {
        let bin = owl_shared::env_reader::str("OWL_LLAMA_TOKENIZE").unwrap_or_else(|| {
            "/home/div/Documents/codes/repos/llama.cpp/build-new/bin/llama-tokenize".to_string()
        });
        if !PathBuf::from(&bin).is_file() {
            eprintln!("skip: llama-tokenize 不在(OWL_LLAMA_TOKENIZE 可指路)");
            return;
        }
        let libdir = owl_shared::env_reader::str("OWL_LLAMA_LIBS").unwrap_or_else(|| {
            "/home/div/Documents/codes/repos/llama.cpp/build-new/bin:/home/div/Documents/codes/packages/llama.cpp/build-master2/bin"
                .to_string()
        });
        let cases: &[&str] = &[
            "Hello, world!",
            "你好，世界",
            "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
            "The quick brown fox jumps over the lazy dog. 1234567890",
            "  leading and trailing   spaces  ",
            "emoji: 🚀🔥 混合\tTab 与\n换行",
            "don't can't I'll we've they're",
            "数学公式 x_{i}² + y² = z²（测试）",
        ];
        for key in ["gguf:4b", "gguf:27b"] {
            let Some(path) = res(key) else {
                eprintln!("skip: 资源清单缺 \"{key}\"");
                continue;
            };
            let g = Gguf::open(&path).expect("open");
            let v = build_vocab(&g).unwrap_or_else(|e| panic!("[{key}] build_vocab: {e}"));
            eprintln!(
                "[{key}] 词表 {:?} n={} pre={:?} specials={:?} add_bos={} template={}",
                v.model,
                v.n_tokens,
                v.pre,
                v.specials,
                v.add_bos,
                v.template.as_ref().map(|t| t.len()).unwrap_or(0)
            );
            for text in cases {
                let mine = v.inner.encode(*text, false).expect("encode").get_ids().to_vec();
                let out = Command::new(&bin)
                    .arg("-m")
                    .arg(&path)
                    .arg("-p")
                    .arg(text)
                    .env("LD_LIBRARY_PATH", &libdir)
                    .output()
                    .expect("spawn llama-tokenize");
                assert!(
                    out.status.success(),
                    "[{key}] llama-tokenize 失败({text:?}):\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let stdout = String::from_utf8_lossy(&out.stdout);
                let theirs: Vec<u32> = stdout
                    .lines()
                    .filter_map(|l| l.trim().split("->").next())
                    .filter_map(|id| id.trim().parse::<u32>().ok())
                    .collect();
                assert_eq!(
                    mine, theirs,
                    "[{key}] tokenize 不一致,文本 {text:?}\n我方: {mine:?}\nllama: {theirs:?}"
                );
                // 往返:decode 回原文(byte-level BPE 应无损)
                assert_eq!(v.inner.decode(&mine, false).expect("decode"), *text, "[{key}] 往返 {text:?}");
            }
            eprintln!("[{key}] tokenize 对拍绿({} 例)", cases.len());
        }
    }

    /// 词表事实自证:token_type 直方图 / 特殊 id 与 token 原文一致 / eos 名可读
    #[test]
    fn vocab_facts_are_consistent() {
        for key in ["gguf:4b", "gguf:27b"] {
            let Some(path) = res(key) else {
                eprintln!("skip: 资源清单缺 \"{key}\"");
                continue;
            };
            let g = Gguf::open(&path).expect("open");
            let v = build_vocab(&g).expect("build_vocab");
            assert_eq!(v.n_tokens, 248320, "[{key}] 词表大小");
            assert_eq!(v.model, VocabModel::Bpe, "[{key}] Qwen 系 = gpt2/BPE");
            assert_eq!(v.pre.as_deref(), Some("qwen35"));
            assert!(!v.add_bos, "[{key}] add_bos_token=false");
            assert!(v.template.as_ref().is_some_and(|t| t.contains("add_generation_prompt")), "[{key}] 内嵌模板");
            // eos = ChatML `<|im_end|>`(家族事实;id 由文件自带)
            let eos = v.specials.eos.expect("qwen35 必有 eos_token_id");
            assert_eq!(v.token_str(eos).unwrap(), "<|im_end|>", "[{key}] eos 词面");
            assert_eq!(v.token_str(248046).unwrap(), "<|im_end|>");
            assert_eq!(v.token_str(248044).unwrap(), "<|endoftext|>");
            // eos 判定链:encode 后 token id 落在 eos 集
            let ids = v.inner.encode("hi<|im_end|>", false).unwrap().get_ids().to_vec();
            assert!(ids.contains(&248046), "[{key}] <|im_end|> 应编码为 control token: {ids:?}");
            assert!(v.eos_ids().contains(&248046));
        }
    }

    /// 响亮拒:未知 `pre` / 非 BPE-Unigram 模型 / 越界 tokenid
    #[test]
    fn rejects_unknown_vocab_shapes() {
        assert!(pre_regex(Some("qwen35")).is_ok());
        assert!(pre_regex(Some("qwen2")).is_ok());
        assert!(pre_regex(None).is_ok());
        assert!(pre_regex(Some("no-such-pre")).is_err(), "未知 pre 必须拒");
        assert_eq!(VocabModel::parse("gpt2"), VocabModel::Bpe);
        assert_eq!(VocabModel::parse("llama"), VocabModel::Unigram);
        assert_eq!(VocabModel::parse("rwkv"), VocabModel::Other("rwkv".into()));
    }
}
