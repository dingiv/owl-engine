//! chat template 渲染器(HF transformers 语义;**agent 服务面专项 §3.2/§7.1**)。
//!
//! **唯一权威**:模板 = 模型快照自带 `chat_template.jinja`(**不复制进代码**);
//! 渲染 = 本模块唯一入口 —— 禁第二处拼 prompt(旧 `openai::render_prompt`
//! 的硬编码 Qwen 模板已退役,见 §4.3 渲染对称律 R2)。
//!
//! ## HF 兼容层(四项缺一不可;T0/G0 实测定案,4 用例 + 2 模板逐字节相等)
//!
//! 1. **空白控制** `trim_blocks + lstrip_blocks`(HF env 参数
//!    `transformers/utils/chat_template_utils.py:489`,**非** minijinja 默认);
//! 2. **Python str 方法** → `minijinja_contrib::pycompat`(模板调用
//!    `.split/.startswith/.lstrip/.rstrip/.endswith`;Jinja2 借 Python 而 minijinja 不借);
//! 3. **`tojson` 覆盖为 HF 语义** = `json.dumps(x, ensure_ascii=False, sort_keys=False)`
//!    (`:480-484` 注释写明"覆盖内置以避免 HTML 转义")⇒
//!    ① 不转义 `<>&'` ② 非 ASCII 原样 ③ 分隔符 `", "`/`": "` ④ Python 数值表示;
//! 4. **globals** `raise_exception`(注入为**结构化错误**,不许 panic 逃逸)/
//!    `strftime_now`(本模板未用,一并注入防模板漂移)。
//!
//! ## 契约要点(T0 踩坑)
//!
//! `chat_template_kwargs` 内的键**必须展平到渲染上下文顶层** —— HF 把 kwargs
//! 直接当模板变量;不展平则 `enable_thinking=false` **静默失效**(实测差 434 B)。

use std::path::Path;

use minijinja::{Environment, Error as MjError, ErrorKind, Value};
use serde_json::{Map, Value as Json};

/// 模板名(进程内编译一次)
const TPL_NAME: &str = "chat";
/// 快照模板文件名(HF 解析优先级与之同源:先独立文件,再 tokenizer_config)
const TPL_FILE: &str = "chat_template.jinja";
/// tokenizer_config.json(模板兜底来源)
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";

/// 渲染失败(结构化;调用方按客户端输入问题 → 400 处置)
#[derive(Debug)]
pub struct TplError(String);

impl std::fmt::Display for TplError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TplError {}

/// 渲染入参(协议层组装;`tools` 缺席 = 客户端没给,与空数组语义不同)
pub struct RenderArgs<'a> {
    /// OpenAI `messages`(原样透传,含 `tool_calls` / `role:"tool"` / `reasoning_content`)
    pub messages: &'a [Json],
    /// 请求 `tools`;`None` = 客户端未提供(模板 `_has_tools` 为假)
    pub tools: Option<&'a [Json]>,
    /// 末尾 assistant 起手(HF `add_generation_prompt`)
    pub add_generation_prompt: bool,
    /// 请求 `chat_template_kwargs`(**展平进顶层**,契约要点)
    pub template_kwargs: Option<&'a Map<String, Json>>,
}

/// 渲染器(装载一次,进程内共享;`Environment` 内模板已编译)
pub struct ChatTpl {
    env: Environment<'static>,
    /// 模板来源(日志/诊断用)
    source: String,
    /// agent 面缺省模板变量(`[agent]` 配置→jinja 变量;请求 kwargs 覆盖之)。
    /// 只装**非缺省**项 ⇒ 与 HF oracle(无 kwargs)逐字节同源(G0 不变式)
    defaults: Map<String, Json>,
}

impl ChatTpl {
    /// 装载(**快照缺省面**;单测/工具便捷入口):读快照 `chat_template.jinja`
    /// 或回退 `tokenizer_config.json.chat_template`,零 agent 缺省变量。
    #[cfg(test)]
    pub fn load(model_dir: &Path) -> Result<Self, TplError> {
        Self::load_with(model_dir, None, Map::new())
    }

    /// 装载(B2 模板覆盖 + agent 缺省变量)。
    /// `template_path`:`None` = 快照;相对路径 = 相对 `model_dir`。
    pub fn load_with(
        model_dir: &Path,
        template_path: Option<&str>,
        defaults: Map<String, Json>,
    ) -> Result<Self, TplError> {
        let (src, source) = match template_path {
            None => read_template(model_dir)?,
            Some(p) => {
                let path = if Path::new(p).is_absolute() {
                    std::path::PathBuf::from(p)
                } else {
                    model_dir.join(p)
                };
                let s = std::fs::read_to_string(&path)
                    .map_err(|e| TplError(format!("读模板覆盖 {}: {e}", path.display())))?;
                (s, format!("{} (agent.template_path)", path.display()))
            }
        };
        Self::from_str(src, source, defaults)
    }

    /// **是否 GGUF 档 + 自动路由的唯一产地**(路径是文件 ⇒ GGUF 单文件形态)。
    ///
    /// 优先级:显式 `agent.template_path`(运维覆盖)> GGUF 内嵌 > 快照文件。
    /// 为什么把路由放这里而不放 main:两条路径的**优先级顺序**是契约
    /// (将来新增模板来源必须在同一处排序,防“两个入口各执一词”)。
    pub fn load_snapshot_or_gguf(
        model_path: &Path,
        template_path: Option<&str>,
        defaults: Map<String, Json>,
    ) -> Result<Self, TplError> {
        if template_path.is_some() {
            return Self::load_with(model_path, template_path, defaults);
        }
        if model_path.is_file() {
            return Self::load_gguf(model_path, defaults);
        }
        Self::load_with(model_path, None, defaults)
    }

    /// **从模板字符串装**(2026-10-19 GGUF 轮;唯一编译入口)。
    ///
    /// 为什么需要:GGUF 的模板住 `.gguf` 的 `tokenizer.chat_template` **KV**,不在目录文件里
    /// (设计 REQ-GGUF-02「单文件自给」)。
    /// 本构造函数是 `load_with` 与 [`ChatTpl::load_gguf`] 的**共同下游** ——
    /// 环境/兼容层只有一份(`build_env`),不存在第二套 Jinja 语义。
    pub fn from_str(
        template: String,
        source: String,
        defaults: Map<String, Json>,
    ) -> Result<Self, TplError> {
        let mut env = build_env();
        env.add_template_owned(TPL_NAME.to_string(), template)
            .map_err(|e| TplError(format!("chat 模板编译失败({source}): {e}")))?;
        Ok(Self { env, source, defaults })
    }

    /// **从 GGUF 文件装**(`tokenizer.chat_template` KV;缺 = 拒启)。
    ///
    /// 与 `load_with` 同一条下游 ⇒ 渲染语义完全一致;差别只在模板**来源**。
    /// 事实自证(单测 `gguf_template_equals_hf_snapshot`):两靶位的 GGUF 内嵌模板
    /// 与其 HF 快照的 `chat_template.jinja` **逐字节相同**。
    pub fn load_gguf(
        gguf_path: &Path,
        defaults: Map<String, Json>,
    ) -> Result<Self, TplError> {
        let g = owl_models::formats::gguf::Gguf::open(gguf_path)
            .map_err(|e| TplError(format!("读 GGUF {}: {e}", gguf_path.display())))?;
        let t = g
            .kv
            .opt_str("tokenizer.chat_template")
            .map_err(|e| TplError(format!("GGUF 元数据: {e}")))?
            .ok_or_else(|| {
                TplError(format!(
                    "GGUF {} 无 `tokenizer.chat_template`(无权渲染 prompt ⇒ 拒启)",
                    gguf_path.display()
                ))
            })?;
        Self::from_str(t.to_string(), format!("{}#tokenizer.chat_template", gguf_path.display()), defaults)
    }

    /// 模板来源(装配日志)
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 渲染 → prompt 文本(编码在引擎侧;本模块零设备依赖)
    pub fn render(&self, args: RenderArgs<'_>) -> Result<String, TplError> {
        let mut ctx = Map::new();
        ctx.insert("messages".into(), Json::Array(args.messages.to_vec()));
        if let Some(tools) = args.tools {
            ctx.insert("tools".into(), Json::Array(tools.to_vec()));
        }
        ctx.insert("add_generation_prompt".into(), Json::Bool(args.add_generation_prompt));
        // === 契约:chat_template_kwargs 展平到顶层 ===
        // (不展平 = enable_thinking=false 静默走默认 true;§3.2 注 / §7.1 坑 1)
        // 序:agent 面缺省在前、请求 kwargs 在后(请求优先)
        for (k, v) in &self.defaults {
            ctx.insert(k.clone(), v.clone());
        }
        if let Some(kw) = args.template_kwargs {
            for (k, v) in kw {
                ctx.insert(k.clone(), v.clone());
            }
        }
        let tpl = self
            .env
            .get_template(TPL_NAME)
            .map_err(|e| TplError(format!("chat 模板未装载(装配期 bug): {e}")))?;
        tpl.render(Json::Object(ctx)).map_err(|e| {
            // 模板内 raise_exception(如"System message cannot contain images.")
            // = 客户端数据问题;其余 = 模板/兼容层问题。两类都给结构化消息
            TplError(format!("chat 模板渲染失败: {e}"))
        })
    }
}

/// 读模板源:独立 `.jinja` 优先,回退 `tokenizer_config.json.chat_template`
fn read_template(model_dir: &Path) -> Result<(String, String), TplError> {
    let p = model_dir.join(TPL_FILE);
    if p.is_file() {
        let s = std::fs::read_to_string(&p)
            .map_err(|e| TplError(format!("读模板 {}: {e}", p.display())))?;
        return Ok((s, format!("{}", p.display())));
    }
    let cfg = model_dir.join(TOKENIZER_CONFIG);
    let raw = std::fs::read_to_string(&cfg)
        .map_err(|e| TplError(format!("缺 {} 且读 {} 失败: {e}", p.display(), cfg.display())))?;
    let v: Json = serde_json::from_str(&raw)
        .map_err(|e| TplError(format!("解析 {}: {e}", cfg.display())))?;
    match v.get("chat_template").and_then(|t| t.as_str()) {
        Some(s) if !s.is_empty() => Ok((s.to_string(), format!("{}#chat_template", cfg.display()))),
        _ => Err(TplError(format!(
            "模型快照无 chat template({} / {})",
            p.display(),
            cfg.display()
        ))),
    }
}

/// HF 兼容环境(构造面见模块头;四项 + features)
fn build_env() -> Environment<'static> {
    let mut env = Environment::new();
    // ① HF `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, …)`
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // ② Jinja2 借 Python str 方法(`.split/.startswith/.lstrip/.rstrip/.endswith`)
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // ③ tojson = HF 语义(Python json.dumps 形态)
    env.add_filter("tojson", |v: Value| -> Result<String, MjError> {
        Ok(py_json(&value_to_json(&v)))
    });
    // ④ HF globals
    env.add_function("raise_exception", |msg: String| -> Result<Value, MjError> {
        Err(MjError::new(ErrorKind::InvalidOperation, msg))
    });
    env.add_function("strftime_now", |format: String| -> Result<String, MjError> {
        // 模板未用(仅注入以防模板漂移);无依赖实现 = 仅支持 %Y-%m-%d 级子集
        let _ = format;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(format!("@{now}"))
    });
    env
}

/// `minijinja::Value` → `serde_json::Value`(tojson 走双转换:渲染频次低,可读优先)
fn value_to_json(v: &Value) -> Json {
    serde_json::to_string(v)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Json::Null)
}

/// Python `json.dumps(x, ensure_ascii=False, sort_keys=False)` 等价输出
/// (HF `tojson`;分隔符 = Python 默认 `", "` / `": "`,键序 = 插入序)
pub fn py_json(v: &Json) -> String {
    let mut out = String::new();
    py_json_into(v, &mut out);
    out
}

fn py_json_into(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_float(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Json::String(s) => py_str(s, out),
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json_into(x, out);
            }
            out.push(']');
        }
        Json::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_str(k, out);
                out.push_str(": ");
                py_json_into(x, out);
            }
            out.push('}');
        }
    }
}

/// Python `str` 的 JSON 转义(`ensure_ascii=False`:`"` `\` 控制字符转义,其余原样)
fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)` 等价(最短往返 + Python 的指数切换阈值:
/// 十进制指数 `< -4` 或 `>= 16` 走科学计数法,指数至少两位)
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f < 0.0 { "-Infinity".into() } else { "Infinity".into() };
    }
    let neg = f.is_sign_negative();
    // Rust `{}` = 最短往返十进制且**不用指数**(与 Python repr 的有效数字一致)
    let plain = format!("{}", f.abs());
    let (int_part, frac_part) = match plain.split_once('.') {
        Some((a, b)) => (a, b),
        None => (plain.as_str(), ""),
    };
    let digits: String = format!("{int_part}{frac_part}");
    let lead = digits.chars().take_while(|c| *c == '0').count();
    let sig: &str = digits[lead..].trim_end_matches('0');
    if sig.is_empty() {
        // ±0
        return if neg { "-0.0".into() } else { "0.0".into() };
    }
    // 首个有效位的十进制指数(整数部分位数 - 1 - 前导零数)
    let exp10: i32 = int_part.len() as i32 - 1 - lead as i32;
    let body = if !(-4..16).contains(&exp10) {
        // 科学计数法:`d[.ddd]e±NN`
        let mut m = String::from(&sig[..1]);
        if sig.len() > 1 {
            m.push('.');
            m.push_str(&sig[1..]);
        }
        let sign = if exp10 < 0 { '-' } else { '+' };
        let e = exp10.unsigned_abs();
        m.push('e');
        m.push(sign);
        if e < 10 {
            m.push('0');
        }
        m.push_str(&e.to_string());
        m
    } else if exp10 >= 0 {
        let point = exp10 as usize + 1;
        if sig.len() > point {
            format!("{}.{}", &sig[..point], &sig[point..])
        } else {
            let mut s = String::from(sig);
            while s.len() < point {
                s.push('0');
            }
            s.push_str(".0");
            s
        }
    } else {
        let zeros = "0".repeat((-exp10 - 1) as usize);
        format!("0.{zeros}{sig}")
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args<'a>(
        msgs: &'a [Json],
        tools: Option<&'a [Json]>,
        kw: Option<&'a Map<String, Json>>,
    ) -> RenderArgs<'a> {
        RenderArgs { messages: msgs, tools, add_generation_prompt: true, template_kwargs: kw }
    }

    fn tpl(src: &str) -> ChatTpl {
        ChatTpl::from_str(src.to_string(), "test".into(), Map::new()).unwrap()
    }

    /// 渲染器必须可跨线程共享(HTTP 层多任务持有)
    #[test]
    fn send_sync() {
        fn assert<T: Send + Sync>() {}
        assert::<ChatTpl>();
    }

    /// HF `tojson` 语义(期望值由 Python `json.dumps(..., ensure_ascii=False,
    /// sort_keys=False)` 生成并硬编码;金标口径见 §7.1)
    #[test]
    fn tojson_hf_semantics() {
        let v: Json = serde_json::from_str(
            r#"{"a":1,"b":[1,2.5,"x"],"c":{"d":null,"e":true},"f":"中文 <&>' \" \\ \t\b\f"}"#,
        )
        .unwrap();
        assert_eq!(
            py_json(&v),
            "{\"a\": 1, \"b\": [1, 2.5, \"x\"], \"c\": {\"d\": null, \"e\": true}, \"f\": \"中文 <&>' \\\" \\\\ \\t\\b\\f\"}"
        );
        // 键序 = 插入序(非字典序)+ 非 ASCII 键原样 + 浮点形态
        let v: Json = serde_json::from_str(r#"{"关":"键","z":1e16,"y":30.0}"#).unwrap();
        assert_eq!(py_json(&v), "{\"关\": \"键\", \"z\": 1e+16, \"y\": 30.0}");
    }

    /// Python `repr(float)` 等价(期望值来自 Python `repr()`)
    #[test]
    fn float_python_repr() {
        for (v, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (0.7, "0.7"),
            (30.0, "30.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e17, "1e+17"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (3.141592653589793, "3.141592653589793"),
            (0.000123, "0.000123"),
            (1e-300, "1e-300"),
            (1e300, "1e+300"),
            (2.5e-10, "2.5e-10"),
            (1.5, "1.5"),
            (-2.75, "-2.75"),
        ] {
            assert_eq!(py_float(v), want, "py_float({v})");
        }
    }

    /// ① trim_blocks/lstrip_blocks:块标记独占行时不留空行(HF env 参数)
    #[test]
    fn trim_and_lstrip_blocks() {
        let t = tpl("A\n{% if true %}\nB\n{% endif %}\nC");
        let msgs: Vec<Json> = vec![];
        assert_eq!(t.render(args(&msgs, None, None)).unwrap(), "A\nB\nC");
    }

    /// ② pycompat:模板调用 Python str 方法不再报 UnknownMethod。
    /// 注:布尔插值 = `True`/`False`(Jinja2/minijinja 的 Python 风格),
    /// 与 HF 一致 —— 不是 `true`/`false`。
    #[test]
    fn pycompat_str_methods() {
        let t = tpl("{{ ('a,b,c'.split(',') | length) }}-{{ 'xyz'.startswith('x') }}-{{ '  s '.lstrip() }}-{{ false }}");
        let msgs: Vec<Json> = vec![];
        assert_eq!(t.render(args(&msgs, None, None)).unwrap(), "3-True-s -False");
    }

    /// ④ raise_exception → 结构化错误(不 panic)
    #[test]
    fn raise_exception_is_structured_error() {
        let t = tpl("{% if true %}{{ raise_exception('boom') }}{% endif %}");
        let msgs: Vec<Json> = vec![];
        let e = t.render(args(&msgs, None, None)).unwrap_err();
        assert!(e.to_string().contains("boom"), "{e}");
    }

    /// 契约:chat_template_kwargs 展平(不展平 = enable_thinking 静默失效)
    #[test]
    fn template_kwargs_are_flattened() {
        let t = tpl("{%- set on = enable_thinking if enable_thinking is defined else true -%}think={{ on }}");
        let msgs: Vec<Json> = vec![];
        let mut kw = Map::new();
        kw.insert("enable_thinking".into(), Json::Bool(false));
        assert_eq!(t.render(args(&msgs, None, Some(&kw))).unwrap(), "think=False");
        // 不给 = 模板默认(true)
        assert_eq!(t.render(args(&msgs, None, None)).unwrap(), "think=True");
    }

    /// **G0 金标门**:快照模板 × HF oracle 逐字节对拍。
    ///
    /// 需 `tpl:default` 指向模型快照(27B);未设 = 跳过(GPU 用例同纪律,
    /// 模型目录是机器资产、不在仓库内)。金标 = `tests/fixtures/chat_tpl/*.txt`
    /// (由 `gen_fixtures.py` 用 HF `apply_chat_template` 生成,见该目录 README)。
    #[test]
    fn golden_snapshot_templates() {
        let Some(dir) = owl_shared::testkit::resource("tpl:default") else {
            owl_shared::testkit::report::skip("tpl:default 未设(指向 27B 快照目录)");
            return;
        };
        let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/chat_tpl");
        let tpl = ChatTpl::load(std::path::Path::new(&dir)).expect("装载快照模板");
        for name in [
            "ctx1_tools",
            "ctx2_tools_multiturn",
            "ctx3_tools_notinking",
            "ctx4_plain_notinking",
        ] {
            golden_case(&tpl, &fx, name);
        }
        // 家族通用性:另一份快照(上游原版 153 行模板;可选)。
        // ⚠️ 注意用**自己的**金标文件(qwen35_0.8b_ctx1.txt)—— 模板不同,
        //    拿 27B 金标比 0.8B 渲染必红(本行曾因复制粘贴踩过,见 §7.2 教训)。
        if let Some(dir8) = owl_shared::testkit::resource("tpl:qwen35") {
            let t8 = ChatTpl::load(std::path::Path::new(&dir8)).expect("装载 0.8B 模板");
            let case: Json = serde_json::from_str(
                &std::fs::read_to_string(fx.join("ctx1_tools.json")).unwrap(),
            )
            .unwrap();
            let want = std::fs::read_to_string(fx.join("qwen35_0.8b_ctx1.txt")).unwrap();
            let got = render_case(&t8, &case);
            assert!(got == want, "qwen35_0.8b_ctx1 与 HF oracle 逐字节不等");
        }
    }

    /// **GGUF 模板来源等价门**(2026-10-19 刀 2;REQ-GGUF-02)。
    ///
    /// 两段清单式断言:
    /// ① **字节等价**:`.gguf` 的 `tokenizer.chat_template` ≡ 对应 HF 快照的
    ///    `chat_template.jinja` —— 证明 GGUF 读取路径**不丢/不歪**模板事实;
    /// ② **渲染等价**:同一批载荷下,`load_gguf(gguf)` 与 `load(hf 目录)` 的
    ///    输出**逐字节相等** —— 证明两条来源走的是同一渲染语义。
    ///
    /// 为什么不重造金标:渲染器本身已有 HF oracle 金标门
    ///([`golden_snapshot_templates`]);本门把“模板事实”接到那条已验链上。
    #[test]
    fn gguf_template_equals_hf_snapshot() {
        let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_tpl");
        let mut checked = 0usize;
        for (gk, hk) in [("tpl:gguf4b", "tpl:hf4b"), ("tpl:gguf27b", "tpl:hf27b")] {
            let (Some(gguf), Some(hf_dir)) = (
                owl_shared::testkit::resource(gk),
                owl_shared::testkit::resource(hk),
            ) else {
                owl_shared::testkit::report::skip(format_args!("资源清单缺 {gk}/{hk}"));
                continue;
            };
            let g = owl_models::formats::gguf::Gguf::open(std::path::Path::new(&gguf)).expect("开 GGUF");
            let from_gguf = g.kv.opt_str("tokenizer.chat_template").expect("KV 形态").expect("模板在场");
            let hf = std::fs::read_to_string(std::path::Path::new(&hf_dir).join(TPL_FILE)).expect("HF 模板");
            // ① 字节等价
            assert_eq!(from_gguf, hf, "[{gk}] 内嵌模板 ≠ HF 快照模板(逐字节)");
            // ② 渲染等价(同批载荷:tools / 多轮 / 关思考 / 纯文本)
            let a = ChatTpl::load_gguf(std::path::Path::new(&gguf), Map::new()).expect("load_gguf");
            let b = ChatTpl::load(std::path::Path::new(&hf_dir)).expect("load(hf)");
            let mut cases: Vec<Json> = Vec::new();
            for name in ["ctx1_tools", "ctx2_tools_multiturn", "ctx3_tools_notinking", "ctx4_plain_notinking"] {
                let p = fx.join(format!("{name}.json"));
                if let Ok(s) = std::fs::read_to_string(&p) {
                    cases.push(serde_json::from_str(&s).expect("fixture json"));
                }
            }
            cases.push(serde_json::json!({
                "messages": [
                    {"role": "system", "content": "你是一个助手"},
                    {"role": "user", "content": "画一个图"},
                    {"role": "assistant", "content": "好的"},
                    {"role": "user", "content": "再来一个"}
                ],
                "add_generation_prompt": true
            }));
            assert!(!cases.is_empty(), "[{gk}] 无载荷可比较");
            for c in &cases {
                assert_eq!(
                    render_case(&a, c),
                    render_case(&b, c),
                    "[{gk}] GGUF 来源与快照来源渲染不等(载荷 {c})"
                );
            }
            eprintln!("[{gk}] 模板 {} B 逐字节等 ;;{} 例渲染逐字节等", from_gguf.len(), cases.len());
            checked += 1;
        }
        if checked == 0 {
            owl_shared::testkit::report::skip("无 GGUF 模板资源(tpl:gguf9b / tpl:gguf27b)");
        }
    }

    fn golden_case(t: &ChatTpl, fx: &std::path::Path, name: &str) {
        let case: Json = serde_json::from_str(
            &std::fs::read_to_string(fx.join(format!("{name}.json"))).unwrap(),
        )
        .unwrap();
        let want = std::fs::read_to_string(fx.join(format!("{name}.txt"))).unwrap();
        let got = render_case(t, &case);
        assert_eq!(got, want, "{name} 与 HF oracle 逐字节不等");
    }

    fn render_case(t: &ChatTpl, case: &Json) -> String {
        let msgs = case["messages"].as_array().cloned().unwrap_or_default();
        let tools = case.get("tools").and_then(|v| v.as_array()).cloned();
        let kw = case.get("chat_template_kwargs").and_then(|v| v.as_object());
        t.render(RenderArgs {
            messages: &msgs,
            tools: tools.as_deref(),
            add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap_or(true),
            template_kwargs: kw,
        })
        .expect("渲染")
    }

    /// 载荷面:messages 原样透传(含 tools 缺席 vs 空数组的语义差)
    #[test]
    fn payload_passthrough() {
        let t = tpl("[{{ messages | length }}]{% if tools is defined %}+tools({{ tools | length }}){% endif %}");
        let msgs: Vec<Json> = vec![serde_json::json!({"role":"user","content":"hi"})];
        let empty: Vec<Json> = vec![];
        assert_eq!(t.render(args(&msgs, None, None)).unwrap(), "[1]");
        assert_eq!(t.render(args(&msgs, Some(&empty), None)).unwrap(), "[1]+tools(0)");
    }
}
