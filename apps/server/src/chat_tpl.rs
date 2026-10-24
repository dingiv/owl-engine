//! chat template 渲染器（HF transformers 语义；agent 服务面 §3.2/§7.1）。
//!
//! **唯一权威**：模板 = 模型快照自带 `chat_template.jinja`（不复制进代码）；
//! 渲染引擎 = [`hf_chat_template`]（底层 minijinja + transformers 兼容层）。
//! 本模块只保留 owl 的**模板来源**（快照 / GGUF / 覆盖）与 **agent 缺省变量**接口。
//!
//! 2026-10-24：移除本模块自带的 minijinja Env / `py_json` / `py_float` 兼容实现，
//! 模板语义收敛到 `hf-chat-template`，避免两套 HF 兼容层漂移。
//!
//! 契约要点：`chat_template_kwargs` 必须展平到渲染上下文顶层 —— HF 把 kwargs
//! 直接当模板变量；不展平则 `enable_thinking=false` **静默失效**。

use std::path::Path;

use hf_chat_template::{ChatTemplate, TokenField, TokenizerConfig};
use serde_json::{Map, Value as Json};

/// 快照模板文件名（HF 解析优先级与之同源：先独立文件，再 tokenizer_config）
const TPL_FILE: &str = "chat_template.jinja";
/// tokenizer_config.json（模板兜底来源）
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";

/// 渲染失败（结构化；调用方按客户端输入问题 → 400 处置）
#[derive(Debug)]
pub struct TplError(String);

impl std::fmt::Display for TplError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TplError {}

/// 渲染入参（协议层组装；`tools` 缺席 = 客户端没给，与空数组语义不同）
pub struct RenderArgs<'a> {
    /// OpenAI `messages`（原样透传，含 `tool_calls` / `role:"tool"` / `reasoning_content`）
    pub messages: &'a [Json],
    /// 请求 `tools`；`None` = 客户端未提供（模板 `_has_tools` 为假）
    pub tools: Option<&'a [Json]>,
    /// 末尾 assistant 起手（HF `add_generation_prompt`）
    pub add_generation_prompt: bool,
    /// 请求 `chat_template_kwargs`（**展平进顶层**，契约要点）
    pub template_kwargs: Option<&'a Map<String, Json>>,
}

/// 渲染器（装载一次，进程内共享；`hf-chat-template` 内模板已编译）
pub struct ChatTpl {
    template: ChatTemplate,
    /// 模板来源（日志/诊断用）
    source: String,
    /// agent 面缺省模板变量（`[agent]` 配置→jinja 变量；请求 kwargs 覆盖之）。
    /// 只装**非缺省**项 ⇒ 与 HF oracle（无 kwargs）逐字节同源（G0 不变式）
    defaults: Map<String, Json>,
    /// 快照 `tokenizer_config.json` 的 `bos_token`/`eos_token`/…（HF 同款注入；
    /// 模板不引用时零影响）。GGUF 单文件路径没有该配置 ⇒ 空。
    special_tokens: Map<String, Json>,
}

impl ChatTpl {
    /// 装载（**快照缺省面**；单测/工具便捷入口）：读快照 `chat_template.jinja`
    /// 或回退 `tokenizer_config.json.chat_template`，零 agent 缺省变量。
    #[cfg(test)]
    pub fn load(model_dir: &Path) -> Result<Self, TplError> {
        Self::load_with(model_dir, None, Map::new())
    }

    /// 装载（B2 模板覆盖 + agent 缺省变量）。
    /// `template_path`：`None` = 快照；相对路径 = 相对 `model_dir`。
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
                    .map_err(|e| TplError(format!("读模板覆盖 {}：{e}", path.display())))?;
                (s, format!("{} (agent.template_path)", path.display()))
            }
        };
        let special_tokens = read_special_tokens(model_dir)?;
        Self::from_str_with_specials(src, source, defaults, special_tokens)
    }

    /// **是否 GGUF 档 + 自动路由的唯一产地**（路径是文件 ⇒ GGUF 单文件形态）。
    ///
    /// 优先级：显式 `agent.template_path`（运维覆盖）> GGUF 内嵌 > 快照文件。
    /// 为什么把路由放这里而不放 main：两条路径的**优先级顺序**是契约
    /// （将来新增模板来源必须在同一处排序，防“两个入口各执一词”）。
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

    /// **从模板字符串装**（唯一编译入口；GGUF 内嵌路径也走这里）。
    ///
    /// 为什么需要：GGUF 的模板住 `.gguf` 的 `tokenizer.chat_template` **KV**，
    /// 不在目录文件里（设计 REQ-GGUF-02「单文件自给」）。
    pub fn from_str(
        template: String,
        source: String,
        defaults: Map<String, Json>,
    ) -> Result<Self, TplError> {
        Self::from_str_with_specials(template, source, defaults, Map::new())
    }

    fn from_str_with_specials(
        template: String,
        source: String,
        defaults: Map<String, Json>,
        special_tokens: Map<String, Json>,
    ) -> Result<Self, TplError> {
        let template = ChatTemplate::from_str(&template)
            .map_err(|e| TplError(format!("chat 模板编译失败({source}): {e}")))?;
        Ok(Self { template, source, defaults, special_tokens })
    }

    /// **从 GGUF 文件装**（`tokenizer.chat_template` KV；缺 = 拒启）。
    ///
    /// 与 `load_with` 同一条下游 ⇒ 渲染语义完全一致；差别只在模板**来源**。
    /// 事实自证（单测 `gguf_template_equals_hf_snapshot`）：两靶位的 GGUF 内嵌模板
    /// 与其 HF 快照的 `chat_template.jinja` **逐字节相同**。
    pub fn load_gguf(
        gguf_path: &Path,
        defaults: Map<String, Json>,
    ) -> Result<Self, TplError> {
        let g = owl_models::formats::gguf::Gguf::open(gguf_path)
            .map_err(|e| TplError(format!("读 GGUF {}：{e}", gguf_path.display())))?;
        let t = g
            .kv
            .opt_str("tokenizer.chat_template")
            .map_err(|e| TplError(format!("GGUF 元数据：{e}")))?
            .ok_or_else(|| {
                TplError(format!(
                    "GGUF {} 无 `tokenizer.chat_template`（无权渲染 prompt ⇒ 拒启）",
                    gguf_path.display()
                ))
            })?;
        Self::from_str(
            t.to_string(),
            format!("{}#tokenizer.chat_template", gguf_path.display()),
            defaults,
        )
    }

    /// 模板来源（装配日志）
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 渲染 → prompt 文本（编码在引擎侧；本模块零设备依赖）
    pub fn render(&self, args: RenderArgs<'_>) -> Result<String, TplError> {
        let mut ctx = Map::new();
        ctx.insert("messages".into(), Json::Array(args.messages.to_vec()));
        if let Some(tools) = args.tools {
            ctx.insert("tools".into(), Json::Array(tools.to_vec()));
        }
        ctx.insert("add_generation_prompt".into(), Json::Bool(args.add_generation_prompt));
        // HF 同款顺序：tokenizer special tokens → agent 缺省 → 请求 kwargs。
        // 模板不引用这些变量时零影响；GGUF 单文件路径为空。
        for (k, v) in &self.special_tokens {
            ctx.insert(k.clone(), v.clone());
        }
        // 契约：chat_template_kwargs 展平到顶层（不展平 = enable_thinking 静默失效）
        for (k, v) in &self.defaults {
            ctx.insert(k.clone(), v.clone());
        }
        if let Some(kw) = args.template_kwargs {
            for (k, v) in kw {
                ctx.insert(k.clone(), v.clone());
            }
        }
        self.template
            .render_context(&Json::Object(ctx))
            .map_err(|e| TplError(format!("chat 模板渲染失败: {e}")))
    }
}

/// 读 `tokenizer_config.json` 的特殊 token（标准 `hf-chat-template` 视图）。
/// 缺文件 = 空；文件在但解析失败 = 响亮拒（配置面事实不能静默丢）。
fn read_special_tokens(model_dir: &Path) -> Result<Map<String, Json>, TplError> {
    let path = model_dir.join(TOKENIZER_CONFIG);
    if !path.is_file() {
        return Ok(Map::new());
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| TplError(format!("读 tokenizer 配置 {}：{e}", path.display())))?;
    let cfg: TokenizerConfig = serde_json::from_str(&raw)
        .map_err(|e| TplError(format!("解析 tokenizer 配置 {}：{e}", path.display())))?;
    let mut out = Map::new();
    for (name, field) in [
        ("bos_token", &cfg.bos_token),
        ("eos_token", &cfg.eos_token),
        ("pad_token", &cfg.pad_token),
        ("unk_token", &cfg.unk_token),
    ] {
        if let Some(v) = field.as_ref().and_then(TokenField::as_str) {
            out.insert(name.to_string(), Json::String(v.to_string()));
        }
    }
    Ok(out)
}

/// 读模板源：独立 `.jinja` 优先，回退 `tokenizer_config.json.chat_template`
fn read_template(model_dir: &Path) -> Result<(String, String), TplError> {
    let p = model_dir.join(TPL_FILE);
    if p.is_file() {
        let s = std::fs::read_to_string(&p)
            .map_err(|e| TplError(format!("读模板 {}：{e}", p.display())))?;
        return Ok((s, format!("{}", p.display())));
    }
    let cfg = model_dir.join(TOKENIZER_CONFIG);
    let raw = std::fs::read_to_string(&cfg)
        .map_err(|e| TplError(format!("缺 {} 且读 {} 失败：{e}", p.display(), cfg.display())))?;
    let v: Json = serde_json::from_str(&raw)
        .map_err(|e| TplError(format!("解析 {}：{e}", cfg.display())))?;
    match v.get("chat_template").and_then(|t| t.as_str()) {
        Some(s) if !s.is_empty() => Ok((s.to_string(), format!("{}#chat_template", cfg.display()))),
        _ => Err(TplError(format!(
            "模型快照无 chat template（{} / {}）",
            p.display(),
            cfg.display()
        ))),
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

    /// HF `tojson` 语义（由 `hf-chat-template` 提供；此处只验接缝）
    #[test]
    fn tojson_hf_semantics() {
        let t = tpl("{{ v | tojson }}");
        let msgs: Vec<Json> = vec![];
        let mut kw = Map::new();
        let v: Json = serde_json::from_str(
            r#"{"a":1,"b":[1,2.5,"x"],"c":{"d":null,"e":true},"f":"中文 <&>' \" \\ \t\b\f"}"#,
        )
        .unwrap();
        kw.insert("v".into(), v);
        assert_eq!(
            t.render(args(&msgs, None, Some(&kw))).unwrap(),
            "{\"a\": 1, \"b\": [1, 2.5, \"x\"], \"c\": {\"d\": null, \"e\": true}, \"f\": \"中文 <&>' \\\" \\\\ \\t\\b\\f\"}"
        );
        // 键序 = 插入序(非字典序)+ 非 ASCII 键原样 + 浮点形态
        let v: Json = serde_json::from_str(r#"{"关":"键","z":1e16,"y":30.0}"#).unwrap();
        kw.insert("v".into(), v);
        assert_eq!(
            t.render(args(&msgs, None, Some(&kw))).unwrap(),
            "{\"关\": \"键\", \"z\": 1e+16, \"y\": 30.0}"
        );
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

    /// tokenizer special tokens 注入（HF 同款；请求 kwargs 可覆盖）
    #[test]
    fn tokenizer_special_tokens_are_injected() {
        let mut special = Map::new();
        special.insert("bos_token".into(), Json::String("<s>".into()));
        let t = ChatTpl::from_str_with_specials(
            "{{ bos_token }}|{{ x }}".into(),
            "test".into(),
            Map::new(),
            special,
        )
        .unwrap();
        let msgs: Vec<Json> = vec![];
        let mut kw = Map::new();
        kw.insert("x".into(), Json::String("ok".into()));
        assert_eq!(t.render(args(&msgs, None, Some(&kw))).unwrap(), "<s>|ok");
        let mut kw = Map::new();
        kw.insert("bos_token".into(), Json::String("<override>".into()));
        assert_eq!(
            t.render(args(&msgs, None, Some(&kw))).unwrap(),
            "<override>|"
        );
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
