//! `#[owl_test]` —— 用例参数化测试属性过程宏(设计:
//! `docs/arch/owl-test-macro.md`)。
//!
//! **本 crate = 工作区过程宏统一收容地**(2026-10-15 用户裁定:包名
//! `owl-macros`,目录 `crates/macros` —— 后续新宏一律进这里,不再
//! 一宏一 crate)。当前唯一导出 = `#[owl_test]`。
//!
//! **纯样板搬运,零逻辑**:配置解析/门控逻辑全在
//! `owl_shared::testkit::case_config`;本 crate 只做签名校验 + 展开。
//!
//! **选项**(逗号分隔、顺序无关;判据与 skip 文案均在
//! `owl_shared::testkit`,本 crate 零逻辑):
//! - `gpu`:`runtime.test_device` 无值(= `OWL_TEST_DEVICE` 未设)⇒ skip;
//! - `hf_parity`:`OWL_HF_PARITY=1` 未开 ⇒ skip(与手写 `parity_enabled`/
//!   `skip_note` 同文案);可与 `gpu` 叠用(`#[owl_test(gpu, hf_parity)]`)。
//!
//! 消费方约定(与 `#[tokio::test]` 同款纪律):
//! - 加本 crate 为 **dev-dependency**;
//! - 调用点必须在 `cfg(test)` 之下(`#[cfg(test)]` 写在 `#[owl_test]`
//!   **上面** —— cfg 先求值,非测试构建不触宏名;`mod tests` 整模块
//!   cfg 亦可);
//! - 展开引用 `::owl_shared`(唯一 loader)与 `::tokio`(runtime)⇒
//!   消费 crate 需有 owl-shared 依赖与 tokio dev-dep(全仓测试已满足)。
//!
//! 展开形态(见设计文档 §2.3):原 async fn 降为 `__owl_body_<名>`(私有),
//! 同名 `#[test]` 同步 wrapper 负责 隐藏 env → 配置注入 →(可选 GPU
//! 门控)→ 手建 current_thread runtime → block_on body。
//! 函数名保持原样 ⇒ `cargo test <名>` 过滤 / `--list` / `#[ignore]` 透传不变。

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, ItemFn, Pat, ReturnType, Type};

/// 属性选项(`#[owl_test(gpu, hf_parity)]`;逗号分隔,顺序无关)。
#[derive(Clone, Copy, Default)]
struct Opts {
    /// GPU 门控:`runtime.test_device` 无值(= `OWL_TEST_DEVICE` 未设)⇒ skip
    gpu: bool,
    /// HF 对拍门控:`OWL_HF_PARITY=1` 未开 ⇒ skip(与手写 `parity_enabled`/`skip_note` 同文案)
    hf_parity: bool,
}

/// 选项解析(未知选项 = 响亮拒;逻辑零 —— 判据都在 `owl_shared::testkit`)
fn parse_opts(attr: &TokenStream) -> Result<Opts, String> {
    let mut o = Opts::default();
    let s = attr.to_string();
    if s.trim().is_empty() {
        return Ok(o);
    }
    for part in s.split(',') {
        match part.trim() {
            "gpu" => o.gpu = true,
            "hf_parity" => o.hf_parity = true,
            other => {
                return Err(format!(
                    "#[owl_test] 未知选项 {other:?}(支持:`gpu`、`hf_parity`,逗号分隔)"
                ))
            }
        }
    }
    Ok(o)
}

/// 属性宏入口。选项:`gpu`(GPU 门控)、`hf_parity`(HF 对拍门控);
/// 逗号分隔,顺序无关。其余属性透传到生成的 `#[test]`。
#[proc_macro_attribute]
pub fn owl_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let opts = match parse_opts(&attr) {
        Ok(o) => o,
        Err(msg) => {
            return syn::Error::new(proc_macro2::Span::call_site(), msg)
                .to_compile_error()
                .into()
        }
    };
    let func = parse_macro_input!(item as ItemFn);
    match expand(opts, func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// 返回类型是否为 `()`(缺省或显式空元组)—— gpu 门控要裸 `return`,
/// 非 unit 拒绝(响亮,不猜 Ok/Err 构造)。
fn ret_is_unit(ret: &ReturnType) -> bool {
    match ret {
        ReturnType::Default => true,
        ReturnType::Type(_, ty) => matches!(&**ty, Type::Tuple(t) if t.elems.is_empty()),
    }
}

/// 类型路径末段(支持 `OwlConfig` 与全限定 `…::config::OwlConfig`)
fn type_last_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn expand(opts: Opts, f: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let sig = &f.sig;
    let name = &sig.ident;
    let err = |msg: String| syn::Error::new(sig.ident.span(), msg);

    // ── 校验(响亮拒绝清单;设计文档 §2.3)──
    if sig.asyncness.is_none() {
        return Err(err(format!(
            "#[owl_test] 只收 async fn(宏负责建 runtime 包装);`{name}` 是同步 fn → 用普通 #[test]"
        )));
    }
    if !sig.generics.params.is_empty() {
        return Err(err(format!(
            "#[owl_test] 不支持泛型 fn(`{name}`);测试用例请写具体类型"
        )));
    }
    if !ret_is_unit(&sig.output) && (opts.gpu || opts.hf_parity) {
        return Err(err(format!(
            "#[owl_test(…, …)] 有门控时要求返回 ()(门控跳过用裸 return);`{name}` 返回非 unit —— \
             门内自行处理 Result,或去掉门控选项"
        )));
    }

    // ── 参数:0 个或 1 个;1 个必须是 &OwlConfig ──
    let mut it = sig.inputs.iter();
    let first = it.next();
    if it.next().is_some() {
        return Err(err(format!(
            "#[owl_test] 最多 1 个参数(cfg: &OwlConfig);`{name}` 声明了多个 —— \
             其余状态请从 cfg 派生或用模块常量"
        )));
    }
    let param = match first {
        None => None,
        Some(FnArg::Receiver(_)) => {
            return Err(err(format!("#[owl_test] 不支持方法(`{name}` 带 self)")));
        }
        Some(FnArg::Typed(pt)) => {
            if !matches!(&*pt.pat, Pat::Ident(_)) {
                return Err(err(format!(
                    "#[owl_test] 参数模式必须是简单标识符(`cfg: &OwlConfig`),得 `{}`",
                    quote::ToTokens::to_token_stream(&pt.pat)
                )));
            }
            let is_cfg_ref = matches!(&*pt.ty, Type::Reference(r)
                if type_last_ident(&r.elem).as_deref() == Some("OwlConfig"));
            if !is_cfg_ref {
                return Err(err(format!(
                    "#[owl_test] 参数类型必须是 `&OwlConfig`(全限定路径亦可),得 `{}`",
                    quote::ToTokens::to_token_stream(&pt.ty)
                )));
            }
            Some((&pt.pat, &pt.ty))
        }
    };

    // ── 展开 ──
    let body = format_ident!("__owl_body_{}", name);
    let case = name.to_string();
    let attrs = &f.attrs; // 收到的其余属性(#[ignore]/doc/…)→ 透传到 wrapper
    let vis = &f.vis;
    let block = &f.block;
    let ret = &sig.output;
    let inputs = &sig.inputs;
    let where_c = &sig.generics.where_clause;

    // 配置加载:有参数或需要门控时才读(纯 runtime 包装用例零 env 读取)
    let load = if param.is_some() || opts.gpu {
        quote! { let __owl_cfg = ::owl_shared::testkit::case_config(#case, file!()); }
    } else {
        quote! {}
    };
    // 门控(S2 收编;skip 文案与旧样板逐字同,保 grep 兼容)。
    // 判据/文案全在 `owl_shared::testkit`;宏零逻辑。
    let gpu_gate = if opts.gpu {
        quote! {
            if __owl_cfg.runtime.test_device.is_none() {
                ::std::eprintln!("skip: OWL_TEST_DEVICE 未设");
                return;
            }
        }
    } else {
        quote! {}
    };
    let parity_gate = if opts.hf_parity {
        quote! {
            if !::owl_shared::testkit::hf_parity_enabled() {
                ::std::eprintln!("{}", ::owl_shared::testkit::HF_PARITY_SKIP);
                return;
            }
        }
    } else {
        quote! {}
    };
    let gate = quote! { #parity_gate #gpu_gate };
    let arg = match param {
        Some(_) => quote! { (&__owl_cfg) },
        None => quote! { () },
    };

    Ok(quote! {
        // ① 原 async fn 降为具名 body(cfg(test) 双保险:宏调用点已门控,
        //    输出再钉一层 ⇒ 非测试构建绝不触 tokio/owl-shared 测试面)
        #[cfg(test)]
        async fn #body(#inputs) #where_c #ret {
            #block
        }

        // ② 同名 #[test] wrapper(函数名不变 ⇒ 过滤/清单/ignore 全兼容)
        #[cfg(test)]
        #[test]
        #(#attrs)*
        #vis fn #name() #ret {
            #load
            #gate
            let __owl_rt = ::tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("owl_test: tokio runtime 构建失败");
            __owl_rt.block_on(#body #arg)
        }
    })
}
