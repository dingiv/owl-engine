//! models 视图的 native kernel 登记表垫子(2026-10-12 M4 目录治理:
//! 本体已迁 owl_kernels::list(原 native.rs,2026-10-12 更名)——
//! 登记表/lookup/Kernel 值是 kernels 语义;
//! 本文件只剩 re-export 与 boot 胶水(kv_manifest_gate:KvQuant/ModelError
//! 是 models 词汇)。旧路径 `crate::kernel::*` 全量续用,调用点零改动。

pub use owl_kernels::list::{
    kernel, kernel_with, launch_shape, lookup, source, with_pick, Entry, Kernel, LaunchShape,
    REGISTRY,
};

// ======================================================================
// KV 写者/读者清单门(收口律配套;pitfall §14 机器化,2026-10-10)
// ======================================================================

/// 按活跃 KV quant 声明每族的内核名,boot 逐名校验:名字不在登记表、
/// 或 .cu 源里无同名定义 = 装配失败。三次漏网(decode 融合插池 f16 臂 /
/// K0-DUAL classic 臂 / NC 悬空名)的机器拦截:新增池写者/读者 =
/// 登记表行 + KvEnv 选择子 + 本清单,三处缺一启动即拦。
pub fn kv_manifest_gate(
    quant: crate::env::KvQuant,
    fi_on: bool,
) -> Result<(), crate::contract::ModelError> {
    let fp8: &[&str] = &[
        "owl_reshape_and_cache_fp8kv",                   // K0 写(classic)
        "owl_reshape_and_cache_dual_f16_fp8kv",          // K0 双写(classic+影子,双臂 e4m3)
        "vllm_chunked_prefill_paged_attn_opt_fp8_hd128", // chunked 读 hd128
        "vllm_chunked_prefill_paged_attn_opt_fp8_hd256", // chunked 读 hd256
        "vllm_paged_attention_v2_fp8_hd128bs32",         // v2 decode hd128
        "vllm_paged_attention_v2_fp8_hd256bs32",         // v2 decode hd256
        "owl_qknorm_rope_kv_insert_f16_fp8kv",           // decode 融合插池
        "owl_naive_attn_nc_f16",                         // 草稿池 f16 NC
        "owl_naive_attn_nc_fp8kv_f16",                   // 草稿池 fp8 NC
    ];
    let f16: &[&str] = &[
        "vllm_reshape_and_cache_f16",                    // K0 写(classic)
        "owl_reshape_and_cache_dual_f16",                // K0 双写(classic+影子)
        "vllm_chunked_prefill_paged_attn_opt_f16_hd128",
        "vllm_chunked_prefill_paged_attn_opt_f16_hd256",
        "vllm_paged_attention_v2_f16_hd128bs32",
        "vllm_paged_attention_v2_f16_hd256bs32",
        "owl_qknorm_rope_kv_insert_f16",                 // decode 融合插池
        "owl_naive_attn_nc_f16",                         // 草稿池 f16 NC
        "owl_naive_attn_nc_bf16",                        // 草稿池 bf16 NC
    ];
    let family: Vec<&str> = match quant {
        crate::env::KvQuant::Fp8E4M3 => fp8.to_vec(),
        crate::env::KvQuant::None => f16.to_vec(),
    };
    // FI prefill 虚核**不入本清单**:FI 走 server plan 缓存路径(适配器
    // 自校验 is_fi),不在 models 登记表(零 kernels 依赖律)。仅信息行。
    if fi_on {
        let fi_name = match quant {
            crate::env::KvQuant::Fp8E4M3 => "flashinfer_prefill_paged_fp8kv",
            _ => "flashinfer_prefill_paged_f16",
        };
        eprintln!("[kv-manifest] FI prefill = {fi_name}(server plan 路径,适配器自校验)");
    }
    let mut bad: Vec<&str> = Vec::new();
    for name in &family {
        let Some(e) = REGISTRY.iter().find(|e| e.name == *name) else {
            bad.push(name);
            continue;
        };
        // .cu 源里必须有同名 extern 入口(登记表行在而源缺定义 = 悬空名)
        if !e.source.contains(&format!("extern \"C\" __global__ void {name}")) {
            bad.push(name);
        }
    }
    if !bad.is_empty() {
        return Err(crate::contract::ModelError::Msg(format!(
            "kv_manifest_gate(quant={quant:?}): 清单内核缺失或 .cu 源无定义: {bad:?} —— 池 dtype 改造漏网(pitfall §14 写者穷举律)"
        )));
    }
    eprintln!(
        "[kv-manifest] quant={quant:?} fi={fi_on}:{} 族内核全部在场 ✓",
        family.len()
    );
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{lower_kernel, KernelArg};
    use owl_iface::contract::Arg;
    use owl_kernels::contract::Bytes;

    #[test]
    fn lower_kernel_rejects_scalar_mismatch() {
        // 标量槽宽度与登记 args 不符 → 组合期 panic(机器拦截 u32/sz 错位类雷)
        let k = kernel_with("owl_narrow_strided_f32", (0, 0, 0), (256, 1, 1), 0);
        let scalars = vec![
            crate::ops::KernelArg::I32(2), // 首标量应为 sz(Bit)—— 故意错
            crate::ops::KernelArg::Bits(3),
            crate::ops::KernelArg::Bits(4),
            crate::ops::KernelArg::Bits(5),
        ];
        let ins = vec![Arg::Block { id: 1 }];
        let out = Bytes::new(9, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 0);
        }));
        assert!(result.is_err(), "标量宽度错位应 panic");
    }

    #[test]
    fn lower_kernel_rejects_parent_count_mismatch() {
        // 父依赖数 != 签名 T 槽数 → 组合期 panic
        let k = kernel_with("owl_narrow_strided_f32", (0, 0, 0), (256, 1, 1), 0);
        let scalars = vec![
            crate::ops::KernelArg::Bits(1),
            crate::ops::KernelArg::Bits(2),
            crate::ops::KernelArg::Bits(3),
            crate::ops::KernelArg::Bits(4),
        ];
        let ins = vec![Arg::Block { id: 1 }, Arg::Block { id: 2 }]; // 应为 1 个父
        let out = Bytes::new(9, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 0);
        }));
        assert!(result.is_err(), "父数不符应 panic");
    }

    #[test]
    fn lower_kernel_happy_path_orders_slots() {
        // 正确声明 → 槽序 = 签名序(T 块 + 标量交错,输出末位)
        let k = kernel_with("owl_narrow_strided_f32", (0, 0, 0), (256, 1, 1), 0);
        let scalars = vec![
            crate::ops::KernelArg::Bits(10),
            crate::ops::KernelArg::Bits(20),
            crate::ops::KernelArg::Bits(30),
            crate::ops::KernelArg::Bits(40),
        ];
        let ins = vec![Arg::Block { id: 7 }];
        let out = Bytes::new(9, 8);
        let msg = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 8);
        assert_eq!(msg.args.len(), 6); // T + sz×4 + out
        assert!(matches!(&msg.args[0], owl_iface::contract::Arg::Block { id: 7 }));
        assert!(matches!(&msg.args[1], owl_iface::contract::Arg::U64(10)));
        assert!(matches!(&msg.args[5], owl_iface::contract::Arg::Block { id: 9 }));
        assert_eq!(msg.out_elems, 8);
    }
}
