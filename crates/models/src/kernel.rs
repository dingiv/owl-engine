//! models 视图的 native kernel 登记表垫子(2026-10-12 M4 目录治理:
//! 本体已迁 owl_kernels::list(原 native.rs,2026-10-12 更名)——
//! 登记表/lookup/Kernel 值是 kernels 语义;
//! 本文件只剩 re-export 与 boot 胶水(kv_manifest_gate:KvQuant/ModelError
//! 是 models 词汇)。旧路径 `crate::kernel::*` 全量续用,调用点零改动。

/// **GDN 序列表单源出口**(P-6b;2026-10-16 B6 收敛:原挂 `layers::gdn`
/// —— 层文件直引 kernels 违反 facedir_lint,故搬到本垫子)。
/// 单源在 `owl_kernels::family::gdn_chunked`(那里与 triton AOT 配置表锁死);
/// engine/models 经这里取,免得出现第二份布局常量。
pub use owl_kernels::family::gdn_chunked::{
    table_bytes as gdn_table_bytes, table_host_bytes as gdn_table_host_bytes, BT as GDN_BT,
};

pub use owl_kernels::cu_sig::{cu_text, parse_cu_sig, source_defines};
pub use owl_kernels::kernel::{
    kernel, kernel_with, launch_shape, lookup, source, with_pick, Kernel, LaunchShape,
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

    // FIXME: 非常不优雅的实现, 需要处理, 字面量在后续的维护处理中，非常的麻烦
    let fp8: &[&str] = &[
        "owl_reshape_and_cache_fp8kv_knhd",              // K0 写(kNHD 单池)
        "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128", // chunked 读 hd128
        "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256", // chunked 读 hd256
        "vllm_paged_attention_v2_fp8_knhd_hd128bs32",    // v2 decode hd128
        "vllm_paged_attention_v2_fp8_knhd_hd256bs32",    // v2 decode hd256
        "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",      // decode 融合插池
        "owl_naive_attn_nc_f16",                         // 草稿池 f16 NC(classic 岛,D5)
        "owl_naive_attn_nc_fp8kv_f16",                   // 草稿池 fp8 NC(classic 岛,D5)
    ];
    let f16: &[&str] = &[
        "vllm_reshape_and_cache_f16_knhd",               // K0 写(kNHD 单池)
        "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd128",
        "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256",
        "vllm_paged_attention_v2_f16_knhd_hd128bs32",
        "vllm_paged_attention_v2_f16_knhd_hd256bs32",
        "owl_qknorm_rope_kv_insert_f16_knhd",            // decode 融合插池
        "owl_naive_attn_nc_f16",                         // 草稿池 f16 NC(classic 岛,D5)
        "owl_naive_attn_nc_bf16",                        // 草稿池 bf16 NC(classic 岛,D5)
    ];
    // 注(统一契约 P6-A/D,2026-10-13):主池 = **kNHD 单布局**(KvLayout 枚举
    // 与 classic 偏移数学已删),故本清单只列 kNHD 名;影子双写核
    // `owl_reshape_and_cache_dual_*` 与 K0-dual 语义**已整体退役**(核 + 登记
    // 表 + driver 臂 + FI 隔离探针全删;FI 正确性改由 engine e2e
    // `gpu_longctx_fi_direct_gate` 覆盖)。草稿池(P6-C 待翻)暂留 classic 岛,
    // 其写核(硬编码 K0Write)+ 读核(NaiveAttnNc)自成闭环。
    let family: Vec<&str> = match quant {
        crate::env::KvQuant::Fp8E4M3 => fp8.to_vec(),
        crate::env::KvQuant::None => f16.to_vec(),
    };
    // FI prefill 虚核**不入本清单**:FI 走 server plan 缓存路径(适配器
    // 自校验 is_fi),不在 models 登记表(零 kernels 依赖律)。仅信息行。
    if fi_on {
        let fi_name = match quant {
            crate::env::KvQuant::Fp8E4M3 => owl_kernels::family::flashinfer::PREFILL_FI_FP8KV,
            _ => owl_kernels::family::flashinfer::PREFILL_FI,
        };
        owl_shared::log_info!("kv-manifest", "FI prefill = {fi_name}(server plan 路径,适配器自校验)");
    }
    let mut bad: Vec<&str> = Vec::new();
    for name in &family {
        let Some(e) = owl_kernels::ops::meta(name) else {
            bad.push(name);
            continue;
        };
        // .cu 源里必须有该核定义(字面 or 单层宏实例化;判据单源 =
        // owl_kernels::cu_sig::source_defines —— chunked kNHD 族走宏)
        if !owl_kernels::cu_sig::source_defines(e.src, name) {
            bad.push(name);
        }
    }
    if !bad.is_empty() {
        return Err(crate::contract::ModelError::Msg(format!(
            "kv_manifest_gate(quant={quant:?}): 清单内核缺失或 .cu 源无定义: {bad:?} —— 池 dtype 改造漏网(pitfall §14 写者穷举律)"
        )));
    }
    owl_shared::log_info!("kv-manifest", "quant={quant:?} fi={fi_on}:{} 族内核全部在场 ✓",
        family.len()
    );
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    
    use owl_iface::contract::{Arg, Dtype};
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
        let ins = vec![Arg::block(1, Dtype::F16, 2)];
        let out = Bytes::new(9, 0, Dtype::F16, 2);
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
        let ins = vec![Arg::block(1, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2)]; // 应为 1 个父
        let out = Bytes::new(9, 0, Dtype::F16, 2);
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
        let ins = vec![Arg::block(7, Dtype::F16, 2)];
        let out = Bytes::new(9, 8, Dtype::F16, 2);
        let msg = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 8);
        assert_eq!(msg.args.len(), 6); // T + sz×4 + out
        assert!(matches!(&msg.args[0], Arg::Block { id: 7, .. }));
        assert!(matches!(&msg.args[1], owl_iface::contract::Arg::U64(10)));
        assert!(matches!(&msg.args[5], Arg::Block { id: 9, .. }));
        assert_eq!(msg.out_elems, 8);
    }
}
