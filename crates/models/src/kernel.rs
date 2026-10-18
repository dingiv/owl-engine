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

/// **GDN host 槽臂形状谓词**(单源在 `owl_kernels::family::gdn_chunked` 的验证
/// 清单;engine 经本垫子消费 —— 与 `gdn_table_bytes` 同款"核词汇从核的家来")
pub use owl_kernels::family::gdn_chunked::host_slot_arm_shape_ok;

/// **首轮拼接输入槽数**(8)与 GDN 序列常量同律:单源在 kernels(`ops::elems`,
/// 两张 `owl_concat_rows_*` 表逐槽核过),engine 的固定槽树 / 装配期门经这里取
/// —— 免得两处各写一个字面量 8。
pub use owl_kernels::ops::elems::CONCAT_ROWS_SLOTS;

pub use owl_kernels::cu_sig::{cu_text, parse_cu_sig, source_defines};
pub use owl_kernels::kernel::{
    kernel, kernel_with, launch_shape, lookup, source, with_pick, Kernel, LaunchShape,
};

/// **KV 池页字节格式的 host 参考**(2026-10-19:索引采集/诊断回读要用)。
///
/// 单源在 `owl_kernels::ops::oracles` —— 那里是**独立手写、已被 GPU 逐字节
/// 验证过**的 e4m3 参考(`k0_fp8_knhd` oracle:池字节 == host e4m3 RNE 参考)。
/// 引擎侧**不得**再写第二份 e4m3 解码(AGENTS:接口不唯一 = 两套规范)。
pub use owl_kernels::ops::oracles::{e4m3_decode, e4m3_encode, e4m3_lut};

// ======================================================================
// KV 写者/读者清单门(收口律配套;pitfall §14 机器化,2026-10-10)
// ======================================================================

/// 按活跃 KV quant 声明每族的内核名,boot 逐名校验:名字不在登记表、
/// 或 .cu 源里无同名定义 = 装配失败。三次漏网(decode 融合插池 f16 臂 /
/// K0-DUAL classic 臂 / NC 悬空名)的机器拦截:新增池写者/读者 =
/// 登记表行 + KvEnv 选择子 + 本清单,三处缺一启动即拦。
/// **池元素 dtype**(配置 × 模型 dtype → 池实际 dtype;**单一映射**)。
///
/// - fp8 两族:池 dtype = 配置点名的那个格式(与模型 dtype 无关);
/// - 否则:池**随模型**(f16 模型 ⇒ f16 池,bf16 模型 ⇒ bf16 池)
///   —— 与 `StatePool::alloc` 的存储分支同一事实(那里 `else` 用 `dims.dtype`),
///   两处同源 ⇒ “池类声明”与“池实际存什么”不会再各说各话。
pub fn kv_pool_dtype(
    quant: crate::env::KvQuant,
    model: crate::contract::Dtype,
) -> crate::contract::Dtype {
    if quant.size_bytes() == 1 {
        quant.dtype()
    } else {
        model
    }
}

/// **KV 池清单门**(派生版;E1 2026-10-17,取代手抄字面量清单)。
///
/// 期望核名集合**从声明面 `POOL` 投影**出来 —— 每个触池算子在自己的
/// `KernelSpec::POOL` 里声明 `(池名, 布局, dtype)`(单源),本门只做投影 +
/// 两道校验。原手抄清单的漏网形状:新池核忘了写进清单 ⇒ 门静默变窄。
///
/// 口径:
/// - **主池**:`kv.main` 且声明 dtype == 本档池 dtype(布局不限 —— kNHD 现役
///   与 classic 岛都在册,退役不等于无源定义);
/// - **草稿池**:`kv.draft` 的**全部**声明类(BF16/F16/F8E4M3)—— 草稿池类别
///   独立于主池,且原清单行为就是“全类都查”(超集检查,只强不弱);
/// - **外族虚核**(family ≠ Native:FI/XQA)不入本门 —— 符号不在 AOT 资产
///   索引里,由各自族 runtime 自校验(设计如此);
/// - 每个期望名必须①在声明表里 ②`.cu` 源里有同名定义(单源 =
///   `cu_sig::source_defines`;宏实例化的 chunked kNHD 族也走它)。
pub fn kv_manifest_gate(
    pool: crate::contract::Dtype,
    fi_on: bool,
) -> Result<(), crate::contract::ModelError> {
    use owl_kernels::contract::{Dtype, OpId};
    // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 清单门响亮拒(不静默按 e4m3 族核);
    // 派生口径下它投影出空集 = 静默通过,故这一条必须显式留在这里。
    if pool == Dtype::F8E5M2 {
        return Err(crate::contract::ModelError::Msg(
            "kv_manifest_gate: fp8-e5m2 暂无核族清单(本仓只有 e4m3)".into(),
        ));
    }
    let mut expected: Vec<&'static str> = Vec::new();
    let mut foreign: Vec<&'static str> = Vec::new();
    for m in owl_kernels::ops::all() {
        let touches = m.pool.iter().any(|p| {
            (p.pool == owl_kernels::spec::pools::KV_MAIN && p.dtype == Some(pool))
                || p.pool == owl_kernels::spec::pools::KV_DRAFT
        });
        if !touches {
            continue;
        }
        if m.family == OpId(owl_kernels::spec::NATIVE) {
            expected.push(m.entry);
        } else {
            foreign.push(m.entry);
        }
    }
    expected.sort_unstable();
    expected.dedup();
    foreign.sort_unstable();
    foreign.dedup();
    // FI prefill 虚核**不入本清单**:FI 走 server plan 缓存路径(适配器
    // 自校验 is_fi),不在 models 登记表(零 kernels 依赖律)。仅信息行。
    if fi_on {
        let fi_name = match pool {
            Dtype::F8E4M3 => owl_kernels::family::flashinfer::PREFILL_FI_FP8KV,
            _ => owl_kernels::family::flashinfer::PREFILL_FI,
        };
        owl_shared::log_info!("kv-manifest", "FI prefill = {fi_name}(server plan 路径,适配器自校验)");
    }
    let bad: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|name| match owl_kernels::ops::meta(name) {
            // .cu 源里必须有该核定义(字面 or 单层宏实例化)
            Some(e) => !owl_kernels::cu_sig::source_defines(e.src, name),
            None => true,
        })
        .collect();
    if !bad.is_empty() {
        return Err(crate::contract::ModelError::Msg(format!(
            "kv_manifest_gate(pool={pool:?}): 声明面 POOL 投影出的池面内核缺失或 .cu 源无定义: {bad:?} —— 池 dtype 改造漏网(pitfall §14 写者穷举律)"
        )));
    }
    owl_shared::log_info!("kv-manifest", "pool={pool:?} fi={fi_on}:声明面投影 {} 名池面内核全部在场 ✓(外族自理 {} 名:{:?})",
        expected.len(),
        foreign.len(),
        foreign
    );
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    
    use owl_iface::contract::{Arg, Dtype};
    use owl_kernels::contract::Bytes;

    /// **KV 清单门 = 声明面投影**(E1,2026-10-17):三档池类都必须通过
    /// (f16 / bf16 模型池 + fp8),e5m2(无核臂)必须响亮拒;且投影非空
    /// —— 防"POOL 声明被清空 ⇒ 门静默通过"。
    #[test]
    fn kv_manifest_gate_is_derived_from_pool_declarations() {
        use crate::contract::Dtype;
        for dt in [Dtype::F16, Dtype::BF16, Dtype::F8E4M3] {
            kv_manifest_gate(dt, false).expect("声明面投影应全绿");
            kv_manifest_gate(dt, true).expect("FI 信息行不应影响判定");
        }
        let e = kv_manifest_gate(Dtype::F8E5M2, false).expect_err("e5m2 无核臂必须拒");
        assert!(e.to_string().contains("e5m2"), "{e}");
        let n = owl_kernels::ops::all().filter(|m| !m.pool.is_empty()).count();
        assert!(n >= 30, "池面声明数异常少({n})—— 投影断了门就静默失效");
        // **覆盖地板**(不是真源;真源恒为 `POOL` 声明):下列 = 原手抄清单里的
        // **生产池路**核。派生的残留风险 = “谁把某条的 POOL 声明删了 ⇒ 投影
        // 静默变窄(门失去它而不报)” —— 本行就是那种删除的机器拦截。
        let floor: &[&str] = &[
            // 主池 fp8(写 / chunked 读 / v2 decode 读 / 融合插池)
            "owl_reshape_and_cache_fp8kv_knhd",
            "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128",
            "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
            "vllm_paged_attention_v2_fp8_knhd_hd128bs32",
            "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
            "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
            // 主池 f16
            "vllm_reshape_and_cache_f16_knhd",
            "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd128",
            "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256",
            "vllm_paged_attention_v2_f16_knhd_hd128bs32",
            "vllm_paged_attention_v2_f16_knhd_hd256bs32",
            "owl_qknorm_rope_kv_insert_f16_knhd",
            // 草稿池(classic 岛:写核硬编 K0Write + 读核 NaiveAttnNc 自成闭环)
            "owl_naive_attn_nc_f16",
            "owl_naive_attn_nc_bf16",
            "owl_naive_attn_nc_fp8kv_f16",
        ];
        let declared: Vec<&str> = owl_kernels::ops::all()
            .filter(|m| !m.pool.is_empty())
            .map(|m| m.entry)
            .collect();
        for name in floor {
            assert!(
                declared.contains(name),
                "池面声明面丢了 {name}(POOL 声明被删?)—— 派生门的覆盖会静默变窄"
            );
        }
    }

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
