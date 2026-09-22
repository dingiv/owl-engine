//! 杂项工具(从 xinfer `utils/mod.rs` 按触点挑选;搬运期增量增补)。

/// xinfer `is_qwen3_hybrid_arch_name` 原样搬运(scheduler 触点 1 处)。
pub fn is_qwen3_hybrid_arch_name(arch: &str) -> bool {
    matches!(
        arch,
        "Qwen3_5ForCausalLM"
            | "Qwen3_5MoeForCausalLM"
            | "Qwen3NextForCausalLM"
            | "Qwen3_5ForConditionalGeneration"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForConditionalGeneration"
            | "Qwen4ExpForConditionalGeneration"
            | "Qwen4ExpForCausalLM"
    )
}
