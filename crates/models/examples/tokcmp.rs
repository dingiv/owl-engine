//! 诊断:owl chat_wrap 全文落盘(vLLM raw 对照用;保证逐字节同一 prompt)
use owl_models::tokenizer::{ChatFormat, Tokenizer, TokenizerSpec};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("模型目录"));
    let spec = TokenizerSpec {
        eos_tokens: vec!["<|im_end|>", "<|endoftext|>"],
        chat: ChatFormat {
            prefix: "<|im_start|>system\nReasoning effort is set to medium. Think through the task at a moderate depth: cover the key steps and verify the result, but keep the reasoning concise.<|im_end|>\n<|im_start|>user\n".into(),
            suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n".into(),
            // 2026-10-13 补齐(多轮模板字段;取值同 specs/qwen35.rs 的规范档)
            im_open: "<|im_start|>".into(),
            im_close: "<|im_end|>\n".into(),
            assistant_open: "<|im_start|>assistant\n".into(),
            think_prefill: "<think>\n".into(),
        },
    };
    let tok = Tokenizer::from_spec(&dir, &spec).expect("tokenizer");
    let user = std::env::args().nth(2).expect("prompt 文本");
    let wrapped = tok.chat_wrap(&user);
    std::fs::write("/tmp/owl_wrapped.txt", &wrapped).expect("落盘");
    let ids = tok.encode(&wrapped);
    println!("wrapped 字节={} tokens={}", wrapped.len(), ids.len());
}
