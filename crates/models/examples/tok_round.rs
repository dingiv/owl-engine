//! 诊断:数字 token 往返(量化数字幻觉 vs 引擎判别)
use owl_models::tokenizer::{ChatFormat, Tokenizer, TokenizerSpec};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("模型目录"));
    let spec = TokenizerSpec {
        eos_tokens: vec!["<|im_end|>", "<|endoftext|>"],
        chat: ChatFormat {
            prefix: String::new(),
            suffix: String::new(),
            im_open: "<|im_start|>".into(),
            im_close: "<|im_end|>\n".into(),
            assistant_open: "<|im_start|>assistant\n".into(),
            think_prefill: "<think>\n".into(),
        },
    };
    let tok = Tokenizer::from_spec(&dir, &spec).expect("tokenizer");
    for s in ["1360", "2640", "2298919", "port is 1360", "port is 1360."] {
        let ids = tok.encode(s);
        let back = tok.decode(&ids);
        let roundtrip_ok = back == s;
        println!("{s:?} -> {ids:?} -> {back:?} 往返一致={roundtrip_ok}");
    }
}
