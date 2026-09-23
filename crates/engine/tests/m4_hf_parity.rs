//! M-Ⅳ 终极对拍:owl vs HF transformers 逐层数值对拍(Qwen3.5-0.8B)。
//!
//! 参考面:/tmp/hf_dump.npz(由 /tmp/hf_dump.py 生成;transformers 5.17.0,
//! float32,attn_implementation=eager,GDN 用 transformers 内置纯 torch 参考
//! 实现,设备 CPU)。owl 侧:真权重 f32、eager prefill(复刻 m1 样板)。
//!
//! 输入统一:ids = [100000, 7, 42, 1, 999],prefill 5 token,位置 0..4,
//! 单序列,无历史。
//!
//! 对齐方案(hf_dump.py 实测确认,非假设):
//! - HF output_hidden_states 共 25 条:
//!     hs[0]     = embed 输出(capture_initial_hidden_state,实测 max|Δ|=0 vs embed_tokens)
//!     hs[1..23] = decoder layer0..layer22 输出
//!     hs[24]    = **final norm 后** last_hidden_state(transformers 5.x
//!                 tie_last_hidden_states=True 用 norm 后结果替换原始
//!                 layer23 输出;实测 max|Δ|=0 vs last_hidden_state)
//! - owl collected(collect ids = 0..24)共 25 条:
//!     collected[0]  = embed 输出(forward_inner embed 后 clone)
//!     collected[1..24] = decoder layer0..layer23 输出
//! ⇒ 对齐:owl collected[i] ↔ HF hs[i](i = 0..23,embed + layer0..22);
//!   owl final norm 后 hidden(forward_embedding,fresh mamba slot)↔ hs[24];
//!   owl collected[24](layer23 原始输出)无 HF 直接对手,用 host RMSNorm
//!   复算交叉验证自身一致性,再经 final-norm 对比传导到 hs[24];
//!   owl 末 token logits ↔ HF logits_last。
//!
//! 判定律:首个相对误差 > 1% 且显著大于前层的层 = 发散层;发散时本测试
//! 红且 panic 消息携带全部数字证据(不 ignore)。

use owl_cuda::CudaDevice;
use owl_engine::config::Config;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::distributed::Comm;
use owl_engine::models::layers::{ctx_scope, VarBuilderX};
use owl_engine::models::qwen3_5::{InputMetadata, Qwen3_5ForCausalLM};
use owl_nn::cublas::NnBlas;
use owl_iface::{Device as _, PoolConfig, PoolKind};
use std::sync::Arc;

const MODEL_DIR: &str = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B";
const ST_MODEL: &str =
    "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
const NPZ_PATH: &str = "/tmp/hf_dump.npz";
const SEQ_LEN: usize = 5;
const HIDDEN: usize = 1024;
const IDS: [u32; SEQ_LEN] = [100000, 7, 42, 1, 999];

// ---------------- npz(ZIP STORED)+ npy(f4 LE)最小读取器 ----------------
// 仅支持 np.savez 产物:ZIP STORED(无压缩)+ NPY v1 '\x93NUMPY' + '<f4' C 序。

struct NpyArray {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl NpyArray {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

fn rd_u16(b: &[u8], off: usize) -> usize {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap()) as usize
}

fn rd_u32(b: &[u8], off: usize) -> usize {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as usize
}

fn rd_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn parse_npy(bytes: &[u8]) -> NpyArray {
    assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic 不符");
    let major = bytes[6] as usize;
    let (hlen, hdr_off) = if major == 1 {
        (rd_u16(bytes, 8), 10)
    } else {
        (rd_u32(bytes, 8), 12)
    };
    let header = std::str::from_utf8(&bytes[hdr_off..hdr_off + hlen]).expect("npy 头非 UTF-8");
    assert!(
        header.contains("'<f4'") || header.contains("\"<f4\""),
        "仅支持 <f4 npy,实得 header = {header}"
    );
    assert!(
        header.contains("fortran_order': False"),
        "需 C 序 npy,实得 header = {header}"
    );
    let shape_part = header
        .split("'shape':")
        .nth(1)
        .expect("npy 头缺 shape")
        .split('}')
        .next()
        .unwrap();
    // 形如 " (25, 5, 1024), "/" (248320, )":剔除括号后按逗号切,空段丢弃
    let inner: String = shape_part
        .chars()
        .filter(|c| !matches!(c, '(' | ')'))
        .collect();
    let shape: Vec<usize> = inner
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.parse::<usize>().expect("shape 解析"))
            }
        })
        .collect();
    let data_bytes = &bytes[hdr_off + hlen..];
    let data: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let arr = NpyArray { shape, data };
    assert_eq!(
        arr.len(),
        arr.data.len(),
        "npy 数据量与 shape 积不符"
    );
    arr
}

fn npz_load(path: &str) -> std::collections::HashMap<String, NpyArray> {
    let bytes = std::fs::read(path).expect("读 npz");
    let mut map = std::collections::HashMap::new();
    let mut off = 0usize;
    while off + 30 <= bytes.len() {
        if rd_u32(&bytes, off) != 0x0403_4b50 {
            break; // 本地文件头签名 PK\x03\x04 结束
        }
        let method = rd_u16(&bytes, off + 8);
        let mut csize = rd_u32(&bytes, off + 18);
        let mut usize_ = rd_u32(&bytes, off + 22);
        let name_len = rd_u16(&bytes, off + 26);
        let extra_len = rd_u16(&bytes, off + 28);
        // 本地头布局:30B 头 + 文件名(name_len)+ 扩展(extra_len)+ 数据
        let name_off = off + 30;
        let extra_off = name_off + name_len;
        let extra_end = extra_off + extra_len;
        let mut e = extra_off;
        while e + 4 <= extra_end {
            let eid = rd_u16(&bytes, e);
            let esz = rd_u16(&bytes, e + 2);
            if eid == 0x0001 {
                let mut f = e + 4;
                if usize_ == 0xFFFF_FFFF && f + 8 <= e + 4 + esz {
                    usize_ = rd_u64(&bytes, f) as usize;
                    f += 8;
                }
                if csize == 0xFFFF_FFFF && f + 8 <= e + 4 + esz {
                    csize = rd_u64(&bytes, f) as usize;
                }
                let _ = f;
            }
            e += 4 + esz;
        }
        let name =
            String::from_utf8(bytes[name_off..name_off + name_len].to_vec()).expect("zip 文件名");
        let data_off = extra_off + extra_len;
        if name.ends_with(".npy") {
            assert_eq!(method, 0, "np.savez 应为 STORED(无压缩),实得 method={method}");
            let raw = &bytes[data_off..data_off + usize_];
            map.insert(name.trim_end_matches(".npy").to_string(), parse_npy(raw));
        }
        off = data_off + csize;
    }
    assert!(!map.is_empty(), "npz 未解析到任何 .npy 成员");
    map
}

// ---------------- 设备侧基建(m1 样板) ----------------

fn dtoh_f32(dev: &CudaDevice, ptr: *mut f32, n: usize) -> Vec<f32> {
    use owl_cuda::ffi::sys;
    dev.ctx().bind_to_thread().unwrap();
    dev.ctx().synchronize().unwrap();
    let mut out = vec![0f32; n];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            ptr as sys::CUdeviceptr,
            out.len() * 4,
        )
        .result()
        .unwrap();
    }
    out
}

/// host RMSNorm 复算(candle 语义:逐 token 行 x * rsqrt(mean(x²)+eps),
/// 再逐元素乘 w;w 为 [H] 按行平铺;全 f32)。用于 owl collected[24]
/// (layer23 原始输出)的自洽交叉验证。
fn host_rms_norm(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let h = w.len();
    assert_eq!(x.len() % h, 0, "x 必须是 h 的整数倍");
    x.chunks(h)
        .flat_map(|row| {
            let mean_sq = row
                .iter()
                .fold(0f64, |acc, &v| acc + (v as f64) * (v as f64))
                / h as f64;
            let inv = ((mean_sq + eps).sqrt()).recip() as f32;
            row.iter()
                .zip(w.iter())
                .map(|(&xi, &wi)| wi * (xi * inv))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn max_abs(a: &[f32]) -> f64 {
    a.iter().fold(0f64, |m, &v| m.max((v as f64).abs()))
}

/// NaN/Inf 感知对比:非有限对不参与 max,但单独计数(禁止折叠吞没)。
fn diff_stats(a: &[f32], b: &[f32]) -> (f64, usize, usize) {
    assert_eq!(a.len(), b.len(), "长度不一致");
    let mut m = 0f64;
    let mut bad_a = 0usize;
    let mut bad_b = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        match (x.is_finite(), y.is_finite()) {
            (true, true) => m = m.max((x as f64 - y as f64).abs()),
            (true, false) => bad_b += 1,
            (false, true) => bad_a += 1,
            (false, false) => bad_a += 1,
        }
    }
    (m, bad_a, bad_b)
}

#[test]
fn m4_hf_parity_full_chain() {
    // ---- 真配置(text_config 解壳)----
    let cfg_text =
        std::fs::read_to_string(format!("{MODEL_DIR}/config.json")).expect("读 config.json");
    let config = Config::from_json_str(&cfg_text).expect("Config 反序列化");
    assert_eq!(config.num_hidden_layers, 24);
    let vocab = config.vocab_size.unwrap();
    assert_eq!(vocab, 248320);
    let hybrid = owl_engine::hybrid::resolve_qwen3_hybrid_config(&config);
    assert_eq!(hybrid.layer_types.len(), 24);
    let layer_tag = |i: usize| -> &'static str {
        if hybrid.layer_types[i] == "linear_attention" {
            "GDN"
        } else {
            "ATTN"
        }
    };

    // ---- 参考面加载(npz)----
    let npz = npz_load(NPZ_PATH);
    let hs_ref = npz.get("hidden_states").expect("npz 缺 hidden_states");
    assert_eq!(hs_ref.shape, vec![25, SEQ_LEN, HIDDEN], "hidden_states 形状");
    let hf_final = npz.get("hidden_final").expect("npz 缺 hidden_final");
    assert_eq!(hf_final.shape, vec![SEQ_LEN, HIDDEN]);
    let hf_logits = npz.get("logits_last").expect("npz 缺 logits_last");
    assert_eq!(hf_logits.shape, vec![vocab]);
    let hf_row = |l: usize| -> &[f32] { &hs_ref.data[l * SEQ_LEN * HIDDEN..(l + 1) * SEQ_LEN * HIDDEN] };

    // ---- 设备与池 ----
    let dev = Arc::new(CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备"));
    let scratch = Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("m4-scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: 2 << 30,
        })
        .unwrap(),
    );
    let wpool = Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("m4-weights-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 6 << 30,
        })
        .unwrap(),
    );
    let ops = owl_nn::OpsCtx::new(&dev).unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let dry = DryKernels::new(dev.ctx()).unwrap();
    ctx_scope::install(ops, blas, dry, scratch.clone(), wpool.clone(), &dev);

    // ---- 真权重构造(m1 样板)----
    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(),
            tokenizer_config_filename: Default::default(),
            config_filename: Default::default(),
            generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from(ST_MODEL)],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        },
        false,
        owl_nn::Dtype::F32,
        &dev,
    )
    .expect("VarBuilderX 打开真 safetensors")
    .with_pool(wpool.clone());
    let model = Qwen3_5ForCausalLM::new(
        &vb,
        Comm::new(),
        &config,
        owl_nn::Dtype::F32,
        false,
        &dev,
    )
    .expect("0.8B 真权重 24 层构造");
    assert_eq!(model.full_attention_count(), 6);
    assert_eq!(model.gdn_layer_count(), 18);
    model.preallocate_mamba_cache(4).expect("MambaCache 预分配");
    model.ensure_mamba_slots_for_sequences(&[0, 1]).expect("槽分配");

    // ---- 输入(ids / 位置 0..4)----
    let ids = owl_engine::models::layers::ctor::from_vec(
        IDS.to_vec(),
        (SEQ_LEN,),
        &dev,
    )
    .unwrap();
    let pos = owl_engine::models::layers::ctor::from_vec(
        (0..SEQ_LEN as u32).collect(),
        (SEQ_LEN,),
        &dev,
    )
    .unwrap();

    // ---- 6 个 full_attention 层 KV(flat [slots, 2*256],slot 直排)----
    let kv_dim = config.num_key_value_heads * config.head_dim.unwrap();
    let slots_cap = 8usize;
    let kv_caches: Vec<(
        owl_nn::DynTensor<owl_cuda::CudaDevice>,
        owl_nn::DynTensor<owl_cuda::CudaDevice>,
    )> = (0..model.full_attention_count())
        .map(|_| {
            let k = owl_nn::TensorPoolOps::from_vec_tensor(
                wpool.as_ref(),
                &[slots_cap, kv_dim],
                vec![0f32; slots_cap * kv_dim],
            )
            .unwrap();
            let v = owl_nn::TensorPoolOps::from_vec_tensor(
                wpool.as_ref(),
                &[slots_cap, kv_dim],
                vec![0f32; slots_cap * kv_dim],
            )
            .unwrap();
            (owl_nn::DynTensor::from_f32(&k), owl_nn::DynTensor::from_f32(&v))
        })
        .collect();

    // ---- 第一趟:collecting forward(收集 24 层输出 + 末 token logits)----
    let meta = InputMetadata {
        seqlens: Some(vec![SEQ_LEN]),
        is_prefill: true,
        is_mtp_verify: false,
        mamba_slot_mapping: None,
        sequence_ids: Some(vec![0]),
        decode_ptrs: None,
    };
    let (logits, collected) = model
        .forward_collecting_layers(
            &ids,
            &pos,
            Some(&kv_caches),
            &meta,
            false,
            &(0..24usize).collect::<Vec<_>>(),
        )
        .expect("collecting forward");
    dev.ctx().synchronize().unwrap();
    assert_eq!(collected.len(), 25, "collect ids 0..24 应收集 embed + 24 层");
    assert_eq!(logits.shape(), &[1usize, vocab]);

    let owl_logits = dtoh_f32(&dev, logits.device_ptr() as *mut f32, vocab);
    let owl_layers: Vec<Vec<f32>> = collected
        .iter()
        .map(|l| {
            let n: usize = l.shape().iter().product();
            dtoh_f32(&dev, l.device_ptr() as *mut f32, n)
        })
        .collect();

    // ---- 对齐锚点:collected[0] 必须就是 embed 输出(逐位)----
    {
        let e = model.embed_forward(&ids).unwrap();
        dev.ctx().synchronize().unwrap();
        let ev = dtoh_f32(&dev, e.device_ptr() as *mut f32, SEQ_LEN * HIDDEN);
        let (d, bad_a, _) = diff_stats(&owl_layers[0], &ev);
        assert_eq!((d, bad_a), (0.0, 0), "collected[0] 与 embed_forward 不一致(对齐锚点失守): max|Δ|={d}, 非有限={bad_a}");
    }

    // ---- 第二趟:fresh mamba slot(1)的 forward_embedding 取 final norm 后 hidden ----
    let meta2 = InputMetadata {
        seqlens: Some(vec![SEQ_LEN]),
        is_prefill: true,
        is_mtp_verify: false,
        mamba_slot_mapping: None,
        sequence_ids: Some(vec![1]),
        decode_ptrs: None,
    };
    let hidden2 = model
        .forward_embedding(&ids, &pos, Some(&kv_caches), &meta2, false)
        .expect("final norm hidden(第二趟,槽 1)");
    dev.ctx().synchronize().unwrap();
    assert_eq!(hidden2.shape(), &[SEQ_LEN, HIDDEN]);
    let owl_final = dtoh_f32(&dev, hidden2.device_ptr() as *mut f32, SEQ_LEN * HIDDEN);

    // ---- host RMSNorm 复算 vs owl final norm(自洽交叉验证)----
    let norm_w = {
        // final norm 权重从 safetensors 直读(model.norm.weight)
        let direct = owl_engine::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();
        direct
            .tensor_f32("model.language_model.norm.weight")
            .expect("final norm 权重直读")
    };
    let host_final = host_rms_norm(&owl_layers[24], &norm_w, config.rms_norm_eps);
    let (host_vs_owl, host_bad, _) = diff_stats(&host_final, &owl_final);
    eprintln!(
        "[m4] 自洽:host RMSNorm(collected[24]) vs owl final norm:max|Δ| = {host_vs_owl:.3e}, 非有限 = {host_bad}(≤1e-3 视为同一算子)"
    );

    // ---- 逐层统计(预计算,NaN/Inf 感知)----
    // 每层:(max|owl-hf|, max|hf|, 相对误差, owl 非有限数, hf 非有限数, owl max|v|, owl sum)
    let mut stats: Vec<(f64, f64, f64, usize, usize, f64, f64)> = Vec::with_capacity(24);
    for i in 0..24 {
        let hf = hf_row(i);
        let (d, bad_a, bad_b) = diff_stats(&owl_layers[i], hf);
        let m = max_abs(hf);
        let rel = if m > 0.0 { d / m } else { d };
        let owl_max = max_abs(&owl_layers[i]);
        let owl_sum: f64 = owl_layers[i].iter().filter(|v| v.is_finite()).map(|&v| v as f64).sum();
        stats.push((d, m, rel, bad_a, bad_b, owl_max, owl_sum));
    }

    // ---- 逐层对比表:owl collected[i] ↔ HF hs[i](i = 0..23)----
    eprintln!("[m4] ---- owl vs HF 逐层对比(collected[i] ↔ hs[i])----");
    eprintln!("[m4] {:>5}  {:>4}  {:>10}  {:>12}  {:>10}  {:>8}  {:>12}  {:>12}  {:>10}",
        "层", "型", "owl max|v|", "owl sum", "owl 非有限", "hf 非有限", "max|owl-hf|", "max|hf|", "相对误差");
    for (i, s) in stats.iter().enumerate() {
        eprintln!(
            "[m4] {:>5}  {:>4}  {:>10.6}  {:>12.4}  {:>10}  {:>8}  {:>12.4e}  {:>12.6}  {:>10.3e}",
            if i == 0 { "embed".to_string() } else { format!("{:02}", i - 1) },
            if i == 0 { "EMB" } else { layer_tag(i - 1) },
            s.5, s.6, s.3, s.4, s.0, s.1, s.2
        );
    }

    // ---- final norm 与 logits 对比 ----
    let (d_final, bad_final_owl, _) = diff_stats(&owl_final, &hf_final.data);
    let m_final = max_abs(&hf_final.data);
    let rel_final = d_final / m_final;
    eprintln!(
        "[m4] final norm: max|owl-hf| = {d_final:.4e}  owl 非有限 = {bad_final_owl}  max|hf| = {m_final:.6}  相对误差 = {rel_final:.3e}  (host 复算自洽 |Δ| = {host_vs_owl:.3e}, 非有限 = {host_bad})"
    );

    let (d_logits, bad_logits_owl, _) = diff_stats(&owl_logits, &hf_logits.data);
    let m_logits = hf_logits
        .data
        .iter()
        .fold(0f64, |m, &v| m.max((v as f64).abs()));
    let rel_logits = d_logits / m_logits;
    eprintln!(
        "[m4] logits: max|owl-hf| = {d_logits:.4e}  owl 非有限 = {bad_logits_owl}/{}  max|hf| = {m_logits:.6}  相对误差 = {rel_logits:.3e}",
        owl_logits.len()
    );

    let top5 = |v: &[f32]| -> Vec<(usize, f32)> {
        // torch.topk 语义:按值降序(logits 取最大者);NaN/Inf 过滤并另行报数
        let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.into_iter().take(5).map(|i| (i, v[i])).collect()
    };
    let top_owl = top5(&owl_logits);
    let top_hf = top5(&hf_logits.data);
    let overlap = top_owl
        .iter()
        .filter(|(i, _)| top_hf.iter().any(|(j, _)| j == i))
        .count();
    eprintln!("[m4] owl logits top-5(非有限已过滤):");
    for (i, v) in &top_owl {
        eprintln!("[m4]     idx={i:7}  logit={v:+.6}");
    }
    eprintln!("[m4] hf   logits top-5:");
    for (i, v) in &top_hf {
        eprintln!("[m4]     idx={i:7}  logit={v:+.6}");
    }
    eprintln!("[m4] top-5 重叠 = {overlap}/5");

    // ---- 判定:首个发散层(相对误差 > 1% 且显著大于前层;非有限计数升级为发散)----
    let mut first_div: Option<usize> = None;
    for i in 1..24 {
        let rel = stats[i].2;
        let bad_a = stats[i].3;
        if bad_a > 0 || (rel > 0.01 && rel > 4.0 * stats[i - 1].2) {
            first_div = Some(i);
            break;
        }
    }
    // embed(i=0)特判:应逐位相等(同源 bf16→f32 gather)
    if stats[0].2 > 0.0 || stats[0].3 > 0 {
        first_div = Some(0);
    }

    let build_table = || {
        let mut s = String::from(
            "层 | 型 | owl max|v| | owl sum | owl 非有限 | max|owl-hf| | max|hf| | 相对误差\n",
        );
        for (i, st) in stats.iter().enumerate() {
            s.push_str(&format!(
                "{} | {} | {:.6} | {:.4} | {} | {:.4e} | {:.6} | {:.3e}\n",
                if i == 0 { "embed".to_string() } else { format!("{}", i - 1) },
                if i == 0 { "EMB" } else { layer_tag(i - 1) },
                st.5, st.6, st.3, st.0, st.1, st.2
            ));
        }
        s.push_str(&format!(
            "final norm | owl 非有限 {bad_final_owl} | rel = {rel_final:.3e}\nlogits | owl 非有限 {bad_logits_owl} | rel = {rel_logits:.3e}, top-5 重叠 = {overlap}/5\n"
        ));
        s
    };

    if let Some(i) = first_div {
        let kind = if i == 0 { "EMB".to_string() } else { layer_tag(i - 1).to_string() };
        panic!(
            "M-Ⅳ 发散立案:首个发散层 = {i}({kind},相对误差 {:.3e} > 1% 且显著大于前层 {:.3e},owl 非有限 {});             final norm rel = {rel_final:.3e}(owl 非有限 {bad_final_owl}),logits rel = {rel_logits:.3e}(owl 非有限 {bad_logits_owl},重叠 {overlap}/5)。\n逐层证据表:\n{}",
            stats[i].2,
            if i > 0 { stats[i - 1].2 } else { 0.0 },
            stats[i].3,
            build_table()
        );
    }

    // ---- 全对齐:final norm 与 logits 同门限判定 ----
    assert!(
        rel_final < 0.01 && bad_final_owl == 0,
        "M-Ⅳ:24 层对齐但 final norm 相对误差 {rel_final:.3e} / 非有限 {bad_final_owl} 超限。\n{}",
        build_table()
    );
    assert!(
        rel_logits < 0.01 && bad_logits_owl == 0,
        "M-Ⅳ:24 层对齐但 logits 相对误差 {rel_logits:.3e} / 非有限 {bad_logits_owl} 超限。\n{}",
        build_table()
    );
    assert!(
        host_vs_owl < 1e-3 && host_bad == 0,
        "M-Ⅳ:host RMSNorm 复算与 owl final norm 偏差 {host_vs_owl:.3e} / 非有限 {host_bad} 过大(自洽性)"
    );
    eprintln!(
        "[m4] ✅ 24 层 + final norm + logits 全对齐(各相对误差 < 1%)——M-Ⅱ 数值收官"
    );
}
