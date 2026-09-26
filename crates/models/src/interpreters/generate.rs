//! 生成解释器(第四执行域,2026-09-26):计算(TensorOps→eval)/
//! 装载(Loadable→eval_load)/ 观测(Tap→解释器代执行)之外,把
//! **步循环**也从上层命令式代码收编为「声明 + 解释器」同构。
//!
//! 边界三条:
//! - 步循环不进 TensorOps:停机/采样是**数据依赖控制流**,声明世界明确
//!   无控制流(tensor.rs §0.0)—— 它是解释器的机制,不是声明的表达力缺口;
//! - tokenizer(BPE)属 host 字符串域,不进张量解释器;其参数已声明化
//!   (ModelSpec.tokenizer → Tokenizer::from_spec);
//! - 采样今日 host argmax;设备采样(Argmax op)挂账 —— 届时步树 =
//!   ids→ids 全声明,host 每步只碰 4 字节标量。
//!
//! 变体定位:计算/装载/观测之外的新域,另起文件(变体表见 super)。

use crate::contract::{DeviceClient, Dtype, ModelError};
use crate::interpreters::eval_ops;
use crate::layers::gdn::GdnBuffers;
use crate::layers::rope::Rope;
use crate::model::Model;
use crate::module::{ForwardCtx, KvBuffers, Module};
use crate::TensorOps;

/// 采样策略(今日仅贪心;top-k/top-p 随设备采样 kernel 立项)
#[derive(Clone, Copy, Debug)]
pub enum Sampling {
    Greedy,
}

/// 生成声明(停机/预算/采样;数据依赖的控制流参数全部在此)
#[derive(Clone, Debug)]
pub struct GenSpec {
    /// 生成 token 上限(不含 prompt)
    pub max_new: usize,
    /// 终止 id 族(Tokenizer::eos_ids;命中即停,不入列)
    pub eos_ids: Vec<u32>,
    pub sampling: Sampling,
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// 生成执行:prompt 逐 token teacher-forcing(decode 核即因果核:kv_len/
/// pos 递增 + GDN 状态逐步滑,数学等价 prefill)→ 末 prompt 起采样 →
/// eos 停机。返回生成 token(不含 prompt;eos 不入列)。
pub async fn eval_generate<D: DeviceClient>(
    model: &Model,
    face: &mut D,
    rp: &Rope,
    kvs: &[KvBuffers],
    gdns: &[GdnBuffers],
    prompt: &[u32],
    spec: &GenSpec,
) -> Result<Vec<u32>, ModelError> {
    // 槽位预算(prompt + 生成 ≤ 直排 KV 槽位;decode 按槽连续写)
    let slots = kvs
        .first()
        .map(|k| k.k_cache.shape()[0])
        .ok_or_else(|| ModelError::Msg("eval_generate: 无 KV 缓冲".into()))?;
    if prompt.len() + spec.max_new > slots {
        return Err(ModelError::Msg(format!(
            "eval_generate: prompt{} + 生成{} 超槽位 {slots}",
            prompt.len(),
            spec.max_new
        )));
    }

    let mut out: Vec<u32> = Vec::new();
    let mut fed = 0usize; // 已喂 token 数
    while out.len() < spec.max_new {
        let tid = if fed < prompt.len() { prompt[fed] } else { *out.last().expect("生成中") };
        let pos = fed;
        // 每步动态 ctx:slots 恒 0 号槽;kv_lens 含本步(kernel 先写后打分)
        let ids_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[tid as f32]));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[pos as f32]));
        let kvs_step: Vec<KvBuffers> = kvs
            .iter()
            .map(|kv| KvBuffers {
                k_cache: kv.k_cache.clone(),
                v_cache: kv.v_cache.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(pos + 1) as f32])),
            })
            .collect();
        let gdns_step: Vec<GdnBuffers> = gdns
            .iter()
            .map(|g| GdnBuffers {
                conv_q: g.conv_q.clone(),
                conv_k: g.conv_k.clone(),
                conv_v: g.conv_v.clone(),
                rec: g.rec.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            })
            .collect();
        let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, rp, &gdns_step);
        let tree = model.forward(&ids_t, &ctx);
        let logits = eval_ops(tree.step(), face).await?;
        fed += 1;
        if fed < prompt.len() {
            continue; // prompt 中段:只推进状态,不采样
        }
        // 采样(Greedy = host argmax;设备采样挂账,见模块头)
        let mut buf = vec![0u8; model.vocab_size() * 4];
        face.dtoh(&logits, &mut buf).await?;
        let (ti, _) = buf
            .chunks_exact(4)
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, c)| {
                let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if x > a.1 { (i, x) } else { a }
            });
        let nt = ti as u32;
        if spec.eos_ids.contains(&nt) {
            break;
        }
        out.push(nt);
    }
    Ok(out)
}
