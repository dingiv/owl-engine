//! repack — compressed-tensors pack-quantized W4A16 → marlin(U4B8)布局转换。
//!
//! 源格式(2026-09-20 对 Arahide/Nex-N2.5-mini-INT4-W4A16 实测定谳,探针层
//! shared_expert.gate_proj,BF16 全量对照,残差 = 4bit RTN 噪声地板):
//! - `weight_shape`: i64 [2] = [out, in]
//! - `weight_packed`: i32 [out, in/8] 行主序;每 i32 = 8 连续 nibble,LSB-first
//! - `weight_scale`: bf16 [out, in/128] per (out 通道, 128-k 分组)
//! - q ∈ [0,15],W = (q - 8) × scale(对称,+8 偏置 = U4B8 语义)
//! - 无 actorder perm 烘焙(config 里 actorder:static 仅溯源元数据)
//!
//! 目标格式 = marlin P1 parity 已验证的 pack 语义(port upstream
//! marlin/__init__.py Layer.pack):B (k/16, n*16/8) i32 tile-perm 打包,
//! s (k/g, n) fp16 scale_perm 重排。

use half::f16;
use rayon::prelude::*;

/// marlin 权重 perm(1024 元素,port upstream `_get_perms()`)。
pub fn marlin_perm() -> Vec<usize> {
    let mut perm = Vec::with_capacity(1024);
    for i in 0..32usize {
        let col = i / 4;
        let mut perm1 = Vec::with_capacity(8);
        for block in 0..2usize {
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1.push(16 * row + col + 8 * block);
            }
        }
        for j in 0..4usize {
            for p in &perm1 {
                perm.push(p + 256 * j);
            }
        }
    }
    // interleave [0,2,4,6,1,3,5,7]
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let grouped: Vec<usize> = perm
        .chunks(8)
        .flat_map(|c| interleave.iter().map(|i| c[*i]))
        .collect();
    grouped
}

/// marlin scale perm(64 元素,g128 分组档;port upstream `scale_perm`)。
pub fn marlin_scale_perm() -> Vec<usize> {
    let mut sp = Vec::with_capacity(64);
    for i in 0..8usize {
        for j in 0..8usize {
            sp.push(i + 8 * j);
        }
    }
    sp
}

/// marlin scale perm_single(32 元素,a8 档;port upstream `scale_perm_single`)。
/// W4A8 的 scale 布局用这张表,不是 W4A16 的 64 表(marlin_permute_scales
/// `is_a_8bit=True` 走 scale_perm_single,源:packages/vllm .../utils/marlin_utils.py)。
pub fn marlin_scale_perm_single() -> Vec<usize> {
    let mut sp = Vec::with_capacity(32);
    for i in 0..4usize {
        for j in [0usize, 1, 8, 9, 16, 17, 24, 25] {
            sp.push(2 * i + j);
        }
    }
    sp
}

/// W4A8 b_scales 打包(vLLM 契约,2026-09-20 定谳):
/// 1. permute 用 scale_perm_single(32-chunk;marlin_permute_scales is_a_8bit=True);
/// 2. int16 归一:s/max(s)*4096 → round → int16 位型塞回 fp16 槽——内核 matmul_a8
///    把 scale 当**整数**乘 int32 累加器(marlin_template.h `a_type == kS8` 分支:
///    `frag_c += frag_c_tmp * (int16)scale_bits`,输出段再 int2float × a_scales);
/// 3. 全局因子 max/4096 不进内核(该内核 global_scale_ptr 仅 nvfp4 读),由调用方
///    折进 a_scales:apply_gptq_marlin_linear `a_scales = act_scale * input_global_scale`。
/// 返回 (位型 u16 序列(= i16 LE 字节即上卡字节), 全局因子)。max==0 时全零 + 0.0。
pub fn pack_marlin_s_a8(scales: &[f32], out: usize, groups: usize) -> (Vec<u16>, f32) {
    assert_eq!(scales.len(), out * groups);
    let sp = marlin_scale_perm_single();

    // (out, groups) 行主 → (groups, out) 行主展平(= vLLM 的 (k/g, n) 布局)
    let mut flat = vec![0f32; groups * out];
    for o in 0..out {
        for gi in 0..groups {
            flat[gi * out + o] = scales[o * groups + gi];
        }
    }
    let max = flat.iter().cloned().fold(0f32, f32::max);
    if max == 0.0 {
        return (vec![0u16; flat.len()], 0.0);
    }
    for chunk in flat.chunks_mut(32) {
        let orig = chunk.to_vec();
        for (dst, src) in sp.iter().enumerate() {
            chunk[dst] = orig[*src];
        }
    }
    let bits: Vec<u16> = flat
        .iter()
        .map(|v| {
            let q = (v / max * 4096.0).round();
            (q as i64).clamp(0, i16::MAX as i64) as i16 as u16
        })
        .collect();
    (bits, max / 4096.0)
}

/// compressed-tensors packed i32 → q(u8),行主序 (out, k),q ∈ [0,15]。
/// S3-P2:rayon 按行并行(行间零依赖,纯 gather,输出与串行版字节一致)。
pub fn unpack_nibbles(packed: &[i32], out: usize, k: usize) -> Vec<u8> {
    assert_eq!(packed.len(), out * (k / 8), "packed len mismatch");
    let kpr = k / 8;
    let mut q = vec![0u8; out * k];
    q.par_chunks_mut(k)
        .enumerate()
        .for_each(|(r, row)| {
            for ci in 0..kpr {
                let v = packed[r * kpr + ci] as u32;
                for nib in 0..8 {
                    row[ci * 8 + nib] = ((v >> (4 * nib)) & 0xF) as u8;
                }
            }
        });
    q
}

/// q(u8 行主序 out×k)→ marlin B((k/16, n*16/8) i32),n = out。
/// port upstream Layer.pack 的 w 打包段(逐字;P1 parity 全绿验证过)。
///
/// S3-P2:原四遍标量(转置→tile 重排→1024-PERM→nibble 打包,~2.5× 内存
/// 流量放大)融合为单遍解析索引链,rayon 按输出字块并行:
///   输出字 W → shuffled 字节 S=W*8+j → chunk=S/1024, dst=S%1024,
///   src=perm[dst] → res 下标 R → (kb,nb,t1,t2) → w[i*n+o] → q[o*k+i]
/// 纯 gather,输出与四遍版逐字节一致(repack_parity.rs 黄金对拍验证)。
pub fn pack_marlin_b(q: &[u8], k: usize, n: usize) -> Vec<i32> {
    assert!(k % 128 == 0 && n % 256 == 0, "k%128==0 n%256==0 required");
    assert_eq!(q.len(), k * n);

    let perm = marlin_perm();
    assert_eq!(perm.len(), 1024);
    let words = k * n / 8;
    let n16 = n * 16;
    let mut b = vec![0i32; words];
    // 按输出字块并行(块内逐字独立);块大小取 64K 字避免 par 调度开销
    const WORDS_PER_TASK: usize = 1 << 16;
    b.par_chunks_mut(WORDS_PER_TASK)
        .enumerate()
        .for_each(|(task, out)| {
            let w0 = task * WORDS_PER_TASK;
            for (wi, word) in out.iter_mut().enumerate() {
                let base = (w0 + wi) * 8; // shuffled 字节基址
                let chunk = base / 1024;
                let dst0 = base % 1024;
                let mut v: u32 = 0;
                for j in 0..8usize {
                    let src = perm[dst0 + j];
                    let r = chunk * 1024 + src; // res 下标
                    let kb = r / n16;
                    let rem = r % n16;
                    let nb = rem / 256;
                    let t1 = (rem % 256) / 16;
                    let t2 = rem % 16;
                    let i = kb * 16 + t1;
                    let o = nb * 16 + t2;
                    v |= (q[o * k + i] as u32) << (4 * j);
                }
                *word = v as i32;
            }
        });
    assert_eq!(b.len(), (k / 16) * (n * 16 / 8));
    b
}

/// q(u8 行主序 out×k,n=out)→ marlin B a8 布局((K/16, N*2) i32 位型)。
/// port vLLM gptq_marlin_repack.cu is_a_8bit=true 分支(4bit,重排 tile 32×32;
/// a16 版则重排 16×64 + 1024-PIPE,两者输出长度同为 K*N/8 i32,但 nibble 排布不同)。
/// 每输出 word(128 word/tile = th_id*4+warp_id):
///   n = (warp/2)*16 + th/4 + (warp%2)*8;nibble r → k_local = tc_row + r/2 + 16*(r%2)
///   (tc_row = (th%4)*4)——与探针判决“词内同 n,k={bk..bk+3,bk+16..bk+19}”一致。
/// 探针黄金数据:tests/data_repack/vllm.b_marlin_a8.i32.bin(gen_a8_bmap_probe 判决)。
pub fn pack_marlin_b_a8(q: &[u8], k: usize, n: usize) -> Vec<i32> {
    assert!(k % 32 == 0 && n % 64 == 0, "k%32==0 n%64==0 required (a8 repack tile)");
    assert_eq!(q.len(), k * n);

    // w(in, out=n) 行主序:q 是 (out, k) → 转置(S3-P2:按 i 行并行,分区无交叉写)
    let mut w = vec![0u8; k * n];
    w.par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, wrow)| {
            for o in 0..n {
                wrow[o] = q[o * k + i];
            }
        });

    let n_tiles = n / 32;
    let mut b = vec![0i32; k * n / 8];
    // S3-P2:按 k-tile 并行(每 tile 写自己的 128 字区间,输出确定性)
    b.par_chunks_mut(128 * n_tiles)
        .enumerate()
        .for_each(|(kt, btile)| {
            for nt in 0..n_tiles {
                for word in 0..128usize {
                    let warp = word % 4;
                    let th = word / 4;
                    let tc_row = (th % 4) * 4;
                    let nn = (warp / 2) * 16 + th / 4 + (warp % 2) * 8;
                    let mut v: u32 = 0;
                    for r in 0..8usize {
                        let k_local = tc_row + r / 2 + 16 * (r % 2);
                        let qv = w[(kt * 32 + k_local) * n + (nt * 32 + nn)] as u32;
                        v |= qv << (4 * r);
                    }
                    btile[nt * 128 + word] = v as i32;
                }
            }
        });
    assert_eq!(b.len(), (k / 16) * (n * 16 / 8));
    b
}

/// scales(f32 行主序 out×groups)→ marlin s((groups, n) fp16 位型,n = out)。
/// port upstream Layer.pack 的 s 段(flatten → (-1,64)[:, scale_perm] → 还原)。
pub fn pack_marlin_s(scales: &[f32], out: usize, groups: usize) -> Vec<u16> {
    assert_eq!(scales.len(), out * groups);
    let sp = marlin_scale_perm();
    assert_eq!(sp.len(), 64);

    // (out, groups) → 转置 (groups, out) → flatten(groups×out,out 连续)
    let mut flat = vec![0f32; groups * out];
    for o in 0..out {
        for gi in 0..groups {
            flat[gi * out + o] = scales[o * groups + gi];
        }
    }
    // (-1, 64)[:, scale_perm]
    assert_eq!(flat.len() % 64, 0);
    for chunk in flat.chunks_mut(64) {
        let orig = chunk.to_vec();
        for (dst, src) in sp.iter().enumerate() {
            chunk[dst] = orig[*src];
        }
    }
    flat.iter().map(|v| f16::from_f32(*v).to_bits()).collect()
}

/// scales(f32 行主序 out×groups)→ marlin s((groups, n) f32,未转 fp16)。
/// 供引擎按激活 dtype 转 f16/bf16(candle to_dtype);permutation 与
/// [`pack_marlin_s`] 相同。
pub fn pack_marlin_s_f32(scales: &[f32], out: usize, groups: usize) -> Vec<f32> {
    assert_eq!(scales.len(), out * groups);
    let sp = marlin_scale_perm();

    let mut flat = vec![0f32; groups * out];
    for o in 0..out {
        for gi in 0..groups {
            flat[gi * out + o] = scales[o * groups + gi];
        }
    }
    for chunk in flat.chunks_mut(64) {
        let orig = chunk.to_vec();
        for (dst, src) in sp.iter().enumerate() {
            chunk[dst] = orig[*src];
        }
    }
    flat
}

// ---------------------------------------------------------------------------
// MoE marlin(fused marlin moe / vLLM marlin_moe_wna16)布局支持
// ---------------------------------------------------------------------------
// 契约(权威 = packages/vllm/vllm/model_executor/layers/fused_moe/experts/
// marlin_moe.py + csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu,2026-09-20
// 提取):
// - b_q_weight: (E, size_k/16, size_n*16/8) i32 连续 —— 每专家与 dense marlin B
//   完全同构,专家堆叠在 dim0;w13 的 size_n = 2*intermediate(gate|up 沿 n 拼接,
//   gate 在前);w2 的 size_k = intermediate, size_n = hidden
// - b_scales: (E, size_k/g, size_n) 激活 dtype,逐专家 dense scale_perm
// - gate/up 中间维需 %64==0(marlin_moe_padded_intermediate 保证)
// - 无 actorder/g_idx(与 dense 同)
// - workspace: i32 零量 sms*max_blocks_per_sm(vLLM 默认 ×4)
// - c_tmp: f32 min(size_n*sorted_ids_len, sms*4*moe_block_size*max_thread_n),
//   moe_block_size==8 时再 ×2;use_fp32_reduce=true + use_atomic_add=false

/// MoE 三组权重一次性 repack:ct 解包后的 q(u8, out×in 行主序)+ scales(f32,
/// out×groups)→ vLLM fused marlin moe 堆叠布局。
///
/// - `gate_q/up_q`: E 个元素,各 (intermediate, hidden)
/// - `down_q`: E 个元素,各 (hidden, intermediate)
/// - scales 同形(out, groups)
/// - 输出 w13 (E, k/16, 4I) i32 [gate;up] 沿 n 拼接后按专家打包堆叠;
///   w2 (E, I/16, 2*hidden);scales 同堆叠(f32,激活 dtype 由调用方转)
///
/// 注:整专家拼接后打包(而非打包后拼接)—— marlin 打包的 1024-chunk PERM
/// 会跨 n-tile 边界混合,仅当单侧 n%64==0 时两者等价;拼接前置无此约束。
pub struct MoeRepacked {
    pub w13: Vec<i32>,
    pub w2: Vec<i32>,
    pub w13_s: Vec<f32>,
    pub w2_s: Vec<f32>,
}

pub fn repack_moe(
    gate_q: &[Vec<u8>],
    up_q: &[Vec<u8>],
    down_q: &[Vec<u8>],
    gate_s: &[Vec<f32>],
    up_s: &[Vec<f32>],
    down_s: &[Vec<f32>],
    hidden: usize,
    intermediate: usize,
    group_size: usize,
) -> MoeRepacked {
    let e = gate_q.len();
    assert!(e > 0 && up_q.len() == e && down_q.len() == e);
    assert!(gate_s.len() == e && up_s.len() == e && down_s.len() == e);
    let k = hidden;
    assert!(k % 128 == 0 && intermediate % 128 == 0,
        "moe marlin: k%128==0 and intermediate%128==0 required (got k={k}, i={intermediate})");

    // S3-P2:专家级并行(每专家独立打包,按序扁平化,输出与串行版一致)
    let packed: Vec<(Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>)> = (0..e)
        .into_par_iter()
        .map(|x| {
            // w13:[gate;up] 沿 out(n) 拼接 → (2I, k) → 转置逻辑由 pack_marlin_b 内部处理
            let mut q_cat = vec![0u8; 2 * intermediate * k];
            q_cat[..intermediate * k].copy_from_slice(&gate_q[x]);
            q_cat[intermediate * k..].copy_from_slice(&up_q[x]);
            let w13 = pack_marlin_b(&q_cat, k, 2 * intermediate);

            let mut s_cat = Vec::with_capacity(2 * intermediate * (k / group_size));
            s_cat.extend_from_slice(&gate_s[x]);
            s_cat.extend_from_slice(&up_s[x]);
            let w13_s = pack_marlin_s_f32(&s_cat, 2 * intermediate, k / group_size);

            // w2:(hidden, intermediate)
            let w2 = pack_marlin_b(&down_q[x], intermediate, hidden);
            let w2_s = pack_marlin_s_f32(&down_s[x], hidden, intermediate / group_size);
            (w13, w13_s, w2, w2_s)
        })
        .collect();

    let mut w13 = Vec::with_capacity(e * (k / 16) * (4 * intermediate));
    let mut w13_s = Vec::with_capacity(e * (k / group_size) * (2 * intermediate));
    let mut w2 = Vec::with_capacity(e * (intermediate / 16) * (2 * k));
    let mut w2_s = Vec::with_capacity(e * (intermediate / group_size) * k);
    for (a, b_s, c, d) in packed {
        w13.extend(a);
        w13_s.extend(b_s);
        w2.extend(c);
        w2_s.extend(d);
    }

    MoeRepacked { w13, w2, w13_s, w2_s }
}

/// vLLM `moe_align_block_size` 的 CPU 参考实现(语义 port 自
/// vllm/model_executor/layers/fused_moe/moe_align_block_size.py docstring 示例:
/// flatten(topk_ids) 按 expert 稳定排序,每段 pad 到 block_size 倍数,
/// pad 值 = topk_ids 总长哨兵;expert_ids 每块一个,尾部填充块 = num_experts 哨兵)。
pub struct MoeAlign {
    pub sorted_token_ids: Vec<i32>,
    pub expert_ids: Vec<i32>,
    pub num_tokens_post_padded: Vec<i32>,
}

pub fn moe_align_block_size(
    topk_ids: &[i32],
    num_tokens: usize,
    topk: usize,
    block_size: usize,
    num_experts: usize,
) -> MoeAlign {
    assert_eq!(topk_ids.len(), num_tokens * topk);
    let sentinel = (num_tokens * topk) as i32;
    let mut counts = vec![0usize; num_experts];
    for v in topk_ids {
        counts[*v as usize] += 1;
    }
    let total_padded: usize = counts.iter().map(|c| c.div_ceil(block_size) * block_size).sum();

    let mut sorted_token_ids = vec![sentinel; total_padded];
    let mut expert_ids = vec![num_experts as i32; total_padded.div_ceil(block_size)];
    let mut seg_start = 0usize;
    for exp in 0..num_experts {
        let cnt = counts[exp];
        let padded = cnt.div_ceil(block_size) * block_size;
        let mut written = 0usize;
        // 稳定:按 flatten 线性序号顺序收集(与 vLLM 排序语义一致)
        for t in 0..num_tokens {
            for kx in 0..topk {
                if topk_ids[t * topk + kx] == exp as i32 {
                    sorted_token_ids[seg_start + written] = (t * topk + kx) as i32;
                    written += 1;
                }
            }
        }
        debug_assert_eq!(written, cnt);
        for b in seg_start / block_size..(seg_start + padded) / block_size {
            expert_ids[b] = exp as i32;
        }
        seg_start += padded;
    }
    MoeAlign {
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded: vec![total_padded as i32],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_shapes() {
        assert_eq!(marlin_perm().len(), 1024);
        assert_eq!(marlin_scale_perm().len(), 64);
    }
}

// ---------------------------------------------------------------------------
// AWQ 非对称(compressed-tensors pack-quantized,gs=32)→ marlin kU4
// ---------------------------------------------------------------------------
// 契约(2026-09-20 对 cyankiwi/Qwen3.8-27B-AWQ-INT4 定谳,与 compressed_tensors
// unpack_from_int32 官方解码逐位对证):
// - weight_packed: i32 (out, in/8),nibble i(LSB)of wp[r,t] = in 元素 8t+i,
//   q ∈ [0,15] 无符号(与 GPTQ ct 同序,复用 unpack_nibbles)
// - weight_scale: bf16/f16 (out, in/g)
// - weight_zero_point: i32 (out/8, in/g),nibble i of zp[t,c] = out 元素 8t+c 行,
//   zp ∈ [0,15] 无符号(打包维 = out)
// - 反量化 W = (q - zp_group) × scale;marlin 侧走 kU4 臂(has_zp,q - zp 在
//   内核内做),故 B 打包与 u4b8 同一套(pack_marlin_b,q 原值无 -8)
// - zeros 的 marlin 布局 = port upstream marlin_utils.marlin_zero_points:
//   (g, n) → flat 64-chunk scale_perm → n-interleave [0,2,4,6,1,3,5,7] →
//   沿 n 8-nibble/i32 LSB-first → (k/g, n/8) i32

/// ct zero_point((out/8, groups) i32)→ zp(out, groups)u8。
pub fn unpack_zp_ct(packed: &[i32], out: usize, groups: usize) -> Vec<u8> {
    assert_eq!(packed.len(), (out / 8) * groups, "zp packed len mismatch");
    let mut z = vec![0u8; out * groups];
    for t in 0..out / 8 {
        for c in 0..groups {
            let v = packed[t * groups + c] as u32;
            for nib in 0..8 {
                z[(t * 8 + nib) * groups + c] = ((v >> (4 * nib)) & 0xF) as u8;
            }
        }
    }
    z
}

/// zp(out, groups)u8 → marlin zeros((k/g, n/8) i32),n = out。
/// port upstream marlin_zero_points(逐段;不含 is_a_8bit 分支)。
pub fn pack_marlin_z(zp: &[u8], out: usize, groups: usize) -> Vec<i32> {
    assert_eq!(zp.len(), out * groups);
    let sp = marlin_scale_perm();

    // (out, groups) → 转置 (groups, n=out)
    let mut z = vec![0u8; groups * out];
    for o in 0..out {
        for g in 0..groups {
            z[g * out + o] = zp[o * groups + g];
        }
    }
    // flat 64-chunk scale_perm
    assert_eq!(z.len() % 64, 0, "groups*n % 64 != 0");
    for chunk in z.chunks_mut(64) {
        let orig = chunk.to_vec();
        for (dst, src) in sp.iter().enumerate() {
            chunk[dst] = orig[*src];
        }
    }
    // n-interleave [0,2,4,6,1,3,5,7]
    const IL: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut il = vec![0u8; z.len()];
    for (blk, ch) in z.chunks(8).enumerate() {
        for (dst, src) in IL.iter().enumerate() {
            il[blk * 8 + dst] = ch[*src];
        }
    }
    // 沿 n 打包 8 nibble/i32 LSB-first → (groups, out/8)
    let n = out;
    assert_eq!(il.len(), groups * n);
    assert!(n % 8 == 0, "n % 8 != 0");
    let mut packed = vec![0i32; groups * (n / 8)];
    for g in 0..groups {
        for jc in 0..n / 8 {
            let mut v: u32 = 0;
            for nib in 0..8 {
                v |= (il[g * n + jc * 8 + nib] as u32) << (4 * nib);
            }
            packed[g * (n / 8) + jc] = v as i32;
        }
    }
    packed
}
