# Qwen3.8-27B 算子盘点(owl 生产路径;W4A16 权重 + KV fp8/f16 双档)

> 数据源:gpu_prof 逐核实测(T=2151 prefill + 8 decode 步,prof2048.json)+
> specs/qwen35.rs / driver.rs / list.rs 代码面。日期 2026-10-09。
> 模型:64 层 = **48 GDN + 16 full**(3:1);hidden 5120,inter 17408,
> vocab 248320;GDN nk=16×128 / nv=48×128;full q 24 头 × 256(门控融合)
> + kv 4×256;权重 AWQ W4A16(marlin v2)全模型含 lm_head(int4 头)。

## 一、逐层算子链(prefill,split 生产通道)

### GDN 层(48 层)

| 步骤 | 算子/核 | 形状(m×n×k,m=tokens) |
|---|---|---|
| in_proj_qkvz | marlin_gemm_w4a16_awq | 512×16384×5120 |
| in_proj_b / in_proj_a | cublas_gemm_f16(T>16)/ gemv_dual(T≤16) | 512×48×5120 ×2 |
| gating_g | owl_gdn_gating_g_f16 | [T,48] |
| beta | sigmoid(SSA 原生) | [T,48] |
| causal conv(q/k/v 三段) | owl_gdn_conv_fwd_f16(状态槽更新) | [T,10240] |
| q/k l2norm | owl_gdn_l2norm_f16 | [T×16,128] ×2 |
| **delta rule 递推** | **gdn_chunked_delta_rule_fwd**(fork 六核;备臂 scalar_delta / recurrence_varlen) | [T,48,128] |
| gated norm + silu | owl_gdn_norm_act_f16 | [T,6144] |
| out_proj | marlin_gemm_w4a16_awq | 512×5120×6144 |
| 残差加 | owl_add_f16 | [T,5120] |

### full-attention 层(16 层)

| 步骤 | 算子/核 | fp8 KV | f16 KV |
|---|---|---|---|
| input norm(融合残差) | owl_fused_add_rmsnorm_f16 | 同 | 同 |
| q/gate 投影 | marlin | 512×12288×5120 | 同 |
| k/v 投影 | marlin | 512×1024×5120 ×2 | 同 |
| q/k norm + rope | owl_norm_rope_f16 | 同 | 同 |
| **KV cache 写** | owl_qknorm_rope_kv_insert_**fp8kv** | ✓ | owl_qknorm_rope_kv_insert_f16 |
| **attention 主体(split 生产)** | owl_prefill_split_f16_hd256(K1)+ owl_prefill_split_reduce_f16_hd256(K2);读 f16 影子(k0_dual 双写) | 影子 f16 | 直读主池 f16 |
| attention 主体(备选 chunked_opt) | vllm_chunked_prefill_paged_attn_opt_**fp8**_hd256 | ✓ | vllm_chunked_prefill_paged_attn_opt_f16_hd256 |
| o_proj | marlin | 512×5120×6144 | 同 |
| MLP gate/up | marlin | 512×17408×5120 ×2 | 同 |
| MLP act | owl_silu_and_mul_f16 | 同 | 同 |
| MLP down | marlin | 512×5120×17408 | 同 |

### 共享

| 步骤 | 算子 |
|---|---|
| embedding 查表 | 窄切视图(零算子) |
| 残差/归一 | owl_fused_add_rmsnorm_f16 / owl_rmsnorm_f16 |
| lm_head | marlin 1×248320×5120(**int4 头**) |
| 采样 | owl_topk16_f16 / argmax(E3 设备采样) |

## 二、逐层算子链(decode,T=1)

### GDN 层

| 步骤 | 算子/核 |
|---|---|
| in_proj_qkvz / b / a | marlin 1×16384 / 1×48 ×2 |
| gating_g | owl_gdn_gating_g_f16 |
| **conv 更新 + delta + norm_act** | owl_gdn_decode_step_v2_f16(D1-v2 融合核)+ owl_gdn_conv_upd_dual_f16(状态双写) |
| out_proj | marlin 1×5120×6144 |

### full-attention 层

| 步骤 | fp8 KV | f16 KV |
|---|---|---|
| q/k norm + rope | owl_norm_rope_f16 | 同 |
| KV 写 | owl_qknorm_rope_kv_insert_f16_**fp8kv**(+k0_dual 影子写,split/FI 通道) | owl_qknorm_rope_kv_insert_f16 |
| attention | vllm_paged_attention_v2_**fp8**_hd256bs32 + vllm_paged_attention_v2_reduce_f16_hd256 | vllm_paged_attention_v2_f16_hd256(bs32/16) |
| o_proj | marlin 1×5120×6144 | 同 |

### 共享

qkv 投影 marlin 1×12288(k/v 1×1024 ×2)、MLP marlin 1×17408×5120 ×2 / 1×5120×17408、lm_head marlin 1×248320、argmax。

## 三、prefill attention 通道对照(T=块,读 KV)

| 通道 | 核 | KV 供给 | 实测 |
|---|---|---|---|
| chunked_prefill_opt | vllm_chunked_prefill_paged_attn_opt_**fp8**_hd256 | 直读 fp8 主池 | p50 4.8ms/发射(2% 峰值,慢核) |
| **prefill_split(生产)** | owl_prefill_split_f16_hd256(K1)+ reduce(K2) | **f16 影子**(k0_dual 双写) | 2048 墙钟 +26~51% |
| FlashInfer | FI batch prefill(影子) | 影子 | 核快 21%,整体持平(弃) |

## 四、装载期算子(非运行时)

owl_ct_repack_u32(AWQ ct 权重重排)、marlin repack、owl_cast_*(f16/f32/bf16)、owl_state_kv_to_vk/vk_to_kv(GDN 状态转置,chunked 配套)。

## 五、与 KV 档位的差异总结

**KV fp8 ↔ f16 只影响两处**:①KV 写核(fp8kv 变体 ↔ f16 变体);②attention 读核(fp8 变体直读 ↔ f16 变体 / split 通道经 f16 影子)。GDN 递推、GEMM、MLP、采样与 KV 档位无关。

## 六、运行时核实(launch_time 实测,2026-10-09)

**观测点三层**:①driver.rs KernelPick(静态选型);②`[cuda] launch_time=true`
的 server 逐发射打印(**仅 eager 发射**;图回放 verify/fold/dflash/decode 图
走 graph_launch 不可见);③`cap_prof`(捕获期插桩,图内算子)。

**实测触发集(生产配置 d7+fp8+chunked,461 tok prompt + 16 tok 输出,
5723 发射)**:30+ 核,与盘点表主体一致;修正/补充两处:

1. **fold eager 链**:spec 部分接受重放虽捕获了 fold 桶形图(m0-7),
   实测仍有 `owl_gdn_recurrence_varlen_gqa_f16` ×384(48 层 × 8 轮,每发
   ~6µs,合计 2.3ms/请求 —— 可忽略,不动);prefill 尾块(<64)走
   recurrence 臂一次性 31ms;
2. **DFlash2 草稿模型算子族(BF16,盘点表原缺)**:owl_dflash_conv_bf16、
   cublas_gemm_bf16、owl_norm_rope_bf16、owl_fused_add_rmsnorm_bf16、
   marlin_gemm_w4a16_bf16(草稿主干 W4A16 bf16 配方)、owl_sigmoid_gate_mul_f16
   (full 层门控乘)。

**结论**:盘点表与运行时一致;草稿族已补。性能上本轮无新刀位(attention
opt 核仍为唯一重病号,split 已缓解;tg4 重写存案)。

## 七、fp8 纯度审计(launch_time 生产完整形态,split+fp8KV)

**结论:主模型 KV 链全 fp8,无误发射。** 写(qknorm_rope_kv_insert_f16_fp8kv
+ reshape_and_cache_fp8kv)、prefill 读(chunked_prefill_opt_fp8)、decode 读
(paged_attention_v2_fp8_hd256bs32)全部 fp8 变体。

**非 fp8 触点(均为设计,非误发射)**:
| 触点 | 核 | 说明 |
|---|---|---|
| 草稿模型 KV | vllm_reshape_and_cache_**bf16** + owl_naive_attn_nc_**bf16** | 草稿 5 层 bf16 设计;draft_kv_fp8 选项未开 |
| decode partial 归并 | paged_attention_v2_reduce_**f16** | v2 架构的中间量归并 |
| split 通道影子 | k0_dual_fp8kv(主 fp8+影子 f16 一次发射)→ split 核读 f16 影子 | split 核仅 f16 变体;fp8 直读变体=优化立项 |
| GDN 状态/激活 | *_f16 / rec f32 | 设计(激活 f16、状态 f32) |

**附带发现:prefill_split 触发条件** = `nparts ≥ 4`(nparts=ceil(ctx/512),
即 ctx ≥ 1536)且 hd256;短 prompt 自动回落 chunked_opt —— 此前 A/B 的
512/1024 点实为 chunked 路径(两臂同),**split 真实收益仅 2048 点
(+26~51%)**,与 §二十七 记录一致。

## 八、勘误与现状对齐(2026-10-12,布局怪癖审计轮)

对照现行代码重审本盘点,两处已漂移:

1. **§三/§七 "split 读 f16 影子" 已过时**。§二十九(10-12)落地 split
   fp8kv 直读变体后,现行 split 通道 = **K0 单写主池 + fp8kv 直读**
   (`env.rs prefill_split_op`:fp8 池 → `PrefillSplitFp8kv`;K0 =
   `k0_write_op` 单写,attention.rs split 臂)。**影子池(k_fis/v_fis)
   现为 FI 档专属**(engine.rs:274,`fi_quant` 需 `flashinfer=true` 才分配);
   FI-off 生产档零影子。§七 表"split 通道影子"行同此勘误。
2. **§四 "owl_state_kv_to_vk/vk_to_kv = 装载期" 已过时**。gdn_chunked
   fork 集成(10-11)后,状态转置是**运行时税**:每次 chunked 调用
   转置入(池 [HV,K,V] → fork [V,K])+ 转置回(server/gdn_chunked.rs
   :238/:303),48 层 × 每 chunk;另有 cast-in(q/k/v/beta→bf16、g→f32)
   边界税。转置/写域转换沿用的同名核现在是热路径算子。

3. **怪癖总表(审计结论,补入)**:真·格式边界税 = ①FI kNHD 影子池
   (内存+双写+正确性依赖,FI 档限定)+ ②GDN chunked 运行时
   转置/cast/g 语义律(速度税,类型面疫苗已封)。非怪癖:marlin
   (装载期 repack 一次性)、cublas(列主序 flag 映射零拷贝)、
   gemv/conv_dual(融合良品)、草稿独立池(本质所需)、
   attn_v2 scratch(vLLM 同构自洽)。DFlash2 草稿 BF16 族
   (§六)= 主 f16/草 bf16 双 dtype 管线,propose 边界 cast 属结构税。
