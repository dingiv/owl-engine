# Qwen3.5-0.8B mini-demo —— 层与算子实施清单

> 2026-09-25 立项。目标:在 owl 新架构(models 声明式 layers + owl-cpu/owl-cuda
> 双后端)上跑通 `models/Qwen/Qwen3.5-0.8B` 文本主干 decode。
> 上游:`async-runtime.md`(四层架构)、`charter.md`(公理);
> 施工现场:`crates/models/src/layers/`(试水批已落)。
> **策略:小步迁移(2026-09-25 用户裁决)—— 每层带对拍落地,禁止一把梭。**

---

## 一、模型实测结构(权重清单反推,488 张量)

`config.json` 关键数字(写层时直接引用):

| 参数 | 值 | 备注 |
|---|---|---|
| 层构成 | **24 层 = 18 GDN + 6 full**(3 线性 + 1 全循环) | `layer_types` |
| hidden / intermediate | 1024 / 3584 | SwiGLU MLP |
| full attn 头 | 8 q / 2 kv(GQA 4:1)× head_dim 256 | `num_key_value_heads=2` |
| GDN 头 | 16 key × 128 / 16 value × 128(**无 GQA**) | q/k/v 各 2048 |
| GDN conv | kernel 4,q\|k\|v 共 6144 通道 | `linear_conv_kernel_dim=4` |
| RMSNorm eps | 1e-6 | qk-norm **×(1+w)**(add_one);主干 norm ×w |
| vocab | 248320,**tied**(无 lm_head.weight) | embedding 兼 lm_head |
| rope | theta **1e7**,partial **0.25** → rotary_dim 64,**interleaved**(相邻对 2i/2i+1) | mrope 文本路径退化为一份 pos |
| attn_output_gate | **true** | q_proj 输出 = value\|gate 拼接(×2) |
| mamba_ssm_dtype | f32 | GDN 状态/卷积恒 f32 |
| max_position | 262144 | rope 表预生成规模 |

权重名(文本主干,`model.language_model.` 前缀略):

```text
embed_tokens [248320, 1024]                          ×1(tied)
layers.N.input_layernorm / post_attention_layernorm  ×24
layers.N.mlp.{gate,up,down}_proj                     ×24
  18 × layers.N.linear_attn.{in_proj_qkv [4096,1024], in_proj_z [2048,1024],
        in_proj_a [16,1024], in_proj_b [16,1024], conv1d [6144,1,4],
        A_log [16], dt_bias [16], norm [128], out_proj [1024,2048]}
   6 × layers.N.self_attn.{q_proj [2048,1024], k_proj [512,1024],
        v_proj [512,1024], o_proj [1024,2048], q_norm [256], k_norm [256]}
model.norm [1024]                                    ×1(终局 norm)
```

不在最小路径(明确挂账):`model.visual.*`(ViT 12 层,多模态)、
`mtp.*`(1 层 MTP,投机解码)、chunked prefill、量化(bf16/int4)。

---

## 二、Layer 清单(9 个;试水批 ✅ 已落 5)

| # | 层 | 组成 | 状态 | 对拍 |
|---|---|---|---|---|
| 1 | `layers/linear` | 装载期转置 + Op::Matmul | ✅ | host 参考 ✓ |
| 2 | `layers/rmsnorm` | Op::Rmsnorm(多行 per-channel) | ✅ | host 参考 ✓ |
| 3 | `layers/mlp` | gate/up → silu×mul → down(全语义) | ✅ | host 参考 ✓ |
| 4 | `layers/embedding` | Kernel `owl_embed_f32`(port) | ✅ | 声明构造 ✓ |
| 5 | `layers/rope` | Kernel `rope_interleaved_partial`(新写) | ✅ | 声明构造 ✓ |
| 6 | `layers/attention` | 3 GEMM + qk-norm(K)+ rope 复用 + naive decode attn(K)+ gate sigmoid(K)+ mul(语义) | ⏳ 下一批 | GPU 端到端 |
| 7 | `layers/gdn` | 4 GEMM + conv_upd3(K)+ gating(K×2)+ l2norm(K)+ delta_dec(K)+ rmsnorm_act(K) | ⏳ | GPU 端到端 |
| 8 | `layers/decoder` | DecoderLayer(Full/Gdn 枚举)+ 残差编排 + 24 层主干 + tied lm_head | ⏳ | 层级组装后 |
| 9 | loader | safetensors → host f32(转置/重复通道)→ layer `new` | ⏳ | 权重指纹 |

(K = Kernel 节点,.cu 随层携带,server nvrtc 懒编译零改动)

## 三、算子清单

**语义算子(动作表;双 face 可跑)** —— 5 个全绿:

| 算子 | 状态 | 备注 |
|---|---|---|
| Matmul / Add / Silu / Rmsnorm | ✅ 已有 | Rmsnorm 本轮修多行缺陷(原 grid 写死 (1,1,1)) |
| **Mul** | ✅ 本轮新增 | 六刀全链:Op → lower → owl_mul_f32 → eval → Interpreter → CpuFace/ops |

**Kernel 节点(每后端语义见各层内嵌源)** —— 约 10 个:

| Kernel | 状态 | 出处 |
|---|---|---|
| `owl_embed_f32` | ✅ | port dry_kernels.cu(ids f32 过线) |
| `owl_rope_interleaved_partial_f32` | ✅ | 新写(相邻对;partial 直通尾段) |
| `owl_qknorm_addone`(per-head ×(1+w)) | ⏳ attention 批 | 新写简版 |
| `owl_naive_decode_attn`(slot 直排 KV) | ⏳ | port dry_kernels.cu |
| `owl_sigmoid`(attn_output_gate) | ⏳ | port dry_kernels.cu |
| `gdn_conv1d_upd3`(q/k/v 三段入,零拼接算子) | ⏳ gdn 批 | port gdn_kernels.cu 改签名 |
| `gdn_gating_g` / `gdn_gating_beta` | ⏳ | port 公式:`g=-exp(A_log)·softplus(a+dt_bias)`、`beta=sigmoid(b)` |
| `gdn_l2norm` | ⏳ | port(per-head,eps 1e-6) |
| `gdn_delta_dec_gqa`(单步;g 核内 exp;state [maxB,Hv,K,V] f32) | ⏳ | port gdn_kernels.cu |
| `gdn_rmsnorm_act`(silu(z) 门 × per-group gamma) | ⏳ | port gdn_kernels.cu |

## 四、形态契约(试水批已验证,勿反工)

1. **构造期 `new`**:唯一保留 Result 的位置;权重 host 声明 `from_host`;
   `[out,in]` 行主序**装载期转置** → 声明 `[in,out]`(decode 零转置);
2. **`forward` 同步 · total · 纯描述**:返回 TensorOps;执行 = `interpreter::eval`;
3. **Kernel 节点发射配置声明期显式**:grid 哨兵自动 1D 只适配 ceil/256 核;
   行核(embed/rope/attn)必须 `with_launch((tokens,…),…)` —— forward 带
   显式 `tokens` 参数;
4. **槽序契约**:kernel 签名序 = T 槽序,输出块固定末尾;权重子树先 eval
   物化,Block id 自动传入(embedding 的 w 曾漏传 —— 测试抓到);
5. **索引/位置量 f32 数值过线**(<2^24 无损;kernel 内 cast;
   S4 u32 dtype 扩展挂账);
6. **常驻块复用 = `TensorOps::of_block(id)`**(KV cache / GDN state /
   rope 表;构造一次,声明引用,块只增不减);
7. **对拍锚纪律**:models 同步参考实现与 owl-cpu/后端实现互为独立副本,
   禁止互引;
8. **kernel 源与业务解耦(垫子层)**:.cu 源只住 owl-kernels(`cu/`);
   `models::kernels` 注册表按名登记导出;layers/TensorOps 组合时只出现
   名字与发射配置,零源码感知(2026-09-25 用户裁决;
   owl-kernels 的 cudarc 已 feature 化,引源不拖链接);
9. **权重跨设备统一定义(2026-09-25 拍板:句柄落线 A)**:
   `Weight = shape 标注 + Option<contract::Bytes>`(与 Tensor<D> 同构:
   容器只持句柄,存放归设备 —— CpuFace 解析为 host 值块,GpuClient
   解析为显存池块,层代码零设备感知)。装载生命周期一个钩子:
   `load(&self, src) -> LoaderOps`(同步零 Result;违约 → 毒值随 ops);
   执行 = `interpreter::eval_load(ops, face)`(唯一副作用点,与 eval
   同签名形态);`mount(blocks)` = 纯内存回填;数据值语义内联进 ops。
   线 B(Tensor<D>/Device trait/rt_cpu)冻结,归一另立项
   (实锤遗留:Tensor::as_declaration 的 id 错位 bug 随线 B 立项修复)。

## 五、验收里程碑

- **M-a(试水)✅**:5 层 + Mul 算子,`tests/layers.rs` 5/5;
- **M-b(attention 层)**:qk-norm/rope/naive-attn/gate 全链,GPU 端到端
  单层对拍(host softmax 参考);
- **M-c(gdn 层)**:五 kernel 全链,单层对拍(gdn_exemplars /
  recurrence_parity 旧世界判例迁移);
- **M-d(decoder)**:24 层组装 + tied lm_head,随机权重 forward 收敛;
- **M-e(真权重)**:loader 灌 0.8B safetensors,decode 冒烟
  (prompt → token 流,数值抽样对比 vLLM/transformers 基准);
- **M-f(图)**:decode FULL 图捕获(server graph 三原语已就绪)。

## 六、变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 立项:结构盘点(488 张量反推)+ 试水批落地(linear/rmsnorm/mlp/embedding/rope + Mul 算子),`tests/layers.rs` 5/5 绿 |
| 2026-09-25 | 垫子层:`models::kernels` 注册表(owl-kernels cu/ → 登记 → layers 按名组合);kernel 源零内嵌;owl-kernels cudarc feature 化 |
