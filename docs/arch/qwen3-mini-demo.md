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
| RMSNorm eps | 1e-6 | **全系零中心 ×(1+w)**(2026-09-26 HF 实证:Qwen3_5RMSNorm weight 零初始化 = offset 形态,主干/q_norm/k_norm 同一类;原"主干 ×w"说法作废) |
| vocab | 248320,**tied**(无 lm_head.weight) | embedding 兼 lm_head |
| rope | theta **1e7**,partial **0.25** → rotary_dim 64,**rotate-half**(pair i/i+half;⚠️ 09-26 翻案:config `mrope_interleaved` 指三网格频率维交错,非相邻对 —— HF 探针 6e-8 vs 2.5) | mrope 文本路径退化为一份 pos |
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
| 5 | `layers/rope` | Kernel `owl_rope_half_partial_f32`(rotate-half+partial) | ✅ | HF parity ✓(GPU vs HF 类直跑) |
| 6 | `layers/attention` | 3 GEMM + qk-norm(复用 rmsnorm w_off)+ rope 复用 + naive decode attn(K)+ gate sigmoid(语义)+ mul + o_proj | ✅ | GPU 端到端两步 decode vs host 全链 ✓(含 bisect 13 段 maxdiff 全零) |
| 7 | `layers/gdn` | 4 GEMM + conv_upd3(K)+ gating(K×2)+ l2norm(K)+ delta_dec(K)+ rmsnorm_act(K) | ⏳ | GPU 端到端 |
| 8 | `layers/decoder` | DecoderLayer(Full/Gdn 枚举)+ 双残差;Model 主干 + tied lm_head | 批7 ✅ / 批8 ⏳ | GPU vs host 全链 ✓(两分支) |
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
| `owl_rope_half_partial_f32` | ✅ | 新写(rotate-half pair i/i+half;原相邻对版 09-26 翻案更名) |
| `owl_qknorm_addone`(per-head ×(1+w)) | ✅ 改判复用 | `owl_rmsnorm_f32` w_off=1(per-head 行 = [T×H, HD]),不新写 |
| `owl_narrow_strided_f32`(q gate 切分) | ✅ | 新写(非连续窄切物化;GPU-only vs host 对拍 ✓) |
| `owl_naive_decode_attn_f32`(slot 直排 KV) | ✅ | port dry_kernels.cu + 新世界契约改造(连续槽窗 [slot-kv_len+1, slot]、bs 上界 guard) |
| `owl_sigmoid`(attn_output_gate) | ✅ 改判复用 | `ops.cu owl_sigmoid_f32` 语义算子,不 port |
| `owl_gdn_gating_g_f32` | ✅ | port 旧世界 fused_gating 拆臂;**beta 臂改判复用 owl_sigmoid_f32** |
| `owl_gdn_l2norm_f32` | ✅ | port 块内归约变体(一 block 一行;HF l2norm 同式) |
| `owl_gdn_conv_upd_f32` | ✅ | port + **w_offset 段基址**(三段独立发射,单权重块不切) |
| `owl_gdn_delta_dec_f32` | ✅ | port gqa decode(g log 空间核内 exp;q 核内乘 1/√kd;kd≤128 寄存器硬上界) |
| `owl_gdn_norm_act_f32` | ✅ | port(×w **非零中心** × silu(z);HF Qwen3_5RMSNormGated 同式) |

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
10. **rmsnorm 语义权威(C8/C12 定案,2026-09-26)**:归一化宽度由 alpha
    定义 —— x 任意前导维折叠为行([T, H×HD] × alpha [HD] = per-head 归一
    化,HF flatten(0,1) 同构);权威 = `ops.rs Op::Rmsnorm` 变体注记 +
    `ops.cu owl_rmsnorm_f32`,reference.rs / owl-cpu ops 为对拍副本。
    w_off 留 flag 不拆 op。**Qwen3.5 全系 norm 走 add_one(HF 实证:
    主干也零中心)**,装载时 RmsNorm 一律 new_add_one + checkpoint 存
    offset 形态;×w 语义保留给旧系(Qwen3)与通用算子面;
11. **维度源头单一律(C1,2026-09-26 定案)**:解释器一切维度推导只读
    声明 shape(t.shape / t.parents[i].shape);`Bytes.len` 不参与语义
    (Block 叶子 len=0),仅 debug 构建边界断言(matmul 多行 k 雷即违此律);
12. **kernel 签名 schema(C2,2026-09-26 定案)**:注册表 Entry.args =
    机器可读形参序(`T/sz/i32/f32`;arg_usize ↔ sz 8 字节;Kernel 节点
    路径输出块末参),lower_kernel 对表校验 + .cu 实签名解析互证测试
    (u32 错位 / out 位置两雷机器拦截)。narrow 未实装已封雷显式毒(C3);
    reshape 纯元数据视图已立(C6,eval 透传零拷贝)。
13. **KV 写副作用(C7 定案,2026-09-26)**:decode naive attn kernel
    内联写 cache = 已知副作用;跨步依赖 = cache 块 id + 单流保序。
    **捕获期约束**:slots/kv_lens/pos 必须是常驻 Block 引用(runner 步间
    改写块内容),不得烤成图内标量;SlotWrite 事件边留给 paged 路线。
14. **编排(C5 定案,2026-09-26)**:整模单树 —— Model = Module,一步
    decode = 一次 eval;捕获 = warmup/捕获/重放同一棵树;层间物化
    (eval_ops 子树收割)仅调试用。
15. **rope 配对语义实证律(2026-09-26 翻案)**:Qwen3.5 text rope =
    **rotate-half(pair i/i+half)+ partial 直通**;config `mrope_interleaved`
    指三网格(T/H/W)在**频率维**交错(recomposition_frequencies
    slice(·,·,3)),非 GPT-J 相邻对 —— HF 类直跑探针裁决(6e-8 vs 2.5),
    modeling 源码同证(apply_rotary_pos_emb = rotate_half 形态,attention
    forward 无重排)。inv_freq 同式不变(theta^(-2i/rotary_dim))。
    权威 = HF `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb`,
    金标准 = `layers/rope.py`(GPU vs HF parity 已接)。**教训:config 字段
    名不等于实现语义(与 rmsnorm 零中心同族误读),新层必须先跑 HF
    探针再写 kernel。**
16. **host 参考内维硬编码雷(2026-09-26 M-b 结案附带)**:attention 端到端
    GPU 发散的根因不在引擎 —— bisect 13 段 maxdiff 全零,发散 = example
    `HostRef::lin` 把内维写死 HIDDEN,o_proj(内维 HQ*HD=16)只加了前
    3 项。host 参考的一切形状参数必须随调用点传参,禁止借用全局常数。
    (M-c 批 6 同雷再踩一次:host_gdn_step/示例 lin 同样硬编码 —— 本条
    升级为律:新层 host 参考审阅必查内维传参。)
17. **narrow start 语义(2026-09-26 M-c 批 6 定案)**:`owl_narrow_strided_f32`
    的 `start` 是**行内列偏移**(dst[r·out+d] = src[r·src_dim+start+d]),
    只能切列,**做不了行块偏移** —— conv 权重行切片串行(实测 wk 全错)。
    行块偏移走段基址标量(conv 核 w_offset),勿用 narrow。
18. **CSE / DAG 求值定律(2026-09-26 M-d 批 7 定案;用户裁决保留)**:TensorOps 树是
    **DAG**(共享节点克隆保留同一 id),解释器必须按 DAG 求值——
    `_eval` 带 memo(键 = 节点 id,作用域 = 单次 eval),同节点只执行一次。**无 CSE 时共享
    子树(残差 h:既喃 ln2 又喃残差加)被重复求值,状态副作用 kernel
    (conv/delta)二次滑状态 → 语义破坏**(实测 mlp 段恒定 ×1.017 偏差,
    bisect 定位)。语义根因:值形态(深拷贝 clone)与引用语义(残差复用)
    不一致——值出度被树语义钉死为 1,残差网络需要出度 ≥ 2 的 SSA。
    **用户裁决:CSE memo 即正式求值语义,句柄化/arena 暂不立项**,
    列为 M-f 图捕获期再评估(图 = DAG 的物化,memo 可升级为节点表)。
19. **节点 id 空间分离律(2026-09-26 M-d 批 7 附带)**:`of_block` 声明
    叶子的 id 必须走 `next_id()` 全局节点空间,块 id 是 server 侧另一套
    计数 —— 直接复用会撞 CSE memo 键(实测 Rmsnorm gamma 撞 [8] 声明的
    64 账长块,C1 断言拦截)。两套 id 空间的映射 = Op::Block{id} 字段。
17. **narrow start 语义(2026-09-26 M-c 批 6 定案)**:`owl_narrow_strided_f32`
    的 `start` 是**行内列偏移**(dst[r·out+d] = src[r·src_dim+start+d]),
    只能切列,**做不了行块偏移** —— conv 权重行切片串行(wk 全错)。
    行块偏移走段基址标量(conv 核 w_offset),勿用 narrow。

## 五、验收里程碑

- **M-a(试水)✅**:5 层 + Mul 算子,`tests/layers.rs` 5/5;
- **M-b(attention 层)✅(2026-09-26)**:qk-norm(复用 w_off)/rope
  (翻案 rotate-half)/narrow/naive-attn/gate 全链;GPU 两步 decode vs
  host 全链 1e-4 ✓;rope GPU vs HF parity ✓;发散定位用 bisect 分段收割
  (临时件已删,方法论录 §四 16);
- **M-c(gdn 层)✅(2026-09-26)**:五 kernel 批 1-5 逐核小批移植(每批
  host+GPU parity,gating/l2norm/norm_act 另接 HF golden;delta 核对拍 HF
  `torch_recurrent_gated_delta_rule` out+state 双对拍)+ 批 6 层组装
  (examples/gdn.rs 两步 decode vs host 全链 1e-4 ✓);
- **M-d(decoder 层)✅(2026-09-26)**:DecoderLayer(Full/Gdn 枚举 +
  双残差)两分支 GPU vs host 全链 1e-4 ✓;CSE/DAG 求值 + 节点 id 空间
  分离两条新律落定(§四 18/19);批 8(Model 主干 + tied lm_head)随
  M-e loader 一并立项;
- **M-e(真权重)**:loader 灌 0.8B safetensors,decode 冒烟
  (prompt → token 流,数值抽样对比 vLLM/transformers 基准);
- **M-f(图)**:decode FULL 图捕获(server graph 三原语已就绪)。

## 六、变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 立项:结构盘点(488 张量反推)+ 试水批落地(linear/rmsnorm/mlp/embedding/rope + Mul 算子),`tests/layers.rs` 5/5 绿 |
| 2026-09-25 | 垫子层:`models::kernels` 注册表(owl-kernels cu/ → 登记 → layers 按名组合);kernel 源零内嵌;owl-kernels cudarc feature 化 |
| 2026-09-26 | **HF parity 管道落地**(crates/models:pyproject uv 项目 + src/layers/<层名>.py(就近)(common.py 复用框架)+ tests/{common/mod.rs,parity_hf.rs} 测试套件;进程边界 safetensors 交换,OWL_HF_PARITY=1 门控);首战即修文档语义:Qwen3.5 全系 norm = 零中心 ×(1+w)(含主干),"主干 ×w"作废;顺带发现 Qwen3_5RMSNormGated = M-c gdn norm 的 HF 参照 |
| 2026-09-26 | M-b attention 层落地(narrow/naive-attn/sigmoid);四雷回溯(u32 错位/槽序/Block len/matmul 多行 k);src 重组(12 碎文件→9,client/types/shape/error→contract.rs,plan/actions→ops.rs,module/loader→module.rs,interpreter 拆 reference.rs);API 稳定化快速批:C1/C2/C3/C6/C8/C12/C13 落地(详见 roadmap.local/api-stabilize-plan.md) |
