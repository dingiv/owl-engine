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
| 8 | `layers/decoder` | DecoderLayer(Full/Gdn 枚举)+ 双残差 | 批7 ✅ | GPU vs host 全链 ✓(两分支) |
| 9 | `model.rs` + `specs/qwen35.rs` | **共有主干**(embed + N 层 + final norm + tied lm_head;C10 装载;整模单树)与 **Qwen3.5 特有**(0.8B 维度档、3:1 层型表)分离 —— 拆分律 §四 23 | 批8 ✅ | GPU 两步 decode vs host 全链 1e-4 ✓(GDN conv/rec 跨步 + KV 增行) |
| 10 | `loader.rs` + specs 装载 | SafeTensorsSource(**流式**:mmap + 条目索引,take/take_range 即取走)+ Qwen35Convention(子前缀/裸键)+ load_0_8b 入口 | ✅ M-e 冒烟 | GPU 真权重两步 decode ✓(**6.6s** / 峰值 **2.8GB**:mmap+流式+4 协程流水+nt 直读+w_t 灭+128MB 分块) |

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
18. **CSE / DAG 求值定律(2026-09-26 M-d 批 7 定案;同日 Arc 修订)**:TensorOps 是
    **DAG**,解释器必须按 DAG 求值——`_eval` 带 memo(键 = 节点 id,作用域 = 单次
    eval),同节点只执行一次。~~值语义深拷贝~~ **已废弃(批 8 实证)**:残差双引用使
    声明树每层 ×2 指数膨胀,真 24 层 2^24 节点直接爆炸(浅层 fixture 测不出);
    **parents 改 Arc 共享**,克隆 O(1),树 = 物理 DAG,构造/遍历/求值全线性,
    “同 id = 同节点”升级为“同节点 = 同一物理对象”。memo 仍保留(DAG 菱形
    两次入口第二次命中)。跨 eval 状态推进是特性。
19. **节点 id 空间分离律(2026-09-26 M-d 批 7 附带)**:`of_block` 声明
    叶子的 id 必须走 `next_id()` 全局节点空间,块 id 是 server 侧另一套
    计数 —— 直接复用会撞 CSE memo 键(实测 Rmsnorm gamma 撞 [8] 声明的
    64 账长块,C1 断言拦截)。两套 id 空间的映射 = Op::Block{id} 字段。
20. **C10 装载律(2026-09-26 批 8 定案,M-e 可插拔化)**:Want 键恒层内
    短键,`Model` 实现 `Loadable`:layout 内三段子清单(embed / layers.{i} /
    norm)经 `LoaderOps::map_keys` 应用 [`KeyConvention`] 改写 —— 改写
    发生在**声明期**,Want 清单里已是 checkpoint 最终键,解释器零约定
    感知,装载 = 一次 `eval_load(&model, ...)`。默认约定
    [`LlamaFamily`]({base}.layers.{i}.{local}.weight);模型特有约定住
    specs/<model>.rs(拆分律 §四 23),如 Qwen3.5 的
    `linear_attn.`/`self_attn.` 子前缀、mlp. 前缀与 `A_log`/`dt_bias` 裸键。
24. **装载域流式律(2026-09-26 M-e 定案,两轮演进)**:数据单副本
    流动 —— mmap 条目区间 `take_range` 查到才转换;**≥64MB 大张量
    分块流式上传**(alloc 一次 + 128MB 块循环 `write_block_f32`
    写入,主机在途 = 单块);`htod_f32(Vec<f32>)` 所有权移入
    (禁 f32→LE→f32 往返);转置 TILE² 分块(写侧 1MB 步长跳页雷)。
    实测:0.8B 装载主机峰值从 3.4GB 常驻 + 2GB 在途 → **全程 <1GB
    匿名堆**。mmap 打开即整文件映射但**物理页按缺页逐页进来**;
    convert_chunk_into 消费完即 `madvise(MADV_DONTNEED)` 归还页 —— RSS
    恒定在"在途块"量级,不随仓增长(实测 **2.25GB**)。装载段实测
    5.1s/3.9GB f32 ≈ 0.8GB/s(4 协程,转换瓶颈);27B 外推见 §五。
    **二拷贝预算(用户定案,2026-09-26)**:一份权重从 mmap 到显存
    只许两次拷贝 —— ① BF16→F32 转换(直写 pinned 租约)② DMA 进
    显存。为此新增 `PinnedRegion` 租约 + `alloc_pinned/upload_pinned`
    能力(GpuClient:HtodPinned 命令 + 页锁池回收;CpuFace:堆租约拷块),
    装载域 Direct 路径 = alloc → convert_chunk_into(直写租约)→
    upload_pinned(move)—— 原 to_vec 消息拷贝与 staging 码头拷贝
    均已消灭。转置小件(≤25MB)容忍 3 拷,设备转置立项时归一。
21. **tied nt 直读律(2026-09-26 M-e 修订;原"单源键双 Want"作废)**:
    Embedding 单 Want 单槽,w 与 lm_head 经 **`Op::MatmulNt`**
    (`owl_matmul_nt_f32`,B 按 [n,k]=[vocab,D] 直读,对固定输出列 B 行
    连续、naive 核访存反而更优)复用同一份 checkpoint 原布局权重 ——
    host 零转置、显存零第二份(抄 candle/mistral "W 保持 [out,in]"
    惯例;原 w_t 转置槽 -1GB 显存 -2GB 主机在途)。Want 聚合(chain)
    语义 = 多重集合(键不需唯一)仍适用于历史形态。
22. **整模 ctx 派生律(2026-09-26 批 8 定案)**:`ForwardCtx::model_decode`
    携 `kvs: &[KvBuffers]` / `gdns: &[GdnBuffers]` 序列(full/gdn 层
    各自的出现序),`Model::last_hidden` 按混型游标派生单层子 ctx ——
    Attention/Gdn 层零感知(仍读 kv/gdn 单引用),混型表分派只在
    Model 一处。
23. **共有/特有拆分律(2026-09-26 批 8 拆分,用户裁决)**:`src/model.rs`
    只住**所有模型共有**的机制(主干结构/单树求值/序列 ctx 派生/C10
    分组件装载/fixture 测试),零模型特定参数;`specs/<model>.rs` 只住
    **该模型特有**的参数事实(维度预设/层型表约定/检查点特例),零机制。
    新模型 = 新 spec 文件;mixer 词汇超出 Full|Gdn(MoE/MLA/…)时另立
    DecoderLayer 扩展口(需求基线:arch 分发表不写死),不动主干。
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
  分离两条新律落定(§四 18/19);
- **M-d2(Model 主干,批 8)✅(2026-09-26)**:`src/model.rs`(共有主干,
  拆分律 §四 23)+ `src/specs/qwen35.rs`(Qwen3.5 特有:hybrid_3to1 /
  qwen3_5_0_8b 真档预设)——
  ModelSpec(hybrid_3to1 / qwen3_5_0_8b 真档预设)+ Model 整模单树
  (C5)+ C10 PrefixedSource 装载 + tied 单源键双 Want +
  ForwardCtx::model_decode 序列派生(§四 20/21/22 三条新律);
  fixture 四层 [G,G,G,F] GPU 两步 decode vs host 全链 1e-4 ✓
  (GDN conv/rec 跨步推进 + KV 增行 + 混型分派);52/52 测试绿;
- **M-e(真权重)首战 ✅(2026-09-26)**:`loader.rs` SafeTensorsSource
  (F32/BF16→f32,目录全 *.safetensors 合并,无需 index)+
  `Qwen35Convention`(linear_attn./self_attn. 子前缀、mlp. 前缀、
  A_log/dt_bias 裸键 —— 488 张量实测)+ `load_0_8b(dir, face)` 入口
  (specs/qwen35.rs);**两步 decode 冒烟 ✓**(step0 top@84 /
  step1 top@279,状态推进,16.7s 峰值 6.7GB);**loader 流水化三连**(2026-09-26,16.7→11.6→**8.9s**,峰值
  6.7→7.6→**3.8GB**):① libc::mmap 只读映射(打开 1.7s→27µs);
  ② `loader_faces(k)` 能力钩子 + LPT 分桶 4 协程流水;③ **流式装载
  (用户裁决:读一点装一点,主机只有"在途"份)**:WeightSource
  `get(&[f32])`→`take(owned)`、`DeviceClient::htod_f32(Vec<f32>)`
  所有权移入(消灭 f32→LE→f32 字节税)、SafeTensorsSource 改 mmap+
  条目索引(0.8B 全量常驻 3.4GB 作废,visual/mtp 0.5GB 不再发生)、
  tied 双槽按键组原子取数(转置先行、直读收尾 move)。④ **w_t 灭 + nt 直读**(抄 candle/mistral [out,in] 惯例,§四 21 修订):
  `owl_matmul_nt_f32` 内核 + `Op::MatmulNt` 全链 + Embedding 单槽;
  ⑤ **128MB 分块流式上传**:`write_block_f32` 原语(iface/GpuClient
  HtodChunk/CpuFace)+ ≥64MB 张量 alloc+分块写。最终 16.7→**6.6s**,
  峰值 6.7→**2.8GB**(含 1.7GB 可回收 mmap 页)。挂账:tokenizer
  接入、数值基准对比 transformers/vLLM、config.json→spec 解析、
  量化 source;
- **M-f(图)**:decode FULL 图捕获(server graph 三原语已就绪)。

## 六、变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 立项:结构盘点(488 张量反推)+ 试水批落地(linear/rmsnorm/mlp/embedding/rope + Mul 算子),`tests/layers.rs` 5/5 绿 |
| 2026-09-25 | 垫子层:`models::kernels` 注册表(owl-kernels cu/ → 登记 → layers 按名组合);kernel 源零内嵌;owl-kernels cudarc feature 化 |
| 2026-09-26 | **HF parity 管道落地**(crates/models:pyproject uv 项目 + src/layers/<层名>.py(就近)(common.py 复用框架)+ tests/{common/mod.rs,parity_hf.rs} 测试套件;进程边界 safetensors 交换,OWL_HF_PARITY=1 门控);首战即修文档语义:Qwen3.5 全系 norm = 零中心 ×(1+w)(含主干),"主干 ×w"作废;顺带发现 Qwen3_5RMSNormGated = M-c gdn norm 的 HF 参照 |
| 2026-09-26 | M-b attention 层落地(narrow/naive-attn/sigmoid);四雷回溯(u32 错位/槽序/Block len/matmul 多行 k);src 重组(12 碎文件→9,client/types/shape/error→contract.rs,plan/actions→ops.rs,module/loader→module.rs,interpreter 拆 reference.rs);API 稳定化快速批:C1/C2/C3/C6/C8/C12/C13 落地(详见 roadmap.local/api-stabilize-plan.md) |
| 2026-09-26 | **M-d2 批 8 Model 主干落地**:`src/specs/qwen35.rs`(specs/ 每档一文件;ModelSpec.hybrid_3to1 / qwen3_5_0_8b 真档预设 + Model 整模单树 C5 + 分组件 C10 装载)+ module.rs(PrefixedSource / Weight::layout_as / ForwardCtx kvs-gdns 序列 + model_decode)+ embedding tied 单源键双 Want(局部键 "w"→"weight",三处测试同步);fixture 四层 GPU 两步 vs host 1e-4 ✓;52/52;新律 §四 20/21/22 |

| 2026-09-26 | **批 8 拆分**:model.rs(共有主干机制)与 specs/qwen35.rs(Qwen3.5 特有参数事实:0.8B 维度 + 3:1 层型表)分离;specs/mod.rs 立拆分律;新律 §四 23;53/53 + GPU 两步 ✓ |
| 2026-09-26 | **批 8 拆分 + M-e 首战**:model.rs(共有主干)与 specs/qwen35.rs(Qwen3.5 参数+键名约定+load_0_8b)分离;**Want.key → String + LoaderOps::map_keys,Model 实现 Loadable 整模单清单装载(load 函数废弃,用户裁决)**;**Arc parents 修订(§四 18)**:值语义深拷贝在真 24 层下 2^24 指数爆炸(实测 48ms/层翻倍曲线),改物理 DAG 后构造/遍历/求值全线性;装载域单缓冲分块转置(§四 24);SafeTensorsSource(BF16→f32);GPU 真权重两步 decode 冒烟 ✓ 16.7s/6.7GB;新律 §四 24 |
| 2026-09-26 | **M-e 补记:观测面 tap 落地 + step1 塔零案定谳**(设计 docs/arch/interpreter-tap.md):interpreters/observe.rs(事件词汇/Want 协议/StatsTap/TapChain)+ eval 三事件点(eval_ops_tap)+ TensorOps label/tag 标注族(同 id 必同 label)+ Model 层根自动打标 + reference reduce_tap(双锚同 id 对齐,挂账);**塔零定谳 = D2H/COMPUTE 跨流竞速**(server handle_dtoh 先排空 COMPUTE,修复后窗口读数与 sync 重读逐位一致)+ harvest 重放污染(从犯,带状态观测禁重放入律候选 25);单遍曲线 24 层全非零,step0 top@84 / **step1 top@279 logit=14.91 塔零消失**,59/59 全绿(OWL_TEST_DEVICE 钉空闲卡)。**顺带:DeviceClient 顺序语义成文**(iface contract:dtoh=读语义 issue 排空 COMPUTE / H2D 族回执即落地已核 / sync=三流全排空) |
| 2026-09-26 | **M-e 端到端跑通:tokenizer 接入 + 文本生成**:`src/tokenizer.rs`(tokenizers crate 直载 tokenizer.json;encode/decode/eos 判定/chat_user 文本路径硬包装,免 jinja)+ `examples/generate.rs`(prompt 逐 token teacher-forcing 喂入 —— decode 核即因果核:kv_len/pos 递增 + GDN 状态逐步滑,数学等价 prefill;greedy 采样 + 全量重解差分增量打印)。实测(3080):英文事实问答正确("I am Qwen3.5, a large language model developed by Tongyi Lab")、中文多字节无乱码、**61 步连续前向状态零腐化**;0.12 s/tok(host 轮询主导,提速归 M-f 图);`<think></think>` 空思考块 = Qwen3.5 模板默认,恰证 chat 格式正确 |

## 七 移植参考
- 旧世界母本:`crates/engine/src/models/qwen3_5.rs`(Qwen3_5ForCausalLM:
  new_with_prefix 前缀装载 = C10 原型;forward_inner 主干循环 = C5 原型;
  mamba_cache/KV per-layer 分派 = ForwardCtx 序列派生原型)——
  本 crate layers/* 与 kernels(cu/)均自此移植;
- 旁证:`packages/xinfer/crates/core/src/models`(qwen3_5.rs 同构实现,
  candle 命令式;结构对照用,勿引代码)