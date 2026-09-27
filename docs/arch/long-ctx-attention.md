# 长 ctx attention —— paged 家族 port 路线(定稿)

> 2026-09-27 立项;同日用户裁决改道:**弃手写 naive fatt 方案,直接对标
> 现役算子方案(FlashInfer / paged attention / chunked prefill),以 vendor
> attention.rs(rev c0f19f2)为 port 源、attention-kernel-port.md K0-K4 为
> 既定路径**。本文档 = 长 ctx 目标与 port 路线的合流定盘。

## 一、目标(不变)

解锁 OWL_MAX_KV=256 硬顶:max_seq_tokens 抬到 4k/8k 级;>256 段从
"滑窗截断近似"回归真全局注意力语义。

## 二、路线(用户裁决定盘)

- **手写 naive fatt 三只核已废弃回退**(decode/写核/prefill 各一;曾完成
  初稿,查重发现与 vendor paged 家族全面同构,且社区实现更优——为避免
  重复造过时方案,整线撤销);
- **主路 = vendor attention.rs paged 家族 file-level 复用**(classic 布局,
  K1/K2 同树同布局):
  | owl 里程碑 | vendor 源 | 角色 |
  |---|---|---|
  | K0 ✅ | reshape_and_cache_kernel.cu | KV 散写(物理槽语义) |
  | K1 | paged_attention_v1.cu + pagedattention.cuh | decode attention(分页) |
  | K2 | paged_attention_v2.cu | decode 大 bs 分片归约 |
  | prefill | prefill_paged_attn_opt.cu | chunked prefill(在线 softmax + 滑窗可选) |
  | 性能线 | flashinfer_adapter_*.cu | FA2 adapter(flashinfer_common.cuh 依赖链,预编 PTX 备选) |
- 上游对照:vLLM 树(d5d2e53e)—— 注意其 NVIDIA 线已删 paged 核
  (cache_kernels.cu 仍在),paged 家族 vendor 是工作区唯一源;
- llama.cpp(无分页 KV)/ sglang(triton+python 壳)对 attention 不可复用,
  分别留 GGUF dequant/MMQ 与调度语义参考。

## 三、分页布局契约(K0 定谳,vLLM classic)

- key_cache `[num_blocks, Hkv, D/x, block_size, x]`(f16 x=8)/
  value_cache `[num_blocks, Hkv, D, block_size]`;
- slot_mapping = **物理槽**(block_idx·block_size + offset);单序列连续分配
  时 block_table 恒等 ⇒ 物理槽 = 逻辑 pos —— **恒等分页桥**:先以恒等表接
  入 paged 核(存储布局从 token 直排 `[s,Hkv,D]` 换 block 直排,总元素数
  不变),分页分配器/多会话块表(M2)在契约之上自然生长,attention 核
  零改动;
- slot/fd 槽表 f32 过线(契约 5);显式 grid;哑尾参(契约 4)——
  nvrtc 约束清单全文见 attention-kernel-port.md §五 K0 条。

## 四、里程碑(并入 K0-K4;本文档只留长 ctx 侧验收)

- P3(engine 侧,与 K1 并行):KvBuffers 布局换 block 直排 + 恒等块表;
  prefill chunk 去 `min(256-base)` 钳制;server OWL_MAX_SEQ 默认抬升;
- P4(端到端):4k 长 ctx 前缀记忆 QA(长文首句提问,真全局注意力下长程
  记忆应优于旧滑窗截断)+ 计时对比;
- P5:结案(计时入 bench;flashinfer adapter 性能线单独立项)。

## 五、P4/P5 结案(2026-09-27)

- **4k e2e 全绿**(`engine::tests::gpu_longctx_4k_prefix_qa`):4045 tok
  prompt(暗号埋首,问题/补全探针在尾)+ 16 greedy 生成,分页 prefill
  127 chunk + v1 长程 decode + 账本收口,全链零崩;同会话 turn2(≤1024
  档)全量回退路径稳。
- **途中定谳三案**:
  1. **v1 动态 smem 越界**(decode 图回放 ILLEGAL_ADDRESS;包装层漏算
     上下文项,2048B 隐形 512 顶)→ 按 .cu 头注契约公式修复,详见
     attention-kernel-port.md 结案条目;
  2. **GDN 状态块解耦**:原按 max_seq_tokens 整表分配(s=4096 → rec
     72GB),改 GDN_SLOTS=4 常量(格语义 = 每会话一格,核内恒 slot 0,
     逐核验证);M2 多会话时提升为 EngineConfig 字段;
  3. **server 中间块零回收**:eval 产生的中间块全量常驻,4k 双 turn
     累计 ~24GB OOM(量化在案)—— **E2 块分配器 + Free 路线的立项
     动机**。
- **计时(P5;debug 与 release 同值 ⇒ host 发射开销主导)**:
  | 档 | prefill @4k | TTFT | decode |
  |---|---|---|---|
  | 4k 全链(debug=release) | 33 tok/s(4057 tok / 123s) | 123.7s | ~40 tok/s(16 tok / 0.4s) |
  - host-bound 判据:release 无增益;127 chunk × 24 层 × 数发射,
    ~15ms/launch —— 削发射数(大 chunk / prefill 图化)是 E4 靶面;
  - 短 ctx 对照(27 tok):prefill 108 tok/s,TTFT 392ms。
- **记忆质量金标挂 S3**:裸模板 + greedy 下模型答句不可靠(短/长 ctx
  行为一致 ⇒ 机械等价已证);召回金标待 chat template 正式渲染后立项。
- **OWL_MAX_SEQ 默认 256 → 4096**(apps/server;paged 布局 + carve 对齐
  + v1 smem 修复后解除 256 顶)。
