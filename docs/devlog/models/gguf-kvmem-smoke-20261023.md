# GGUF / GGUF+KVMem 冒烟（2026-10-23）

> 模型：`models/ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`
> 卡：单张 3090 Ti 41:00.0（24G）；f16 KV；无投机。
> 档：
> - `profiles/qwen38_27b-3090-gguf.toml`（baseline，32k，无 KVMem）
> - `profiles/qwen38_27b-3090-gguf-kvmem.toml`（KVMem，max_seq 131k，
>   自动池 48k，workspace 8192，`gqa_aggregation=max`）

## 1. baseline

| prompt | 结果 |
|---:|---|
| 16k 随机词 | 连贯（识别为 word salad） |
| 56k coherent needle（code@~6.2k） | `BLUEBIRD-42` |
| 60k coherent needle（fp8 KV 以装下） | `BLUEBIRD-42` |

## 2. GGUF + KVMem

| prompt | 结果 |
|---:|---|
| 32k coherent needle | `BLUEBIRD-42` |
| 56k coherent needle | `: BLUEBIRD-42`（已召回） |
| 60k coherent needle | **`:`**（退化） |
| 60k + workspace 16384 | **`:`**（仍退化） |

60k 档取证：

```text
final bids ctx=60000 n=256/512
block 207（code@~6.6k）在选中集
target-hash block=207 hash=cfc5a2a59abf133f
```

⇒ 选择器选中了针块，**不是选块/页表失配**；60k baseline（dense）正常，
KVMem 56k 正常、60k 退化。当前判断是 **GGUF + 稀疏视图在 ~60k 的
交互问题**，尚未定谳（需要对比 60k dense stage-in 哈希/逐层 attention）。

## 3. 资源/稳定性

- 60k KVMem：host KvBlock **7.5 GiB**（3751 块），host 总用 9.1 GiB；
- 零 `CUDA_ERROR_ILLEGAL_ADDRESS`、零 Xid、零页失守；
- 60k 时 N6 压力 WARN 较多，但 prefill/decode 完成；
- GGUF+KVMem 32k/56k 可用；**60k 暂列已知退化档，不应作为验收通过**。

## 4. 复现

```bash
./target/release/server -p profiles/qwen38_27b-3090-gguf.toml \
  --set runtime.max_seq=61000 --set pool.kv_fp8=true --log-level info

./target/release/server -p profiles/qwen38_27b-3090-gguf-kvmem.toml \
  --set probes.dflash_probe=true --set probes.view_hash_block=207 --log-level info
```

固定 coherent needle 生成器见 `/tmp/needle_coherent10.py`（90% 自然段落，
code 在约 1/9 处）。
