# TP2 host-relay 通信通道(无 P2P 驱动兜底)

> 2026-11-04 立项并当日落地。目标:**没有 P2P 驱动的机器也能跑 TP2**。
> 不是替代现役 P2P;是同一套 AR 协议的第二条传输腿。
>
> 2026-11-04 晚更新:**CE D2H push + 本地 host reduce + f8 紧凑载荷** 已落地;
> 27B release/std 相对 P2P 的差距压到 **prefill 10.8% / decode 15.9% /
> TTFT 11.8%**,达到 30% 目标。

## 1. 结论(当前实现)

- 新配置键:`[parallel] tp_comm = "p2p" | "host"`(缺省 `p2p`)。
- `host` = `cudaHostAlloc(MAPPED|PORTABLE)` 的 pinned host staging;
  rank0 分配两段,rank1 `host_attach` 同一段。
- **数据腿 = CE push**:
  - 生产者用 copy engine 把 payload **D2H 写到对端(消费者)的 host 槽**;
  - 消费者 `ar_reduce` 读**本端** host 槽累加;
  - 双方都不再让“对端 GPU 直接读本端 CE 写出的 host”。
- host staging 固定 **1 slot / rank**(32MiB + 64B header):
  CE 目标偏移必须主机侧静态可知,多槽的 `produced % slots` 只有设备知道。
- 大包(≥256KiB、F16/BF16)走 **f8 紧凑载荷**:
  - 本端 device scratch 上先量化(payload → fp8 + scales,约 0.515×);
  - CE 只搬紧凑载荷到对端 host;
  - reduce 读本地 host 并反量化。
  - 量化复用现役 `ar_stage_f8` 核 + 一个 **dummy header**(64B device block),
    不触碰真实 `produced/consumed`。
- 小包(decode)走 raw CE push;`tp_ar_f8` 仍由配置控制。
- **不调用** `cudaDeviceCanAccessPeer` / `cuMemSetAccess(peer)` / BAR1,
  因此无 P2P 驱动的消费卡环境可直接起。
- **保留 CUDA Graph**:host 段/device scratch 地址固定,header 在 host 内存;
  G3 门实测 capture → replay 两次逐位一致。
- 当前 `host` 只接 `tp_ar = "v1"`;`fused/ar2` 在配置门响亮拒启。

## 2. 参考实现(工作区已有)

| 实现 | 机制 | 我们取什么 |
|---|---|---|
| `llama.cpp/ggml/src/ggml-cuda/allreduce.cu` | 小包:单核把本端 payload 写 pinned host,host-memory arrival token 轮询对端,读回求和;大包:CE D2H/H2D + 事件 + add 核 | **pinned host + 设备内轮询**的可行性;无 P2P 消费卡的实证 |
| `repos/ninfer_duo_3080/src/ops/common/allreduce.cu` | `cudaMemcpyAsync(D2D)` 走 UVA;有 peer access 时直传,否则 CUDA 透明 host-stage;`cudaMemcpyPeerAsync` 被 capture 拒 | 确认“无 P2P 时 host 中转是 CUDA 支持的合法 transport” |
| vLLM / SGLang `custom_all_reduce` | CUDA IPC + peer access(P2P) | 不能作为无 P2P 兜底;保留为 P2P 参照 |

CE push/f8 仍复用现役 stage/reduce 核与 header 协议;新增的是
**数据搬运腿**(`Command::CopyBlock`)和 **f8 device scratch**。

> **与 charter A2.2 的关系**:本通道是“pinned host-mapped 段 + GPU 核/CE 读写”,
> 不是 CPU 中转/`cudaMemcpy` 驱动的 host relay,所以它仍是 **InGraph**
> (A2.2 的 `SegmentBoundary` 留给将来的 CPU 归约腿)。

## 3. 机制

### 3.1 两段 host staging(单 slot)

```text
rank0: host_alloc(H0) -> owner 账
       host_alloc(H1) -> owner 账
rank1: host_attach(H1) -> attach 账(own)
       host_attach(H0) -> attach 账(peer)

rank0: own=H0, peer=H1
rank1: own=H1, peer=H0
```

- `HostAlloc` = `cudaHostAlloc(MAPPED|PORTABLE)` + `cuMemHostGetDevicePointer_v2`;
  返回 `(块 id, host 地址)`。
- `HostAttach` = 同进程/UVA 下把同一 host 地址登记成第二份块账,不拥有、不 free。
- `HostRelease`:owner `cudaFreeHost`;attach 只销账;图引用租约钉住时**拒收**。
- 生命周期顺序:先收 rank1 attach,再收 rank0 owner(`shutdown`/`shutdown_tp`
  已按 host 模式反转顺序)。
- staging = **`1 × 32MiB + 64B header` / rank**(每进程 64MiB pinned + 64B)。

### 3.2 CE push(raw / decode)

生产者 rank A:

```text
CopyBlock(payload[0..n] -> A.peer(H_B)[HDR])
→ ar_stage(bytes=16, data=A.peer, own=A.H_A, peer=A.H_B)
  只做 16B 幂等写 + produced 发布(真实数据已由 CE 搬走)
```

消费者 rank B:

```text
ar_reduce(out=payload, data=B.own(H_B), own=B.H_B, peer=B.H_A)
  等 A.produced;读 B 自己的 host 槽;发布 B.consumed
```

- stage 的 credit/尾部 rendezvous 仍由现役 `ar_stage` 核执行;
- `CopyBlock` 与 stage/reduce 同 COMPUTE 流,顺序由流 FIFO 保证;
- 图内 `CopyBlock` 是合法 memcpy 节点(`handle_copy_block` 已登记图引用租约);
- 单 slot ⇒ CE 目标偏移固定 `HDR`(64),无需知道 `produced` 奇偶。

### 3.3 f8 紧凑载荷(prefill)

```text
payload(本端 device)
  → ar_stage_f8(dummy header, data=本端 device scratch)
     量化写 [q n_elems B][scales ng×f16](~0.515×)
  → CopyBlock(scratch[HDR..] -> peer host[HDR], compact_len)
  → ar_stage(bytes=16, data=peer, local=scratch F16 view)
     真实 produced 发布
  → peer ar_reduce_f8(data=own host)
     读 q/scales 反量化 + 累加 + 真实 consumed 发布
```

- device scratch = `32MiB + 64B`/rank;`dummy header` = 64B device block
  (`slot_bytes=0`, `n_slots=1`),只服务量化核的协议形状。
- `host_f8_compact_bytes` 与 `.cu` 的 `OWL_AR_FP8_SC_OFF` 同式:
  `sc_off=(n+1)&~1`,`ng=ceil(n/64)`,`total=sc_off+ng*2`。
- f8 只用于 `n_bytes ≥ 256KiB` 且 dtype F16/BF16;decode/小包走 raw。
- f8 的数值语义与现役 P2P f8 腿一致(对称量化 + 两 rank 同一对值相加)。

### 3.4 图安全

- 图捕获期 `HostAlloc/HostAttach/HostRelease` 由 `capture_gate` 结构化拒;
  分配/attach 必须在建图前完成。
- host 块/device scratch 在 `block_ptr` 里登记图引用租约;图存活时
  `host_release` 拒收;device scratch 是 raw `GpuClient` 分配,收尾用 raw `free`。
- 游标是相对量(produced/consumed),不烘绝对值 ⇒ replay 不需 host 复位。

## 4. 配置与运行

```toml
[parallel]
mode = "tensor"
device = ["GPU-…", "GPU-…"]
tp_comm = "host"   # p2p(缺省)| host
tp_ar = "v1"       # host 当前只支持 v1
tp_ar_f8 = true    # host: 大包走 CE+f8;false = raw CE
```

无 P2P 的机器可直接用;有 P2P 的机器若要 A/B:

```bash
# host 腿(f8 CE)
./target/release/server -p profiles/qwen38_27b-3080_tp2.toml \
  --set parallel.tp_comm=host --set parallel.tp_ar=v1

# host raw CE(关 f8)
./target/release/server -p profiles/qwen38_27b-3080_tp2.toml \
  --set parallel.tp_comm=host --set parallel.tp_ar=v1 --set parallel.tp_ar_f8=false

# P2P 回归(缺省)
./target/release/server -p profiles/qwen38_27b-3080_tp2.toml
```

`--check` 已覆盖:host 单卡拒、host×fused/ar2 拒、host×v1 通过。

## 5. 门与实测

### 5.1 低层门 `crates/backends/cuda/tests/host_relay_gate.rs`

两张 3080(无需 P2P),`host_alloc`/`host_attach` + 现役 stage/reduce:

- G1 5 case:4B F32 / 10KB F16 / 1MB F32 / 1.31MB F16(27B chunk128)/
  8MB BF16(单 slot 满格),逐位对 host 参考;
- G2 同输入两跑逐位一致;
- G3 capture → replay 两次逐位一致;图存活时 `host_release` 被租约拒收;
- G4 CE push(2 slot 形状)正确性;
- G5(仅计时)8MiB:CE copy 0.68ms / CE push 1.47ms / SM push 3.11ms。
- 结果:**全绿,约 4.4s**。

运行:

```bash
CUDA_VISIBLE_DEVICES='<3080-uuid>,<3080-uuid>' \
OWL_PEER_TEST_DEV_A=0 OWL_PEER_TEST_DEV_B=1 \
  cargo test -p owl-cuda --test host_relay_gate -- --nocapture --test-threads=1
```

### 5.2 TP2 等价门 `tp_g8_gate`

`OWL_TP_COMM_HOST=1` 走 host 腿;0.8B greedy 16 token:

| 卡对 | 结果 |
|---|---|
| 双 3080(20G) | L1 末步 logits **0 项不同**;16/16 token 与 TP1 一致 |
| 双 3090 Ti | L1 **0 项不同**;16/16 token 一致 |

同一门不设 `OWL_TP_COMM_HOST` 走 P2P,仍全绿(回归)。

> 0.8B 默认 prompt 只有 57 token,prefill 包 <256KiB ⇒ **不覆盖 f8 腿**;
> f8 腿由 27B release/std(15/15 成功、无 AR err/Xid)覆盖。
> 若要在 0.8B 门强制放大包,可设 `OWL_TP_PREFILL_CHUNK`,但当前 prompt 长度
> 仍可能不足以过阈值 —— 专门 f8 低位门尚未单独立。

### 5.3 带宽拆解

同机同对 3080 的 CUDA 探针(5 MiB):

```text
SM 直写 host-mapped        1092.65 us    4.80 GB/s   ← 旧 SM stage 的瓶颈
SM 读 host-mapped           405.02 us   12.94 GB/s   ← reduce 不慢
copy engine D2H->pinned     398.11 us   13.17 GB/s
copy engine D2D->hostmapped 398.11 us   13.17 GB/s
copy engine H2D<-pinned     401.91 us   13.04 GB/s
```

低层 CE push 计时(8MiB,5 轮):

```text
CE 单搬                     0.68 ms
CE push + local-host reduce 1.47 ms
SM push + local-host reduce 3.11 ms
```

⇒ 把 stage 从 SM 写 host(4.8GB/s)换成 CE(13GB/s),AR 约快 2.1×;
f8 再把 CE 与 host reduce 的字节数减半。

同机同对 3080 的 **VMM 映射探针**(`tools/fixture/vmm-peer-probe.cu`,5 MiB,
与 owl `peer_import` 同款 PINNED+POSIX_FD+`cuMemSetAccess(PROT_READ)`):

```text
RT-read-1way      405.73 us   12.92 GB/s   (cudaMalloc + enablePeerAccess)
VMM-read-1way     435.96 us   12.03 GB/s   (owl 同款 VMM)
VMM-read+fence    444.56 us   11.79 GB/s   (owl ar_reduce 同款 fence)
VMM-read-2way     551.87 us    9.50 GB/s
```

⇒ **VMM 导入的别名确实是直连 PCIe P2P**,不是 host-staged:
12.03 GB/s 与 runtime peer 12.92 GB/s 同档。

### 5.4 release std 五点三轮 A/B(2026-11-04 晚,最终)

同 release 二进制、同 profile、同 spec d7 / w4a8 / int8 KV / 单并发;
P2P 用生产档;host 强制 `tp_comm=host` + `tp_ar=v1` + `tp_ar_f8=true`。
`llm_speedtest --flavor std`(512/1024/2048/4096/8192 × 3,输出 256)。

| 长度 | prefill P2P | prefill host f8 | host 慢 | decode P2P | decode host f8 | host 慢 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2099.6 | 1853.8 | **11.7%** | 252.7 | 194.5 | **23.0%** |
| 1024 | 2474.8 | 2158.3 | **12.8%** | 188.4 | 172.2 | **8.6%** |
| 2048 | 2706.0 | 2471.0 | **8.7%** | 186.5 | 190.1 | **-1.9%** |
| 4096 | 2691.9 | 2418.7 | **10.1%** | 133.7 | 110.5 | **17.3%** |
| 8192 | 2825.6 | 2518.5 | **10.9%** | 129.1 | 81.0 | **37.3%** |

汇总:

| 指标 | P2P | host f8 | host 慢 |
|---|---:|---:|---:|
| prefill 均值 | **2559.5 t/s** | 2284.1 t/s | **10.8%** |
| decode 均值 | **178.1 t/s** | 149.7 t/s | **15.9%** |
| TTFT 均值 | **1169.7 ms** | 1307.5 ms | **11.8%** |

> decode 逐点受 spec 接受率/降级态影响呈双峰(75~201 t/s);配对汇总
> 仍慢 15.9%,但绝对差距已从 33.6 t/s 缩到 28.4 t/s。prefill/TTFT 均
> **达到 30% 目标**。f8 只影响大包 prefill;decode 仍走 raw CE。
> 15/15 成功,无 Xid,无 AR err,停服无 host-drop。

旧 SM stage 基线(仅历史对照):prefill 1739.8 / decode 144.5 / TTFT 1737.6,
分别慢 32.0% / 18.8% / 48.5%。

## 6. 历史失败:DMA stage 直读(未采用,勿原样重做)

### 6.1 做过什么

曾实现:

```text
host block 检测 → ar_host_wait(真 credit)
→ memcpy_dtod_async(payload → own+HDR)(copy engine)
→ ar_host_signal(fence + produced++)
→ 原 ar_reduce 读**对端** host
```

还试过 `stage_seq`、`res.sync()`、capture 期回退 SM、f8 强制关闭。

### 6.2 失败判据

0.8B TP2 host:

```text
rank0 vs rank1 末步 logits:248036 项不同
staging: produced=1536 consumed=768 err=0(无超时)
```

判断:**对端 GPU 直接读 copy-engine 写出的 host** 的可见性/时序没有闭合。
低层单/少次 AR 能过,128 AR/图不能过。

### 6.3 与当前 CE push 的区别

失败档的数据腿是“CE 写本端 host → 对端 GPU 跨卡读本端 host”。
当前档改成 **CE 写对端 host → 对端 GPU 读自己的 host 槽**(push):
读者是 host 段的 owner,CE 写与本地 host 读在同一段内存上;
再叠 f8 后 27B std 15/15 成功、无 AR err。**这不是把失败档原样贴回**。

当前保留的历史候选(未做):
- H2D pull(CE D2H + CE H2D + device add);
- eager prefill 的 host sync 版 DMA;
- CPU relay(`SegmentBoundary`,最稳但最慢)。

## 7. 改动面

| 文件 | 内容 |
|---|---|
| `crates/backends/iface/src/command.rs` | `Command::CopyBlock`(设备块→块字节拷贝,可进批/图) |
| `crates/backends/cuda/src/command.rs` | `HostAlloc/HostAttach/HostRelease` + `CopyBlock` 信封映射 |
| `crates/backends/cuda/src/state.rs` | `Block::HostMapped`、host 分配/attach/release、host 账、Drop 兜底 |
| `crates/backends/cuda/src/server/host.rs` | 三个 host handler |
| `crates/backends/cuda/src/server/{mod,capture_gate,trace_map}.rs` | 派发/捕获门/发射账 |
| `crates/backends/cuda/src/gpu_client.rs` | 客户端门面 |
| `crates/backends/cuda/src/ffi.rs` | `cuMemHostGetDevicePointer_v2` + MAPPED/PORTABLE flags(A4 登记) |
| `crates/engine/src/parallel/tp.rs` | `TpOpts.comm`、host 握手、单槽、f8 scratch/dummy、shutdown 顺序 |
| `crates/models/src/{module.rs,interpreters/eval.rs}` | `TpCtx.host_own/host_quant/host_dummy`;CE push/f8 发射 |
| `crates/shared/src/config/mod.rs` | `TpCommMode`、`[parallel] tp_comm`、校验 |
| `apps/server/src/{config,main}.rs` | 配置搬运/TpOpts;executor 收尾显式 `shutdown_tp` |
| `profiles/full.toml` | 新键缺省 p2p |
| `crates/backends/cuda/tests/host_relay_gate.rs` | 低层门 G1~G5 |
| `docs/devlog/comm/host-relay-2026-11-04.md` | 本文 |

## 8. 限制与下一步

1. **只支持 `tp_ar = "v1"`**;host×fused/ar2 暂拒。
2. **单 slot**:CE 目标偏移必须主机侧静态可知;多槽的 `produced % slots`
   只有设备知道。要上多槽必须把 slot 选择也做成 host 侧可静态展开的序。
3. **host pinned 限额**:32MiB/rank + 64B header = 64MiB/进程;
   低 `ulimit -l` 环境会 `host_alloc` 响亮失败。
4. **f8 只覆盖大包**:<256KiB 或 F32 走 raw CE。decode 的 host 差距
   (15.9%)主要是协议/图节点/启动税,不是带宽。
5. **f8 低位门尚未单独立**:0.8B 默认 prompt 过不了 256KiB 阈值;
   27B std 已覆盖真实 f8 数据路径。后续可加一个低层 f8 CE 门或加大
   0.8B prompt。
6. **CPU relay** 是另一条后续腿:D2H→CPU sum→H2D,`SegmentBoundary`,
   不如图路;本次不做。
7. **27B std 已过**:15/15,无 Xid;停服无 host-drop;release 二进制
   已重建。
