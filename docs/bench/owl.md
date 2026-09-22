# owl 计速汇编(一期 nn 链路)

> 2026-09-22。引擎:**owl**(M1 前哨,graph 治理未接入;T0-T4 一期工程产物)。
> 硬件:单卡 RTX 3090 Ti(3090_dual 槽位 GPU-e565c505,CUDA 13.2,sm86)。
> 口径:std 单点(链尺寸 M=64 K=128 N=64,f32;ITER=200 取 median,每轮
> `ctx.synchronize`)。**一期为单档图**(单 shape、单 bs,多档/池治理属 M1)。

## 一、链路定义

```
mm_out   = W[64×128] @ A[128×64]        (cublas Sgemm,workspace 预钉 32MiB)
add_out  = mm_out + bias                (owl_add_f32,ported candle-kernel)
silu_out = silu(add_out)                (owl_silu_f32,ported)
graph_out= rmsnorm(silu_out,α,ε=1e-5)   (owl_rmsnorm_f32,ported)
```

## 二、数据

| 路径 | median (ms) | 备注 |
|---|---|---|
| eager 全链(4 算子) | **0.0232** | 4 次 launch + 末尾同步 |
| matmul(eager)+ 图(add/silu/rmsnorm) | 0.0234 | 1 次 cublas + 1 次 graph.launch;**多一道 legacy→non-blocking 显式栅栏** |
| graph 相对 eager | **-0.9%** | 见 §三判读 |

正确性:捕获 replay ×4 与 eager 输出最大偏差 **0.000e0**(逐位一致);
eager 链与 CPU f64 参考:matmul 2.1e-6 / silu 2.2e-6 / rmsnorm 7.7e-7。

## 三、判读(诚实版)

1. **本链尺寸下 graph 无收益(-0.9%)**:链总时长 ~23µs 中 cublas GEMM
   占大头,3 个微 kernel 的 launch 开销(各 ~2-3µs)本就占比小;且 graph
   路径多一道显式栅栏(non-blocking 与 legacy 无隐式同步,重构正确性
   所需),把节省吃平。**结论:一期验证的是"捕获/重放/一致性/账本零漂移"
   的机制正确性,不是加速**——提速预期留给 M1 后(图治理接入 + 大链 +
   cublas 入图)。
2. **结构性发现(本日最有价值)**:cudarc `default_stream()` = legacy
   NULL stream,**不可捕获**(begin_capture 直接
   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`)。T2/T3 的 cublas 与 OpsCtx
   均绑 legacy 流 → **cublas 入图被 block**,须 M1 改绑显式
   non-blocking stream(T2 移交事项的根源实锤)。一期 example 的捕获段
   用 T0 Kernels 直发自建流绕行(未改任何既有源码)。
3. **A5 账本零漂移**:两 example 生命周期末 `bytes_alive` 精确回到基线
   (P 阶段分配的 32.2 MiB 全程在册,E 阶段与 replay 循环零变动)——
   "严格显存预期"在一期链路上成立。

## 四、M1 交接清单(由本实验确认)

- [ ] cublas 改绑 non-blocking stream(T2 遗留,捕获 block 的唯一根因)
- [ ] OpsCtx/Kernels 流选择显式化(default_stream 遗留不可捕获)
- [ ] 图治理(A1):捕获前 warmup 制度化、AUTO_FREE_ON_LAUNCH 与池治理联动
- [ ] 多档 shape(A1.4 预检)与 BAR1 维度对账(A2.8 PeerShared 池)

## 五、复现

```bash
cargo run --release -p owl-nn --example 01_eager
cargo run --release -p owl-nn --example 02_capture
```
