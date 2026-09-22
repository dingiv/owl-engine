# attention-rs kernel port 专项(立项)

> 版本:v0.1(2026-09-22)。裁决依据:graph-model-seam.md §五·3
> (保留 attention-rs 裸 kernel,只 port kernel 不 port 运行时;
> flashinfer 后续引入)。参照源:`packages/xinfer/vendor/attention.rs`
> (rev c0f19f2,冻结只读)。

## 一、资产盘点(实测)

### Rust 面(逻辑/分派/元数据)
| 文件 | 行数 | 内容 | 依赖 candle? |
|---|---:|---|---|
| paged_attention.rs | 1383 | PagedAttention 结构 + 分派(v1/v2)+ 元数据 | **是**(candle Tensor/CudaDType,~10 处 DevicePtr) |
| gdn.rs | 2748 | GDN 线性注意力封装 | 是 |
| mamba_cache.rs | 1225 | Mamba/GDN 状态缓存 | 是 |
| cache.rs | 543 | KV cache 工具 | 是 |
| silu_and_mul.rs / sort.rs | 340 | 融合激活 / 排序 | 是 |

### CUDA 面(kernels/src/,nvrtc/编译产物)
| 文件 | 行数 | 说明 |
|---|---:|---|
| paged_attention_v1/v2.cu + pagedattention.cuh | ~3k | vLLM 系 paged attention(模板宏展开 dtype)|
| reshape_and_cache_kernel.cu | ~200 | KV 写入缓存块(decode 必经)|
| gdn.cu | 2173 | GDN 全核 |
| ffi.rs | 4749 | 全 kernel 的 driver-FFI 绑定(load 直通)|
| mask.cu / silu_and_mul.cu / sort.cu / fast_topk.cu | ~1.5k | 配套 |

**port 总面 ≈ 9k 行 Rust + 5k 行 CUDA**;decode FULL 图最小集 =
paged_attention_v1(or v2)+ reshape_and_cache + 配套 header,约
2.5k CUDA + 1.5k Rust 绑定。

## 二、技术路线(nvrtc,对齐 owl-nn 既有设施)

owl-nn 已有 nvrtc 设施(`compile_ptx_with_opts`,owl_nn_kernels.cu 模式)。
port 方式:

1. **.cu/.cuh 文件级拷贝**进 `crates/kernels/cu/attention/`(A4:文件头
   标注出处 rev + 所有权转移;不改 kernel 语义);
2. **dtype 收窄**:vLLM 模板宏按 owl 需要只实例化 f16/bf16 两档
   (f32 调试档可选)——nvrtc 编译时间与 PTX 体积的主要来源就是
   dtype 展开,砍到 2 档;
3. **绑定层**:仿 owl-cuda ffi 白名单(准入铁律:load 直通、零分配
   零懒状态),wrapper 放 `crates/kernels/src/attention.rs`;
   参数一律裸指针 + 元数据 host 结构(A1.5:动态量 device 张量);
4. **可捕获性契约**:port 时逐 kernel 核对——无 cudaMalloc/无同步/
   无 D2H/无 host 分支(attention-rs 裸 kernel 历来满足,核对是
   形式化留证,进 CaptureSafe 声明);
5. **candle 依赖剥离**:paged_attention.rs 里 ~10 处 DevicePtr/candle
   类型改 owl DynTensor 裸指针 + shape;分派逻辑原样。

## 三、阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| K0 | reshape_and_cache 单 kernel port(最简,验证 .cu→nvrtc→绑定→真机全链)| 合成 KV 块写入对拍 host 参考 |
| K1 | paged_attention_v1(f16/bf16)+ mask/scale 元数据 | 已知 Q/KV 合成输入对拍(数值容差按 vLLM 测试口径)|
| K2 | paged_attention_v2(大 bs 分片归约变体)| 同上 |
| K3 | GDN 全家(gdn.cu 2173 + mamba_cache 语义)| qwen3_5 hybrid 层接入 |
| K4 | silu_and_mul / sort / fast_topk(采样链)| 对拍 |

## 四、风险

1. pagedattention.cuh 的模板宏在 nvrtc 下的编译时长/编译器兼容
   (vLLM 原生走 nvcc 离线编译;nvrtc 对 `#include` 链和 C++ 模板
   支持有限——K0 先验证此风险,若 nvrtc 不可行,备选 = 预编译
   PTX/cubin 入库(构建期 nvcc,运行期 load),工程模式与
   trtllm_cubin_loader.rs 同款);
2. v2 的 split-k 归约依赖 workspace 临时缓冲——须走 owl scratch 池
   (A5.2),port 时把 workspace 参数显式化;
3. bf16 数学在 sm86(GeForce)无加速,性能口径以 f16 为主。

## 五、状态

- [x] 盘点 + 路线设计(本文档)
- [ ] K0 reshape_and_cache(下一步,验证 nvrtc 风险)
- [ ] K1-K4
