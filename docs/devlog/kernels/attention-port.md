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
- [x] **K0 reshape_and_cache(2026-09-27 完工;用户裁决同步:直接对标 paged/
  flashinfer 现役方案,弃手写 naive 路线)**
  - 复用源:vendor reshape_and_cache_kernel.cu(classic 布局,与 K1/K2 同树同布局;
    vLLM 上游新树已删 paged 核,vendor 是工作区唯一 paged 源);
  - 产出:`cu/attention/reshape_and_cache.cu`(f16 特化)+ registry
    `vllm_reshape_and_cache_f16` + GPU 对拍(恒等分页/跳块物理槽/padding 跳写/
    空洞保持零,位型全绿);
  - **nvrtc 约束清单(K0 实测,后续 port 逐条对齐)**:
    ① stdint.h 不可开 → long long 内建替;
    ② 模板需特化为 extern "C" 入口(nvrtc 按名取核);
    ③ 槽表/索引量 → f32 过线(契约 5);
    ④ 末位 T = 解释器新分配输出块(槽序契约 4)→ 就地写核需哑尾参;
    ⑤ 发射 grid 显式(哨兵自动 1D 会越界读表);
  - K1 真风险待验:pagEDattention.cuh 模板宏体系(652 行)在 nvrtc 下的
    编译行为 —— ①-⑤ 之外的新增约束在此暴露。
- [x] **K1/K2 + prefill 批量 port(2026-09-27 完工,子代理施工;验收口径 =
  cargo check 编译通过,不要求执行/测试 —— 用户裁决)**
  - 产出:`pagedattention_f16.cu`(6 核 = v1/v2/v2_reduce × hd128/hd256)、
    `prefill_paged_attn_f16.cu`(2 核 = chunked prefill opt × hd128/hd256),
    共 9 个 extern "C" 核 + K0 的 reshape_and_cache,全部括号配平 0;
  - registry 9 条目(hd 双档后缀;prefill 真实函数名
    `vllm_chunked_prefill_paged_attn_opt_f16_*`);registry 签名自测过;
  - **nvrtc 风险清单(静态不可证,真机对拍时收口)**:
    ① pagedattention.cuh 模板宏体系的扁平化特化在 nvrtc 下的实际编译行为
       未验;若挂 → 备选 = build.rs nvcc 预编 PTX(marlin .a 先例);
    ② PTX 内联 asm(cvt.f32.f16 / shfl_xor_sync / tanh.approx.f32)在
       compute_86 下的展开;
    ③ extern shared 动态 smem:v1/v2 >48KB 需 cudaFuncSetAttribute,owl
       发射器暂无此通道 → **ctx 上限 ≈12k @HD256**(超出列 launcher 扩展挂账);
    ④ f32 表过线精度前提:槽/块号 <2²⁴(消费卡池规模内恒成立);
    ⑤ softscapping/fast_tanh 路径 owl 未消费(默认 1.0 直通),未验证。
- [x] **执行验证(2026-09-27 真机;OWL_TEST_DEVICE 实测)**:三家族 9 核全绿 ——
  K0 位型全绿;K1 v1(hd128)nvrtc 编译 + classic 布局 + 数值 2e-2 全过;
  K2 v2+reduce 链同过;prefill chunked 批核因果语义(查询 token t 看
  [0,t])对拍过。**扁平化 port 路线(nvrtc 通道)正式成立**,预编 PTX 备选
  不启用。port 修复实录:v1 wrapper 漏传 use_alibi;核内 block_table 指
  针型未随契约⑤改 f32;prefill helper 在 namespace vllm 而核体在全局域 →
  `using namespace vllm` 桥接 + fast_tanh_opt 补落(vendor 逐字)。
  未验:hd256 双档(仅 smoke hd128)、v2 多分片(>512 ctx)、softscapping≠1。
- [ ] K3 GDN 全家(已在 owl text/gdn.cu 独立落地;此条转对照审计)
- [ ] K4 silu_and_mul / sort / fast_topk(采样链)
- [ ] flashinfer adapter 线(flashinfer_common.cuh 依赖链,预编 PTX 备选)
- [x] **v1 动态 smem 越界定谳与修复(2026-09-27;E1 4k e2e 排查战果)**
  - 现象:引擎 4k e2e decode 图回放即 `CUDA_ERROR_ILLEGAL_ADDRESS`
    (eager 全绿;seq≤512 幸存,1024+ 必炸;崩溃点随负载漂移);
  - 归因:`vllm_paged_attention_v1` 动态 smem = **logits f32 × 上下文
    token 数**(核内逐上下文 token 一格),而发射包装只传固定 2048B
    (= 契约第二项"固定地板",漏了第一项"上下文项")→ **kv_len > 512
    必然 smem 越界**;2048B 预算被 capture 烘焙进 decode 图,eager 冒烟
    (小 ctx)永远踩不到 → 单测全绿、4k 必炸;
  - 修复:发射侧按 .cu 头注契约计算
    `smem = max(ceil(max_ctx,BLOCK)·BLOCK·4B, (NUM_WARPS/2)·HD·4B)`;
    上限 ≈ ctx 12k @HD256(>48KB 需 cudaFuncSetAttribute,launcher
    扩展挂账不变);
  - **教训:契约公式写在 .cu 头注 ≠ 发射方真的按公式算;包装层魔数
    (2048 "诊断值")带病入库的代价 = 全链 4k 崩溃。发射参数必须由
    契约公式推导,禁手抄常量。**
- [x] **OWL_LAUNCH_SYNC 逐发射归因开关(2026-09-27;诊断基建)**:env 门控
  的逐 launch/gemm/alloc/htod/graph 同步,sticky CUDA 错误就地归因到
  具体核/操作(本次 4k 排查即靠它定位 graph 回放)。保留为常驻诊断面。
- [x] **防再发守卫(2026-09-27;两层)**:
  1. **契约单源函数** `paged_v1_smem(hd, nb, page)`(layers/attention.rs,
     与 v1_name 同址):smem 公式唯一推导点 + 纯数学单测(地板项/上下文项
     双主导锚);包装层禁手写 smem 常量;
  2. **server 硬顶守卫**(launch.rs L2.5):动态 smem > 设备 opt-in 上限
     (boot 查一次缓存)→ 结构化拒绝,不再依赖驱动泛化 INVALID_VALUE;
     开销 = 一次字段比较(~ns),decode 热路径(图回放)不经此处,
     捕获期发射被覆盖一次。两条均非热路径税。
