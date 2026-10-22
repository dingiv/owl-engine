# candle_core::Tensor API → owl 翻译映射表

> 版本:v0.1(2026-09-22,T1 机械普查产物)。
> 参照源:`packages/xinfer/crates/core/src`(**108 个 .rs 文件,只读**)。
> owl 侧现状(2026-09-22 快照):`crates/nn/src/tensor.rs`(Tensor 仅 f32;
> 现有方法 shape/dtype/device_ptr/token/elems/dims/reshape/t2/transpose/narrow/
> broadcast_as/persistent/to_vec)+ `crates/nn/src/ops.rs`(fill_from_host/
> add/mul/softmax/rmsnorm/matmul,全部 f32)+ `crates/backends/iface`(
> Device/Pool/DevBuf/MemValue/PoolKind/BackendError)。
>
> ## 统计口径(必读)
>
> 1. 所有次数 = 在 `crates/core/src` 下 `grep -rEo "\.<方法名>\b" --include="*.rs" | wc -l`
>    的**原始命中数**;静态方法 = `grep -rF "<方法名>(" | wc -l`。
> 2. **原始计数含误报**:`.get`/`.to_vec`/`.zip`/`.to_u32`/`.to_f32`/`.to_bool`/
>    `.device` 等同时命中 `Vec`/`Option`/`HashMap`/标量/`Device` 对象上的同名
>    方法,不全是 Tensor 调用。已人工抽样的标注"含误报"及抽样口径;
>    未抽样者一律标注"未抽样"。
> 3. dtype 次数 = `grep -rEho "DType::[A-Z0-9_]+" | sort | uniq -c`,无歧义。
> 4. 非 Tensor 依赖次数 = `grep -rEho "candle_core::[a-zA-Z_:]+" | sort | uniq -c`。

---

## 一、Tensor 方法面(按原始命中数降序)

| # | 方法(candle 签名形态) | 原始次数 | 语义一句话 | owl 目标 |
|---|---|---:|---|---|
| 1 | `Tensor::get(&self, items: impl Shapes) -> Result<Tensor>` | 440(含误报,未抽样:Vec/HashMap/Option 同名) | 按索引取元素(取标量需再 `to_scalar`) | ops::index 系列(待定义);owl 无标量语义,需定义 `Tensor::get_item` |
| 2 | `Tensor::to_dtype(&self, dtype: DType) -> Result<Tensor>` | 420 | dtype 转换(cast),全库最热调用 | **待定义** `Tensor::to_dtype`(owl 无 F16/BF16 等 dtype,依赖 T1 dtype 泛型化) |
| 3 | `Tensor::contiguous(&self) -> Result<Tensor>` | 288 | 强制行主序连续(布局整理) | **待定义**(owl 目前隐式连续,方法可先恒等+debug_assert) |
| 4 | `Tensor::narrow(&self, dim: Dim, start, len) -> Result<Tensor>` | 249 | 某维上取子段(视图或拷贝) | 已有 `Tensor::narrow`(现拷贝语义) |
| 5 | `Tensor::reshape(&self, shape: S) -> Result<Tensor>` | 239 | 重排形状(元素序不变) | 已有 `Tensor::reshape` |
| 6 | `VarBuilder::get_with_hints_dtype(&self, shape, name, hint, dtype) -> Result<Tensor>` | 202 | **权重装载**(GGUF 按名取张量并 cast 到指定 dtype;是 VarBuilder 方法,非 Tensor) | **待定义** 权重装载通道(xinfer 的 gguf_varbuilder 整体搬运,owl 侧 = 池直连 from_vec_tensor + 量化解码) |
| 7 | `Tensor::max(&self, dim: Dim) / max_all()` | 124(含误报,未抽样:i32/f32 值同名) | 某维或全张量最大 | **待定义**(ops::max 归约) |
| 8 | `Tensor::min(...) / min_all()` | 75(含误报,未抽样) | 最小值归约 | **待定义** ops::min |
| 9 | `Tensor::to_vec1<T>/to_vec2<T>` | 72 + 53 + 19(含误报:Vec 同名 `.to_vec(`;`to_vec1` 基本为 Tensor) | D2H 回读为 host 向量 | 已有 `Tensor::to_vec`(EagerOnly,禁入捕获段) |
| 10 | `Tensor::transpose(&self, a: Dim, b: Dim) -> Result<Tensor>` | 59 | 交换两维 | 已有 `Tensor::transpose` |
| 11 | `Tensor::norm(ord)` | 49(抽样:命中含 `layer_metrics.rs` 的指标名"norm",真 Tensor::norm 远少于 49,未精算) | 向量范数(l2) | **待定义**(rmsnorm 已有专用 kernel;通用 norm 待) |
| 12 | `Tensor::to_u32()` 等标量转换 | 48(抽样:多为 gguf metadata 的 i64→u32,非 Tensor) | 取标量值 | 随 owl 标量读取 API 一并定义(运行期非关键,编译期占位) |
| 13 | `Tensor::matmul(&self, rhs: &Tensor) -> Result<Tensor>` | 37 | 矩阵乘 | 已有 `ops::matmul`(f32/cublas;泛型待) |
| 14 | `Tensor::broadcast_mul/broadcast_add/broadcast_div/broadcast_sub(&self, rhs)` | 34/21/11/1 | 逐元素二元运算 + 右对齐 broadcast | 已有 add/mul(f32,无 broadcast 泛化);broadcast 规则表 = T1 语义表(风险 #1) |
| 15 | `Tensor::to_device(&self, dev: &Device) -> Result<Tensor>` | 33 | 换设备搬运(单卡路径基本恒等) | **待定义**:owl 单卡 → 可降为 debug_assert 恒等;多卡走 Device::map_remote |
| 16 | `Embedding::embed_forward(&self, xs)` | 33 | 词表查表(xinfer 模型层自定义方法,非 candle API) | 已有 `ops` 的 embedding kernel(rope_embed);搬运时映射到 nn embedding |
| 17 | `VarBuilder::get_with_hints(shape, name, hint) -> Result<Tensor>` | 30 | 权重装载(保持源 dtype) | 同 #6 |
| 18 | `Tensor::index_select(&self, index, dim) -> Result<Tensor>` | 29 | 按索引向量取行/列 | **待定义** ops::index_select |
| 19 | `Tensor::dim(i) / rank() / shape() / dtype()` | 264/166/58/335(元数据访问,含误报未抽样) | 形状/秩/dtype 元数据 | 已有 shape/dims/dtype;`dim(i)` 薄封装 |
| 20 | `Tensor::to_f32()/to_bool()` | 15/8(含误报) | 标量/整型转换 | 待定义(随 dtype 泛型化) |
| 21 | `Tensor::gather(&self, indexes, dim) -> Result<Tensor>` | 12 | 高级索引收集(任意维 index 张量) | **待定义** ops::gather |
| 22 | `Tensor::sort(dim) / arg_sort` | 11/4 | 排序/排序索引 | **待定义**(采样链用) |
| 23 | `Tensor::powf(f) / pow` | 20/10 | 幂 | **待定义** |
| 24 | `Tensor::floor() / round()` | 10/7 | 取整 | **待定义** |
| 25 | `Tensor::broadcast_as(shape)` | 10 | 广播到指定形状 | 已有 `Tensor::broadcast_as` |
| 26 | `Attention::apply_rotary_emb_qkv(q,k,v,cos,sin)` | 10 | RoPE 融合(xinfer attention-rs 层自定义方法) | owl rope_f32 kernel 已有;搬运时改接 nn ops |
| 27 | `Tensor::ln() / sqrt() / exp() / tanh() / log()` | 9/40/3/3/6 | 逐元素超越函数 | **待定义** unary ops 族(silu/gelu 已有) |
| 28 | `Tensor::clamp(min,max)` | 9 | 截断 | **待定义** |
| 29 | `Tensor::sqr() / abs() / sum() / mean() / sum_all()` | 7/7/35/1/1 | 逐元素平方/绝对值/归约和/均值 | **待定义**(sum 归约 + unary 族) |
| 30 | `Tensor::repeat(n)` | 7 | 沿新维度复制 | **待定义** |
| 31 | `Tensor::is_contiguous()` | 6 | 布局查询 | **待定义**(owl 恒 true,先恒等) |
| 32 | `Tensor::conv1d(...)` | 6(抽样:全为 config 里的**权重名字符串** `"ple.conv1d.weight"`,真 Tensor::conv1d 调用 = 0) | — | **不搬运**(纯字符串误报) |
| 33 | `Tensor::scatter_add(index, src, dim)` | 7 | 索引散射累加(deepstack/moe mask 用) | **待定义** ops::scatter_add |
| 34 | `Tensor::slice(dim, start, len) / slice_scatter` | 38(slice,含误报未抽样;candle 有 slice_scatter 但 core 内 0 调用) | 维上切片(=narrow 的别名) | 映射到已有 narrow |
| 35 | `Tensor::apply(&Module) / apply_t` | 21 | 经 Module trait 转发(= 各模型 forward 的泛型入口) | **不搬运**:owl 无 Module trait 语义,搬运时展开为直接函数调用 |
| 36 | `Tensor::bias(&bias)` | 28(含误报,未抽样) | 偏置加法 | 映射到 ops::add(broadcast) |
| 37 | `Tensor::from_vec(shape, vec) [静态]` | 142 | 从 host 向量建张量(H2D) | **Pool 工厂** `pool.from_vec_tensor`(裁决 5:分配入口在 Pool) |
| 38 | `Tensor::zeros(shape, dtype, dev) [静态]` | 85 | 零张量 | **Pool 工厂** `pool.zeros_tensor` |
| 39 | `Tensor::cat(&[&Tensor], dim) [静态]` | 74 | 沿维拼接 | **待定义** ops::cat |
| 40 | `Tensor::stack(&[&Tensor], dim) [静态]` | 39 | 沿**新**维堆叠 | **待定义** ops::stack |
| 41 | `Tensor::arange(start, end, step, dtype, dev) [静态]` | 17 | 等差序列 | **待定义**(池工厂 + 填充 kernel) |
| 42 | `Tensor::empty(shape, dtype, dev) [静态]` | 12 | 未初始化张量 | **Pool 工厂** `pool.scratch_tensor`/`zeros_tensor`(owl 无未初始化语义,统一清零) |
| 43 | `Tensor::new(...) [静态]` | 7 | 通用构造(旧 API) | **Pool 工厂** |
| 44 | `Tensor::full(shape, fill, dtype, dev) [静态]` | 6 | 常数张量 | **Pool 工厂** + fill kernel(待定义) |
| 45 | `Tensor::where_cond(cond, a, b) [静态]` | 4 | 条件选择 | **待定义** ops::where_cond |
| 46 | `Tensor::arg_sort / argmax / argmin` | 4/5/— | 排序/极值索引(采样链) | **待定义**(与 #22 同族) |
| 47 | `Tensor::cumsum(dim)` | 2 | 前缀和 | **待定义** |
| 48 | `Tensor::chunk/chunks/chunks_exact(dim, n)` | 12/7/2 | 均分块 | **待定义**(可用 narrow 组合实现,编译期可先映射) |
| 49 | `Tensor::zeros_like / ones_like / clone` | 3/2/1204 | 形状同值克隆 | zeros_like→zeros_tensor;`clone` = Arc 租约克隆(owl Tensor Clone = 租约计数 +1,语义对齐,**注意:owl clone 带强租约语义,与 candle 浅拷贝语义一致**) |
| 50 | `Tensor::device()` | 311(含误报,未抽样:大量为 `let device = ...` 的变量名) | 查询宿主设备 | **待定义** `Tensor::device() -> &D`(owl Tensor 自带 D 类型参数,直接返回) |
| 51 | `Tensor::eq/ne/lt/le/gt/ge` | 4/—/2/—/1/— | 逐元素比较 | **待定义** compare 族 |
| 52 | `Tensor::neg / mul_scalar / add_scalar` | 4/—/— | 负号/标量乘法 | **待定义** |
| 53 | `Tensor::affine(&bias, scale)` | 1 | y = x*scale + bias(融合) | **待定义**(可展开为 mul+bias) |
| 54 | `Tensor::topk` | 0(未命中) | — | 不搬运 |
| 55 | `Tensor::unfold` | 3 | 滑动窗口视图(视觉塔) | **待定义**(低频,运行里程碑再议) |
| 56 | `Tensor::pad` | 1 | 边界填充 | **待定义**(低频) |
| 57 | `Tensor::is_cuda()` | 1 | 设备查询 | 不搬运(owl 单后端,恒真) |
| 58 | `Tensor::concat` | 2(含误报:Vec 同名) | — | 映射到 cat |

## 二、DType 使用面

> 口径:`grep -rEho "DType::[A-Z0-9_]+"` 原始命中(无歧义)。

| DType | 次数 | 典型用途(抽样自真实调用点) |
|---|---:|---|
| F32 | 451 | 主计算路径(对拍/精度敏感段)、logits、KV 元数据 |
| BF16 | 68 | 权重主 dtype(GGUF bf16 档)、激活 |
| U8 | 63 | 量化字节流(Q 系列权重装载)、字节 buffer |
| U32 | 62 | token ids 张量、位置索引 |
| Q8_0 | 35 | GGUF Q8_0 量化权重(moe.rs 量化表) |
| F16 | 28 | fp16 权重/激活档 |
| I64 | 15 | graph 捕获期的 positions/slot_mapping 设备张量(utils/graph.rs:704) |
| F8E8M0 | 12 | FP8 权重档(moe.rs:3139 get_with_hints_dtype FP8 路径;runner 的 SerializableDType 映射) |
| F8E4M3 | 11 | FP8(e4m3)权重档 |
| Q4K/Q6K/Q5K/Q3K/Q2K | 10/7/6/6/6 | GGUF K-quant 量化权重(量化路径) |
| F64 | 5 | 对拍参考计算 |
| Q5_1/Q5_0/Q4_1/Q4_0 | 3/3/3/3 | GGUF 旧 K-quant |
| IQ 系列(IQ4_XS 等 8 种) | 各 2 | GGUF i-quant 权重 |

**owl 缺口结论**:owl-nn 现状仅 F32(Scalar 只实现 f32)。T1 最小 dtype 面 =
F16/BF16/U8/U32/I64(覆盖 94% 触点);Q8/QxK/IQ/F8 系列(量化权重路径)
按 §五 风险登记以类型占位延后。

## 三、非 Tensor 的 candle 依赖面

> 口径:`grep -rEho "candle_core::[a-zA-Z_:]+" | sort | uniq -c` 原始命中。

| 依赖 | 次数 | 搬运替换建议 |
|---|---:|---|
| `candle_core::bail!` | 369 | 引入 **thiserror + `bail` 微宏**:owl-engine 内定义 `bail!` 宏转 `Err(EngineError::Msg)`(保持源码形态最接近,搬运成本最低);错误类型 = owl-iface `BackendError` + 引擎层扩展(建议 `engine::Error` 枚举,带 `#[from] BackendError`) |
| `candle_core::Error::wrap` | 94 | `impl From<BackendError> for engine::Error`(wrap = from) |
| `candle_core::Error::Msg / ::msg / ::debug` | 73/34/6 | engine::Error::Msg(String) 变体 |
| `candle_core::Result` | 21 + 别名 | `pub type Result<T> = std::result::Result<T, engine::Error>` |
| `candle_core::Device::Cpu / ::Cuda` | 27/4 | **不搬运 Cpu 分支**:owl 单 CUDA 后端;`Device::Cpu` 出现点(对拍/初始化)改 owl host 向量 + 池工厂 |
| `candle_core::D::Minus1`(shape 负维枚举) | 27 | 在 engine 内定义 `Dim` 新类型(`enum Dim { Pos(usize), Neg(usize) }`)或统一用 i32 + 归一化 helper |
| `candle_core::Layout / Shape` | 9/5 | owl 用 `Vec<usize>` shape(现状);若搬运代码依赖 Layout 概念,补 `Shape` 薄类型 |
| `candle_core::quantized::gguf_file::Content` | 4+3+2 | **整体搬运** gguf 装载层(纯 host 解析,无 candle 依赖);GGUF→owl 池直连装载通道(T3 前置) |
| `candle_core::quantized::QMatMul / GgmlDType` | 2/2 | 量化 GEMM 走 owl 侧 marlin-ffi 路线;GgmlDType 枚举随 gguf 层搬运 |
| `candle_core::cuda_backend::*`(CudaStorage/CudaDevice/WrapErr/cudarc 直用等) | ~30 处合计 | **不搬运**:owl 的 A4 纪律——FFI 只存在于 owl-cuda;对应能力 = Device/Pool/CaptureSession/DevBuf 契约 |
| `candle_core::Module` | 2 | 不搬运(见方法 #35,展开为直接调用) |
| `candle_core::Tensor`(类型本身) | 14(+ 遍布各文件的 `use`) | owl `nn::Tensor<T, D>`;`Tensor<Tensor>` 泛型参数在 owl 侧 = 具体 dtype 类型 |

## 四、翻译映射优先级建议(供 T1-T3 排期)

1. **P0(编译路径必过)**:dtype 面(F16/BF16/U8/U32/I64)、to_dtype、
   contiguous、narrow、reshape、transpose、broadcast_as、to_vec、matmul、
   add/mul + broadcast、sum/max/min、zeros/from_vec/cat/stack/arange 池工厂。
2. **P1(模型 forward 主体)**:index_select、gather、scatter_add、embed、
   rope、softmax/rmsnorm(已有)、sqr/ln/sqrt/exp/tanh/abs/clamp、compare 族、
   repeat、chunk 组合、bias(=add)、where_cond。
3. **P2(采样/调度层)**:sort/arg_sort/argmax/argmin、cumsum、norm、
   full/empty(=zeros 变体)、pow/powf/floor/round、neg、to_scalar 族。
4. **P3(低频/视觉塔)**:unfold、pad、conv1d(实际 0 调用)、is_contiguous
   (恒等)、device()(直接返回)。
