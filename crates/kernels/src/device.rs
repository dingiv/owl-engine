//! 服务端面执行原语 —— [`DeviceRes`] 资源面(server 实现,仅治理状态)
//! × [`Exec`] 执行引擎(kernels 实现,全部机械)。
//!
//! 切分判据(operator-contract 设计 v0.4 §4.4)= **状态 vs 机械**:
//! - 状态留 server:块表/账本(A5.2)、捕获状态机与图治理(A1.6/A1.7)、
//!   三流模型 —— DeviceRes 五方法即全部,server 侧再无执行代码;
//! 机械下沉 kernels:资产(cubin)模块表、launch 组装、
//!   捕获窗守卫与登记的机械部分、staging 不变盒、体检核发射。
//!
//! **零运行期编译**(2026-10-13 裁决,NVRTC 已退役):设备码一律构建期
//! `build.rs` cubin 资产([`Exec::load_aot`] 是唯一装载入口);源不在
//! `include_str!` 闭包内 = 装载即拒(旧 `load_nvrtc` 已删,勿复活)。
//!
//! P3 无例外:资源获取/状态读取/副作用一律 `Result`;裸返回仅限纯且全
//! 函数。零 `()`、零 log-and-continue、零 ack 侧信道。
//!
//! feature = "device"(cudarc driver/nvrtc;独立于已退役的 cuda_ops 预编链;
//! `nvrtc` feature 仅提供 `Ptx::from_binary` 数据类型包装,零编译)。

use crate::contract::{FieldStats, InvariantBox, Law, OpError, OpId, Stage};
/// 发射参数值(权威定义在 [`crate::contract`] —— 自检面(零 feature 门)也要用它)
pub use crate::contract::LaunchVal;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

/// 设备指针(CUDA 虚地址;由 [`DeviceRes::resolve`] 从块句柄解析)
pub type DevPtr = u64;

/// `&mut dyn DeviceRes` 自身也满足 DeviceRes(转发;RunEnv 拆借后直用)
impl DeviceRes for &mut dyn DeviceRes {
    fn context(&self) -> Result<Arc<CudaContext>, OpError> {
        (**self).context()
    }
    fn stream(&self) -> Result<Arc<CudaStream>, OpError> {
        (**self).stream()
    }
    fn resolve(&self, b: &crate::contract::Bytes) -> Result<DevPtr, OpError> {
        (**self).resolve(b)
    }
    fn alloc(&mut self, bytes: usize, tag: &'static str) -> Result<ScratchBuf, OpError> {
        (**self).alloc(bytes, tag)
    }
    fn capturing(&self) -> Result<bool, OpError> {
        (**self).capturing()
    }
    fn record_capture(&mut self, note: LaunchNote) -> Result<(), OpError> {
        (**self).record_capture(note)
    }
    fn upload(&mut self, dst: DevPtr, src: &[u8]) -> Result<(), OpError> {
        (**self).upload(dst, src)
    }
    fn device_ordinal(&self) -> Result<i32, OpError> {
        (**self).device_ordinal()
    }
    fn download(&mut self, src: DevPtr, dst: &mut [u8]) -> Result<(), OpError> {
        (**self).download(src, dst)
    }
    fn alloc_block(&mut self, bytes: usize, tag: &'static str) -> Result<crate::contract::Bytes, OpError> {
        (**self).alloc_block(bytes, tag)
    }
    fn upload_block(&mut self, b: &crate::contract::Bytes, src: &[u8]) -> Result<(), OpError> {
        (**self).upload_block(b, src)
    }
}

/// scratch 分配回执(ptr + 语义尺寸;生命周期归 server 账本,A1.7 租约)
#[derive(Debug, Clone, Copy)]
pub struct ScratchBuf {
    pub ptr: DevPtr,
    pub bytes: usize,
}

/// 捕获登记原材料(A1.6 机械面;server 侧组装进自己的 CaptureRecord)
#[derive(Debug, Clone)]
pub struct LaunchNote {
    pub op: String,
    pub kernel: &'static str,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

/// 资源面:server 是唯一实现者;只含治理状态,零执行逻辑。
pub trait DeviceRes {
    /// CUDA 上下文(AOT cubin 装载 / nvrtc 编译产物挂靠点)
    fn context(&self) -> Result<Arc<CudaContext>, OpError>;
    /// 计算流(三流模型属 server;COMPUTE 流)
    fn stream(&self) -> Result<Arc<CudaStream>, OpError>;
    /// 块句柄 → 设备指针(账房解析;Bytes 纯 ID 纪律的唯一出口)
    fn resolve(&self, b: &crate::contract::Bytes) -> Result<DevPtr, OpError>;
    /// scratch 分配:账本 + 捕获 slab 感知 + A1.7 强租约
    fn alloc(&mut self, bytes: usize, tag: &'static str) -> Result<ScratchBuf, OpError>;
    /// 捕获状态查询(当前是否处于图捕获窗)
    fn capturing(&self) -> Result<bool, OpError>;
    /// A1.6 登记钩子(捕获窗内的发射原材料;server 组装 CaptureRecord)
    fn record_capture(&mut self, note: LaunchNote) -> Result<(), OpError>;
    /// host 字节上传到设备指针(meta 表/小常量)。
    ///
    /// **序语义(2026-10-15 A13 更正)**:实现侧一律**与目标块所在流同序**
    /// (分配 → 上传 → 后续核 在该流上严格有序);**回执 ≠ host 侧落地**。
    /// 旧文档写「同步语义,回执即落地」—— 那句是错的,且**正是 A13 的病根**:
    /// 实现曾用 NULL 流同步拷贝写一个由 `stream.alloc`(流序异步分配)刚给的
    /// 地址,两者零序 ⇒ 写落进尚未落定的内存被**静默丢弃**
    /// (compute-sanitizer 判据:`Use-before-alloc @ cuMemcpyHtoD ← cuMemAllocAsync`)。
    /// 调用方不得依赖「upload 返回后 host 就能读到」;要读请走
    /// [`download`](Self::download)(它自带回读前排空)。
    fn upload(&mut self, dst: DevPtr, src: &[u8]) -> Result<(), OpError>;
    /// 设备序号(FFI 面 dev 注入;marlin/cublas 等直调家族用)
    fn device_ordinal(&self) -> Result<i32, OpError>;
    /// **账本块分配**(装配期自检专用;2026-10-13 P5c)。
    ///
    /// 与 [`alloc`](Self::alloc) 的区别:`alloc` 给的是**裸 scratch 指针**
    /// (不进账房 ⇒ 不能被 `resolve` 解析);而 FFI/家族 runtime 的算子参数
    /// 是**块句柄** ⇒ 自检要跑 foreign 算子(marvin/xqa/FI/cublas)就必须能
    /// 拿到"真在册的块"。返回 [`Bytes`](crate::contract::Bytes) 句柄。
    fn alloc_block(&mut self, bytes: usize, tag: &'static str) -> Result<crate::contract::Bytes, OpError>;

    /// **按块句柄上传**(装配期自检专用;2026-10-13 P5c)。
    ///
    /// 为什么与 [`upload`](Self::upload) 分开:`upload` 的 dst 是**裸指针**且
    /// 服务端只认"指针在 scratch 账本"(写者白名单);而自检要往**账本块**灌
    /// 输入(块句柄 → `resolve` → 写)。两条语义不同,不合并。
    fn upload_block(&mut self, b: &crate::contract::Bytes, src: &[u8]) -> Result<(), OpError>;

    /// **设备 → host 字节回读**(装配期自检/诊断专用;2026-10-13 P5)。
    ///
    /// 为什么单列一个方法:自检(oracle)必须能把算子输出**读回 host 与独立
    /// 参考逐字节比对** —— 这是"静默错值"类唯一能机器判定的手段。
    /// 与 `upload` 对称:裸指针 + 字节口径,元素语义归调用方。
    ///
    /// **序语义(2026-10-15 A13 更正)**:实现侧在回读前**排空生产流**,否则
    /// NULL 流同步 D2H 与 COMPUTE 流无序 ⇒ 读到半写入/旧值/全零
    /// (本仓 2026-10-09 / `xqa.rs` / `gdn_chunked.rs:453` 三次本地绕行的病根)。
    fn download(&mut self, src: DevPtr, dst: &mut [u8]) -> Result<(), OpError>;

    /// **家族面设备动作入账钩子**(2026-10-15 A13 补漏)。
    ///
    /// 为什么需要它:家族 runtime 直连设备(scratch 分配 / 上传 / 回读 /
    /// 核发射 / FFI 直发)**不产生 `Command`**,而发射账(perf-framework
    /// §十三)只在命令派发处记一条 ⇒ **家族面在账上完全隐身** —— A13 就是
    /// 这样藏了一整轮,最后靠 compute-sanitizer 才挖出来。
    ///
    /// 用法(有界词表):`fam_alloc` / `fam_upload` / `fam_upload_block` /
    /// `fam_download` / `fam_launch` / `fam_ffi_launch`。
    ///
    /// **纪律**:
    /// 1. 这是**通知**(best-effort),不是可失败操作 —— 记账失败绝不拖垮执行,
    ///    故返回 `()` 且缺省空实现(关账时零成本);
    /// 2. `ptrs` = 设备地址(生命周期判据的钥匙),没有就给空切片;`name`
    ///    允许空串;**不得在其中做分配/同步/阻塞**。
    fn note_device(&mut self, _kind: &'static str, _name: &str, _bytes: usize, _ptrs: &[u64]) {}

    /// **入账面是否武装**(2026-10-15 审查补)。
    ///
    /// 为什么需要它:`note_device` 缺省是空体,但**调用方要先把载荷建出来**
    /// (`Exec::launch` 每次发射都 `Vec::with_capacity` 收集指针)⇒ 关账时那笔
    /// 分配是白做的(热路径:prefill 实测 3376 次发射/轮)。本方法让调用方
    /// **先问再建**:缺省 `false`(非 device 面/未武装),`GpuRes` 覆写为
    /// "trace 环已开"。
    fn note_enabled(&self) -> bool {
        false
    }
}


/// 设备码块身份(**缓存键的唯一真源**)。
///
/// 为什么不能只用"资产名":`gdn_chunked` 一族**六个 Triton cubin 共用一个
/// tag**,按 tag 做模块缓存会让后五个 cubin 静默拿到第一个的模块
/// (实测踩坑:装配期报 `named symbol not found`)。
/// 键 = (tag, 字节块首址, 长度):AOT 资产(`include_bytes!` 静态)与自管
/// cubin(`static` 表项)各自唯一 ⇒ 同名不同块物理上不可能撞。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BlobKey {
    tag: &'static str,
    ptr: usize,
    len: usize,
}

impl BlobKey {
    fn of(tag: &'static str, bytes: &[u8]) -> Self {
        Self { tag, ptr: bytes.as_ptr() as usize, len: bytes.len() }
    }
}

/// **值表发射(唯一映射点)**:`LaunchVal` 表 → cudarc 参数槽 → 发射。
///
/// 为什么提出来:`Exec::launch`(生产路径)与测试探针(直用 cudarc)必须
/// **逐位同构**地压槽,否则"测试绿 ≠ 生产绿"。指针参数以 u64 位型直入 8 字节槽
/// (与 handler `&u64` 同 ABI);`arg()` 不可败(参数空间按值拷贝),失败面收敛
/// 在一处。生命周期一律取同一个 `'a`(`LaunchArgs` 对其类型参数**不变**,
/// 拆两个生命周期无法统一 —— 首试教训)。
///
/// **by-value 结构体参数**走 [`LaunchVal::Blob`]:驱动按内核参数元数据里的
/// `paramSize` 从**一个指针**拷字节 ⇒ 必须给完整镜像的地址(拆成多个槽是错的,
/// 见 `LaunchVal::Blob` 注)。镜像布局由调用方 `#[repr(C)]` + size 断言保证,
/// 并可经"资产自描述"回报的 `sizeof` 机器锁死(XQA `KVCacheList` 即此)。
pub fn launch_vals<'a>(
    stream: &'a Arc<CudaStream>,
    f: &'a CudaFunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    vals: &'a [LaunchVal],
) -> Result<(), cudarc::driver::DriverError> {
    use cudarc::driver::PushKernelArg;
    let mut b = stream.launch_builder(f);
    for v in vals {
        match v {
            LaunchVal::Ptr(p) => {
                b.arg(p);
            }
            LaunchVal::U64(x) => {
                b.arg(x);
            }
            LaunchVal::U32(x) => {
                b.arg(x);
            }
            LaunchVal::I32(x) => {
                b.arg(x);
            }
            LaunchVal::F32(x) => {
                b.arg(x);
            }
            // 结构体镜像:取内联数组首字节地址(驱动按 paramSize 拷 len 字节);
            // 数组随 `vals` 切片存活 ⇒ 指针在 launch 期间有效
            LaunchVal::Blob { len, bytes } => {
                debug_assert!(*len as usize <= bytes.len());
                b.arg(&bytes[0]);
            }
        }
    }
    // SAFETY: 槽序与宽度由调用方按算子契约给(表驱动;L3 哨兵已查)
    unsafe {
        b.launch(cudarc::driver::LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared_mem,
        })
    }
    .map(|_| ())
}

/// 执行引擎:kernels 实现;全部机械,吃资源面干活。
/// 私有态:设备码块 → 模块表 / 函数缓存。
#[derive(Default)]
pub struct Exec {
    /// 函数缓存:(块身份, 符号) → 函数
    fns: HashMap<(BlobKey, &'static str), Arc<CudaFunction>>,
    /// 模块表:块身份 → 已载模块(同一 `.cu`/同一 cubin 的多核共享)
    modules: HashMap<BlobKey, Arc<cudarc::driver::CudaModule>>,
}

impl Exec {
    pub fn new() -> Self {
        Self::default()
    }

    /// **构建期资产装载**(native 核唯一路径;资产 id 查表 → fatbin → 符号)。
    ///
    /// 查表键 = **资产 id**(`cu/` 相对路径),与 `KernelSpec::SRC` 一一对应。
    /// E 轮(2026-10-13)起**不再携带源文本**:文本只在构建期参与编译,运行期
    /// 只需要资产身份(设计 §15.5 的内容键立意改由「构建期目录扫描 + 符号
    /// 清单」承担)。模块按资产共享 —— 同一 `.cu` 的多个核只载一次。
    ///
    /// 未命中 = `OpError::Asset`(**不是**降级到运行期编译:自 2026-10-13
    /// 裁决起 NVRTC 已退役,本函数是唯一的设备码装载入口)。
    pub fn load_aot(
        &mut self,
        res: &mut dyn DeviceRes,
        src_id: &str,
        symbol: &'static str,
        op: OpId,
    ) -> Result<Arc<CudaFunction>, OpError> {
        let asset = crate::family::cubins::asset(src_id).ok_or_else(|| OpError::Asset {
            op: op.0.to_string(),
            detail: format!(
                "symbol {symbol}: 资产 id {src_id:?} 无构建期 fatbin —— 该路径不在 build.rs 的 \
                 目录扫描结果内(零运行期编译硬门:NVRTC 已退役,见设计 §15.3/§15.5)"
            ),
        })?;
        let key = BlobKey::of(asset.id, asset.bytes);
        if let Some(f) = self.fns.get(&(key, symbol)) {
            return Ok(f.clone());
        }
        let module = match self.modules.get(&key) {
            Some(m) => m.clone(),
            None => {
                let ctx = res.context()?;
                let m = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
                    .map_err(|e| OpError::Asset {
                        op: op.0.to_string(),
                        detail: format!("asset {}: {e:?}", asset.id),
                    })?;
                self.modules.insert(key, m.clone());
                m
            }
        };
        let f = module.load_function(symbol).map_err(|e| OpError::Asset {
            op: op.0.to_string(),
            detail: format!("符号 {symbol} 不在资产 {}({e:?})——资产/代码代际失配", asset.id),
        })?;
        let f = Arc::new(f);
        self.fns.insert((key, symbol), f.clone());
        Ok(f)
    }

    /// 裸 cubin 装载(旧族自管资产路径;`gdn_scalar` 先例)。
    /// 新族一律走 [`load_aot`](Self::load_aot)(资产登记表统一面)。
    pub fn load_cubin(
        &mut self,
        res: &mut dyn DeviceRes,
        asset: &'static str,
        bytes: &[u8],
        symbol: &'static str,
        op: OpId,
    ) -> Result<Arc<CudaFunction>, OpError> {
        // ⚠️ 块身份含字节首址:同 tag 的多份 cubin(如 gdn_chunked 六核)必须
        // 各自成模块,否则后续 cubin 的符号查不到(实测:装配期 named symbol not found)
        let key = BlobKey::of(asset, bytes);
        if let Some(f) = self.fns.get(&(key, symbol)) {
            return Ok(f.clone());
        }
        let module = match self.modules.get(&key) {
            Some(m) => m.clone(),
            None => {
                let ctx = res.context()?;
                let m = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_binary(bytes.to_vec()))
                    .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("cubin {asset}: {e:?}") })?;
                self.modules.insert(key, m.clone());
                m
            }
        };
        let f = module
            .load_function(symbol)
            .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("符号 {symbol} 不在 {asset}({e:?})——资产/代码代际失配,manifest 门拦截") })?;
        let f = Arc::new(f);
        self.fns.insert((key, symbol), f.clone());
        Ok(f)
    }

    /// 动态 sharedmem 上限预置(>48K 核的 opt-in;handler 同名单由家族自管)
    pub fn set_shared_limit(f: &CudaFunction, bytes: u32) -> Result<(), OpError> {
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32)
            .map_err(|e| OpError::Asset { op: OpId("exec").0.to_string(), detail: format!("setattr {bytes}: {e:?}") })
    }

    /// 捕获窗守卫 + 登记的统一入口(横切纪律一次写成)。
    /// 两路都**真实发射**:捕获窗内由驱动把发射写进图(CUDA stream
    /// capture 原生语义),同时记 [`LaunchNote`] 交 res 账面(A1.6;是否
    /// 结构化 CaptureRecord 归 graph 治理层立项);窗外 = 纯直发。
    /// 参数 = **owned 值表**([`LaunchVal`]):HRTB 闭包与 cudarc arg 的
    /// 值生命周期相克(E0597,首试教训);owned 值表零生命周期参数,
    /// 调用点也不再需要 builder 闭包样板。
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &mut self,
        res: &mut dyn DeviceRes,
        op: OpId,
        f: &CudaFunction,
        kernel: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        vals: &[LaunchVal],
    ) -> Result<(), OpError> {
        if res.capturing()? {
            res.record_capture(LaunchNote { op: op.0.to_string(), kernel, grid, block, shared_mem })?;
        }
        let stream = res.stream()?;
        launch_vals(&stream, f, grid, block, shared_mem, vals).map_err(|e| {
            OpError::Launch {
                op: op.0.to_string(),
                stage: Stage::Compute,
                detail: format!("{kernel}: {e:?}"),
            }
        })?;
        // **家族面入账**(2026-10-15 A13 补漏;审查后修正两点):本函数是
        // gdn_chunked / gdn_scalar / xqa / flashinfer 四族核发射的唯一出口 ——
        // 在此一次性覆盖全员(值表里的 Ptr 参数顺带落账 = 设备地址判据)。
        //   ① `note_enabled()` 守门 ⇒ 关账时**不建载荷**(零分配,不再白建 Vec);
        //   ② 入账挪到**发射成功之后** ⇒ 账与事实一致(失败不留幽灵条目)。
        // 直连 FFI 的族(marlin/cublas/FI run)自行调 `note_device("fam_ffi_launch", ..)`。
        if res.note_enabled() {
            let mut ptrs: Vec<u64> = Vec::with_capacity(vals.len());
            for v in vals {
                if let LaunchVal::Ptr(p) = v {
                    ptrs.push(*p);
                }
            }
            res.note_device("fam_launch", kernel, 0, &ptrs);
        }
        Ok(())
    }

    /// htod(不变盒;staging 纪律:构造后不可变,捕获窗引用安全)。
    /// 字节口径:dst 为 u8 视图,元素类型语义归调用方(与生产 staging 同式)。
    pub fn memcpy_htod_bytes(
        &mut self,
        res: &mut dyn DeviceRes,
        dst: &mut cudarc::driver::CudaSlice<u8>,
        src: &InvariantBox,
        op: OpId,
    ) -> Result<(), OpError> {
        let stream = res.stream()?;
        stream
            .memcpy_htod(src.as_bytes(), dst)
            .map_err(|e| OpError::Launch { op: op.0.to_string(), stage: Stage::Store, detail: format!("htod: {e:?}") })
    }

    /// 数值体检:取回样本算 FieldStats(违例证据自带数值)。
    /// EagerOnly:含 D2H,捕获窗调用被守卫拒(A1.5/A1.6)。
    pub fn sample_stats(
        &mut self,
        res: &mut dyn DeviceRes,
        src: &cudarc::driver::CudaSlice<f32>,
        op: OpId,
        law: Law,
    ) -> Result<FieldStats, OpError> {
        if res.capturing()? {
            return Err(OpError::Launch { op: op.0.to_string(), stage: Stage::Epilogue, detail: "sample_stats 禁入捕获段(EagerOnly)".into() });
        }
        let stream = res.stream()?;
        let host = stream.clone_dtoh(src).map_err(|e| OpError::Launch { op: op.0.to_string(), stage: Stage::Epilogue, detail: format!("sample dtoh: {e:?}") })?;
        let (mut min, mut max, mut nan, mut inf) = (f32::INFINITY, f32::NEG_INFINITY, 0u64, 0u64);
        for v in &host {
            if v.is_nan() {
                nan += 1;
            } else if v.is_infinite() {
                inf += 1;
            } else {
                min = min.min(*v);
                max = max.max(*v);
            }
        }
        let stats = FieldStats { min, max, nan_count: nan, inf_count: inf };
        // 律判定(此处只产 Err;处置归引擎策略层)
        let violated = match law {
            Law::GateCumsumNonPositive => stats.max > 1e-6 || stats.nan_count > 0 || stats.inf_count > 0,
            Law::IntermediateFinite | Law::OutputFinite => stats.nan_count > 0 || stats.inf_count > 0,
        };
        if violated {
            return Err(OpError::Invariant { op: op.0.to_string(), law, stats });
        }
        Ok(stats)
    }
}
