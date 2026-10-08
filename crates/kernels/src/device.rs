//! 服务端面执行原语 —— [`DeviceRes`] 资源面(server 实现,仅治理状态)
//! × [`Exec`] 执行引擎(kernels 实现,全部机械)。
//!
//! 切分判据(operator-contract 设计 v0.4 §4.4)= **状态 vs 机械**:
//! - 状态留 server:块表/账本(A5.2)、捕获状态机与图治理(A1.6/A1.7)、
//!   三流模型 —— DeviceRes 五方法即全部,server 侧再无执行代码;
//! - 机械下沉 kernels:nvrtc 编译缓存、cubin 模块表、launch 组装、
//!   捕获窗守卫与登记的机械部分、staging 不变盒、体检核发射。
//!
//! P3 无例外:资源获取/状态读取/副作用一律 `Result`;裸返回仅限纯且全
//! 函数。零 `()`、零 log-and-continue、零 ack 侧信道。
//!
//! feature = "cuda"(cudarc driver/nvrtc;与 cuda_ops 同门)。

use crate::contract::{FieldStats, InvariantBox, Law, OpError, OpId, Stage};
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
    /// host 字节上传到设备指针(meta 表/小常量;同步语义,回执即落地)
    fn upload(&mut self, dst: DevPtr, src: &[u8]) -> Result<(), OpError>;
    /// 设备序号(FFI 面 dev 注入;marlin/cublas 等 StaticLib 家族用)
    fn device_ordinal(&self) -> Result<i32, OpError>;
}

/// 发射参数值(owned 值表元素;指针参数 = u64 位型直入指针槽)
#[derive(Debug, Clone, Copy)]
pub enum LaunchVal {
    /// 设备指针(块解析产物/scratch;8 字节指针槽)
    Ptr(u64),
    U64(u64),
    I32(i32),
    F32(f32),
}

/// 函数解析键(Linkage 三链路一个入口)
#[derive(Debug, Clone)]
pub enum FunctionKey {
    /// AOT cubin 资产符号(boot 期已装载,此处查表)
    Cubin { asset: &'static str, symbol: &'static str },
    /// NVRTC 源码(首用编译,(source hash, name) 缓存)
    Nvrtc { name: &'static str, source: &'static str },
    // StaticLib 不走这里:FFI 直调(链接期已解析),init 期只做
    // 句柄/workspace 预钉 —— ensure_blas / marlin workspace / FI plan ws。
}

/// 执行引擎:kernels 实现;全部机械,吃资源面干活。
/// 私有态:nvrtc 编译缓存 / cubin 模块表 / staging 盒池(家族接入时填充)。
#[derive(Default)]
pub struct Exec {
    /// 函数缓存:(来源标签, 符号名) → 函数(cubin 与 nvrtc 同表)
    fns: HashMap<(&'static str, &'static str), Arc<CudaFunction>>,
    /// nvrtc 编译缓存(源码标签 → PTX 模块;编译一次全符号共享)
    nvrtc: HashMap<&'static str, Arc<cudarc::driver::CudaModule>>,
    /// nvrtc include 路径(缺省 /usr/local/cuda/include)
    nvrtc_include: Vec<String>,
}

impl Exec {
    pub fn new() -> Self {
        Self { nvrtc_include: vec!["/usr/local/cuda/include".to_string()], ..Default::default() }
    }

    /// AOT cubin 装载(缓存键 = (资产标签, 符号);重复请求查表直回)
    pub fn load_cubin(
        &mut self,
        res: &mut dyn DeviceRes,
        asset: &'static str,
        bytes: &[u8],
        symbol: &'static str,
        op: OpId,
    ) -> Result<Arc<CudaFunction>, OpError> {
        if let Some(f) = self.fns.get(&(asset, symbol)) {
            return Ok(f.clone());
        }
        let ctx = res.context()?;
        let m = ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(bytes.to_vec()))
            .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("cubin {asset}: {e:?}") })?;
        let f = m
            .load_function(symbol)
            .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("符号 {symbol} 不在 {asset}({e:?})——资产/代码代际失配,manifest 门拦截") })?;
        let f = Arc::new(f);
        self.fns.insert((asset, symbol), f.clone());
        Ok(f)
    }

    /// NVRTC 源码装载(编译一次;符号查表直回)
    pub fn load_nvrtc(
        &mut self,
        res: &mut dyn DeviceRes,
        source_tag: &'static str,
        source: &'static str,
        symbol: &'static str,
        op: OpId,
    ) -> Result<Arc<CudaFunction>, OpError> {
        if let Some(f) = self.fns.get(&(source_tag, symbol)) {
            return Ok(f.clone());
        }
        if !self.nvrtc.contains_key(source_tag) {
            use cudarc::nvrtc::CompileOptions;
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                source,
                CompileOptions {
                    include_paths: self.nvrtc_include.clone(),
                    ..Default::default()
                },
            )
            .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("nvrtc {source_tag}: {e:?}") })?;
            let ctx = res.context()?;
            let m = ctx
                .load_module(ptx)
                .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("nvrtc module {source_tag}: {e:?}") })?;
            self.nvrtc.insert(source_tag, m);
        }
        let m = self.nvrtc.get(source_tag).expect("刚插入").clone();
        let f = m
            .load_function(symbol)
            .map_err(|e| OpError::Asset { op: op.0.to_string(), detail: format!("nvrtc 符号 {symbol}: {e:?}") })?;
        let f = Arc::new(f);
        self.fns.insert((source_tag, symbol), f.clone());
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
        let mut b = stream.launch_builder(f);
        use cudarc::driver::PushKernelArg;
        for v in vals {
            // 指针参数 = u64 位型直入 8 字节槽(与 handler &u64 同 ABI);
            // arg() 不可败(参数空间按值拷贝),失败面收敛在 launch 一处
            match v {
                LaunchVal::Ptr(p) => {
                    b.arg(p);
                }
                LaunchVal::U64(x) => {
                    b.arg(x);
                }
                LaunchVal::I32(x) => {
                    b.arg(x);
                }
                LaunchVal::F32(x) => {
                    b.arg(x);
                }
            }
        }
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared_mem,
        };
        unsafe { b.launch(cfg) }.map_err(|e| OpError::Launch { op: op.0.to_string(), stage: Stage::Compute, detail: format!("{kernel}: {e:?}") })?;
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
