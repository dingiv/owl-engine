//! CUDA f32 基础算子:**自描述发射包**(KernelFn = 名字 + PTX 源 + 参数槽)。
//!
//! 职责分离(2026-09-23 用户裁决):本 crate 只**描述**一次发射
//! (函数 + 参数槽 + 网格配置),真正的发射动作在 GPU 层(server actor
//! 线程)调用 [`KernelLaunch::launch_on`] 完成。
//!
//! - `.cu` 源:cu/ops.cu(build.rs 预编译 PTX,include_str 内嵌);
//! - 参数:裸指针/标量 → u64 槽(裁决 3①:标量+裸指针,禁止 htod 形状数组);
//! - unsafe 边界:指针有效性由调用方保证(server 账房持有池块)。


/// kernel 源(.cu;留作审计/文档——实际编译产物 = build.rs 预编译的 PTX)
pub const OPS_CU: &str = include_str!("../cu/ops.cu");

/// build.rs 预编译的 PTX(构建期内嵌;运行时零编译)
pub const OPS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"));



// ============================================================================
// §1 发射描述(纯数据;不执行)
// ============================================================================

/// 一次 kernel 发射的完整描述。纯数据,无生命周期纠缠——
/// server 拿到后在其流上执行;声明与执行就此分离。
///
/// 参数槽语义:槽 = u64;kernel 签名为指针时槽存指针值(8B),
/// 签名为标量时驱动按签名宽度读槽(LE 低字节 = 正确位型;
/// f32 → to_bits、i32 → 符号扩展,均成立)。
#[allow(dead_code)] // 骨架期:字段消费方 = gpu_server 的懒编译发射(接线后移除)
pub struct KernelFn {
    pub name: &'static str,
    pub func: &'static str,
    pub ptx: &'static str,
    pub slots: Vec<u64>,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

// impl KernelFn {
//     /// 执行(仅 GPU 层/actor 线程调用)。
//     ///
//     /// # Safety
//     /// 指针槽指向的显存必须存活且已就绪(流序由调用方的流纪律保证);
//     /// 参数槽个数/位型必须与 kernel 签名严格一致(漏一个 = INVALID_VALUE)。
//     pub unsafe fn launch_on(&self, stream: &CudaStream) -> Result<(), String> {
//         // 参数槽保活至发射完成;驱动按签名从槽读值(指针 8B / 标量 4-8B,
//         // LE 低字节 = 正确位型)。PushKernelArg<&u64> 存槽指针,
//         // 槽随本函数的 slots 存活,launch 在同一语句块内完成。
//         let mut slots = self.slots.clone();
//         let cfg = LaunchConfig {
//             grid_dim: self.grid,
//             block_dim: self.block,
//             shared_mem_bytes: self.shared_mem,
//         };
//         unsafe {
//             let mut builder = stream.launch_builder(&self.func);
//             for slot in slots.iter_mut() {
//                 builder.arg(&*slot);
//             }
//             builder
//                 .launch(cfg)
//                 .map(|_| ())
//                 .map_err(|e| format!("launch: {e}"))
//         }
//     }
// }

// ============================================================================
// §2 f32 基础算子:描述构造器(返回 KernelLaunch,不发射)
// ============================================================================

// /// f32 基础算子族(绑定 kernel 编译产物;描述构造器不发射)。
// pub struct F32Kernels {
//     funcs: HashMap<&'static str, CudaFunction>,
// }

impl KernelFn {
    /// 装载 build.rs 预编译的 PTX(运行时零编译;驱动按设备 JIT)。
    // pub fn new(stream: Arc<CudaStream>) -> Result<Self, String> {
    //     let ptx = Ptx::from_src(OPS_PTX);
    //     let module = stream
    //         .context()
    //         .load_module(ptx)
    //         .map_err(|e| format!("load_module: {e:?}"))?;
    //     let mut funcs = HashMap::new();
    //     for name in NAMES {
    //         funcs.insert(
    //             name,
    //             module.load_function(name).map_err(|e| format!("{name}: {e:?}"))?,
    //         );
    //     }
    //     Ok(Self { funcs })
    // }

    // fn func(&self, name: &str) -> CudaFunction {
    //     self.funcs[name].clone()
    // }

    /// 一维网格:ceil-div 256 线程/块
    fn grid_1d(n: usize) -> (u32, u32, u32) {
        ((n as u32 + 255) / 256, 1, 1)
    }

    // ---- 描述构造器(纯;不发射)----

    /// 逐元素加:a + b → out(同形,n = 元素数)
    pub fn add(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> KernelFn {
        KernelFn {
            name: "owl_add_f32",
            func: "owl_add_f32",
            ptx: OPS_PTX,
            slots: vec![a as u64, b as u64, out as u64, n as u64],
            grid: Self::grid_1d(n),
            block: (256, 1, 1),
            shared_mem: 0,
        }
    }

    /// silu:x / (1 + e^-x)
    pub fn silu(x: *const f32, out: *mut f32, n: usize) -> KernelFn {
        KernelFn {
            name: "owl_silu_f32",
            func: "owl_silu_f32",
            ptx: OPS_PTX,
            slots: vec![x as u64, out as u64, n as u64],
            grid: Self::grid_1d(n),
            block: (256, 1, 1),
            shared_mem: 0,
        }
    }

    /// 朴素矩阵乘:[m,k] × [k,n] → [m,n](行主序;grid 二维 16×16)
    pub fn matmul(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> KernelFn {
        KernelFn {
            name: "owl_matmul_f32",
            func: "owl_matmul_f32",
            ptx: OPS_PTX,
            slots: vec![a as u64, b as u64, out as u64, m as u64, k as u64, n as u64],
            grid: ((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
            block: (16, 16, 1),
            shared_mem: 0,
        }
    }

    /// rmsnorm(2D [rows,n];parents = [x, alpha];w_off = ×(1+w))
    pub fn rmsnorm(
        x: *const f32,
        alpha: *const f32,
        out: *mut f32,
        rows: usize,
        n: usize,
        eps: f32,
        w_off: bool,
    ) -> KernelFn {
        KernelFn {
            name: "owl_rmsnorm_f32",
            func: "owl_rmsnorm_f32",
            ptx: OPS_PTX,
            slots: vec![
                x as u64,
                alpha as u64,
                out as u64,
                n as u64,
                eps.to_bits() as u64,
                w_off as u64,
            ],
            grid: (rows as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem: 256 * 4,
        }
    }
}
