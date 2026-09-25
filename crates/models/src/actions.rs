//! 预定义算子动作表:声明层具名算子 → LaunchMsg 的唯一 lower 通道。
//!
//! kernel 源**不住这里** —— 经 [`crate::kernels`] 注册表按名取用
//! (源码之家 = owl-kernels cu/;2026-09-25 垫子层裁决)。
//! 用户自定义 kernel 走 TensorOps::of(Kernel) 直带源码(不经注册表,
//! 它是组合面逃生舱)。

use crate::client::{Arg, Bytes, KernelSpec, LaunchMsg};
use crate::kernels;

fn spec(name: &'static str) -> KernelSpec {
    KernelSpec { name: name.to_string(), source: kernels::source(name).to_string() }
}

fn ceil_1d(n: usize) -> (u32, u32, u32) {
    ((n as u32 + 255) / 256, 1, 1)
}

/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
pub fn lower_add(ins: &[Bytes], out: &Bytes) -> LaunchMsg {
    let n: usize = ins[0].len; // Bytes.len = 元素数(alloc/htod 均按元素登记)
    LaunchMsg {
        kernel: spec("owl_add_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::U64(n as u64),
        ],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Mul(同形逐元素乘)
pub fn lower_mul(ins: &[Bytes], out: &Bytes) -> LaunchMsg {
    let n: usize = ins[0].len; // Bytes.len = 元素数(alloc/htod 均按元素登记)
    LaunchMsg {
        kernel: spec("owl_mul_f32"),
        args: vec![Arg::Block { id: ins[0].id }, Arg::Block { id: ins[1].id }, Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Silu(一元)
pub fn lower_silu(ins: &[Bytes], out: &Bytes) -> LaunchMsg {
    let n: usize = ins[0].len; // Bytes.len = 元素数(alloc/htod 均按元素登记)
    LaunchMsg {
        kernel: spec("owl_silu_f32"),
        args: vec![Arg::Block { id: ins[0].id }, Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Matmul([m,k]×[k,n];k 由 server 从 a 账长自推)
pub fn lower_matmul(ins: &[Bytes], out: &Bytes, m: usize, k: usize, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_matmul_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::I32(m as i32),
            Arg::I32(k as i32),
            Arg::I32(n as i32),
        ],
        grid: ((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
        block: (16, 16, 1),
        shared_mem: 0,
        out_elems: m * n,
    }
}

/// lower:Rmsnorm([rows, cols];per-channel alpha([cols] 广播);w_off = ×(1+w))
pub fn lower_rmsnorm(ins: &[Bytes], eps: f32, w_off: bool, out: &Bytes, rows: usize, cols: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_rmsnorm_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::I32(cols as i32),
            Arg::F32(eps),
            Arg::I32(w_off as i32),
        ],
        // 一 block 一行(owl_rmsnorm_f32:row = blockIdx.x)
        grid: (rows as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 256 * 4,
        out_elems: rows * cols,
    }
}

// ============================================================================
// Kernel 节点(动作表二期:用户自定义 kernel 的唯一 lower 通道)
// ============================================================================

/// lower:Kernel 节点。
///
/// 槽序契约:`.arg(t)` 同时追加参数槽与父依赖(同序),故 T 槽按序
/// 对齐归约结果 ins;**输出块固定追加在槽序末尾**(server 按"最后一个
/// Block"回传句柄)。grid = (0,0,0) 哨兵 → 按输出元素数自动 1D ceil/256。
pub fn lower_kernel(
    kernel: &crate::kernel::Kernel,
    decl_args: &[crate::plan::KernelArg],
    ins: &[Bytes],
    out: &Bytes,
) -> LaunchMsg {
    use crate::plan::KernelArg;
    let mut args: Vec<Arg> = Vec::with_capacity(decl_args.len() + 1);
    let mut pi = 0usize; // T 槽 ↔ 父依赖同序计数
    for a in decl_args {
        match a {
            KernelArg::T { .. } => {
                args.push(Arg::Block { id: ins[pi].id });
                pi += 1;
            }
            KernelArg::Bits(v) => args.push(Arg::U64(*v)),
            KernelArg::I32(v) => args.push(Arg::I32(*v)),
            KernelArg::F32(v) => args.push(Arg::F32(*v)),
        }
    }
    args.push(Arg::Block { id: out.id });
    let (grid, block, shared_mem) = if kernel.launch.grid == (0, 0, 0) {
        (auto_grid(out.len), kernel.launch.block, kernel.launch.shared_mem)
    } else {
        (kernel.launch.grid, kernel.launch.block, kernel.launch.shared_mem)
    };
    LaunchMsg {
        kernel: KernelSpec { name: kernel.name.to_string(), source: kernel.source.to_string() },
        args,
        grid,
        block,
        shared_mem,
        out_elems: out.len,
    }
}

/// 自动 1D grid(哨兵展开用)
pub fn auto_grid(out_elems: usize) -> (u32, u32, u32) {
    ceil_1d(out_elems)
}
