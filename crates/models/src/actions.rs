//! 预定义算子动作表:声明层具名算子 → LaunchMsg 的唯一 lower 通道。
//!
//! kernel 源内嵌于此(预定义算子的实现归解释层所有;
//! 用户自定义 kernel 走 TensorOps::of(Kernel) 直带源码)。

use crate::client::{Arg, Bytes, KernelSpec, LaunchMsg};

/// 预定义 f32 算子源(add/silu/matmul/rmsnorm)
pub const PREDEFINED_CU: &str = r#"
extern "C" __global__ void owl_add_f32(
    const float* a, const float* b, float* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}
extern "C" __global__ void owl_silu_f32(
    const float* x, float* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i] / (1.0f + expf(-x[i])); }
}
extern "C" __global__ void owl_matmul_f32(
    const float* a, const float* b, float* out,
    const int m, const int k, const int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; p++) { acc += a[(size_t)r * k + p] * b[(size_t)p * n + c]; }
        out[(size_t)r * n + c] = acc;
    }
}
extern "C" __global__ void owl_rmsnorm_f32(
    const float* x, const float* alpha, float* out,
    const int n, const float eps, const int w_off) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float* xin = x + (size_t)row * n;
    float* y = out + (size_t)row * n;
    float local = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) { local += xin[c] * xin[c]; }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float a = w_off ? (alpha[c] + 1.0f) : alpha[c];
        y[c] = xin[c] * inv * a;
    }
}
"#;

fn spec(name: &str) -> KernelSpec {
    KernelSpec { name: name.to_string(), source: PREDEFINED_CU.to_string() }
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

/// lower:Rmsnorm([rows,n];w_off = ×(1+w))
pub fn lower_rmsnorm(ins: &[Bytes], eps: f32, w_off: bool, out: &Bytes) -> LaunchMsg {
    let n: usize = ins[0].len; // Bytes.len = 元素数(alloc/htod 均按元素登记)
    LaunchMsg {
        kernel: spec("owl_rmsnorm_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::I32(n as i32),
            Arg::F32(eps),
            Arg::I32(w_off as i32),
        ],
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 256 * 4,
        out_elems: n, // rows × n
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
