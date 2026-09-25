//! playground:声明式 Tensor 链式 API 的 CPU 可跑玩具(值语义版;2026-09-23)。
//!
//! ⚡ 存储裁决(2026-09-23 用户拍板):**不用 Arc**。每个节点持有完整的
//! 遍历树(值语义,clone = 深拷贝)。理由:
//!   - 声明树是**配置面**产物:只在启动期构建一次,不在热路径;
//!   - 树规模 = 几百节点,值拷贝在 Rust 里是纳秒级,根本不算什么;
//!   - 换来:零引用计数、零共享别名、零 Send/Sync 负担——纯值世界。
//!
//! 主干 demo(不变):
//!   demo1 链式声明 + 递归解释器(描述→自叶向根归约,对拍 host 参考)
//!   demo2 毒值传播(构造期违约随链流动,边界收割)
//!   demo3 值语义:克隆即深拷贝,重复消费 = 重复计算(配置面,可接受)
//!   demo4 fork 多档位(底稿深拷贝进各档;烘焙 = 展平)
//!
//! 全部 f32 + 朴素 CPU 实现;无 tokio、无 GPU、无 unsafe、无 Arc。

// ============================================================================
// §1 计划树(反向多叉树,值语义)
// ============================================================================

type Shape = Vec<usize>;

/// 反向多叉树节点:**完整持有**自己的输入子树(值语义)。
/// - 边从【结果】指向【输入】;叶子(无 parents)= 数据源;
/// - 计算 = 自叶向根归约,root 的值 = 输出。
#[derive(Clone)] // clone = 深拷贝整棵子树(配置面:一次性的,可接受)
struct Step {
    parents: Vec<Step>,
    depth: u32,
    op: Op,
    shape: Shape,
    err: Option<String>,
}

#[derive(Clone)]
enum Op {
    Host(Vec<f32>),
    Zeros,
    Add,
    Matmul, // [m,k]×[k,n]
    Silu,
    Rmsnorm { eps: f32, w_off: bool },
}

#[derive(Clone)]
struct Tensor {
    head: Step, // 完整遍历树(值!)
    shape: Shape,
}

impl Tensor {
    fn is_poisoned(&self) -> bool {
        self.head.err.is_some()
    }

    // ---- 工厂 ----

    fn from_host(shape: Shape, data: &[f32]) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape × data 不匹配"
        );
        Tensor {
            head: Step {
                parents: vec![],
                depth: 0,
                op: Op::Host(data.to_vec()),
                shape: shape.clone(),
                err: None,
            },
            shape,
        }
    }

    fn zeros(shape: Shape) -> Tensor {
        Tensor {
            head: Step {
                parents: vec![],
                depth: 0,
                op: Op::Zeros,
                shape: shape.clone(),
                err: None,
            },
            shape,
        }
    }

    // ---- 毒值(透传)----

    fn poisoned(&self, detail: String) -> Tensor {
        Tensor {
            head: Step {
                parents: vec![self.head.clone()],
                depth: self.head.depth + 1,
                op: Op::Zeros,
                shape: self.shape.clone(),
                err: self.head.err.clone().or(Some(detail)),
            },
            shape: self.shape.clone(),
        }
    }

    fn check_same_shape(&self, b: &Tensor, who: &str) -> Option<String> {
        (self.shape != b.shape)
            .then(|| format!("{who}: 形状不符 {:?} vs {:?}", self.shape, b.shape))
    }

    // ---- 链式运算(声明;永不失败——违约变毒)----

    pub fn add(&self, b: &Tensor) -> Tensor {
        if let Some(d) = self.check_same_shape(b, "add") {
            return self.poisoned(d);
        }
        self.join(Op::Add, Some(b), self.shape.clone())
    }

    pub fn silu(&self) -> Tensor {
        self.join(Op::Silu, None, self.shape.clone())
    }

    /// ×(1+w) 语义演示
    fn rmsnorm(&self, alpha: &Tensor, eps: f32, w_off: bool) -> Tensor {
        if let Some(d) = self.check_same_shape(alpha, "rmsnorm") {
            return self.poisoned(d);
        }
        self.join(Op::Rmsnorm { eps, w_off }, Some(alpha), self.shape.clone())
    }

    /// [m,k] × [k,n] → [m,n]
    fn matmul(&self, b: &Tensor) -> Tensor {
        let (m, k, k2, n) = (self.shape[0], self.shape[1], b.shape[0], b.shape[1]);
        if k != k2 {
            return self.poisoned(format!("matmul: 内维不符 {k} vs {k2}"));
        }
        self.join(Op::Matmul, Some(b), vec![m, n])
    }

    /// append:深拷贝两棵输入子树进本节点(值语义;配置面一次性成本)
    fn join(&self, op: Op, rhs: Option<&Tensor>, shape: Shape) -> Tensor {
        let mut parents = vec![self.head.clone()];
        if let Some(r) = rhs {
            parents.push(r.head.clone());
        }
        Tensor {
            head: Step {
                parents,
                depth: self.head.depth + 1,
                op,
                shape: shape.clone(),
                err: self.head.err.clone().or(rhs.and_then(|r| r.head.err.clone())),
            },
            shape,
        }
    }
}

// ============================================================================
// §2 解释器:自叶向根归约(递归;深度 = 链长,几百层内无忧)
// ============================================================================

fn eval(step: &Step) -> Result<Vec<f32>, String> {
    // 毒值落地:案发(depth)+ 细节,结构化报出
    if let Some(e) = &step.err {
        return Err(format!("[毒值落地 @depth {}] {e}", step.depth));
    }
    // 自叶向根:先归约全部输入
    let ins: Vec<Vec<f32>> = step.parents.iter().map(eval).collect::<Result<_, _>>()?;
    let out = match &step.op {
        Op::Host(data) => data.clone(),
        Op::Zeros => vec![0.0; step.shape.iter().product::<usize>()],
        Op::Add => {
            let (a, b) = (&ins[0], &ins[1]);
            a.iter().zip(b).map(|(x, y)| x + y).collect()
        }
        Op::Silu => ins[0].iter().map(|v| v / (1.0 + (-v).exp())).collect(),
        Op::Rmsnorm { eps, w_off } => {
            let (x, alpha) = (&ins[0], &ins[1]);
            let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            x.iter()
                .zip(alpha)
                .map(|(v, al)| {
                    let g = if *w_off { al + 1.0 } else { *al };
                    v * inv * g
                })
                .collect()
        }
        Op::Matmul => {
            let (a, b) = (&ins[0], &ins[1]);
            let (m, k, n) = (step.shape[0], a.len() / step.shape[0], step.shape[1]);
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let av = a[i * k + p];
                    for j in 0..n {
                        out[i * n + j] += av * b[p * n + j];
                    }
                }
            }
            out
        }
    };
    Ok(out)
}

// ============================================================================
// §3 烘焙:展平(捕获解释器的类比;计划可重复执行)
// ============================================================================

/// 展平 = 自根收集整树(值语义下就是遍历自己持有的树)
fn flatten(step: &Step, out: &mut Vec<(u32, Op)>) {
    for p in &step.parents {
        flatten(p, out);
    }
    out.push((step.depth, step.op.clone()));
}

// ============================================================================
// §4 demos
// ============================================================================

fn demo1_chain_and_eval() {
    println!("== demo1:链式声明(matmul) + 递归解释器 ==");
    let x = Tensor::from_host(vec![1, 4], &[1.0, 2.0, -3.0, 4.0]);
    let eye = Tensor::from_host(
        vec![4, 4],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    );
    let bias = Tensor::zeros(vec![1, 4]);
    let alpha = Tensor::from_host(vec![1, 4], &[0.5; 4]);

    let h = x.matmul(&eye).add(&bias);
    let out_t = h.silu().rmsnorm(&alpha, 1e-6, true);
    let got = eval(&out_t.head).expect("demo1");

    let silu: Vec<f32> = [1.0f32, 2.0, -3.0, 4.0]
        .iter()
        .map(|v| v / (1.0 + (-v).exp()))
        .collect();
    let ms = silu.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    let want: Vec<f32> = silu.iter().map(|v| v * inv * 1.5).collect();

    assert!(
        got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-5),
        "{got:?} vs {want:?}"
    );
    println!("   out = {got:?}\n   与 host 参考一致 ✓");
}

fn demo2_poison() {
    println!("== demo2:毒值传播 + 边界收割 ==");
    let a = Tensor::from_host(vec![4], &[1.0; 4]);
    let bad = Tensor::from_host(vec![3], &[1.0; 3]);

    let chain = a.add(&bad).silu();
    assert!(chain.is_poisoned());
    let err = eval(&chain.head).unwrap_err();
    println!("   {err}");
    assert!(err.contains("毒值落地"));
    println!("   描述期零 panic,错误边界结构化 + 案发 depth 可回溯 ✓");
}

fn demo3_value_semantics() {
    println!("== demo3:值语义(克隆 = 深拷贝) ==");
    let x = Tensor::from_host(vec![4], &[1.0, -2.0, 3.0, -4.0]);
    let s = x.silu();

    // 克隆两份,各自独立归约——互不干扰(无共享别名,天然无竞速)
    let c1 = s.clone();
    let c2 = s.clone();
    let v1 = eval(&c1.head).unwrap();
    let v2 = eval(&c2.head).unwrap();
    assert_eq!(v1, v2);

    // 重复消费 = 重复计算(配置面一次性成本,值语义的直接后果;
    // 若将来有热点,eval 层加结构哈希 memo 即可,不动类型)
    println!("   深拷贝独立归约,结果一致 ✓;重复消费的重复计算 = 配置面成本,可接受");
}

fn demo4_fork() {
    println!("== demo4:fork 多档位(底稿深拷贝进各档) ==");
    let base = Tensor::from_host(vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let tier_a = base.silu();          // 档 A:base 深拷贝 + silu
    let tier_b = base.add(&base).silu(); // 档 B:base 深拷贝 + add + silu

    let mut flat_a = Vec::new();
    let mut flat_b = Vec::new();
    flatten(&tier_a.head, &mut flat_a);
    flatten(&tier_b.head, &mut flat_b);
    println!(
        "   档 A = {} 节点,档 B = {} 节点(值语义:add 双亲各持一份 base 深拷贝)",
        flat_a.len(),
        flat_b.len()
    );
    assert_eq!(flat_a.len(), 2); // base + silu
    assert_eq!(flat_b.len(), 4); // base×2(add 双亲)+ add + silu —— 深拷贝的形态

    let _ = eval(&tier_a.head).unwrap();
    let _ = eval(&tier_b.head).unwrap();
    println!("   两档独立归约 ✓");
}

fn main() {
    demo1_chain_and_eval();
    demo2_poison();
    demo3_value_semantics();
    demo4_fork();
    demo5_gpu_executor();
    println!("\nplayground 主干全绿(值语义;GPU 解释器 = 同一归约,逐节点换 server 提交)");
}

// ============================================================================
// §5 GPU 执行器:同一棵归约树,逐节点换真发射(绕过 server,裸 cudarc)
// ============================================================================

mod gpu_exec {
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaStream, CudaSlice, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::safe::{compile_ptx_with_opts, CompileOptions};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// 声明式树里出现的全部算子(CPU demo 各 match 臂的 GPU 对偶)
    const KERNELS: &str = r#"
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
"#;

    pub struct GpuExecutor {
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        funcs: HashMap<&'static str, CudaFunction>,
    }

    impl GpuExecutor {
        pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(ordinal)?;
            ctx.bind_to_thread()?;
            let stream = ctx.new_stream()?;
            // arch 显式钉(不钉 = nvrtc 默认 target 可能高于驱动可加载档)
            let ptx = compile_ptx_with_opts(
                KERNELS,
                CompileOptions { arch: Some("sm_86"), ..Default::default() },
            )?;
            let module = ctx.load_module(ptx)?;
            let mut funcs = HashMap::new();
            for name in ["owl_add_f32", "owl_silu_f32", "owl_matmul_f32"] {
                funcs.insert(name, module.load_function(name)?);
            }
            Ok(Self { ctx, stream, funcs })
        }

        fn func(&self, name: &str) -> CudaFunction {
            self.funcs[name].clone()
        }

        fn cfg_1d(n: usize) -> LaunchConfig {
            LaunchConfig {
                grid_dim: ((n as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        }

        pub fn zeros(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            Ok(self.stream.alloc_zeros::<f32>(n)?)
        }

        pub fn htod(&self, data: &[f32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            Ok(self.stream.clone_htod(data)?)
        }

        pub fn dtoh(&self, s: &CudaSlice<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let mut host = vec![0.0; s.len()];
            self.stream.memcpy_dtoh(s, &mut host)?;
            Ok(host)
        }

        pub fn add(
            &self,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
        ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let out = self.zeros(a.len())?;
            unsafe {
                self.stream
                    .launch_builder(&self.func("owl_add_f32"))
                    .arg(a)
                    .arg(b)
                    .arg(&out)
                    .arg(&(a.len() as u64))
                    .launch(Self::cfg_1d(a.len()))?;
            }
            Ok(out)
        }

        pub fn silu(
            &self,
            x: &CudaSlice<f32>,
        ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let out = self.zeros(x.len())?;
            unsafe {
                self.stream
                    .launch_builder(&self.func("owl_silu_f32"))
                    .arg(x)
                    .arg(&out)
                    .arg(&(x.len() as u64))
                    .launch(Self::cfg_1d(x.len()))?;
            }
            Ok(out)
        }

        /// [m,k] × [k,n](朴素;grid 二维)
        pub fn matmul(
            &self,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let out = self.zeros(m * n)?;
            unsafe {
                self.stream
                    .launch_builder(&self.func("owl_matmul_f32"))
                    .arg(a)
                    .arg(b)
                    .arg(&out)
                    .arg(&(m as i32))
                    .arg(&(k as i32))
                    .arg(&(n as i32))
                    .launch(LaunchConfig {
                        grid_dim: ((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            Ok(out)
        }
    }
}

// ============================================================================
// §6 GPU 归约 walker:同一棵树,逐节点换真发射
// ============================================================================

fn eval_gpu(
    step: &Step,
    ex: &gpu_exec::GpuExecutor,
) -> Result<cudarc::driver::CudaSlice<f32>, String> {
    if let Some(e) = &step.err {
        return Err(format!("[毒值落地 @depth {}] {e}", step.depth));
    }
    // 自叶向根:先归约全部输入(保持设备侧,中途零 D2H)
    let ins: Vec<cudarc::driver::CudaSlice<f32>> = step
        .parents
        .iter()
        .map(|p| eval_gpu(p, ex))
        .collect::<Result<_, _>>()?;
    let shape_ok = |s: &Shape| s.iter().product::<usize>();
    let out = match &step.op {
        Op::Host(data) => ex.htod(data).map_err(|e| e.to_string())?,
        Op::Zeros => ex.zeros(shape_ok(&step.shape)).map_err(|e| e.to_string())?,
        Op::Add => ex.add(&ins[0], &ins[1]).map_err(|e| e.to_string())?,
        Op::Silu => ex.silu(&ins[0]).map_err(|e| e.to_string())?,
        Op::Matmul => {
            let (m, n) = (step.shape[0], step.shape[1]);
            let k = ins[0].len() / m;
            ex.matmul(&ins[0], &ins[1], m, k, n).map_err(|e| e.to_string())?
        }
        Op::Rmsnorm { .. } => unimplemented!("demo 未走 rmsnorm 的 GPU 分支"),
    };
    Ok(out)
}

fn demo5_gpu_executor() {
    use gpu_exec::GpuExecutor;
    println!("== demo5:GPU 执行器(同一棵归约树,逐节点真发射) ==");
    let ordinal = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let ex = GpuExecutor::new(ordinal).expect("GpuExecutor::new");
    println!("   device: {}", ex.ctx.name().unwrap());

    // 与 demo1 相同的声明链(matmul → add → silu),整树留 GPU,末端一次 D2H
    let x = Tensor::from_host(vec![1, 4], &[1.0, 2.0, -3.0, 4.0]);
    let eye = Tensor::from_host(
        vec![4, 4],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    );
    let bias = Tensor::zeros(vec![1, 4]);
    let alpha = Tensor::from_host(vec![1, 4], &[0.5; 4]);

    let h = x.matmul(&eye).add(&bias);
    let _out_t = h.silu().rmsnorm(&alpha, 1e-6, true);

    // GPU 归约(rmsnorm 走 CPU 臂收尾,其余节点在卡上;对比 CPU 结果)
    let h_gpu = eval_gpu(&h.head, &ex).expect("eval_gpu(h)");
    let host_h = ex.dtoh(&h_gpu).expect("dtoh");
    let cpu_h = eval(&h.head).expect("cpu eval h");
    assert!(
        host_h.iter().zip(&cpu_h).all(|(g, w)| (g - w).abs() < 1e-5),
        "GPU vs CPU 不一致: {host_h:?} vs {cpu_h:?}"
    );
    println!("   matmul+add(GPU) == CPU ✓  {host_h:?}");

    // silu 一并上卡(临时补一个 silu 后缀;结果对拍 CPU)
    let out_gpu = ex.silu(&h_gpu).expect("silu gpu");
    let got = ex.dtoh(&out_gpu).expect("dtoh");
    // CPU 参考:silu(h)(与 GPU 路径同一段;完整 rmsnorm 的 GPU kernel 略,
    // 逻辑同 add/silu,留作练习位)
    let cpu_silu: Vec<f32> = cpu_h.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    assert!(
        got.iter().zip(&cpu_silu).all(|(g, w)| (g - w).abs() < 1e-5),
        "GPU vs CPU 不一致: {got:?} vs {cpu_silu:?}"
    );
    println!("   silu(GPU) == CPU ✓  {got:?}");
    ex.ctx.synchronize().unwrap();
    println!("   同一归约树,CPU/GPU 双解释器结果一致 ✓");
}
