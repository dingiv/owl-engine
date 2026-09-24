//! 裸 cudarc 参考示例:算子发射(nvrtc 编译 kernel)+ 流捕获 → 图回放。
//!
//! 目的:与 owl 的治理层(owl-cuda 的相位机/池/租约)对照,展示
//! "没有治理时"原厂 safe API 的样子与坑。文件自带全部语义注释,
//! 可作为图捕获语义的速查样板。
//!
//! 运行:`cargo run -p owl-cuda --example cudarc_ops_and_graph`
//! (设备序号:环境变量 OWL_TEST_DEVICE,默认 0)
//!
//! 对照要点(读完本文件再去看 owl-cuda 就知道治理加了什么):
//! 1. cudarc 的 CudaSlice 是 stream-ordered bump 分配——drop 即归还
//!    底层池,但**没有跨分配的账本**,捕获期分配 = 图外悬空的经典坑;
//! 2. 图捕获窗口内只允许本流的 kernel launch(memcpy 事件等都是雷);
//!    本例因此**先分配后捕获**,捕获闭包内只有 launch;
//! 3. CudaGraph::launch 回放到**捕获时绑定的同一批指针**——指针换地址
//!    即回放踩旧地址,这正是 owl "BufToken/租约"要封死的姿势 3。

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::safe::{compile_ptx_with_opts, CompileOptions};
use std::time::Instant;

/// saxpy:y = a * x + y(经典入门核;标量 a 走值参数,无 htod)
const KERNEL_SRC: &str = r#"
extern "C" __global__ void saxpy_f32(
    const float a,
    const float* x,
    const float* y,
    float* out,
    const size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaf(a, x[i], y[i]);
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ordinal: usize = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // ---- 1. 上下文与流(cudarc 每卡一个 CudaContext;clone = Arc 克隆)----
    let ctx = CudaContext::new(ordinal)?;
    ctx.bind_to_thread()?;
    // 关键一击:cudarc 默认开 event-tracking——每次 launch 给 CudaSlice
    // 挂 read/write 事件,下一次发射 = 等上一发射的事件。捕获窗内出现
    // 这种等待 = 依赖未捕获工作 → CAPTURE_ISOLATION(本例实测踩过)。
    // 图捕获场景必须关掉:同步秩序由我们自己管理(owl 也是这么治理的)。
    // Safety:demo 进程内没有第三方代码依赖 cudarc 的事件秩序
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?; // non-blocking 流(与 NULL 流无隐式同步)
    // demo 全程只用这一条流(分配/eager/捕获/回放);跨流引用未捕获工作
    // 或 legacy 依赖,任一都会让 begin_capture 报 CAPTURE_ISOLATION
    println!("[init] device {ordinal}: {}", ctx.name()?);

    // ---- 2. nvrtc 编译 + 装载(源码 → PTX → CudaModule → CudaFunction)----
    // arch 必须显式钉到本卡 compute capability:nvrtc 默认 target 若高于
    // 驱动可加载的最高档,load_module 会报 INVALID_IMAGE(常见坑)。
    let ptx = compile_ptx_with_opts(
        KERNEL_SRC,
        CompileOptions {
            arch: Some("sm_86"), // 本机 = 3090/3080(Ampere sm_86)
            ..Default::default()
        },
    )?;
    let module = ctx.load_module(ptx)?;
    let saxpy = module.load_function("saxpy_f32")?;
    println!("[nvrtc] saxpy_f32 编译装载完成");

    const N: usize = 1 << 20;
    let a = 2.5f32;

    // ---- 3. 分配 + H2D(注意:全部发生在捕获窗口之外)----
    // host 输入用 page-locked 内存:pageable 指针的 memcpy 会被驱动经
    // legacy NULL 流中转,在流上留下 legacy 依赖 → 之后 begin_capture
    // 必报 CAPTURE_ISOLATION(本例实测踩过;这就是 pinned 的意义)
    let mut x_pin = unsafe { ctx.alloc_pinned::<f32>(N)? };
    let mut y_pin = unsafe { ctx.alloc_pinned::<f32>(N)? };
    let x_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.5 - 100.0).collect();
    let y_host: Vec<f32> = (0..N).map(|i| (i % 13) as f32).collect();
    x_pin.as_mut_slice()?.copy_from_slice(&x_host);
    y_pin.as_mut_slice()?.copy_from_slice(&y_host);

    let x = stream.clone_htod(x_pin.as_slice()?)?; // CudaSlice<f32>:stream-ordered 分配
    let y = stream.clone_htod(y_pin.as_slice()?)?;
    let mut out = stream.alloc_zeros::<f32>(N)?;

    // ---- 4. eager 发射:builder 链式 arg → launch ----
    let eager_launch = |x: &cudarc::driver::CudaSlice<f32>,
                        y: &cudarc::driver::CudaSlice<f32>,
                        out: &mut cudarc::driver::CudaSlice<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        // arg 返回 &mut Self 链式;标量 = DeviceRepr 值参数(无 htod;裁决 3① 同源思想)
        // 一维网格:ceil-div;256 线程/块(与 owl LaunchConfig 同构)
        let cfg = LaunchConfig {
            grid_dim: ((N as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&saxpy)
                .arg(&a)
                .arg(x)
                .arg(y)
                .arg(out)
                .arg(&N) // 参数个数必须与 kernel 签名严格一致(漏一个 = INVALID_VALUE)
                .launch(cfg)?;
        }
        Ok(())
    };

    eager_launch(&x, &y, &mut out)?;
    stream.synchronize()?;
    // host 期望值(捕获前不能有任何 pageable memcpy:驱动经 legacy NULL
    // 流中转会给流留依赖,begin_capture 必被 CAPTURE_ISOLATION 拒绝)
    let want: Vec<f32> = (0..N)
        .map(|i| f32::mul_add(a, x_host[i], y_host[i]))
        .collect();
    drop(x_pin); // pinned 块用完即还(Drop 自带 event.synchronize)
    drop(y_pin);

    // ---- 5. 图捕获:begin → (窗口内只许本流 launch)→ end/instantiate ----
    // 本例实测踩过并写入注释的两个坑:
    //   坑 A(event-tracking):cudarc 默认给 launch 挂跨流事件等待,
    //     捕获窗内 = 依赖未捕获工作 → CAPTURE_ISOLATION(已用
    //     disable_event_tracking 关闭);
    //   坑 B(pageable):pageable host 指针的 memcpy 被驱动经 legacy
    //     NULL 流中转,流上留下 legacy 依赖 → begin_capture 被拒
    //     (已改用 page-locked 的 alloc_pinned);
    // 姿势纪律(与 owl 治理层一一对应的"裸"版本):
    //   - 分配/H2D/D2H/同步 绝不入窗(入窗 = 从图里消失或直接错误);
    //   - 窗口内多次 launch 会被录成**线性图**(单流 = 隐式依赖链);
    //   - relaxed 模式允许窗口内出现阻塞调用而不报错——但那正是坑,
    //     生产用 ThreadLocal/Global + 窗口内纯净(owl 用相位机硬封)。
    // 捕获就发生在 stream 自己身上(分配与 last-write 同流:跨流引用
    // 未捕获流的工作 = CAPTURE_ISOLATION)
    ctx.synchronize()?;
    stream.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;

    // 捕获 3 连发:同流 launch 录成依赖链 n1 → n2 → n3
    // (参数 = 第 4 步同一批指针;回放即重写同一批显存)
    unsafe {
        stream
            .launch_builder(&saxpy)
            .arg(&a)
            .arg(&x)
            .arg(&y)
            .arg(&mut out)
            .arg(&N)
            .launch(LaunchConfig {
            grid_dim: ((N as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let graph = stream
        .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY)?
        .expect("捕获窗口内无任何可录制操作时会返回 None(空窗)");
    graph.upload()?; // 预上传资源,消首放 setup 开销
    println!("[capture] 图实例化 + upload 完成");

    // ---- 6. 回放:同一批指针上的整链重写 ----
    // 先把 out 打脏,验证回放真的重写了它(而非读到旧值)
    stream.memcpy_htod(&vec![-1.0f32; N], &mut out)?;

    const REPLAYS: usize = 1000;
    let t0 = Instant::now();
    for _ in 0..REPLAYS {
        graph.launch()?; // 整链一次发射(3 节点 → 1 次 graph launch)
    }
    stream.synchronize()?;
    let dt = t0.elapsed();
    println!(
        "[replay] {REPLAYS} 次回放,平均 {:.2} µs/次(整链 1 节点集)",
        dt.as_micros() as f64 / REPLAYS as f64
    );

    let mut got_replay = vec![0.0f32; N];
    stream.memcpy_dtoh(&out, &mut got_replay)?;
    stream.synchronize()?;
    assert_eq!(want, got_replay, "回放结果必须与 host 期望逐位一致");
    println!("[verify] 回放 == host 期望(逐位一致)");

    // ---- 7. 生命周期:drop 顺序即治理(cudarc 自动:graph exec → graph;
    // CudaSlice drop = stream-ordered free)——但"谁还活着"完全靠人工,
    // 没有哨兵、没有租约、没有世代校验:这就是 owl 治理层存在的理由。
    drop(graph);
    drop(out);
    drop(y);
    drop(x);
    println!("[done] 全部资源已随 drop 归还");
    Ok(())
}
