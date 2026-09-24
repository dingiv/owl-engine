//! GPU server:actor 线程 + 池块账房 + kernel 发射。
//! 实现 owl-models 的 `DeviceClient` 能力契约。

use cudarc::nvrtc::safe::{compile_ptx_with_opts, CompileOptions};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg,
};
use owl_models::client::{Arg, Bytes, DeviceClient, LaunchMsg};
use owl_models::{Dtype, ModelError};
use owl_models::shape::Shape;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;
use std::sync::{Condvar, Mutex};

struct GpuCtx {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    compiled: HashMap<String, CudaFunction>,
    blocks: HashMap<u64, (CudaSlice<f32>, usize)>,
    next_block: u64,
}

impl GpuCtx {
    fn new(ordinal: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e:?}"))?;
        ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            ctx,
            stream,
            compiled: HashMap::new(),
            blocks: HashMap::new(),
            next_block: 1,
        })
    }

    fn new_block(&mut self, slice: CudaSlice<f32>) -> u64 {
        let n = slice.len();
        let id = self.next_block;
        self.next_block += 1;
        self.blocks.insert(id, (slice, n));
        id
    }

    fn block(&self, id: u64) -> Result<&(CudaSlice<f32>, usize), ModelError> {
        self.blocks
            .get(&id)
            .ok_or_else(|| ModelError::DeadBlock { id })
    }

    fn ensure_kernel(&mut self, name: &str, ptx: &str) -> Result<CudaFunction, ModelError> {
        if let Some(f) = self.compiled.get(name) {
            return Ok(f.clone());
        }
        // CUDA C 源 → nvrtc → PTX → module
        let ptx = compile_ptx_with_opts(
            ptx,
            CompileOptions { arch: Some("compute_86"), ..Default::default() },
        )
        .map_err(|e| ModelError::Msg(format!("nvrtc({name}): {e:?}")))?;
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|e| ModelError::Msg(format!("load_module({name}): {e:?}")))?;
        let func = module
            .load_function(name)
            .map_err(|e| ModelError::Msg(format!("load_function({name}): {e:?}")))?;
        self.compiled.insert(name.to_string(), func.clone());
        Ok(func)
    }
}

type Job = Box<dyn FnOnce(&mut GpuCtx) + Send>;

struct GpuActor {
    rx: mpsc::Receiver<Job>,
    ctx: GpuCtx,
}

impl GpuActor {
    fn pump(mut self) {
        while let Ok(job) = self.rx.recv() {
            job(&mut self.ctx);
        }
    }
}

struct Ack<T>(Arc<(Mutex<Option<T>>, Condvar)>);

impl<T> Ack<T> {
    fn pair() -> (Self, Waiter<T>) {
        let inner = Arc::new((Mutex::new(None), Condvar::new()));
        (Self(inner.clone()), Waiter(inner))
    }
    fn send(self, v: T) {
        let (lock, cv) = &*self.0;
        *lock.lock().unwrap() = Some(v);
        cv.notify_all();
    }
}

struct Waiter<T>(Arc<(Mutex<Option<T>>, Condvar)>);

impl<T> Waiter<T> {
    fn wait(self) -> T {
        let (lock, cv) = &*self.0;
        let mut g = lock.lock().unwrap();
        while g.is_none() {
            g = cv.wait(g).unwrap();
        }
        g.take().unwrap()
    }
}

#[derive(Clone)]
pub struct GpuClient {
    tx: mpsc::Sender<Job>,
}

impl GpuClient {
    pub fn spawn(ordinal: usize) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<Job>();
        let (boot_tx, boot_rx) = Ack::<Result<(), String>>::pair();
        std::thread::Builder::new()
            .name(format!("owl-gpu-{ordinal}"))
            .spawn(move || match GpuCtx::new(ordinal) {
                Ok(ctx) => {
                    boot_tx.send(Ok(()));
                    GpuActor { rx, ctx }.pump();
                }
                Err(e) => boot_tx.send(Err(e)),
            })
            .map_err(|e| format!("actor 线程启动失败: {e}"))?;
        boot_rx.wait()?;
        Ok(Self { tx })
    }

    fn submit<R, F>(&self, job: F) -> Result<R, ModelError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuCtx) -> Result<R, ModelError> + Send + 'static,
    {
        let (ack, waiter) = Ack::pair();
        self.tx
            .send(Box::new(move |ctx: &mut GpuCtx| {
                ack.send(job(ctx));
            }))
            .map_err(|_| ModelError::ServerClosed)?;
        waiter.wait()
    }
}

impl DeviceClient for GpuClient {
    async fn alloc(&mut self, n_bytes: usize) -> Result<Bytes, ModelError> {
        let n = n_bytes / 4;
        self.submit(move |ctx: &mut GpuCtx| {
            let slice = ctx.stream.alloc_zeros::<f32>(n).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
            let id = ctx.new_block(slice);
            Ok(Bytes::new(id, n))
        })
    }

    async fn htod(
        &mut self,
        _dtype: Dtype,
        _shape: &Shape,
        src: &[u8],
    ) -> Result<Bytes, ModelError> {
        let f32v: Vec<f32> = src
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.submit(move |ctx: &mut GpuCtx| {
            let slice = ctx.stream.clone_htod(&f32v).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
            let id = ctx.new_block(slice);
            Ok(Bytes::new(id, f32v.len()))
        })
    }

    async fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError> {
        let id = b.id;
        let want_f32 = out.len() / 4;
        self.submit(move |ctx: &mut GpuCtx| {
            let (slice, n) = ctx.block(id).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
            if *n != want_f32 {
                return Err(ModelError::Msg(format!("dtoh: 块 {id} 元素 {n} != 收割 {want_f32}")));
            }
            let mut host = vec![0.0f32; *n];
            ctx.stream
                .memcpy_dtoh(slice, &mut host)
                .map_err(|e| ModelError::Msg(format!("{e:?}")))?;
            let mut bytes = Vec::with_capacity(host.len() * 4);
            for f in &host {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            Ok(bytes)
        })
        .map(|bytes| out.copy_from_slice(&bytes))
    }

    async fn launch(&mut self, msg: LaunchMsg) -> Result<Bytes, ModelError> {
        self.submit(move |ctx: &mut GpuCtx| {
            // 参数槽:类型化入包(与 kernel 形参宽度严格对位)
            enum Slot {
                Ptr(u64),
                U(u64),
                I(i32),
                F(f32),
            }
            let mut slots: Vec<Slot> = Vec::with_capacity(msg.args.len());
            for a in &msg.args {
                match a {
                    Arg::Block { id } => {
                        let (slice, _) = ctx.block(*id).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
                        let (p, _s) = slice.device_ptr(&ctx.stream);
                        slots.push(Slot::Ptr(p as u64));
                    }
                    Arg::U64(v) => slots.push(Slot::U(*v)),
                    Arg::I32(v) => slots.push(Slot::I(*v)),
                    Arg::F32(v) => slots.push(Slot::F(*v)),
                }
            }
            let func = ctx.ensure_kernel(&msg.kernel.name, &msg.kernel.source)?;
            unsafe {
                let mut builder = ctx.stream.launch_builder(&func);
                for slot in &slots {
                    match slot {
                        Slot::Ptr(v) => builder.arg(v),
                        Slot::U(v) => builder.arg(v),
                        Slot::I(v) => builder.arg(v),
                        Slot::F(v) => builder.arg(v),
                    };
                }
                builder
                    .launch(LaunchConfig {
                        grid_dim: msg.grid,
                        block_dim: msg.block,
                        shared_mem_bytes: msg.shared_mem,
                    })
                    .map_err(|e| {
                        ModelError::Msg(format!("launch({}): {e}", msg.kernel.name))
                    })?;
            }
            // out 块由 eval 先 alloc 再进 args,发射后原样回传句柄
            let out_id = msg
                .args
                .iter()
                .rev()
                .find_map(|a| match a {
                    Arg::Block { id } => Some(*id),
                    Arg::U64(_) | Arg::I32(_) | Arg::F32(_) => None,
                })
                .ok_or_else(|| ModelError::Msg("launch: args 中无输出块".to_string()))?;
            Ok(Bytes::new(out_id, msg.out_elems))
        })
    }

    async fn sync(&mut self) -> Result<(), ModelError> {
        self.submit(move |ctx: &mut GpuCtx| {
            ctx.ctx.synchronize().map_err(|e| ModelError::Msg(format!("{e:?}")))?;
            Ok(())
        })
    }
}
