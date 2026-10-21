//! **命令面契约门**(2026-10-21):具名方法(糖)→ `Command` → `exec` 的单源性。
//!
//! 为什么需要它:`DeviceClient` 的 ~35 个具名方法现在全是**缺省糖**
//! (内部构造 `Command` 调 `exec`)。若某一条糖的字段映射漂了(流选择、
//! `zero` 语义、偏移、dtype),真实后端(尤其是 GPU)会以"错流/错清零/
//! 错偏移"的方式静默错值 —— 这正是旧契约里"每个后端各写一遍"的老坑。
//!
//! 本门用**只实现 exec/exec_batch 的假 face**逐个验字段,不依赖任何真卡。

use std::sync::{Arc, Mutex};

use owl_iface::command::{BatchOutcome, Command, Outcome};
use owl_iface::contract::{
    Bytes, DeviceClient, Dtype, LaunchMsg, KernelSource, ModelError, UpStream,
};

/// 只记录命令 + 按命令变体产生**定型结果**的 face(不实现任何具名方法)。
struct RecFace {
    cmds: Arc<Mutex<Vec<Command>>>,
    /// 让 `alloc` 返回 `Outcome::Done`(测变体不符的响亮拒)
    poison_alloc: bool,
    /// 累积批队列(`queue`/`flush`)
    pending: Vec<Command>,
}

impl RecFace {
    fn new() -> (Self, Arc<Mutex<Vec<Command>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (Self { cmds: log.clone(), poison_alloc: false, pending: Vec::new() }, log)
    }
}

impl DeviceClient for RecFace {
    async fn exec(&mut self, cmd: Command) -> Result<Outcome, ModelError> {
        // 契约:队列非空 ⇒ 单条命令响亮拒(各实现自己的守卫)
        if !self.pending.is_empty() {
            return Err(ModelError::Msg(format!(
                "命令面:还有 {} 条命令未提交(先 `flush()`)",
                self.pending.len()
            )));
        }
        let out = match &cmd {
            Command::Alloc { dtype, elems, .. } => {
                if self.poison_alloc {
                    Outcome::Done
                } else {
                    Outcome::Buf(Bytes::new(7, *elems, *dtype, 0))
                }
            }
            Command::Htod { dtype, shape, src, .. } => {
                Outcome::Buf(Bytes::new(8, src.len(), *dtype, shape.len() as u8))
            }
            Command::Dtoh { want_bytes, .. } => Outcome::Bytes(vec![9u8; *want_bytes]),
            Command::Launch { msg } => Outcome::Buf(Bytes::new(9, msg.out_elems, Dtype::F32, 1)),
            Command::GraphEnd => Outcome::Graph(42),
            _ => Outcome::Done,
        };
        self.cmds.lock().expect("log").push(cmd);
        Ok(out)
    }

    fn queue(&mut self, cmd: Command) -> Result<usize, ModelError> {
        let i = self.pending.len();
        self.pending.push(cmd);
        Ok(i)
    }

    async fn flush(&mut self) -> Result<BatchOutcome, ModelError> {
        if self.pending.is_empty() {
            return Ok(BatchOutcome::empty());
        }
        let cmds = std::mem::take(&mut self.pending);
        self.exec_batch(cmds).await
    }

    async fn exec_batch(&mut self, cmds: Vec<Command>) -> Result<BatchOutcome, ModelError> {
        let mut outcomes = Vec::with_capacity(cmds.len());
        for c in cmds {
            outcomes.push(self.exec(c).await);
        }
        let n = outcomes.len();
        Ok(BatchOutcome::collect(outcomes, n))
    }
}

fn msg(n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: KernelSource { name: "k".into(), source: String::new() },
        args: vec![],
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// ① `alloc` / `alloc_uninit`:唯一差别 = `zero` 位。
#[tokio::test]
async fn alloc_sugar_maps_to_zero_flag() {
    let (mut f, log) = RecFace::new();
    let b = f.alloc(Dtype::F16, 64, "w").await.expect("alloc");
    assert_eq!((b.len, b.dtype), (64, Dtype::F16));
    let _ = f.alloc_uninit(Dtype::F16, 32, "scratch").await.expect("uninit");

    let g = log.lock().expect("log");
    match &g[0] {
        Command::Alloc { dtype, elems, zero, name } => {
            assert_eq!((*dtype, *elems, *zero, name.as_str()), (Dtype::F16, 64, true, "w"));
        }
        other => panic!("alloc 应产生 Command::Alloc,得 {}", other.kind()),
    }
    match &g[1] {
        Command::Alloc { zero, elems, .. } => {
            assert_eq!((*zero, *elems), (false, 32), "alloc_uninit = zero:false");
        }
        other => panic!("alloc_uninit 应产生 Command::Alloc,得 {}", other.kind()),
    }
}

/// ② htod 三条入口的**流选择**(批量面 = H2D;参数面 = COMPUTE;f32 直传 = COMPUTE)。
#[tokio::test]
async fn htod_streams_are_explicit() {
    let (mut f, log) = RecFace::new();
    let data = vec![0u8; 8];
    let _ = f.htod(Dtype::F16, &vec![4], &data, "batch").await.expect("htod");
    let _ = f.htod_on(UpStream::Compute, Dtype::F16, &vec![4], &data, "param").await.expect("htod_on");
    let _ = f.htod_f32(&vec![2], vec![1.0, 2.0], "f32").await.expect("htod_f32");

    let g = log.lock().expect("log");
    let streams: Vec<(UpStream, Dtype, String)> = g
        .iter()
        .map(|c| match c {
            Command::Htod { stream, dtype, name, .. } => (*stream, *dtype, name.clone()),
            other => panic!("应全是 Htod,得 {}", other.kind()),
        })
        .collect();
    assert_eq!(streams[0], (UpStream::H2d, Dtype::F16, "batch".into()));
    assert_eq!(streams[1], (UpStream::Compute, Dtype::F16, "param".into()));
    assert_eq!(
        streams[2],
        (UpStream::Compute, Dtype::F32, "f32".into()),
        "htod_f32 是参数面(同流 FIFO),且 dtype = F32"
    );
}

/// ③ dtoh / dtoh_at:偏移如实进命令,回执字节落进调用方缓冲。
#[tokio::test]
async fn dtoh_offset_and_result_copy() {
    let (mut f, log) = RecFace::new();
    let b = Bytes::new(3, 4, Dtype::U8, 0);
    let mut out = [0u8; 4];
    f.dtoh(&b, &mut out).await.expect("dtoh");
    assert_eq!(out, [9u8; 4], "回执字节必须抄进调用方缓冲");
    let mut out2 = [0u8; 2];
    f.dtoh_at(&b, 16, &mut out2).await.expect("dtoh_at");

    let g = log.lock().expect("log");
    match &g[0] {
        Command::Dtoh { id, off_bytes, want_bytes } => {
            assert_eq!((*id, *off_bytes, *want_bytes), (3, 0, 4));
        }
        other => panic!("dtoh 应产生 Command::Dtoh,得 {}", other.kind()),
    }
    match &g[1] {
        Command::Dtoh { off_bytes, want_bytes, .. } => {
            assert_eq!((*off_bytes, *want_bytes), (16, 2), "偏移不得被吞");
        }
        other => panic!("dtoh_at 应产生 Command::Dtoh,得 {}", other.kind()),
    }
}

/// ④ launch / sync / free / graph_end:命令变体 + 结果变体一一对应。
#[tokio::test]
async fn launch_sync_free_graph_map_to_commands() {
    let (mut f, log) = RecFace::new();
    let b = f.launch(msg(12)).await.expect("launch");
    assert_eq!((b.len, b.rank), (12, 1));
    f.sync().await.expect("sync");
    f.free(&[1, 2]).await.expect("free");
    assert_eq!(f.graph_end().await.expect("graph_end"), 42);

    let g = log.lock().expect("log");
    let kinds: Vec<&'static str> = g.iter().map(Command::kind).collect();
    assert_eq!(kinds, vec!["launch", "sync", "free", "graph_end"]);
}

/// ⑤ **变体不符必须响亮**(不许静默取默认值)。
#[tokio::test]
async fn wrong_outcome_variant_is_loud() {
    let (mut f, _log) = RecFace::new();
    f.poison_alloc = true;
    let e = f.alloc(Dtype::F32, 1, "x").await.expect_err("变体不符必须 Err");
    let s = format!("{e}");
    assert!(s.contains("变体不符") && s.contains("Buf") && s.contains("Done"), "得 {s}");
}

/// ⑥ `exec_batch` 的顺序/结果对位(CPU 面语义 = 逐条即序)。
#[tokio::test]
async fn exec_batch_keeps_order_and_reports_errors() {
    let (mut f, log) = RecFace::new();
    let r = f
        .exec_batch(vec![
            Command::Alloc { dtype: Dtype::F16, elems: 1, zero: true, name: "a".into() },
            Command::GraphEnd,
            Command::Sync,
        ])
        .await
        .expect("batch");
    assert_eq!(r.outcomes.len(), 3);
    assert!(r.all_ok(), "本 face 全成");
    assert_eq!(r.segments, 3, "CPU 面 segments 如实报 N");
    match &r.outcomes[1] {
        Ok(Outcome::Graph(42)) => {}
        other => panic!("逐条结果必须与投递顺序对位,得 {other:?}"),
    }
    assert_eq!(log.lock().expect("log").len(), 3);
}

/// ⑦ dtype/字节交叉校验仍在契约咽喉上(2026-10-15 审查项的回归哨)。
#[tokio::test]
async fn htod_rejects_dtype_byte_mismatch() {
    let (mut f, _log) = RecFace::new();
    let e = f
        .htod(Dtype::F32, &vec![4], &[0u8; 8], "bad")
        .await
        .expect_err("4×F32=16B > 8B ⇒ 必须 Err");
    let s = format!("{e}");
    assert!(s.contains("dtype/shape") && s.contains("16B"), "得 {s}");
}

/// ⑧ `DeviceHandle` 的命令面边界:分配/归还命令**响亮拒**(记账不可绕过)。
///
/// 注:`DeviceHandle` 有同名**固有** `exec(handle::Command)`,故这里用 UFCS
/// 显式选 trait 方法(这也是文档里"同名遮蔽"那条坑的回归哨)。
#[tokio::test]
async fn handle_trait_exec_refuses_accounting_commands() {
    use owl_cpu::CpuFace;
    use owl_iface::handle::DeviceHandle;

    let mut dev: DeviceHandle<CpuFace> = DeviceHandle::new(CpuFace::new(), None);
    let e = <DeviceHandle<CpuFace> as DeviceClient>::exec(
        &mut dev,
        Command::Alloc { dtype: Dtype::F16, elems: 4, zero: true, name: "bypass".into() },
    )
    .await
    .expect_err("分配命令必须被拒(记账不可绕过)");
    assert!(format!("{e}").contains("记账"), "得 {e}");

    // 控制类命令直通(不产生/销毁块)
    <DeviceHandle<CpuFace> as DeviceClient>::exec(&mut dev, Command::Sync)
        .await
        .expect("Sync 应直通");
}

/// ⑨ `queue` / `flush`:追加不等待、提交一次收口;队列非空时 `exec` 响亮拒。
#[tokio::test]
async fn queue_then_flush_is_one_batch() {
    let (mut f, log) = RecFace::new();
    // 队列里攒三条(其中 GraphEnd 的结果我们后续从 flush 里取)
    assert_eq!(f.queue(Command::Launch { msg: msg(3) }).expect("queue"), 0);
    assert_eq!(f.queue(Command::Sync).expect("queue"), 1);
    assert_eq!(f.queue(Command::GraphEnd).expect("queue"), 2);
    assert_eq!(log.lock().expect("log").len(), 0, "queue 不得碰后端");

    // 队列非空 ⇒ 单条命令必须响亮拒(不隐式多跑一次往返)
    let e = f.sync().await.expect_err("队列非空必须拒");
    assert!(format!("{e}").contains("未提交"), "得 {e}");

    let out = f.flush().await.expect("flush");
    assert_eq!(out.outcomes.len(), 3);
    assert!(out.all_ok());
    assert_eq!(out.segments, 3, "CPU 面如实报 N");
    match &out.outcomes[2] {
        Ok(Outcome::Graph(42)) => {}
        other => panic!("flush 结果必须按 queue 下标对位,得 {other:?}"),
    }
    // flush 后队列清空 ⇒ 单条命令恢复可用
    f.sync().await.expect("flush 后 exec 恢复");
    assert_eq!(log.lock().expect("log").len(), 4);
}

/// ⑩ 空队列 flush = 零往返空回执(不是错误,也不是隐式空批发包)。
#[tokio::test]
async fn flush_empty_is_noop() {
    let (mut f, log) = RecFace::new();
    let out = f.flush().await.expect("空 flush 应 Ok");
    assert!(out.outcomes.is_empty() && out.first_err.is_none());
    assert_eq!(out.segments, 0, "空批不打冒功段数");
    assert_eq!(log.lock().expect("log").len(), 0);
}

/// ⑪ `DeviceHandle::queue` 对分配/归还/装载命令**响亮拒**(记账不可绕过)。
#[tokio::test]
async fn handle_queue_refuses_accounting_commands() {
    use owl_cpu::CpuFace;
    use owl_iface::handle::DeviceHandle;

    let mut dev: DeviceHandle<CpuFace> = DeviceHandle::new(CpuFace::new(), None);
    let e = <DeviceHandle<CpuFace> as DeviceClient>::queue(
        &mut dev,
        Command::Alloc { dtype: Dtype::F16, elems: 4, zero: true, name: "x".into() },
    )
    .expect_err("分配命令必须被拒");
    assert!(format!("{e}").contains("记账"), "得 {e}");

    // 发射类命令可以入队(不产生/不销毁块)
    <DeviceHandle<CpuFace> as DeviceClient>::queue(&mut dev, Command::Sync)
        .expect("Sync 可入队");
    <DeviceHandle<CpuFace> as DeviceClient>::flush(&mut dev)
        .await
        .expect("flush 应直通");
}
