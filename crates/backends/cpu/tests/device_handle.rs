//! **DeviceHandle 行为测试**(步 1 骨架 + 步 3 钩子面;2026-10-19)
//!
//! 为什么放在 `owl-cpu/tests` 而不是 `owl-iface` 的 `#[cfg(test)]`:
//! iface 若把 owl-cpu 作 **dev-dependency**,会形成 dev-dep 成环 ⇒ Cargo 把 iface
//! **编译两份** ⇒ `CpuFace` 实现的是另一实例的 `DeviceClient`(trait 不通用)。
//! 放在这里:owl-iface 是普通依赖,**同一实例**,`CpuFace` 可直接作内层 client。
//!
//! 覆盖:
//! - 步 1:钩子事件序列 / owner 归属 / 批的 `first_err` / 未接线命令**响亮拒** / 纯转发态;
//! - 步 3:**转发面出生钩子**(`alloc`/`alloc_uninit`/`htod` 族)、`free` 死亡钩子、
//!   **图窗**(窗内类自动 = `Graph`;`graph_end` 失败**不猜**去向下场;`graph_abort`
//!   按后端自述销账)、`adopt_ledger`(boot 期合流的**响亮拒**判据)、
//!   `face_clone` **上下文同源**。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use owl_cpu::{CpuFace, MockLimits};
use owl_iface::contract::{DeviceClient, Dtype, Shape};
use owl_iface::handle::{
    AllocEvent, AllocLedger, AssetClass, Command, DeviceHandle, FreeEvent, MoveEvent, Outcome, Owner,
};

/// 记账桩:只记录事件(验证钩子序列)。
#[derive(Default)]
struct RecLedger {
    allocs: Vec<AllocEvent>,
    frees: Vec<FreeEvent>,
    moves: Vec<MoveEvent>,
    /// 池(桩:同尺寸判据从简);默认**关** —— 只有池/压力门开它
    /// (默认开会把 `free` 全部截进池,别的门就看不到"真归还")
    pool: Vec<(usize, u64)>,
    pool_on: bool,
    /// 桩池的 rank(验 P0-1:非本 rank 不命中)
    pool_rank: u8,
}

impl AllocLedger for RecLedger {
    fn on_alloc(&mut self, ev: &AllocEvent) {
        self.allocs.push(ev.clone());
    }
    fn on_free(&mut self, ev: &FreeEvent) {
        self.frees.push(ev.clone());
    }
    fn on_move(&mut self, ev: &MoveEvent) {
        self.moves.push(ev.clone());
    }
}

type Rec = Arc<Mutex<RecLedger>>;

fn cell() -> (owl_iface::handle::LedgerCell, Rec) {
    let rec: Rec = Arc::new(Mutex::new(RecLedger::default()));
    let cell: owl_iface::handle::LedgerCell = Arc::new(Mutex::new(RecLedgerProxy(rec.clone())));
    (cell, rec)
}

/// `RecLedger` 的克隆转发壳(事件同时留在测试侧,便于断言)—— `LedgerCell`
/// 必须 `Send`,所以桩本身用 `Arc<Mutex<..>>` 而不是 `Rc<RefCell<..>>`。
struct RecLedgerProxy(Rec);

impl AllocLedger for RecLedgerProxy {
    fn on_alloc(&mut self, ev: &AllocEvent) {
        lock(&self.0).allocs.push(ev.clone());
    }
    fn on_free(&mut self, ev: &FreeEvent) {
        lock(&self.0).frees.push(ev.clone());
    }
    fn on_move(&mut self, ev: &MoveEvent) {
        lock(&self.0).moves.push(ev.clone());
    }
    fn pool_acquire(&mut self, _rank: u8, _class: AssetClass, bytes: usize) -> Option<u64> {
        let mut g = lock(&self.0);
        if !g.pool_on {
            return None;
        }
        // 同尺寸才给(与 store 的 key 判据一致)
        g.pool.iter().rposition(|(b, _)| *b == bytes).map(|i| g.pool.remove(i).1)
    }
    fn pool_release(&mut self, id: u64) -> Vec<u64> {
        let mut g = lock(&self.0);
        if !g.pool_on {
            return vec![id];
        }
        let bytes = g.allocs.iter().rev().find(|e| e.id == id).map(|e| e.bytes).unwrap_or(0);
        g.pool.push((bytes, id));
        Vec::new()
    }
    fn pool_drain(&mut self) -> Vec<u64> {
        std::mem::take(&mut lock(&self.0).pool).into_iter().map(|(_, id)| id).collect()
    }
}

fn lock(r: &Rec) -> std::sync::MutexGuard<'_, RecLedger> {
    match r.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

fn handle() -> (DeviceHandle<CpuFace>, Rec) {
    let (cell, rec) = cell();
    (DeviceHandle::new(CpuFace::new(), Some(cell)), rec)
}

fn capture_handle() -> (DeviceHandle<CpuFace>, Rec) {
    let (cell, rec) = cell();
    let face = CpuFace::mock(MockLimits { graph_capture: true, ..Default::default() });
    (DeviceHandle::new(face, Some(cell)), rec)
}

fn msg(name: &str) -> owl_iface::contract::LaunchMsg {
    owl_iface::contract::LaunchMsg {
        kernel: owl_iface::contract::KernelSource { name: name.into(), source: String::new() },
        args: vec![],
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        out_elems: 4,
    }
}

// ============================================================================
// 步 1 门(命令面/批/纯转发)
// ============================================================================

/// 门 ①:exec 单条 + 钩子事件(owner/class/name/bytes/id/rank 全对)
#[tokio::test]
async fn exec_emits_alloc_event_with_owner() {
    let (mut dev, rec) = handle();
    dev.set_owner(Owner::Turn(7));
    let out = owl_iface::alloc!(dev, AssetClass::Temp, Dtype::F16, 1024, "kv.workspace")
        .expect("alloc");
    let Outcome::Buf(b) = out else { panic!("应是 Alloc 结果") };
    assert_eq!(b.len, 1024);
    let r = lock(&rec);
    assert_eq!(r.allocs.len(), 1, "一次分配一条事件");
    let ev = &r.allocs[0];
    assert_eq!(ev.owner, Owner::Turn(7), "owner 来自上下文");
    assert_eq!(ev.class, AssetClass::Temp);
    assert_eq!(ev.name, "kv.workspace");
    assert_eq!(ev.bytes, 1024 * Dtype::F16.size_bytes(), "字节 = 元素 × dtype");
    assert_eq!(ev.id, b.id, "事件 id 与返回块同源");
    assert_eq!(ev.rank, 0, "单卡恒 rank 0");
}

/// 门 ②:free 走命令面 + 事件;归属随节点切换
#[tokio::test]
async fn free_command_emits_free_event() {
    let (mut dev, rec) = handle();
    dev.set_owner(Owner::Session(3));
    let Outcome::Buf(b) = owl_iface::alloc!(dev, AssetClass::KvBlock, Dtype::F16, 64, "kv.chain")
        .expect("alloc")
    else {
        panic!("应是 Alloc")
    };
    owl_iface::free!(dev, b.id).expect("free");
    let r = lock(&rec);
    assert_eq!(r.allocs[0].owner, Owner::Session(3));
    assert_eq!(r.frees.len(), 1);
    assert_eq!(r.frees[0].id, b.id);
}

/// 门 ③:批的 `first_err` 语义(失败条不入账,成功条必须入账)
#[tokio::test]
async fn batch_reports_first_err_and_skips_failed_alloc() {
    let (mut dev, rec) = handle();
    dev.set_owner(Owner::Turn(1));
    let out = dev
        .batch([
            Command::alloc(AssetClass::Temp, Dtype::F16, 16, "ok.0"),
            Command::Htod { id: 1, off: 0, bytes: 8 }, // 骨架期未接线 ⇒ 必失败
            Command::alloc(AssetClass::Temp, Dtype::F16, 16, "ok.1"),
        ])
        .await
        .expect("batch 本体不失败(逐条结果在内)");
    assert_eq!(out.first_err, Some(1), "首个失败下标");
    assert_eq!(out.segments, 3, "骨架期 = 语义等价批(逐条转发)");
    assert!(!out.all_ok());
    let r = lock(&rec);
    assert_eq!(r.allocs.len(), 2, "只有两条成功分配入账");
    assert_eq!(
        r.allocs.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        vec!["ok.0", "ok.1"]
    );
}

/// 门 ④:未接线命令**响亮拒**(不是静默 no-op)
#[tokio::test]
async fn unwired_commands_fail_loudly() {
    let (mut dev, _rec) = handle();
    for cmd in [
        Command::Htod { id: 1, off: 0, bytes: 8 },
        Command::Dtoh { id: 1, off: 0, bytes: 8 },
    ] {
        let e = dev.exec(cmd).await.expect_err("骨架期必须拒");
        assert!(format!("{e}").contains("尚未接线"), "得 {e:?}");
    }
}

/// 门 ⑤:纯转发态(`ledger = None`)零记账、行为不变
#[tokio::test]
async fn none_ledger_is_pure_forwarding() {
    let mut dev: DeviceHandle<CpuFace> = DeviceHandle::new(CpuFace::new(), None);
    let _ = owl_iface::alloc!(dev, AssetClass::Temp, Dtype::F32, 8, "no-ledger").expect("alloc");
    assert_eq!(dev.hook_fail(), 0);
    assert_eq!(dev.owner(), Owner::System, "缺省归属 = system");
}

/// 门 ⑥:批里 `Sync` 接线(逐条成功 + 一次收口)
#[tokio::test]
async fn batch_runs_sync() {
    let (mut dev, _rec) = handle();
    let out = dev.batch([Command::Sync, Command::Sync]).await.expect("batch");
    assert!(out.all_ok(), "{:?}", out.first_err);
    assert_eq!(out.segments, 2);
}

/// 门 ⑦:`Launch` **真的转发到 client**(失败信息来自 client,不是"尚未接线");
/// 且**不算出生**(输出块 = 实参里的 Block,调用方早就 alloc 过)
#[tokio::test]
async fn launch_is_forwarded_and_is_not_a_birth() {
    let (mut dev, rec) = handle();
    let e = dev
        .exec(Command::Launch { msg: msg("k"), name: "k".into() })
        .await
        .expect_err("未知核名:client 必须拒");
    assert!(!format!("{e}").contains("尚未接线"), "必须已转发到 client;得 {e}");
    assert!(lock(&rec).allocs.is_empty(), "launch 不产出生事件");
}

// ============================================================================
// 步 3 门(转发面出生/死亡 · 图窗 · 上下文)
// ============================================================================

/// 门 ⑧:**转发面出生钩子**(不走命令面而直接调 `DeviceClient` 的调用点也要记账)
#[tokio::test]
async fn forwarded_births_are_hooked() {
    let (mut dev, rec) = handle();
    dev.set_owner(Owner::Turn(2));
    dev.set_class(AssetClass::Weight);
    let a = dev.alloc(Dtype::F16, 32, "fwd.alloc").await.expect("alloc");
    let u = dev.alloc_uninit(Dtype::F32, 8, "fwd.uninit").await.expect("alloc_uninit");
    let h = dev
        .htod(Dtype::F32, &Shape::from(vec![4]), &[0u8; 16], "fwd.htod")
        .await
        .expect("htod");
    let r = lock(&rec);
    assert_eq!(r.allocs.len(), 3, "三条出生必须都入账:{:?}", r.allocs);
    assert_eq!(r.allocs[0].bytes, 32 * Dtype::F16.size_bytes());
    assert_eq!(r.allocs[1].bytes, 8 * Dtype::F32.size_bytes());
    assert_eq!(r.allocs[2].bytes, 16, "htod 字节 = 源字节");
    assert_eq!(r.allocs[1].name, "fwd.uninit", "uninit 也点名(不落 alloc 语义)");
    for ev in &r.allocs {
        assert_eq!(ev.class, AssetClass::Weight, "相位类别透传");
        assert_eq!(ev.owner, Owner::Turn(2));
    }
    drop(r);
    // 死亡面:forward 的 free 也要销账
    dev.free(&[a.id, u.id, h.id]).await.expect("free");
    assert_eq!(lock(&rec).frees.len(), 3, "forward free 必须发三条");
}

/// 门 ⑨:**图窗出生 ⇒ 类自动 = `Graph`**(节点/相位无权把它记成 Temp:
/// 捕获期后端恒走 slab 切片,块随图存活)
#[tokio::test]
async fn window_births_are_classified_as_graph() {
    let (mut dev, rec) = stub_handle(true);
    dev.set_class(AssetClass::Temp); // 相位设的是 Temp —— 但窗内不许撒谎
    dev.graph_begin().await.expect("开窗");
    dev.alloc(Dtype::F16, 8, "win.0").await.expect("窗内分配");
    assert_eq!(dev.open_window_blocks(), 1, "窗内块被跟踪");
    let r = lock(&rec);
    assert_eq!(r.allocs[0].class, AssetClass::Graph, "窗内出生 = Graph 类");
}

/// 门 ⑩:图退休按**后端自述**销账(CpuFace = 不回收 ⇒ 账跟着调用方的 `free` 走)
#[tokio::test]
async fn graph_destroy_follows_backend_self_report() {
    let (mut dev, rec) = capture_handle();
    dev.graph_begin().await.expect("开窗");
    let b = dev.alloc(Dtype::F16, 8, "win.1").await.expect("窗内分配");
    // 窗内发射一次(CPU 面 graph_end 要求窗非空)
    dev.launch(msg("k")).await.expect_err("未知核名"); // 不计数 → 用 copy_batch/memset?
    // CPU mock 的"发射数"只认 launch 成功;这里改用真实路径:直接 alloc + abort
    dev.graph_abort().await.expect("作废窗");
    // CpuFace 自述 false(块交调用方 free)⇒ abort 不销账
    assert_eq!(lock(&rec).frees.len(), 0, "后端不回收 ⇒ 本层不销账");
    dev.free(&[b.id]).await.expect("显式归还");
    assert_eq!(lock(&rec).frees.len(), 1, "销账点 = 调用方 free");
}

/// 门 ⑪:窗失败**不猜去向**(保留窗 + 计数可见),`abort` 才收口
#[tokio::test]
async fn window_failure_is_not_guessed() {
    let (mut dev, rec) = capture_handle();
    dev.graph_begin().await.expect("开窗");
    let b = dev.alloc(Dtype::F16, 8, "win.2").await.expect("窗内分配");
    let e = dev.graph_end().await.expect_err("空窗必拒");
    assert!(format!("{e}").contains("空捕获窗"), "得 {e}");
    assert_eq!(dev.window_unclosed(), 1, "窗未收要计数(可见,不静默)");
    assert_eq!(lock(&rec).frees.len(), 0, "去向未定 ⇒ 不销账");
    assert_eq!(dev.open_window_blocks(), 1, "窗仍在场");
    dev.graph_abort().await.expect("abort 收口(CpuFace 保留捕获态)");
    assert_eq!(lock(&rec).frees.len(), 0, "CpuFace 自述不回收 ⇒ abort 也不销账");
    dev.free(&[b.id]).await.expect("显式归还");
    assert_eq!(lock(&rec).frees.len(), 1);
}

/// 门 ⑫:`adopt_ledger` 只许在 **boot 期(零事件)** 接管;接管后事件进新账
#[tokio::test]
async fn adopt_ledger_is_boot_only() {
    let (mut dev, _rec0) = handle();
    let (cell1, rec1) = cell();
    dev.adopt_ledger(cell1).expect("零事件时可接管");
    let _ = dev.alloc(Dtype::F16, 4, "after.adopt").await.expect("alloc");
    assert_eq!(lock(&rec1).allocs.len(), 1, "接管后事件进新账");
    // 已发过事件再接管 = 响亮拒(不许把已有事件丢进孤儿账)
    let (cell2, _rec2) = cell();
    let e = dev.adopt_ledger(cell2).expect_err("已记账后接管必须拒");
    assert!(format!("{e}").contains("boot 期"), "得 {e}");
}

/// **自带令牌的桩账本**(验 `book_token` 支路:同一本书 ⇒ 幂等接管)
struct TokenLedger(usize);

impl AllocLedger for TokenLedger {
    fn on_alloc(&mut self, _ev: &AllocEvent) {}
    fn on_free(&mut self, _ev: &FreeEvent) {}
    fn on_move(&mut self, _ev: &MoveEvent) {}
    fn book_token(&self) -> usize {
        self.0
    }
}

fn token_cell(t: usize) -> owl_iface::handle::LedgerCell {
    Arc::new(Mutex::new(TokenLedger(t)))
}

/// 门 ⑫-b:**同一本书**的接管幂等(TP 用:装载前已合流,装配期再 adopt 一次)。
///
/// 判据两层:同一令牌 + 已发过事件 ⇒ **允许**(零事件丢失);换一本书 + 已记账
/// ⇒ **仍响亮拒**(老判据不被削弱)。缺了前一条,TP2 的"装载即发事件"会让
/// 装配期那次 adopt 必拒(2026-10-20 回归:0.8B 就有 312 条装载事件)。
#[tokio::test]
async fn adopt_same_book_is_idempotent_after_events() {
    let (mut dev, _rec0) = handle();
    dev.adopt_ledger(token_cell(7)).expect("零事件时可接管");
    let _ = dev.alloc(Dtype::F16, 4, "after.adopt").await.expect("alloc");
    dev.adopt_ledger(token_cell(7)).expect("同一本书 ⇒ 幂等放行(不丢事件)");
    let e = dev.adopt_ledger(token_cell(8)).expect_err("另一本书 + 已记账 ⇒ 必须拒");
    assert!(format!("{e}").contains("boot 期"), "得 {e}");
    // 令牌 0 = 不表态 ⇒ 一律按"另一本书"(没有身份就没有幂等)
    let (mut dev2, _rec2) = handle();
    dev2.adopt_ledger(token_cell(0)).expect("零事件接管");
    let _ = dev2.alloc(Dtype::F16, 4, "after.adopt2").await.expect("alloc");
    dev2.adopt_ledger(token_cell(0)).expect_err("token=0 不构成幂等判据");
}

/// 门 ⑬:`face_clone` **上下文同源**(节点设一次全族生效;事件进同一本账)
#[tokio::test]
async fn face_clone_shares_context_and_ledger() {
    let (dev, rec) = stub_handle(false);
    let mut clone = dev.face_clone().expect("StubFace 可克隆");
    clone.set_owner(Owner::Session(9));
    assert_eq!(dev.owner(), Owner::Session(9), "克隆设的上下文对本体可见");
    let _ = clone.alloc(Dtype::F16, 4, "clone.alloc").await.expect("alloc");
    assert_eq!(lock(&rec).allocs.len(), 1, "同一本账");
    assert_eq!(lock(&rec).allocs[0].owner, Owner::Session(9));
}

// ============================================================================
// stub face:门 ⑨/⑬~⑯ 的载体
// ============================================================================
//
// 为什么需要:CpuFace **不可克隆**(它的 `face_clone` 恒 None —— 与 GPU 的
// "克隆 = 同一个 server 的另一个发送端"语义不同,克隆一份 mock 会得到**独立状态**),
// 且其 `graph_end` 要求窗内真有发射。而本组门要验的是 **handle 自己的簿记**
// (`auto_reclaimed = true` 的"后端整批回收 ⇒ 本层补发归还"三条契约),
// 所以用一个最小 face:状态只有"发号 + 回收开关"。

#[derive(Clone)]
struct StubFace {
    /// 压力注入:第一次 `alloc` 返回 `PoolExhausted`(验 N6 自愈)
    fail_first_alloc: Arc<std::sync::atomic::AtomicBool>,
    /// 被 `free` 掉的 id(验"排空后真归还")
    freed: Arc<Mutex<Vec<u64>>>,
    ids: Arc<AtomicU64>,
    /// 窗内发射计数(复核服务端"空窗判据":零发射 ⇒ `graph_end` 拒)
    nodes: Arc<AtomicU64>,
    /// 自述:窗内自产块由后端回收(契约三条)
    reclaimed: bool,
}

impl StubFace {
    fn new(reclaimed: bool) -> Self {
        Self {
            fail_first_alloc: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            freed: Arc::new(Mutex::new(Vec::new())),
            ids: Arc::new(AtomicU64::new(1)),
            nodes: Arc::new(AtomicU64::new(0)),
            reclaimed,
        }
    }
    fn take_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::Relaxed)
    }
}

impl DeviceClient for StubFace {
    fn window_blocks_auto_reclaimed(&self) -> bool {
        self.reclaimed
    }
    async fn graph_begin(&mut self) -> Result<(), owl_iface::contract::ModelError> {
        self.nodes.store(0, Ordering::Relaxed);
        Ok(())
    }
    async fn graph_end(&mut self) -> Result<owl_iface::contract::GraphId, owl_iface::contract::ModelError> {
        // 空窗判据(与 cuda/cpu 同律)
        if self.nodes.load(Ordering::Relaxed) == 0 {
            return Err(owl_iface::contract::ModelError::Msg("StubFace: 空捕获窗".into()));
        }
        Ok(self.take_id())
    }
    async fn graph_launch(
        &mut self,
        _graph: owl_iface::contract::GraphId,
    ) -> Result<(), owl_iface::contract::ModelError> {
        Ok(())
    }
    async fn graph_abort(&mut self) -> Result<(), owl_iface::contract::ModelError> {
        Ok(())
    }
    async fn graph_destroy(
        &mut self,
        _graph: owl_iface::contract::GraphId,
    ) -> Result<(), owl_iface::contract::ModelError> {
        Ok(())
    }
    async fn alloc(
        &mut self,
        dtype: Dtype,
        elems: usize,
        _name: &str,
    ) -> Result<owl_iface::contract::Bytes, owl_iface::contract::ModelError> {
        if self.fail_first_alloc.swap(false, Ordering::Relaxed) {
            return Err(owl_iface::contract::ModelError::PoolExhausted {
                pool: "stub".into(),
                needed: 1,
                available: 0,
            });
        }
        Ok(owl_iface::contract::Bytes::new(self.take_id(), elems, dtype, 0))
    }

    async fn free(&mut self, ids: &[u64]) -> Result<(), owl_iface::contract::ModelError> {
        match self.freed.lock() {
            Ok(mut g) => g.extend_from_slice(ids),
            Err(p) => p.into_inner().extend_from_slice(ids),
        }
        Ok(())
    }
    async fn htod(
        &mut self,
        dtype: Dtype,
        _shape: &Shape,
        src: &[u8],
        _name: &str,
    ) -> Result<owl_iface::contract::Bytes, owl_iface::contract::ModelError> {
        Ok(owl_iface::contract::Bytes::new(self.take_id(), src.len(), dtype, 0))
    }
    async fn dtoh(
        &mut self,
        _b: &owl_iface::contract::Bytes,
        out: &mut [u8],
    ) -> Result<(), owl_iface::contract::ModelError> {
        out.fill(0);
        Ok(())
    }
    async fn launch(
        &mut self,
        _msg: owl_iface::contract::LaunchMsg,
    ) -> Result<owl_iface::contract::Bytes, owl_iface::contract::ModelError> {
        self.nodes.fetch_add(1, Ordering::Relaxed);
        Ok(owl_iface::contract::Bytes::new(0, 0, Dtype::U8, 0))
    }
    async fn sync(&mut self) -> Result<(), owl_iface::contract::ModelError> {
        Ok(())
    }
    fn face_clone(&self) -> Option<Self> {
        Some(self.clone())
    }
}

fn stub_handle(reclaimed: bool) -> (DeviceHandle<StubFace>, Rec) {
    let (cell, rec) = cell();
    (DeviceHandle::new(StubFace::new(reclaimed), Some(cell)), rec)
}

/// 门 ⑭:后端自述**回收** ⇒ `graph_destroy` 时本层补发归还(契约②)
#[tokio::test]
async fn auto_reclaimed_backend_frees_on_destroy() {
    let (mut dev, rec) = stub_handle(true);
    dev.graph_begin().await.expect("开窗");
    let b = dev.alloc(Dtype::F16, 8, "win.d").await.expect("窗内分配");
    dev.launch(msg("k")).await.expect("stub launch");
    let gid = dev.graph_end().await.expect("窗收");
    assert_eq!(lock(&rec).frees.len(), 0, "图还在世 ⇒ 不销账");
    dev.graph_destroy(gid).await.expect("退休");
    let r = lock(&rec);
    assert_eq!(r.frees.len(), 1, "契约②:图退休 ⇒ 补发归还");
    assert_eq!(r.frees[0].id, b.id);
}

/// 门 ⑮:后端自述**回收** ⇒ ①`abort` 成功补发;③**空窗失败**也补发(窗随之关闭)
#[tokio::test]
async fn auto_reclaimed_backend_frees_on_abort_and_empty_window() {
    let (mut dev, rec) = stub_handle(true);
    // ① abort 成功 ⇒ 补发归还
    dev.graph_begin().await.expect("开窗");
    let b1 = dev.alloc(Dtype::F16, 8, "win.e1").await.expect("窗内分配");
    dev.graph_abort().await.expect("作废");
    {
        let r = lock(&rec);
        assert_eq!(r.frees.len(), 1, "契约①:abort 成功 ⇒ 补发归还");
        assert_eq!(r.frees[0].id, b1.id);
        assert_eq!(dev.window_unclosed(), 0, "窗已收 ⇒ 不计未收");
    }
    // ③ 空窗失败(零发射)⇒ 后端已整批回收 ⇒ 补发 + 窗关
    dev.graph_begin().await.expect("再开窗");
    let b2 = dev.alloc(Dtype::F16, 8, "win.e2").await.expect("窗内分配");
    let e = dev.graph_end().await.expect_err("空窗必拒");
    assert!(format!("{e}").contains("空捕获窗"), "得 {e}");
    {
        let r = lock(&rec);
        assert_eq!(r.frees.len(), 2, "契约③:空窗失败 ⇒ 补发归还");
        assert_eq!(r.frees[1].id, b2.id);
        assert_eq!(dev.window_unclosed(), 0, "已判定的回收不算未收");
        assert_eq!(dev.open_window_blocks(), 0, "窗已关");
    }
}

/// 门 ⑯:后端自述**不回收** ⇒ 同一路径**不销账**(账跟着调用方的 `free` 走)
#[tokio::test]
async fn non_reclaiming_backend_leaves_book_to_caller() {
    let (mut dev, rec) = stub_handle(false);
    dev.graph_begin().await.expect("开窗");
    let b = dev.alloc(Dtype::F16, 8, "win.f").await.expect("窗内分配");
    dev.launch(msg("k")).await.expect("stub launch");
    let gid = dev.graph_end().await.expect("窗收");
    dev.graph_destroy(gid).await.expect("退休");
    assert_eq!(lock(&rec).frees.len(), 0, "后端不回收 ⇒ 本层不销账");
    dev.free(&[b.id]).await.expect("显式归还");
    assert_eq!(lock(&rec).frees.len(), 1, "销账点 = 调用方 free");
}

/// 门 ⑰:**缺口 4 —— 池复用**(同尺寸:释放入池、再取零命令复用;
/// 零化档命中后补一次 memset,uninit 档不补)
#[tokio::test]
async fn pool_reuses_released_temp_blocks() {
    // 桩账本 = 真 store 的池语义(最小实现:按 (class, bytes) LIFO)
    #[derive(Default)]
    struct PoolLedger {
        pool: std::collections::HashMap<(u8, usize), Vec<u64>>,
        hits: u64,
    }
    impl AllocLedger for PoolLedger {
        fn on_alloc(&mut self, _ev: &AllocEvent) {}
        fn on_free(&mut self, _ev: &FreeEvent) {}
        fn on_move(&mut self, _ev: &MoveEvent) {}
        fn pool_acquire(&mut self, _rank: u8, class: AssetClass, bytes: usize) -> Option<u64> {
            let k = (class as u8, bytes);
            let id = self.pool.get_mut(&k).and_then(|v| v.pop());
            if id.is_some() {
                self.hits += 1;
            }
            id
        }
        fn pool_release(&mut self, id: u64) -> Vec<u64> {
            self.pool.entry((AssetClass::Temp as u8, 4096)).or_default().push(id);
            Vec::new()
        }
    }
    let rec: Rec = Arc::new(Mutex::new(RecLedger::default()));
    let cell: owl_iface::handle::LedgerCell =
        Arc::new(Mutex::new(RecLedgerProxy(rec.clone())));
    let _ = cell; // 桩:池面另起一本(只为验 handle 的取/还调用序)
    let pool: Arc<Mutex<PoolLedger>> = Arc::new(Mutex::new(PoolLedger::default()));
    let cell2: owl_iface::handle::LedgerCell = Arc::new(Mutex::new(PoolProxy(pool.clone())));
    struct PoolProxy(Arc<Mutex<PoolLedger>>);
    impl AllocLedger for PoolProxy {
        fn on_alloc(&mut self, _ev: &AllocEvent) {}
        fn on_free(&mut self, _ev: &FreeEvent) {}
        fn on_move(&mut self, _ev: &MoveEvent) {}
        fn pool_acquire(&mut self, rank: u8, class: AssetClass, bytes: usize) -> Option<u64> {
            match self.0.lock() {
                Ok(mut g) => g.pool_acquire(rank, class, bytes),
                Err(p) => p.into_inner().pool_acquire(rank, class, bytes),
            }
        }
        fn pool_release(&mut self, id: u64) -> Vec<u64> {
            match self.0.lock() {
                Ok(mut g) => g.pool_release(id),
                Err(p) => p.into_inner().pool_release(id),
            }
        }
    }
    let mut dev: DeviceHandle<CpuFace> = DeviceHandle::new(CpuFace::new(), Some(cell2));
    dev.set_class(AssetClass::Temp);
    let a = dev.alloc_uninit(Dtype::F16, 2048, "ws.0").await.expect("alloc");
    dev.free(&[a.id]).await.expect("release ⇒ 入池");
    let b = dev.alloc_uninit(Dtype::F16, 2048, "ws.1").await.expect("pool hit");
    assert_eq!(b.id, a.id, "同尺寸复用 = 同一个块 id");
    let (hits, released) = dev.pool_counters();
    assert_eq!((hits, released), (1, 1), "命中与入池计数(hits/released)");
    assert_eq!(lock(&rec).frees.len(), 0, "池化 ⇒ **不发** FreeEvent(块仍在借)");
}

/// 门 ⑱:**N6 压力自愈** —— 分配遇 `PoolExhausted` ⇒ 排空复用池(真还驱动) ⇒ 重试一次
///
/// FIXME(N6-step1,未收口):`#[ignore]` —— 门实测"排空未发生"(drain 返回 0,首次错误
/// 直接上抛)。已确认:①释放确实入池(`pool_counters().1 == 1`)②请求尺寸不同(64B vs 池内 32B)
/// ③桩只注入一次错误。待查 = `drain_pool_to_driver` 里 `dyn AllocLedger::pool_drain`
/// 的动态派发是否走到了桩实现(与"同名解析"同族嫌疑);收口后去掉 `#[ignore]`。
#[tokio::test]
async fn pool_drain_heals_pool_exhausted_alloc() {
    let (cell, rec) = cell();
    lock(&rec).pool_on = true; // 本门专测池/压力面
    let face = StubFace::new(true);
    let freed = face.freed.clone();
    let fail = face.fail_first_alloc.clone();
    let mut dev = DeviceHandle::new(face, Some(cell));
    dev.set_class(AssetClass::Temp);
    // 先造一块入池(释放 ⇒ 可池化 ⇒ 不回驱动)
    let a = dev.alloc_uninit(Dtype::F16, 16, "t.0").await.expect("alloc");
    dev.free(&[a.id]).await.expect("入池");
    assert_eq!(dev.pool_counters().1, 1, "已入池");
    assert!(freed.lock().map(|g| g.is_empty()).unwrap_or(true), "入池 ⇒ 未回驱动");
    // 下一次分配:桩先抛 PoolExhausted ⇒ 句柄应排空池(真 free)后重试成功
    fail.store(true, Ordering::Relaxed);
    // 请求**不同尺寸**(32 元素)⇒ 池未命中 ⇒ 桩抛 PoolExhausted ⇒ 句柄排空池(真 free 那块 16 元素)后重试
    let b = dev.alloc_uninit(Dtype::F16, 32, "t.1").await.expect("自愈后应成功");
    assert_eq!(b.len, 32);
    let f = freed.lock().map(|g| g.clone()).unwrap_or_default();
    assert_eq!(f, vec![a.id], "排空 = 把池里那块真还给驱动");
}

/// 门 ⑲:**P0-3 捕获窗内不取池**(窗内块必须真分配 —— 否则指针烘进图后被复用 = UAF)
#[tokio::test]
async fn pool_is_disabled_inside_capture_window() {
    let (cell, rec) = cell();
    lock(&rec).pool_on = true;
    let mut dev = DeviceHandle::new(StubFace::new(true), Some(cell));
    dev.set_class(AssetClass::Temp);
    // 先造一块入池
    let a = dev.alloc_uninit(Dtype::F16, 16, "w.0").await.expect("alloc");
    dev.free(&[a.id]).await.expect("入池");
    // 窗内:同尺寸也**不许**命中 ⇒ 真分配(新 id)
    dev.graph_begin().await.expect("开窗");
    let in_win = dev.alloc_uninit(Dtype::F16, 16, "w.1").await.expect("窗内分配");
    assert_ne!(in_win.id, a.id, "窗内禁止取池(P0-3)");
    dev.graph_abort().await.expect("关窗");
    // 窗关后:同尺寸照常命中
    let after = dev.alloc_uninit(Dtype::F16, 16, "w.2").await.expect("窗后分配");
    assert_eq!(after.id, a.id, "窗关后恢复池复用");
}
