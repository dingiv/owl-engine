//! **DeviceHandle 行为测试**(骨架期;2026-10-19)
//!
//! 为什么放在 `owl-cpu/tests` 而不是 `owl-iface` 的 `#[cfg(test)]`:
//! iface 若把 owl-cpu 作 **dev-dependency**,会形成 dev-dep 成环 ⇒ Cargo 把 iface
//! **编译两份** ⇒ `CpuFace` 实现的是另一实例的 `DeviceClient`(trait 不通用)。
//! 放在这里:owl-iface 是普通依赖,**同一实例**,`CpuFace` 可直接作内层 client。
//!
//! 覆盖(设计卷 §九 步 1 的判据):钩子事件序列 / owner 归属 / 批的 `first_err` /
//! 未接线命令**响亮拒** / 纯转发态零记账。

use std::cell::RefCell;
use std::rc::Rc;

use owl_cpu::CpuFace;
use owl_iface::contract::Dtype;
use owl_iface::handle::{AssetClass, AllocEvent, AllocLedger, Command, DeviceHandle, FreeEvent, MoveEvent, Outcome, Owner};

/// 记账桩:只记录事件(验证钩子序列)。
#[derive(Default)]
struct RecLedger {
    allocs: Vec<AllocEvent>,
    frees: Vec<FreeEvent>,
    moves: Vec<MoveEvent>,
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

fn handle() -> (DeviceHandle<CpuFace>, Rc<RefCell<RecLedger>>) {
    let rec = Rc::new(RefCell::new(RecLedger::default()));
    let dev = DeviceHandle::new(CpuFace::new(), Some(rec.clone()));
    (dev, rec)
}

/// 门 ①:exec 单条 + 钩子事件(owner/class/name/bytes/id 全对)
#[tokio::test]
async fn exec_emits_alloc_event_with_owner() {
    let (mut dev, rec) = handle();
    dev.set_owner(Owner::Turn(7));
    let out = owl_iface::alloc!(dev, AssetClass::Temp, Dtype::F16, 1024, "kv.workspace")
        .expect("alloc");
    let Outcome::Buf(b) = out else { panic!("应是 Alloc 结果") };
    assert_eq!(b.len, 1024);
    let r = rec.borrow();
    assert_eq!(r.allocs.len(), 1, "一次分配一条事件");
    let ev = &r.allocs[0];
    assert_eq!(ev.owner, Owner::Turn(7), "owner 来自上下文");
    assert_eq!(ev.class, AssetClass::Temp);
    assert_eq!(ev.name, "kv.workspace");
    assert_eq!(ev.bytes, 1024 * Dtype::F16.size_bytes(), "字节 = 元素 × dtype");
    assert_eq!(ev.id, b.id, "事件 id 与返回块同源");
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
    let r = rec.borrow();
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
    let r = rec.borrow();
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

/// 门 ⑦:`Launch` **真的转发到 client**(失败信息来自 client,不是"尚未接线")
#[tokio::test]
async fn launch_is_forwarded_to_client() {
    let (mut dev, _rec) = handle();
    let msg = owl_iface::contract::LaunchMsg {
        kernel: owl_iface::contract::KernelSource { name: "k".into(), source: String::new() },
        args: vec![],
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        out_elems: 0,
    };
    let e = dev
        .exec(Command::Launch { msg, name: "k".into() })
        .await
        .expect_err("未知核名:client 必须拒");
    let msg = format!("{e}");
    assert!(!msg.contains("尚未接线"), "必须已转发到 client;得 {msg}");
}
