use super::*;
use crate::budget::BudgetInputs;
use crate::hooks::{EvictOut, Need, NeedWhy, Sink};
use crate::id::ChainHash;
use crate::policy::QueuePolicy;

const BLOCK_BYTES: usize = 1 << 20;

fn inputs() -> BudgetInputs {
    BudgetInputs {
        v_card_bytes: 25_000_000_000,
        weights_bytes: 17_500_000_000,
        runtime_floor_bytes: 2_000_000_000,
        draft_pool_bytes: 0,
        gdn_row_bytes: 162_000_000,
        kv_bytes_per_token: 32 * 1024,
        page_tokens: 32,
        host_bytes: 8 << 20,
        disk_bytes: 0,
        kv_face_bytes: 0,
    }
}

/// 测试用账本单元(预算 = 入参的 host/disk 总量 —— 与 boot 同式)
fn cell_for(i: &BudgetInputs) -> crate::tier::SharedLedger {
    crate::tier::shared_ledger(crate::tier::TierBudget {
        host_bytes: i.host_bytes,
        disk_bytes: i.disk_bytes,
    })
}

fn kv_cfg(slots: usize, cache_cap: usize) -> KvConfig {
    KvConfig {
        slots,
        block_tokens: 32,
        block_bytes: BLOCK_BYTES,
        cache: crate::leg::CachePolicy::Cap(cache_cap),
        host_bytes: 8 << 20,
        class: crate::leg::BlockClass::KvPage,
        label: "test",
    }
}

fn rel(turns: usize) -> Rel {
    let i = inputs();
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(turns, QueuePolicy::Reject),
        kv_cfg(64, 32),
        cell_for(&i),
    )
    .expect("装配");
    // 注册池布局(单片段;测试只跑账/计划,不跑真搬运 —— 搬迁在 store_e2e 覆盖)
    r.leg
        .attach_pool(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("布局");
    r
}

/// 无宿主档（`host_enabled = false`）：用于钉“纯 GPU 出序”。
fn rel_no_host(turns: usize) -> Rel {
    let mut i = inputs();
    i.host_bytes = 0;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 0;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(turns, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    r.leg
        .attach_pool(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("布局");
    r
}

fn toks(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// 装配带盘层的 Rel(测试辅助)。
fn rel_with_disk(i: &mut BudgetInputs, kv_bytes: usize, snap_bytes: usize) -> Rel {
    // 2026-11-18：盘层 = 一层一个总池；测试把两腿旧配额合并成总账。
    i.disk_bytes = kv_bytes.saturating_add(snap_bytes);
    let mut r = Rel::open(
        *i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        kv_cfg(64, 32),
        cell_for(i),
    )
    .expect("装配");
    r.leg
        .attach_pool(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("布局");
    r.attach_disk("/tmp/owl-disk-test-adopt").expect("接盘");
    r
}

mod admin;
mod disk;
mod kv;
mod report;
mod state;
