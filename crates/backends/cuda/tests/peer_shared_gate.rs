//! **peer_shared_gate —— TP2 S1 的 G4(额度墙)/ G5(账本闭合)门**。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §5 门表 / §2.2-2.3 实现面。
//!
//! 判据:
//! - **G4**:≤cap 成功;cap+1 **响亮拒**(错误文本断言 —— 不许静默缩容);
//!   坏凭据(篡改 magic)导入响亮拒;
//! - **G5**:peer 账生命周期闭合 —— 分配/导入/释放后 `shared_bytes` /
//!   `alias_bytes` / `mappings` 归零,引擎 `live_bytes` 无漂移;
//! - 附带(映射可用性的最小实证):A 写 → B 经别名读回,数值逐字节相等;
//!   普通 `Free` 对 peer 块 = 响亮 no-op;重复释放/普通块走 peer_release
//!   一律响亮拒。
//!
//! # 运行
//!
//! ```bash
//! # 两张**空闲**卡(CUDA ordinal;生产占卡时勿跑):
//! OWL_PEER_TEST_DEV_A=0 OWL_PEER_TEST_DEV_B=1 \
//!   cargo test -p owl-cuda --test peer_shared_gate -- --nocapture
//! ```
//! 未设 env = skip(与 `device_rows` 门同风格)。
//!
//! ⚠️ 本门**不做** peer 授予前的分档校验(`classify` 在 `owl-comm` 侧;
//! server 只认单笔 cap)。混插对(3090×3080)上 `import` 可能被驱动拒 ——
//! 那属预期(D5 裁决:v1 拒混插),不算本门失败。

use owl_cuda::{
    DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, Shape,
    PEER_ALLOC_HARD_CAP,
};

// ============================================================================
// 组装(同 gpu_server.rs 家法)
// ============================================================================

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-gpu-peer-{tag}"))
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

// ============================================================================
// LedgerJson 取数(最小解析:只认自家键;gpu_server.rs 已有 JSON 合法性门)
// ============================================================================

/// 抠 `"key":<digits>`(第一个匹配;键不存在 = 测试失败 —— 门要硬)
fn json_u64(json: &str, key: &str) -> u64 {
    let pat = format!("\"{key}\":");
    let i = json.find(&pat).unwrap_or_else(|| panic!("LedgerJson 缺键 {key}: {json}")) + pat.len();
    let rest = &json[i..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap_or_else(|e| panic!("键 {key} 数值非法: {e}"))
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// 门
// ============================================================================

#[tokio::test]
async fn peer_shared_gate() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        eprintln!(
            "skip: 需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B(两张空闲卡的 CUDA ordinal)"
        );
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    let mut a = assemble(oa, "a");
    let mut b = assemble(ob, "b");

    // ── 基线:peer 账空;cap = 硬上限常量(单源) ──────────────────────
    let j0 = a.ledger_json().await.expect("ledger_json(a)");
    assert_eq!(json_u64(&j0, "shared_bytes"), 0, "基线 shared 应为 0");
    assert_eq!(json_u64(&j0, "alias_bytes"), 0, "基线 alias 应为 0");
    assert_eq!(json_u64(&j0, "mappings"), 0, "基线 mappings 应为 0");
    let cap = json_u64(&j0, "cap_bytes");
    assert_eq!(
        cap, PEER_ALLOC_HARD_CAP,
        "server 默认 cap 必须 = 硬上限常量(单源;若改过 env/默认值,同步本断言)"
    );
    let live0 = json_u64(&j0, "live_bytes");

    // ── G4a:≤cap 成功(64 MiB = 施工设计 §6 预算账的 staging 设计值)
    const STAGING: usize = 64 << 20;
    let (pb, cred) = a
        .peer_alloc(STAGING, "gate:staging-a")
        .await
        .expect("PeerAlloc 64MiB 应成功(≤cap)");
    let ja = a.ledger_json().await.expect("ledger_json(a) after alloc");
    assert_eq!(json_u64(&ja, "shared_bytes"), STAGING as u64, "shared 账应 +64MiB");
    assert_eq!(json_u64(&ja, "mappings"), 1);
    assert_eq!(
        json_u64(&ja, "live_bytes"),
        live0 + STAGING as u64,
        "peer 本端分配真实占本卡 FB ⇒ 引擎 live 账必增"
    );

    // ── G4b:cap+1 响亮拒(文本断言;拒绝后账不漂移) ─────────────────
    let err = a
        .peer_alloc((cap + 1) as usize, "gate:over-cap")
        .await
        .expect_err("cap+1 必须响亮拒(不得静默缩容)");
    let msg = format!("{err}");
    assert!(
        msg.contains("超单笔上限"),
        "拒收消息须指明超限语义(不猜);实得: {msg}"
    );
    assert!(
        msg.contains("MiB"),
        "拒收消息须带数值(MiB 口径);实得: {msg}"
    );
    let ja2 = a.ledger_json().await.expect("ledger_json(a) after reject");
    assert_eq!(json_u64(&ja2, "shared_bytes"), STAGING as u64, "拒收不得留半成品/漂移");
    assert_eq!(json_u64(&ja2, "mappings"), 1, "拒收不得留下映射笔数");

    // ── G4c:坏凭据响亮拒(import 侧;不猜格式) ──────────────────────
    let mut bad = cred.clone();
    bad[0] ^= 0xFF; // 篡改 magic
    let e_bad = b
        .peer_import(bad, STAGING, "GPU-bad", "gate:bad-cred")
        .await
        .expect_err("magic 被篡改的凭据必须响亮拒");
    assert!(
        format!("{e_bad}").contains("magic"),
        "拒收消息须指明 magic 不符;实得: {e_bad}"
    );

    // ── import 成功(记 live 基线于 import 之前) ────────────────────
    let live_b_before = json_u64(&b.ledger_json().await.expect("ledger_json(b) pre"), "live_bytes");
    let alias = b
        .peer_import(cred, STAGING, "GPU-A(反查)", "gate:staging-b")
        .await
        .expect("PeerImport 应成功");
    let jb = b.ledger_json().await.expect("ledger_json(b) after import");
    assert_eq!(json_u64(&jb, "alias_bytes"), STAGING as u64, "alias 账应 +64MiB");
    assert_eq!(json_u64(&jb, "mappings"), 1);
    assert_eq!(
        json_u64(&jb, "live_bytes"),
        live_b_before,
        "alias 映射的是对端 FB ⇒ 不占本卡显存(live 不增)"
    );

    // ── 映射可用性最小实证:A 写 → A sync → B 经别名读回 ────────────
    let payload: [f32; 4] = [1.5, -2.25, 3.125, 42.0];
    let src = owl_iface::htod!(a, Dtype::F32, &Shape::from(vec![payload.len()]), &le_f32(&payload))
        .await
        .expect("htod src");
    a.copy_block_at(&src, 0, &pb, 0, 16)
        .await
        .expect("设备内拷贝(src → peer 块)");
    a.sync().await.expect("a sync(src 落地完成)");
    let mut got = vec![0u8; 16];
    b.dtoh(&alias, &mut got).await.expect("b 读别名");
    assert_eq!(got, le_f32(&payload), "跨卡别名读回数值不一致(映射不可用?)");
    a.free(&[src.id]).await.expect("free src");

    // ── 普通 Free 对 peer 块 = 响亮 no-op(不静默销毁) ──────────────
    a.free(&[pb.id]).await.expect("free 对 peer 块:句柄语义仍 Ok");
    let ja3 = a.ledger_json().await.expect("ledger_json(a) after free");
    assert_eq!(
        json_u64(&ja3, "shared_bytes"),
        STAGING as u64,
        "普通 Free 不得碰 peer 块(VMM 生命周期专属 PeerRelease)"
    );

    // ── G5:释放闭合(双方;live 归位) ──────────────────────────────
    a.peer_release(vec![pb.id]).await.expect("a peer_release");
    b.peer_release(vec![alias.id]).await.expect("b peer_release");
    let ja4 = a.ledger_json().await.expect("ledger_json(a) after release");
    let jb4 = b.ledger_json().await.expect("ledger_json(b) after release");
    assert_eq!(json_u64(&ja4, "shared_bytes"), 0);
    assert_eq!(json_u64(&ja4, "mappings"), 0);
    assert_eq!(json_u64(&ja4, "live_bytes"), live0, "释放后引擎 live 账归位(无漂移)");
    assert_eq!(json_u64(&jb4, "alias_bytes"), 0);
    assert_eq!(json_u64(&jb4, "mappings"), 0);

    // ── 负例:重复释放 / 普通块走 peer_release —— 都必须响亮拒 ──────
    assert!(
        a.peer_release(vec![pb.id]).await.is_err(),
        "重复释放应响亮拒(DeadBlock),不许静默 Ok"
    );
    let plain = owl_iface::client_alloc!(a, Dtype::F32, 4).await.expect("alloc plain");
    let e_plain = a
        .peer_release(vec![plain.id])
        .await
        .expect_err("普通块走 peer_release 必须响亮拒(指路 Free)");
    assert!(
        format!("{e_plain}").contains("不是 peer 块"),
        "拒收消息须指明错在哪;实得: {e_plain}"
    );
    a.free(&[plain.id]).await.expect("free plain");

    a.close().await.expect("close a");
    b.close().await.expect("close b");
    eprintln!(
        "peer_shared_gate 全绿:cap={}MiB staging={}MiB 双卡 = {} / {}",
        cap >> 20,
        STAGING >> 20,
        oa,
        ob
    );
}
