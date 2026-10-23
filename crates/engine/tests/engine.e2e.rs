//! engine 端到端入口 —— **每 crate 唯一 e2e 入口**(docs/arch/test-architecture.md §二)。
//!
//! 纪律(§二/§三):
//! - 场景体住在 `crates/engine/src/tests.rs`(看得见私有项),**不挂 `#[test]`**,
//!   以 `pub async fn` 暴露(由 feature `e2e` 门控,生产构建零改动);
//! - 本文件是**唯一** `#[tokio::test]`,手动导入并**顺序调用**全部场景;
//! - env **只在这里读一次**,缺失即整档 skip(场景内不再读 env 的门面在迁移中);
//! - 无 `futures` 依赖 ⇒ 逐场景 `catch_unwind` 收集为**挂账**;现为
//!   **串行 + 标签定位**(`[e2e] <name>` 打印后进入,失败即停在该标签处)。
//!
//! 运行:
//! ```bash
//! OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --features e2e --test engine.e2e -- --test-threads=1
//! (27B/草稿等资源来自 `testdata/resources.toml`;见 docs/dev/fixtures.md)
//! ```
//! **GPU 必须串行**(pitfall §16:并行 = 多 CUDA 上下文同卡并发图捕获 → 假挂)。
//!
//! ## 装载分组(现状 / 下一步)
//!
//! 场景**各自装载**(改前就是如此);「一 crate 一入口」已达成(单进程/单入口/一张标签表)。
//! **下一步(Step B)**:按 boot 期配置把场景归组,组内共用**一次**装载 ——
//! | 组 | 场景 | 改前装载数 → 目标 |
//! |---|---|---|
//! | 0.8B 默认档(默认 knobs,max_seq ≤4096) | schedule_decision_surface / two_turns / session_continuity / multi_session_isolation / prefix_cache_hit / longctx_4k | 6 → 1 |
//! | 27B 默认档(默认 knobs,max_seq 1024) | 27b_chat_inference / awq27b_marlin_e2e | 2 → 1 |
//! | 独立(配置不同,不可合并) | w4a16_marlin_e2e(另一装载器)/ spec_identity_gate(Dumb 草稿)/ spec_identity_gate_27b(MTP 草稿)/ dflash2_gate(DFlash2+fp8)/ decode_marginal_bench(greedy+slots1)/ raw_completion_diag(greedy+Dumb) | 各 1 |
//! 合并做法 = 场景签名改收 `&mut RunningEngine<owl_engine::GpuFace>`(删各自 3 行装配),
//! 由本入口建一次并传入。**改动面大、需逐场景核对借用**,单独立刀。

use owl_engine::tests as scen;

/// 场景过滤(只跑名字含子串的场景;缺省 = 全跑)。
/// 用途:①排障单场景 ②按改动面只跑相关场景(test-architecture.md §六)。
/// 例:`OWL_E2E_ONLY=dflash2 cargo test -p owl-engine --features e2e --test engine_e2e`
fn wanted(name: &str) -> bool {
    match owl_shared::env_reader::str("OWL_E2E_ONLY") {
        Some(f) => name.contains(&f),
        None => true,
    }
}

/// 共享装载的步骤(收 `Option<&mut RunningEngine>`;本入口传 `Some`)。
macro_rules! run_shared {
    ($name:ident, $r:expr) => {{
        if wanted(stringify!($name)) {
            eprintln!("[e2e] {:<34} ...(共享装载)", stringify!($name));
            scen::$name(Some($r)).await;
            eprintln!("[e2e] {:<34} OK", stringify!($name));
        } else {
            eprintln!("[e2e] {:<34} skip(OWL_E2E_ONLY)", stringify!($name));
        }
    }};
}

/// 自带装载的步骤(收 `()`)。**新步骤一律并入共享装载组**(§四.1),
/// 自带装载只留给 boot 期配置不同的(草稿类型/池/fp8/slab)。
macro_rules! run {
    ($name:ident) => {{
        if wanted(stringify!($name)) {
            eprintln!("[e2e] {:<34} ...", stringify!($name));
            scen::$name().await;
            eprintln!("[e2e] {:<34} OK", stringify!($name));
        } else {
            eprintln!("[e2e] {:<34} skip(OWL_E2E_ONLY)", stringify!($name));
        }
    }};
}

// ⚠️ **挂账(2026-10-16 定谳,非本仓逻辑错)**:本入口在**单进程**里连开多个
// 引擎实例,而引擎的设备内存**从不 Free**(设计前提 = "全进程单引擎" ——
// `vram.rs` 头注;`Bytes` 是句柄,drop 不发 `Command::Free`,归池只发生在
// 装载收尾 `mem_pool_trim`)⇒ `vram::live` 只增不减,后续实例的
// `StatePool::alloc` vram 治理把池夹到**几何下限**(0.8B/3080 实测:池槽 0/4)。
// 现象:`gpu_spec_identity_gate` 套件内红"KV 块池耗尽"、单跑绿(2026-10-13 已记
// 症状;2026-10-16 定谳 = 池被夹到 4 块,账目本身自洽)。
// 候选修法(未拍):①收尾给引擎发 Free(`Drop for StatePool/GraphFleet`)②门档
// `vram_target=0` 关治理(有真 OOM 风险)③按场景拆进程。**别**用"等 live 回落"
// 的屏障 —— 实测 20s 也不回落(它压根不释放)。

/// **非 async 入口**:场景含图捕获的声明树走查(深递归),而入口把多层
/// async 内联进一个 future → libtest 默认栈(8MiB)实测溢出
/// (`fatal runtime error: stack overflow` @ dflash propose 图捕获)。
/// 故自建 **64MiB 栈线程 + current_thread runtime**(自包含,不靠 env)。
#[test]
fn e2e() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(e2e_body())
        })
        .expect("spawn e2e 线程")
        .join()
        .expect("e2e 线程 panic");
}

async fn e2e_body() {
    // ── env 一次读(缺失 = 整档 skip;不再往场景里散读) ──
    // §18:CPU mock 档**无卡**也可入档(OWL_CPU_MOCK 在 + 无 OWL_TEST_DEVICE
    // ⇒ 只跑 CPU 档场景,GPU 场景一概不触)。
    let dev = owl_cuda::test_device_ordinal_opt();
    let cpu_only = dev.is_none() && owl_shared::env_reader::flag("OWL_CPU_MOCK");
    if dev.is_none() && !cpu_only {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设(engine e2e 整档跳过)");
        return;
    }
    let dev = dev.unwrap_or(0);
    if cpu_only {
        eprintln!("[e2e] 无卡档(OWL_CPU_MOCK):只跑 CPU mock 场景 —— store 演示台三门口 + S2 状态腿");
        run!(cpu_mock_offload_probe);
        run!(cpu_mock_state_leg_probe);
        run!(cpu_mock_index_probe);
        eprintln!("[e2e] 无卡档完成,用时 {:.1}s", std::time::Instant::now().elapsed().as_secs_f32());
        return;
    }
    let has_27b = owl_shared::testkit::resource("ckpt:27b-awq").is_some();
    let has_draft = owl_shared::testkit::resource("ckpt:dflash2").is_some();
    eprintln!(
        "[e2e] device={dev} 27B_DIR={} DFLASH2_DIR={}(缺失的组会自行 skip)",
        has_27b, has_draft
    );
    let t0 = std::time::Instant::now();

    // ── 0.8B 默认档:装载**一次**,6 步共用(§四.1) ──
    let seq08: usize = owl_shared::env_reader::parse_or("OWL_E2E_SEQ", 4096);
    let want08 = [
        "gpu_two_turns_lifecycle",
        "gpu_session_continuity",
        "gpu_multi_session_isolation",
        "gpu_prefix_cache_hit",
        "gpu_longctx_4k_prefix_qa",
    ]
    .iter()
    .any(|n| wanted(n));
    if want08 {
        let mut r08 = scen::boot_08b(seq08).await;
        run_shared!(gpu_two_turns_lifecycle, &mut r08);
        run_shared!(gpu_session_continuity, &mut r08);
        run_shared!(gpu_multi_session_isolation, &mut r08);
        run_shared!(gpu_prefix_cache_hit, &mut r08);
        run_shared!(gpu_longctx_4k_prefix_qa, &mut r08);
    } else {
        eprintln!("[e2e] 0.8B 默认档 skip(被 OWL_E2E_ONLY 排除)");
    }

    // ── S3-3b 实证(自装载;按需开:`OWL_OFFLOAD_PROBE=1`) ──
    run!(gpu_08b_offload_probe);

    // ── S0 预算实测(自装载;按需开:`OWL_BUDGET_PROBE=1`) ──
    run!(gpu_08b_budget_probe);

    // ── S2 状态腿三层化(自装载;按需开:`OWL_STATE_LEG_PROBE=1`) ──
    run!(gpu_08b_state_leg_probe);

    // ── B-3e stage-in 回灌实测(自装载;按需开:`OWL_B3E_PROBE=1`) ──
    run!(gpu_08b_stage_in_probe);

    // ── B-0 选择层索引采集(自装载;按需开:`OWL_INDEX_PROBE=1`) ──
    run!(gpu_08b_index_probe);

    // ── B 轴覆盖率探针(Q-2;按需开:`OWL_COVERAGE_PROBE=1` + `OWL_INDEX_PROBE=1`) ──
    run!(gpu_08b_coverage_probe);

    // ── A2 门:切块不变性(同 prompt × 两种 `prefill_chunk` 读数必须一致;§8.30)──
    run!(gpu_08b_chunk_invariance_gate);

    // ── B-3d 选择层接进 decode(按需开:另加 `OWL_SELECT_DECODE=1`) ──
    run!(gpu_08b_select_decode_probe);

    // ── VLM 图像输入基线(自装载:需要 `vision = true`)──
    run!(gpu_vlm_image_baseline);

    // ── **前缀无感删除**(图像段;2026-10-23):两图 → 删旧图 → 后缀 K 原地重转 ──
    run!(gpu_image_evict_gate);

    // ── 冷启动取证(fix/vlm F1;诊断,无硬断言;两个场景 = A/B)──
    run!(gpu_vlm_first_turn_probe_textfirst);
    run!(gpu_vlm_first_turn_probe_imagefirst);
    // 同探针 × 27B(生产 VLM 档):0.8B 的结论不能外推
    run!(gpu_vlm_first_turn_probe_27b_textfirst);
    run!(gpu_vlm_first_turn_probe_27b_imagefirst);

    // ── piecewise 段驱动件(不装模型;纯 GraphPlan 面) ──
    run!(gpu_single_graph_capture_and_replay);

    // ── 0.8B 其余档:boot 期配置各不相同,各自装载 ──
    run!(gpu_schedule_decision_surface); // max_seq=64/chunk=4(切块敏感)
    run!(gpu_batch_decode_matches_serial); // B5.1 合批 decode(T=2,0.8B)
    // §18 CPU mock 档:同一套 store 三门口跑在 CpuFace 上(无卡;OWL_CPU_MOCK 门控)
    run!(cpu_mock_offload_probe);
    run!(cpu_mock_state_leg_probe);
    run!(gpu_gdn_slot_cross_probe);   // gdn_slots=2 + prefix_cache=false
    run!(gpu_08b_fuse_text_parity);   // greedy + fused_decode 臂
    run!(gpu_longctx_fi_direct_gate); // fp8 池 × {FI 直读, 自研读核}(自装载两回)

    // ── 0.8B W4A16 装载臂(另一装载器) ──
    run!(gpu_w4a16_marlin_e2e);

    // ── 0.8B 投机恒等门(Dumb 草稿 depth=3) ──
    run!(gpu_spec_identity_gate);

    // ── 27B 默认档:装载**一次**,两步共用(§四.1;装载 ~24s/次) ──
    // boot_27b_default 本身 = 一步测试(构造+装载+装配+捕获态断言+计时)
    if has_27b && (wanted("gpu_27b_chat_inference") || wanted("gpu_awq27b_marlin_e2e")) {
        let mut r27 = scen::boot_27b_default().await;
        run_shared!(gpu_27b_chat_inference, &mut r27);
        run_shared!(gpu_awq27b_marlin_e2e, &mut r27);
    } else {
        eprintln!("[e2e] 27B 默认档 skip(无目录或被 OWL_E2E_ONLY 排除)");
    }

    // ── B-5 Stage 2:27B 单卡 + 选择层 + 部分驻留(自装载;按需开 `OWL_SELECT_27B=1`)──
    run!(gpu_27b_select_probe);

    // ── 27B 其余档:boot 期配置各不相同(草稿类型/池/fp8/slab),各自装载 ──
    run!(gpu_27b_raw_completion_diag);   // greedy + Dumb 草稿 + slab
    run!(gpu_decode_marginal_bench);     // greedy + slots1
    run!(gpu_batch_decode_bench_27b);    // B5.1 合批 decode 吞吐(27B,spec off)
    run!(gpu_head_int8_gate);            // int8 头 ≡ bf16 头(token 级)
    run!(gpu_spec_identity_gate_27b);    // MTP 草稿
    run!(gpu_mtp_kv_parity);             // T3：MTP KV 行 t=1 vs 多行对拍
    run!(gpu_dflash2_identity_gate_27b); // DFlash2 草稿 + fp8 池

    eprintln!("[e2e] 全场景通过,用时 {:.1}s", t0.elapsed().as_secs_f32());
}
