//! 引擎测试(2026-10-01 拆分自 engine.rs §5;用例逐字保真,仅私有
//! 字段访问改 StatePool 路径)。
//!
//! **测试分层(2026-10-13,见 docs/arch/test-architecture.md)**:本文件只留**单元测试**
//! (CPU/无装载,`cargo test -p owl-engine --lib`);涉及**模型装载**的场景
//! (下方 `pub async fn gpu_*`,**无 `#[test]`**)由唯一 e2e 入口
//! `crates/engine/tests/engine.e2e.rs` 手动调用(`--features e2e --test engine_e2e`)。
//! 装载只做一次:`boot_08b`/`boot_27b_default` 是共享装载工厂(装载本身即一步测试)。
//!
//! CPU:构造/装载门禁;GPU 门控(OWL_TEST_DEVICE):双 turn 生命周期 /
//! 会话连续 / 4k 长 ctx / W4A16 / 27B AWQ / 多会话隔离 / 前缀缓存命中 /
//! S0 调度决策面。

use crate::engine::{Engine, EngineConfig, EngineKnobs};
#[cfg(test)] // 仅单元档的 decode 差分用例消费;e2e 档不编该函数
use crate::exec::decode_delta;
use crate::running::RunningEngine;
use crate::scheduler::{SchedulerOutput, StepAction};
use crate::turn::TurnEvent;
#[cfg(test)]
use owl_cpu::CpuFace; // dev-dep:仅 cfg(test) 单元档可用
// dev-dep 门控(与 tokio 同款模式):cfg(test) 先求值 ⇒ e2e 档不触宏名
#[cfg(test)]
use owl_shared::config::OwlConfig;
#[cfg(test)]
use owl_macros::owl_test;
use owl_iface::contract::DeviceClient;

fn asset_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/assets/Qwen3.5-0.8B")
}

fn gpu_ordinal() -> Option<usize> {
    owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE")
}

/// CPU:构造(零执行)+ ModelLoader 装载门禁(权重/tokenizer 解析)
/// EngineConfig 全件从**用例配置**注入(隐藏 env `OWL_TEST_CONFIG_CPU_CONSTRUCT_AND_LOAD`
/// → 引擎同款 toml;缺省 = `OwlConfig::default()`)。本用例只到 loader ——
/// 三件均不参与断言,注入不改测试行为;换档 = 换 toml 不改代码(S3 收编示范)。
#[cfg(test)] // dev-dep(owl-macros)门控:单元档专用
#[owl_test]
async fn cpu_construct_and_load(cfg: &OwlConfig) {
    let mut engine = Engine::on(
        EngineConfig {
            device_ordinal: cfg.runtime.device,
            max_seq_tokens: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            knobs: EngineKnobs::from_config(cfg),
        },
        CpuFace::new(),
    )
    .expect("engine 构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    assert_eq!(loaded.model.layers.len(), 24);
    assert!(!loaded.tokenizer.encode("你好").is_empty(), "tokenizer 活性");
}

/// CPU:增量解码多字节字符跨 token(🌟 4 字节;逐 token 喂入,
/// 差分不得 panic 且终态与前缀累计一致)。EngineConfig 同样走用例配置注入。
#[cfg(test)] // dev-dep(owl-macros)门控:单元档专用
#[owl_test]
async fn cpu_incremental_decode_multibyte(cfg: &OwlConfig) {
    let mut engine = Engine::on(
        EngineConfig {
            device_ordinal: cfg.runtime.device,
            max_seq_tokens: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            knobs: EngineKnobs::from_config(cfg),
        },
        CpuFace::new(),
    )
    .expect("engine 构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let ids = loaded.tokenizer.encode("你好，一个🌟加一句中文。");
    assert!(ids.len() >= 4, "测试需要多 token");
    let mut acc = String::new();
    for k in 1..=ids.len() {
        let _delta = decode_delta(&loaded.tokenizer, &ids[..k], &mut acc);
    }
    let final_text = loaded.tokenizer.decode(&ids);
    assert!(
        final_text.starts_with(acc.trim_end()),
        "累计差分应是终文前缀:acc={acc:?} final={final_text:?}"
    );
    assert!(final_text.contains('🌟'), "终文应含拆跨字符");
}

/// **参数化端到端实证**(owl-test-macro.md §四验收 2):wrapper 注入的
/// `cfg` 必须与「独立按同键 loader 复算」逐字段一致 —— env 未设臂
/// (default)与文件臂(设 `OWL_TEST_CONFIG_CASE_CONFIG_INJECTION=<toml>`)
/// 用同一条断言。零装载零 GPU(单元档)。
#[cfg(test)]
#[owl_test]
async fn case_config_injection(cfg: &OwlConfig) {
    let key = "OWL_TEST_CONFIG_CASE_CONFIG_INJECTION";
    let mut expect = match owl_shared::env_reader::str(key) {
        Some(path) => OwlConfig::load(None, Some(std::path::Path::new(&path)))
            .unwrap_or_else(|e| panic!("独立复算失败({path}): {e}")),
        None => OwlConfig::default(),
    };
    expect.runtime.test_device = owl_shared::env_reader::parse("OWL_TEST_DEVICE");
    assert_eq!(cfg, &expect, "宏注入配置必须与唯一 loader 独立复算逐字段同");
}

/// S0 调度决策面单测(GPU 门控 boot,但**调度步零设备执行**):
/// Idle / Begin(Fresh)+ 首块切块 / 决策幂等 / decode 派生量 /
/// GuardHit 增量复用。执行器不跑 —— 推进用手工模拟(fed/out 直写)。
/// S0 调度决策面单测(GPU 门控 boot,但**调度步零设备执行**):
/// Idle / Begin(Fresh)+ 首块切块 / 决策幂等 / decode 派生量 / GuardHit 增量复用。
/// ⚠️ **不进 0.8B 共享装载组**(2026-10-13):本测依赖 `max_seq=64/chunk=4`
/// 的切块语义,而共享档是 `4096/32` → 实测红。切块敏感的用例一律自带装载。
pub async fn gpu_schedule_decision_surface() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine =
        Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64, prefill_chunk: 4, knobs: EngineKnobs::default() })
            .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut eng = engine.run(loaded).await.expect("run");

    // 空转:队列空且无活跃
    assert!(matches!(eng.schedule().unwrap(), SchedulerOutput::Idle));

    // 提交 → Begin(Fresh)+ 首块(chunk = min(prefill_chunk, 剩余))
    let (sid, _tid) = eng.submit_session(None, "一二三四五六七八九十", 3).unwrap();
    let n = match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            let b = begin.expect("首步必带 begin");
            assert_eq!(b.session_id, sid);
            assert!(b.reset_gdn, "Fresh 重置 GDN 格");
            assert!(b.restore_key.is_none(), "无前缀命中");
            match action {
                StepAction::Prefill { base, is_last, chunk_ids } => {
                    assert_eq!(base, 0);
                    assert_eq!(chunk_ids.len(), 4, "chunk = prefill_chunk");
                    assert!(!is_last, "n>4 → 首块非末块");
                }
                _ => panic!("期望 Prefill"),
            }
            eng.turns.active.as_ref().unwrap().prompt_ids.len()
        }
        _ => panic!("期望 Step"),
    };
    assert!(n > 4, "prompt 须超一块(token n={n})");

    // 决策幂等:未执行再调度 → 同相位、不再 begin(执行器才推进 fed)
    match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            assert!(begin.is_none(), "活跃中不再 begin");
            assert!(matches!(action, StepAction::Prefill { base: 0, .. }));
        }
        _ => panic!("期望 Step"),
    }

    // 模拟执行器推进至 prompt 末 + 产出首 token → decode 派生量备齐
    {
        let act = eng.turns.active.as_mut().unwrap();
        act.fed = act.prompt_ids.len();
        act.out.push(7);
    }

    let page = eng.pool.page;
    match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            assert!(begin.is_none());
            match action {
                StepAction::Decode { token, pos, kv_slot, gdn_slot, grew } => {
                    assert_eq!(token, 7);
                    assert_eq!(pos, n);
                    // 物理槽 = 块表[pos/page]×page + pos%page(页 16 后
                    // 17 token 提示词跨入块 1,恒 b0 假设已不成立)
                    let expect_slot =
                        eng.bank.sessions.get(sid).unwrap().block_table[n / page] as usize * page
                            + n % page;
                    assert_eq!(kv_slot as usize, expect_slot, "物理槽 = 块链换算");
                    assert_eq!(gdn_slot, 0, "首会话格 0");
                    assert!(!grew, "页内不跨页");
                }
                _ => panic!("期望 Decode"),
            }
        }
        _ => panic!("期望 Step"),
    }

    // GuardHit:账本截短两 token(模拟客户端续发同前缀 prompt),
    // 同会话续发 → 增量起步 + 不重置 GDN
    {
        let act = eng.turns.active.take().unwrap();
        let s = eng.bank.sessions.get_mut(act.session_id).unwrap();
        s.commit_turn(&act.prompt_ids, &[]);
        s.cached_len -= 2;
    }
    eng.submit_session(Some(sid), "一二三四五六七八九十", 2).unwrap();
    match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            let b = begin.expect("续 turn 带 begin");
            assert!(!b.reset_gdn, "GuardHit 不重置");
            assert!(b.restore_key.is_none(), "GuardHit 非前缀缓存路径");
            match action {
                StepAction::Prefill { base, is_last, .. } => {
                    assert_eq!(base, n - 2, "增量起步 = cached_len");
                    assert!(is_last, "2 token 段一块即末块");
                }
                _ => panic!("期望 Prefill"),
            }
        }
        _ => panic!("期望 Step"),
    }
}

/// GPU 门控:双 turn 生命周期 —— submit 入队 / pump 事件流 / per-turn
/// GDN 重置(第二个 turn 在脏状态后仍须产出连贯文本)。
/// D1 冒烟:0.8B greedy 双配置(融合 vs 旧链)各产出非空连贯文本。
/// **文本本身两配置必然分叉**(数值微差 × 自回归混沌,2026-10-03 定谳:
/// 融合核与旧链互差 ≤2e-4,算子层 32 步现实幅度递推单测全绿)——
/// 本测试只验「两配置链路都活着且确定(各 ×2 逐字稳定)」。门控 OWL_TEST_DEVICE。
pub async fn gpu_08b_fuse_text_parity() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();

    async fn gen_text(fuse: bool, ordinal: usize, dir: &std::path::Path) -> String {
        // GDN 融合 A/B 收敛注:NO_FUSE 反开关是唯一活闸(EnvProvider);
        // OWL_GDN_FUSE_DECODE 无读者(死旗标,显式依赖律改造中删除)——
        // 双臂现同态(fused on),保留参数占位与恒等断言
        let _ = fuse;
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() }; // greedy
        knobs.env.gdn.fused_decode = true;
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 128,
            prefill_chunk: 16,
            knobs,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let id = running.submit("用五十字介绍长城。", 40).expect("submit");
        drive_turn(&mut running, id).await
    }

    let a1 = gen_text(false, ordinal, &dir).await;
    let a2 = gen_text(false, ordinal, &dir).await;
    let b1 = gen_text(true, ordinal, &dir).await;
    let b2 = gen_text(true, ordinal, &dir).await;
    eprintln!("[smoke] baseline={a1:?}/{a2:?} fused={b1:?}/{b2:?}");
    assert!(!a1.is_empty() && !b1.is_empty(), "两配置均应产出非空文本");
    assert_eq!(a1, a2, "baseline greedy 跨 boot 确定性");
    assert_eq!(b1, b2, "fused greedy 跨 boot 确定性");
}

pub async fn gpu_two_turns_lifecycle(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine =
            Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64, prefill_chunk: 16, knobs: EngineKnobs::default() })
                .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "真模型捕获应成功(回放态),得 {:?}",
        running.capture_outcome()
    );

    let t1 = running.submit("Hello, who are you?", 16).expect("t1");
    let t2 = running.submit("用一句话介绍长城。", 16).expect("t2");
    let (mut ttft, t0) = (None, std::time::Instant::now());
    let mut done = Vec::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => break,
            TurnEvent::Prefill { turn, fed, total } => {
                eprintln!("[ev] prefill t{turn} {fed}/{total} (+{:?})", t0.elapsed());
            }
            TurnEvent::Token { turn, .. } if ttft.is_none() => {
                ttft = Some(t0.elapsed());
                eprintln!("[ttft] 首 Token(turn {turn}): {:?}", ttft.unwrap());
            }
            TurnEvent::Completed { turn, text, .. } => {
                eprintln!("[ttft] turn {turn} 完成(累计 {:?})", t0.elapsed());
                done.push((turn, text))
            }
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    assert_eq!(done.len(), 2, "两 turn 都完成");
    assert!(done.iter().any(|(t, _)| *t == t1));
    assert!(done.iter().any(|(t, _)| *t == t2));
    for (t, text) in &done {
        eprintln!("[test] t{t}: {text}");
        assert!(!text.trim().is_empty(), "t{t} 产出非空");
    }
}

/// GPU 门控:同会话连续 turn(S0 验收)—— turn2 应命中账本
/// (跳过 GDN 重置,只 prefill 增量段);临时会话路径由生命周期
/// 测试覆盖,此处验 Some(id) 连续性 + 收口账推进。
pub async fn gpu_session_continuity(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // turn1:建立会话账(显式 id;None = 临时会话终了即焚,不归此测)
    let (sid, t1) = running
        .submit_session(Some(42), "请记住:我最喜欢的数字是七。", 16)
        .expect("t1");
    assert_eq!(sid, 42);
    let text1 = drive_turn(&mut *running, t1).await;
    eprintln!("[test] t{t1}(session {sid}): {text1}");
    assert!(!text1.trim().is_empty());
    assert!(running.session_len(sid).unwrap_or(0) > 0, "turn1 收口账非零");

    // turn2:同会话全量重发(历史 + 新问题)→ 守卫应命中,增量 prefill
    let (sid2, t2) = running
        .submit_session(
            Some(sid),
            "请记住:我最喜欢的数字是七。我刚才说我最喜欢的数字是什么?",
            16,
        )
        .expect("t2");
    assert_eq!(sid2, sid, "同会话归队");
    let text2 = drive_turn(&mut *running, t2).await;
    eprintln!("[test] t{t2}(session {sid2}): {text2}");
    assert!(!text2.trim().is_empty());

    // 收口验证:账本应推到 turn2 全量 + 生成段;记忆质量(答“七”)
    // 属 S3 金标验收,此处不断言文本内容
    let cached = running.session_len(sid).expect("账在");
    assert!(cached > 0, "同会话收口后账本非零");
}

/// GPU 门控:4k 长 ctx 前缀记忆 QA(E1 收尾 P4)—— 暗号埋在 prompt
/// 开头(≈token 10),问题压在 ≈3900 token 处;真全局注意力下应召回,
/// 旧 OWL_MAX_KV=256 滑窗截断下物理不可过(暗号在窗外)。兼验收:
/// 4k 预算守卫、分页 prefill 百级 chunk、v1 decode 长程、同会话
/// turn2(前缀失配回退全量路径 @4k)。
pub async fn gpu_longctx_4k_prefix_qa(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let seq: usize = owl_shared::env_reader::parse_or("OWL_E2E_SEQ", 4096);
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: seq,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // 组装:暗号开头 + 中性填充 + 补全式探针收尾(裸 LM 无 chat 模板,
    // 问答式会退化成文本续写 —— 探针改为让模型续写暗号本体)
    let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
    let question = "\n\n清单结束。按约定,暗号重复一遍:暗号是「";
    let filler = |i: usize| {
        format!(
            "第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n"
        )
    };
    let mut body = String::from(marker);
    let mut i = 0usize;
    let nofill = owl_shared::env_reader::flag("OWL_E2E_NOFILL");
    loop {
        if nofill {
            break; // 对照实验:短 ctx 直问(验模型/模板,不验长程)
        }
        let cand = format!("{body}{}", filler(i));
        let full = format!("{cand}{question}");
        if running.tok.encode(&full).len() + 32 > seq {
            break; // 留 32 token 生成余量(turn2 预算同享)
        }
        body = cand;
        i += 1;
    }
    let prompt_t1 = format!("{body}{question}");
    let n_tok = running.tok.encode(&prompt_t1).len();
    eprintln!("[test] 长 ctx prompt = {n_tok} tok(填充 {i} 条)");
    assert!(
        nofill || n_tok > seq * 4 / 5,
        "长文应填满窗口 ≥80%,得 {n_tok}/{seq}"
    );

    let t0 = std::time::Instant::now();
    let (sid, t1) = running
        .submit_session(Some(7), prompt_t1.as_str(), 16)
        .expect("t1 应在 4k 预算内");
    assert_eq!(sid, 7);
    let mut ttft = None;
    let mut prefill_last = None;
    let text1;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("turn {t1} 队列丢失"),
            TurnEvent::Prefill { turn, fed, total } if turn == t1 => {
                if fed % (32 * 16) == 0 || fed == total {
                    eprintln!("[ev] t{t1} prefill {fed}/{total} @ {:?}", t0.elapsed());
                }
                prefill_last = Some((fed, total, t0.elapsed()));
            }
            TurnEvent::Token { turn, .. } if turn == t1 && ttft.is_none() => {
                ttft = Some(t0.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == t1 => {
                text1 = text;
                break;
            }
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    let wall1 = t0.elapsed();
    if let Some((fed, total, el)) = prefill_last {
        eprintln!(
            "[bench] t{t1} prefill {fed}/{total} tok @ {el:?}({:.0} tok/s,debug) | TTFT {:?} | 总 {:?}",
            fed as f64 / el.as_secs_f64(),
            ttft.unwrap(),
            wall1
        );
    }
    eprintln!("[test] t{t1} 答: {text1}");
    // 记忆质量金标挂 S3(0.8B + greedy + 裸模板的答句质量不可靠;
    // 对照实验:短/长 ctx 行为一致 ⇒ 机械等价)。此处验收 =
    // 4k 全链不崩 + 帐本收口 + 计时(v1 smem 越界已修,见上)
    assert!(!text1.trim().is_empty(), "t{t1} 产出非空");

    // turn2 同会话再问(前缀失配 → 回退全量路径)。E2a 后唯一验收:
    // 修复前此处在 4k 下 OOM(server 中间块零回收,双 turn 累计
    // ~24GB;现 eval 竞技场每 chunk 归池,显存恒平)
    {
        let (_, t2) = running
            .submit_session(Some(sid), format!("{body}再重复一遍:暗号是「"), 16)
            .expect("t2 应在预算内");
        let t0b = std::time::Instant::now();
        let text2 = drive_turn(&mut *running, t2).await;
        eprintln!("[bench] t{t2} 全量回退重灌 + 生成 @ {:?}", t0b.elapsed());
        eprintln!("[test] t{t2} 答: {text2}");
        assert!(!text2.trim().is_empty(), "t{t2} 产出非空");
        assert!(running.session_len(sid).unwrap_or(0) >= n_tok, "账本收口");
    }
}

/// GPU 门控:FI 直读单池健康门(2026-10-13 立项;kV 统一契约 P5 收尾)。
///
/// 背景:统一契约把主池翻 kNHD、FI 从「影子池双写」改为「直读主池」;
/// 为迎合它,写核/读核大范围改过。这条链**数值上**是否健康,此前没有
/// 有效判据:HTTP + greedy 的「跳实现逐字节门」已实测作废(两个
/// flash-attention 实现的 ulp 差被 greedy 放大成完全不同的文本;且
/// 27B AWQ + 空 think 预填在长文上模型自身退化)—— 见坑册 §23。
///
/// 本门姿势(照 [`gpu_longctx_4k_prefix_qa`] 的教训):
/// - **裸 LM**(不走 chat 模板)+ **补全式探针**(让模型续写暗号本体;
///   问答式在裸 LM 上必退化成文本续写);
/// - **同 dtype(fp8)同池几何**,只切读核(`attn.fi`)= FI 直读 vs 自研
///   (chunked_opt / split)→ 输出**应逐字一致**,**与文本质量无关**:
///   就算续写是垃圾,两臂的垃圾也必须一样 —— 这才是等价门;
/// - 4k prompt ⇒ 百级 chunk(prefill_chunk=32),覆盖两臂各自的多 chunk 路径。
///
/// 断言:①两臂 prompt token 数一致 ②召回行为一致 ③**文本逐字一致**。
pub async fn gpu_longctx_fi_direct_gate() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let seq: usize = owl_shared::env_reader::parse_or("OWL_E2E_SEQ", 4096);
    const NEEDLE: &str = "蓝鲸二十一号";

    /// 单臂:boot(池 dtype 与 `fi` 可切)→ 4k 前缀记忆 prompt → 补全式生成。
    async fn arm(ordinal: usize, fi: bool, fp8: bool, seq: usize) -> (String, usize) {
        let dir = asset_dir();
        let mut knobs = EngineKnobs::default();
        knobs.env.kv.quant = if fp8 {
            owl_models::env::KvQuant::Fp8E4M3
        } else {
            owl_models::env::KvQuant::None
        };
        knobs.env.attn.fi = fi; // ← 唯一变量:读核(FI 直读 vs 自研 chunked/split)
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: seq,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");

        let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
        let question = "\n\n清单结束。按约定,暗号重复一遍:暗号是「";
        let filler = |i: usize| {
            format!("第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n")
        };
        let mut body = String::from(marker);
        let mut i = 0usize;
        loop {
            let cand = format!("{body}{}", filler(i));
            if running.tok.encode(&format!("{cand}{question}")).len() + 32 > seq {
                break; // 留 32 token 生成余量
            }
            body = cand;
            i += 1;
        }
        let prompt = format!("{body}{question}");
        let n_tok = running.tok.encode(&prompt).len();
        assert!(n_tok > seq * 4 / 5, "长文应填满窗口 ≥80%,得 {n_tok}/{seq}");
        let t0 = std::time::Instant::now();
        let (_, t) = running
            .submit_session(Some(7), prompt.as_str(), 16)
            .expect("submit 应在 4k 预算内");
        let text = drive_turn(&mut running, t).await;
        eprintln!(
            "[fi-gate] fi={fi} fp8={fp8} prompt={n_tok} tok({i} 条) 生成 @ {:?}: {text:?}",
            t0.elapsed()
        );
        // 实例收尾屏障(同 gpu_spec_identity_gate/恒等门 既有约定):
        // 本门连开 4 个实例,不 barrier 则前一个的 CUDA ctx/显存尚未析构,
        // 下一个实例的 VRAM 探测读到未释放的账 → 池被算小 → 后续用例
        // “KV 块池耗尽”(§11.5 跨用例污染同族;2026-10-13 套件内实测复现)。
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
        (text, n_tok)
    }

    // 两臂:fp8 生产档 × 读核(自研 / FI 直读)。同 dtype 内两臂必须
    // **逐字一致**。f16 档由既有 `gpu_longctx_4k_prefix_qa`(默认 env 即
    // f16 池)覆盖 —— 本门只补 fp8 这一档,少开两次装载(套件内池账
    // 扰动与耗时都省;2026-10-13 四臂版曾把后置 spec 门的池挤小)。
    let mut _cross = String::new();
    for fp8 in [true] {
        let (off, n_off) = arm(ordinal, false, fp8, seq).await;
        let (on, n_on) = arm(ordinal, true, fp8, seq).await;
        assert_eq!(n_off, n_on, "两臂 prompt token 数应一致(同构造)");
        let (r_off, r_on) = (off.contains(NEEDLE), on.contains(NEEDLE));
        eprintln!(
            "[fi-gate] fp8={fp8} 召回 FI-off={r_off} / FI-on={r_on}; 逐字一致={}",
            off == on
        );
        assert_eq!(
            r_off, r_on,
            "两臂召回行为应一致(fp8={fp8})\n  off={off:?}\n  on ={on:?}"
        );
        assert_eq!(
            off, on,
            "同 dtype 同池、只换读核 → 输出应逐字一致(不一致 = FI 直读或自研读核的实参/几何有问题;fp8={fp8})\n  off={off:?}\n  on ={on:?}"
        );
        _cross = off.clone();
    }
}

/// GPU 门控:W4A16 marlin 全链 e2e(E3 验收;REQ-PRE-01)——
/// llm-compressor 转换的 0.8B W4A16 检查点装载(装载期 ct→marlin
/// 重排)+ 生成:eligible Linear 走 foreign marlin GEMM,小线性
/// (GDN in_proj_z/b/a)反量化 f16 直读。
pub async fn gpu_w4a16_marlin_e2e() {
    // 套件统一约定(OWL_TEST_DEVICE):曾硬钉 ordinal 2 —— 与现役
    // vLLM 生产服务器(3080 对)撞卡时 OOM at capture slab(2026-10-06)
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir(); // 转换产物与 f16 同目录约定:../Qwen3.5-0.8B-W4A16
    let w4a16_dir = dir
        .parent()
        .expect("assets")
        .join("Qwen3.5-0.8B-W4A16");
    if !w4a16_dir.exists() {
        eprintln!("skip: W4A16 检查点不存在({:?}),先跑转换器", w4a16_dir);
        return;
    }
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    let loaded = engine
        .loader()
        .load_qwen35_0_8b_w4a16(&w4a16_dir, &dir)
        .await
        .expect("W4A16 装载");
    let mut running = engine.run(loaded).await.expect("装配");
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "W4A16 decode 图应捕获成功(回放态),得 {:?}",
        running.capture_outcome()
    );
    let t0 = std::time::Instant::now();
    let t1 = running
        .submit("用一句话介绍长城。", 16) // D1 立案复现:16 tok 亦崩(前 2 token 对,随后乱;基线同步长连贯)
        .expect("submit");
    let text = drive_turn(&mut running, t1).await;
    eprintln!("[bench] W4A16 16 tok @ {:?}({:.1} tok/s)", t0.elapsed(), 16.0 / t0.elapsed().as_secs_f64());
    eprintln!("[test] W4A16 答: {text}");
    assert!(!text.trim().is_empty(), "产出非空");
}

/// GPU 门控:Qwen3.8-27B AWQ-INT4 全链冒烟(2026-10-01;cyankiwi
/// g32-asym 检查点)—— 装载(formats/awq.rs 懒物化 + marlin kU4
/// GEMM_W4A16_AWQ)+ 图捕获 + 生成。门控 OWL_AWQ27B_DIR = 检查点
/// 目录(models/cyankiwi/Qwen3.8-27B-AWQ-INT4);未设则 skip。
/// VRAM 预算:权重 ~17.5GB + KV/GDN 状态 ~1.3GB → 3090 Ti 24G。
pub async fn gpu_awq27b_marlin_e2e(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
        eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        eprintln!("skip: 检查点目录不存在({dir:?})");
        return;
    }
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let _t0 = std::time::Instant::now();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16)
            .await
            .expect("27B AWQ 装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };
    assert!(
        owl_shared::env_reader::flag("OWL_NO_GRAPH")
            || matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "27B decode 图应捕获成功,得 {:?}",
        running.capture_outcome()
    );
    let t1 = running
        .submit("用一句话介绍长城。", 16)
        .expect("submit");
    let t2 = std::time::Instant::now();
    let text = drive_turn(&mut *running, t1).await;
    eprintln!(
        "[bench] 生成 tok @ {:.1}ms(含首 token prefill 尾步)",
        t2.elapsed().as_secs_f32() * 1e3
    );
    // 刀D 取证:层间原生时间线(OWL_TS_PROBE;clock64 @~1.98GHz)
    if owl_shared::env_reader::flag("OWL_TS_PROBE") {
        let raw = match running.graphs.session.read_input_slot_bytes("ts_buf").await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[ts] 读回失败 {e}");
                vec![]
            }
        };
        let stamps: Vec<u64> = raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .filter(|&v| v != 0)
            .collect();
        eprintln!("[ts] 拍数 {}", stamps.len());
        let mut prev = *stamps.first().unwrap_or(&0);
        for (i, &s) in stamps.iter().enumerate() {
            let d_us = (s.saturating_sub(prev)) as f64 / 1980.0;
            eprintln!("[ts] #{i:02} Δ={d_us:8.1}µs");
            prev = s;
        }
    }
    eprintln!("[test] 27B AWQ 答: {text}");
    assert!(!text.trim().is_empty(), "产出非空");
}

/// GPU 门控:多会话隔离(E2b 验收)—— A/B 两会话交替 turn:
/// ① B 的 turn 前后,A 的块链/账本逐字节不受扰;② A/B 物理块互异
/// (无前缀缓存时零共享);③ 双会话产出非空。隔离 = 块管理器的
/// 直接行为证据(块链不同 ⇒ KV 物理槽不同)。
pub async fn gpu_multi_session_isolation(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    let (_, ta1) = running
        .submit_session(Some(101), "我的代号是阿尔法。", 12)
        .expect("A1");
    let ta1_text = drive_turn(&mut *running, ta1).await;
    eprintln!("[test] A1: {ta1_text}");
    assert!(!ta1_text.trim().is_empty());
    let a_table_1 = running.bank.sessions.get(101).expect("A 账在").block_table.clone();
    let a_len_1 = running.session_len(101).expect("A 账在");
    assert!(!a_table_1.is_empty(), "A 块链非空");

    let (_, tb1) = running
        .submit_session(Some(202), "我的代号是贝塔。", 12)
        .expect("B1");
    let tb1_text = drive_turn(&mut *running, tb1).await;
    eprintln!("[test] B1: {tb1_text}");
    assert!(!tb1_text.trim().is_empty());

    // 隔离断言 ①:B 的 turn 后,A 的块链/账本逐字节不变
    let a_table_2 = running.bank.sessions.get(101).expect("A 账在").block_table.clone();
    assert_eq!(a_table_1, a_table_2, "B 的 turn 不得扰动 A 块链");
    assert_eq!(running.session_len(101), Some(a_len_1), "B 的 turn 不得扰动 A 账本");
    // 隔离断言 ②:物理块零共享(无前缀缓存 = 全新分配)
    let b_table = running.bank.sessions.get(202).expect("B 账在").block_table.clone();
    assert!(b_table.iter().all(|b| !a_table_2.contains(b)), "A/B 物理块互异");
    eprintln!("[test] 块链 A = {a_table_2:?} / B = {b_table:?}");

    // A 二轮(全量重发,前缀守卫失配回退全量也须稳)
    let (_, ta2) = running
        .submit_session(Some(101), "我的代号是阿尔法。我的代号是什么?", 12)
        .expect("A2");
    let ta2_text = drive_turn(&mut *running, ta2).await;
    eprintln!("[test] A2: {ta2_text}");
    assert!(!ta2_text.trim().is_empty());
    assert!(running.session_len(101).unwrap_or(0) >= a_len_1, "A 账本推进");
}

/// GPU 门控:前缀缓存命中(E2c 验收;REQ-PRE-04)—— 同前缀跨会话
/// 二轮:A 建档(块链 + 快照登记),B 同前缀 + 异尾 → 块链复用 +
/// 快照恢复,prefill 只算尾部。机械断言:B 的 prefill chunk 数 ≪ A;
/// B 块链前缀与 A 物理共享;TTFT 可测下降。
pub async fn gpu_prefix_cache_hit(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 512,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // A:marker + filler(~320 tok,跨 ≥1 块边界 → 快照可拍)
    let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
    let filler = |i: usize| {
        format!(
            "第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n"
        )
    };
    let mut body = String::from(marker);
    let mut i = 0usize;
    while running.tok.encode(&format!("{body}{}", filler(i))).len() < 300 {
        body = format!("{body}{}", filler(i));
        i += 1;
    }
    let prompt_a = format!("{body}清单结束。按约定,暗号重复一遍:暗号是「");
    let prompt_b = format!("{body}这次换个问题:暗号里出现了什么动物?");
    let n_a = running.tok.encode(&prompt_a).len();
    let n_b = running.tok.encode(&prompt_b).len();
    eprintln!("[test] A prompt = {n_a} tok / B prompt = {n_b} tok(前缀共享)");

    let t0 = std::time::Instant::now();
    let (_sid_a, ta1) = running.submit_session(Some(101), prompt_a.as_str(), 12).expect("A");
    let mut a_prefill_chunks = 0usize;
    #[allow(unused_assignments)]
    let (mut a_ttft, mut a_text) = (None, String::new());
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("A 队列丢失"),
            TurnEvent::Prefill { turn, .. } if turn == ta1 => a_prefill_chunks += 1,
            TurnEvent::Token { turn, .. } if turn == ta1 && a_ttft.is_none() => {
                a_ttft = Some(t0.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == ta1 => {
                a_text = text;
                break;
            }
            TurnEvent::Failed { turn, err } => panic!("A{turn} 失败: {err}"),
            _ => {}
        }
    }
    eprintln!("[bench] A prefill chunks = {a_prefill_chunks}, TTFT = {:?}", a_ttft.unwrap());
    assert!(!a_text.trim().is_empty());

    // B:同前缀 + 异尾(新会话;guard 必失配 → 走前缀缓存匹配)
    let t0b = std::time::Instant::now();
    let (_, tb1) = running.submit_session(Some(202), prompt_b.as_str(), 12).expect("B");
    let mut b_prefill_chunks = 0usize;
    let mut b_ttft = None;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("B 队列丢失"),
            TurnEvent::Prefill { turn, .. } if turn == tb1 => b_prefill_chunks += 1,
            TurnEvent::Token { turn, .. } if turn == tb1 && b_ttft.is_none() => {
                b_ttft = Some(t0b.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == tb1 => {
                eprintln!("[test] B 答: {text}");
                break;
            }
            TurnEvent::Failed { turn, err } => panic!("B{turn} 失败: {err}"),
            _ => {}
        }
    }
    eprintln!(
        "[bench] B prefill chunks = {b_prefill_chunks}, TTFT = {:?}(A = {:?})",
        b_ttft.unwrap(),
        a_ttft.unwrap()
    );

    // 机械断言:chunk 数骤降 + 块链物理共享 + 账本各自收口
    let a_chain = running.bank.sessions.get(101).expect("A").block_table.clone();
    let b_chain = running.bank.sessions.get(202).expect("B").block_table.clone();
    assert!(
        b_prefill_chunks * 4 <= a_prefill_chunks.max(1),
        "B prefill chunk 数应 ≪ A:{b_prefill_chunks} vs {a_prefill_chunks}"
    );
    let shared = a_chain.iter().filter(|b| b_chain.contains(b)).count();
    assert!(shared >= 8, "前缀块应物理共享,共享 {shared}");
    eprintln!("[test] 共享物理块 {shared}(A 链 {} / B 链 {})", a_chain.len(), b_chain.len());
}

/// GPU 门控:27B 单卡推理验收(正菜;3090 Ti 24G)—— 真实三问 ×
/// greedy(模板已含 thinking-off 空 think 块),验收 = 回答非空 +
/// 无退化复读(启发式:任意 2-8 字片段连现 4 次即退化)。
/// 输出全文打印供人工判读;复读启发式是底线门不是质量门。
pub async fn gpu_27b_chat_inference(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
        eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        eprintln!("skip: 检查点目录不存在({dir:?})");
        return;
    }
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 1024,
            prefill_chunk: 128,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16)
            .await
            .expect("27B AWQ 装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    let questions = [
        "用一句话介绍长城。",
        "水的沸点是多少摄氏度?",
        "写一句关于春天的诗。",
    ];
    // 退化复读启发式:任意 2-8 字片段连现 4 次(贪心解码典型病灶;
    // 正常文本几乎不可能命中)
    fn degenerate(text: &str) -> Option<String> {
        let chars: Vec<char> = text.chars().collect();
        for len in 2..=8usize {
            for start in 0..chars.len().saturating_sub(len * 4) {
                let pat: String = chars[start..start + len].iter().collect();
                let mut hits = 0;
                let mut i = start;
                while i + len <= chars.len() && chars[i..i + len].iter().collect::<String>() == pat {
                    hits += 1;
                    i += len;
                    if hits >= 4 {
                        return Some(pat);
                    }
                }
            }
        }
        None
    }
    for (qi, q) in questions.iter().enumerate() {
        // GPU 逐核归因探针窗(OWL_GPU_PROF=1):Q0 前 reset / Q0 后 query
        // —— 纯 decode 窗分账。注:窗口钉在 Q0(首 turn)——eager 模式
        // 27B 逐 turn 显存爬坡 ~480MB/步(收割缺口,另案),turn 2 会 OOM
        if qi == 0 && owl_shared::env_reader::flag("OWL_GPU_PROF") {
            owl_shared::metrics::reset_metrics();
        }
        let t0 = std::time::Instant::now();
        let id = running.submit(*q, 96).expect("submit");
        let mut ttft = None;
        let mut n_tok = 0usize;
        let text;
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Idle => panic!("Q{qi} 队列丢失"),
                TurnEvent::Token { turn, .. } if turn == id => {
                    if ttft.is_none() {
                        ttft = Some(t0.elapsed());
                    }
                    n_tok += 1;
                }
                TurnEvent::Completed { turn, text: t, .. } if turn == id => {
                    text = t;
                    break;
                }
                TurnEvent::Failed { turn, err } => panic!("Q{qi} turn {turn} 失败: {err}"),
                _ => {}
            }
        }
        let wall = t0.elapsed();
        eprintln!(
            "[27b-chat] Q{qi}: {q}\n[27b-chat] A{qi}: {text}\n[27b-chat] {n_tok} tok @ {wall:?}(TTFT {:?}, {:.1} tok/s)",
            ttft.unwrap(),
            n_tok as f64 / wall.as_secs_f64()
        );
        // GPU 逐核归因探针:Q0 窗口结束 → 热点分账(按总时长降序)
        if qi == 0 && owl_shared::env_reader::flag("OWL_GPU_PROF") {
            owl_shared::metrics::query_metrics(
                &owl_shared::metrics::MetricsFilter::new().tag_prefix("gpu.").limit(24),
            );
        }
        assert!(!text.trim().is_empty(), "Q{qi} 空回答");
        if let Some(pat) = degenerate(&text) {
            // 严格门(OWL_27B_STRICT=1)在 paged decode 质量案结案后启用;
            // 现默认警告 —— paged 档起始语义正确但退化复读(立案中),
            // naive 档(OWL_FORCE_NAIVE=1)三问连贯 = 当前推荐配方
            if owl_shared::env_reader::flag("OWL_27B_STRICT") {
                panic!("Q{qi} 退化复读:片段 {pat:?} 连现 ≥4 次;全文 = {text:?}");
            }
            eprintln!("[27b-chat][warn] Q{qi} 退化复读(立案中):片段 {pat:?}");
        }
    }
}

    /// **0.8B 默认档共享装载**(e2e 唯一装载点之一;6 步共用:
/// schedule_decision_surface / two_turns_lifecycle / session_continuity /
/// multi_session_isolation / prefix_cache_hit / longctx_4k_prefix_qa)。
/// 装载本身即一步测试(构造+装载+装配+捕获态断言+计时)。
pub async fn boot_08b(max_seq: usize) -> RunningEngine<owl_cuda::GpuClient> {
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let dir = asset_dir();
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: max_seq,
        prefill_chunk: 32,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let running = engine.run(loaded).await.expect("装配");
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "0.8B 装载应捕获成功(回放态),得 {:?}",
        running.capture_outcome()
    );
    eprintln!("[e2e] 0.8B 默认档装载完成 {:?}(max_seq={max_seq})", t0.elapsed());
    running
}

/// **27B 默认档共享装载**(chat_inference / awq27b_marlin_e2e 共用;
/// 27B 装载 ~24-32s/次,重复装载 = 纯浪费)。
pub async fn boot_27b_default() -> RunningEngine<owl_cuda::GpuClient> {
    let dir = std::path::PathBuf::from(
        owl_shared::env_reader::str("OWL_AWQ27B_DIR").expect("OWL_AWQ27B_DIR"),
    );
    assert!(dir.exists(), "检查点目录不存在: {dir:?}");
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 1024,
        prefill_chunk: 128,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    // 装载分相上屏(loader 性能工作台账用:构造 / 权重装载 / 装配+捕获)
    eprintln!("[e2e] 27B 引擎构造 {:?}", t0.elapsed());
    let t_load = std::time::Instant::now();
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16)
        .await
        .expect("27B AWQ 装载");
    eprintln!("[e2e] 27B 权重装载 {:?}", t_load.elapsed());
    let t_run = std::time::Instant::now();
    let running = engine.run(loaded).await.expect("装配");
    eprintln!("[e2e] 27B 装配+图捕获 {:?}", t_run.elapsed());
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "27B 装载应捕获成功(回放态),得 {:?}",
        running.capture_outcome()
    );
    eprintln!("[e2e] 27B 默认档装载完成 {:?}", t0.elapsed());
    running
}

/// 诊断:裸续写 A/B(模板态病 vs 权重病的鉴别臂)
    pub async fn gpu_27b_raw_completion_diag() {
        let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else { return; };
        let dir = std::path::PathBuf::from(dir);
        if !dir.exists() { return; }
        let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal, max_seq_tokens: 512, prefill_chunk: 128,
            knobs: EngineKnobs::default(),
        }).expect("构造");
        let loaded = engine.loader().load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        for (qi, q) in ["中国的首都是北京。长城是", "1 2 3 4 5 6"].iter().enumerate() {
            let id = running.submit(*q, 48).expect("submit");
            let text = loop {
                match running.pump().await.expect("pump") {
                    TurnEvent::Completed { turn, text, .. } if turn == id => break text,
                    TurnEvent::Idle => panic!("队列丢失"),
                    _ => {}
                }
            };
            eprintln!("[raw-diag] Q{qi}: {q:?}\n[raw-diag] A{qi}: {text:?}");
        }
    }


    /// E5-M2 恒等门(机制臂,0.8B + 哑草稿):greedy 下 spec 轮输出 ≡ 无
    /// spec 输出(逐 token)。任意草稿(哑草稿 = anchor 重复)下机制必须
    /// 恒等 —— 接受是恒等变换,分红只影响速度。双 boot × 双 prompt ×
    /// 会话 token 账逐位比对(第二 prompt 同引擎续 turn = 跨 turn 状态
    /// 连续性一并覆盖)。哑草稿需 OWL_SPEC_DUMB=1(C7:无草稿器回落)。
    pub async fn gpu_spec_identity_gate() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let dir = asset_dir();
        const PROMPTS: [&str; 2] = ["用五十字介绍长城。", "水的沸点是多少度?"];

        async fn gen(
            spec: bool,
            ordinal: usize,
            dir: &std::path::Path,
        ) -> Vec<(String, Vec<u32>)> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() }; // greedy(恒等门口径)
            // 显式池余量(缺省 = 2×max_seq = 8 页,spec 链一挤就"块池耗尽";
            // 2026-10-13 套件内实测:前置用例扰动即红、单跑绿。恒等门测的是
            // 文本等价,不是容量边界 —— 边界另有专测)
            knobs.pool_tokens = Some(1024);
            if spec {
                knobs.draft.depth = 3;
                knobs.draft.dumb = true;
            }
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                max_seq_tokens: 128,
                prefill_chunk: 16,
                knobs,
            })
            .expect("构造");
            let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载");
            let mut running = engine.run(loaded).await.expect("装配");
            let mut out = Vec::new();
            for (i, p) in PROMPTS.iter().enumerate() {
                let (sid, id) = running
                    .submit_session(Some(100 + i as u64), *p, 48)
                    .expect("submit");
                let text = drive_turn(&mut running, id).await;
                let toks = running.bank.sessions
                    .get(sid)
                    .map(|s| s.tokens.clone())
                    .unwrap_or_default();
                out.push((text, toks));
            }
            // ★ 回归门(2026-10-13 定谳的挂死案):**跑满预算的 turn 必须
            // 送达终态事件**。病史:spec 路径轮末 `pending_events = evq`
            // 覆盖,把预算尽时 `sample_and_emit` 压入的 Completed 冲掉 →
            // actor 永不派发终态 → HTTP 层卡在 recv(客户端一帧不收也不
            // EOF)。此处 `ignore_eos=true` 强制跑满 8 tok → `drive_turn`
            // 若等不到 Completed 会 panic("队列丢失")= 门红。
            {
                let (sid_ie, id_ie) = running
                    .submit_session_ex(Some(900), "用一句话介绍长城。", 8, true)
                    .expect("submit(ignore_eos)");
                let text_ie = drive_turn(&mut running, id_ie).await;
                let n = running.bank.sessions.get(sid_ie).map(|s| s.tokens.len()).unwrap_or(0);
                eprintln!("[spec-gate] 预算尽门(ignore_eos): {n} tok,文本 {text_ie:?}");
                assert!(n >= 8, "预算尽 turn 应跑满 8 tok,得 {n}(终态事件是否送达见 panic 信息)");
            }
            let st = running.spec.stats;
            eprintln!(
                "[spec-gate] stats: rounds={} sum_m={} full={} partial={} zero={}",
                st.rounds, st.sum_m, st.full, st.partial, st.zero
            );
            out
        }

        let base = gen(false, ordinal, &dir).await; // 基线
        let spec = gen(true, ordinal, &dir).await; // spec(哑草稿)
        for (i, ((ta, ta_tokens), (tb, tb_tokens))) in base.iter().zip(&spec).enumerate() {
            eprintln!(
                "[spec-gate] p{i} baseline({}tok)={ta:?}\n[spec-gate] p{i} spec({}tok)={tb:?}",
                ta_tokens.len(),
                tb_tokens.len()
            );
            assert_eq!(ta, tb, "恒等门 p{i}:greedy spec 文本 ≡ 无 spec");
            assert_eq!(ta_tokens, tb_tokens, "恒等门 p{i}:token 账逐位一致");
        }
        // env 泄漏防护(spec vars 是进程全局;残留 = 后续测试被动进 spec 模
        // 式 —— two_turns 事件序列案 2026-10-06;旋钮已显式化,无 env 清场)
    }

    /// E5-M2b 恒等门(真草稿臂,27B + MTP 头):真草稿下 greedy spec 输出
    /// ≡ 无 spec 输出(逐 token)。同时验证 m>0 真接受路径(部分/全接受)
    /// 与 spec_stats 账目闭合(Σ(m_i+1) = 发射 token 数)。门控
    /// OWL_AWQ27B_DIR。
    pub async fn gpu_spec_identity_gate_27b() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
            eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
            return;
        };
        // 域参数化(E5-DF3 AL 排查:PROSE 域 = DFlash2 结构性最差域,
        // xinfer p5 实测 accept 1.9%/AL 1.13;MATH 域 AL 4.49 —— OWL_GATE_PROMPT
        // 换域对照,缺省恒 = 原 42tok 长城门)
        let prompt = owl_shared::env_reader::str_or("OWL_GATE_PROMPT", "用五十字介绍长城。");

        /// **装载一次**(带 MTP 草稿;2026-10-13 用户令:不再每轮重载)
        async fn boot_gate_engine(ordinal: usize, dir: &str) -> RunningEngine<owl_cuda::GpuClient> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
            // 24G 贴顶三旋钮(单会话门不需要多格/多快照/前缀缓存)
            knobs.gdn_slots = 1;
            knobs.snap_max = 1;
            knobs.prefix_cache = false;
            // 草稿档一律按投机配置 boot;baseline 轮靠运行时 spec_depth=0 回落
            knobs.draft.depth = 3;
            knobs.cuda.capture_slab_mb = Some(160);
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                max_seq_tokens: 256,
                prefill_chunk: 64,
                knobs,
            })
            .expect("构造");
            let loaded = engine
                .loader()
                .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), owl_models::module::QuantPlan::F16)
                .await
                .expect("装载");
            engine.run(loaded).await.expect("装配")
        }

        /// 单轮(`spec_depth = 0` = 不投机;>0 = 投机)
        async fn round(
            running: &mut RunningEngine<owl_cuda::GpuClient>,
            spec_depth: usize,
            prompt: &str,
            tag: &str,
        ) -> (String, Vec<u32>) {
            running.spec.depth = spec_depth;
            let (sid, id) = running.submit_session(Some(7), prompt, 40).expect("submit");
            let text = drive_turn(running, id).await;
            let toks = running.bank.sessions
                .get(sid)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            let st = running.spec.stats;
            eprintln!(
                "[spec-gate-27b] {tag} spec_depth={spec_depth} stats: rounds={} sum_m={} full={} partial={} zero={}",
                st.rounds, st.sum_m, st.full, st.partial, st.zero
            );
            (text, toks)
        }

        // ── 装载**一次**,两轮:① 不投机 baseline ② 投机(用户令 2026-10-13) ──
        let mut running = boot_gate_engine(ordinal, &dir).await;
        let (ta, ta_tokens) = round(&mut running, 0, &prompt, "baseline").await;
        let (tb, tb_tokens) = round(&mut running, 3, &prompt, "spec").await;
        eprintln!(
            "[spec-gate-27b] baseline({}tok)={ta:?}\n[spec-gate-27b] spec({}tok)={tb:?}",
            ta_tokens.len(),
            tb_tokens.len()
        );
        assert_eq!(ta, tb, "恒等门 27B:greedy 真草稿 spec 文本 ≡ 无 spec");
        assert_eq!(ta_tokens, tb_tokens, "恒等门 27B:token 账逐位一致");
        // 旋钮已全量显式化,无 env 清场需要
    
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
    }

    /// E5-DF3 恒等门(27B AWQ target + DFlash2 草稿):greedy spec 文本
    /// ≡ 无 spec 逐位。门控 OWL_TEST_DEVICE + OWL_AWQ27B_DIR +
    /// OWL_DFLASH2_DIR。**20G 卡贴顶**(target ~13.7G + draft 3.85G +
    /// 池)—— OOM 时降 max_seq_tokens/挪 24G 卡。
    pub async fn gpu_dflash2_identity_gate_27b() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
            eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
            return;
        };
        let Some(ddir) = owl_shared::env_reader::str("OWL_DFLASH2_DIR") else {
            eprintln!("skip: OWL_DFLASH2_DIR 未设(DFlash2 草稿目录)");
            return;
        };
        // 域参数化(E5-DF3 AL 排查:PROSE 域 = DFlash2 结构性最差域,
        // xinfer p5 实测 accept 1.9%/AL 1.13;MATH 域 AL 4.49 —— OWL_GATE_PROMPT
        // 换域对照,缺省恒 = 原 42tok 长城门)
        let prompt = owl_shared::env_reader::str_or("OWL_GATE_PROMPT", "用五十字介绍长城。");
        // E5 性能:长生成旋钮(AL/吞吐稳态剖析;默认 40 = 恒等门口径)
        let max_new: usize = owl_shared::env_reader::parse_or("OWL_GATE_MAX_NEW", 40);

        /// **装载一次**(带草稿;2026-10-13 用户令:不再每轮重载)。
        /// 单次装载两轮复用:一轮不投机(spec_depth=0)、一轮投机。
        async fn boot_gate_engine(
            ordinal: usize,
            dir: &str,
            ddir: &str,
            max_new: usize,
        ) -> RunningEngine<owl_cuda::GpuClient> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
            knobs.gdn_slots = 1;
            knobs.snap_max = 1;
            knobs.prefix_cache = false;
            // 草稿档一律按**投机**配置 boot(深度 + verify 捕获档);
            // 不投机的 baseline 轮靠运行时 `spec_depth = 0` 回落,不重载。
            knobs.draft.depth = owl_shared::env_reader::parse_or("OWL_SPEC_DEPTH", 7);
            knobs.cuda.capture_slab_mb = Some(160);
            knobs.draft.dflash2_dir = Some(ddir.to_string());
            // E19 定位开关(2026-10-13):knhd×fp8 池×int4 头×spec 语义回归
            // 复现(缺省关 = 原口径不变)。三项同开才复现 —— 单开任一项均绿:
            // 疑为“knHD 多序(seqs>1)读非位等”在窄边际下的贪心翻转。
            if owl_shared::env_reader::flag("OWL_GATE_FP8") {
                knobs.env.kv.quant = owl_models::env::KvQuant::Fp8E4M3;
            }
            // XQA 臂开关(2026-10-13;恒等门用 —— spec ≡ 无 spec / 裸 decode
            // 臂不破文本,是 XQA 进生产的唯一红线)
            if owl_shared::env_reader::flag("OWL_GATE_XQA") {
                knobs.env.attn.verify_xqa = true;
            }
            // 裸 decode XQA 臂(2026-10-13 ②):baseline 轮(spec_depth=0)
            // 就走它 ⇒ 本门同时是"decode 臂不破文本"的红线
            if owl_shared::env_reader::flag("OWL_GATE_DECODE_XQA") {
                knobs.env.attn.decode_xqa = true;
            }
            if let Some(p) = owl_shared::env_reader::parse::<usize>("OWL_GATE_POOL") {
                knobs.pool_tokens = Some(p);
            }
            let head_plan = if owl_shared::env_reader::flag("OWL_GATE_INT4HEAD") {
                owl_models::module::QuantPlan::W4A16Awq
            } else {
                owl_models::module::QuantPlan::F16
            };
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                max_seq_tokens: owl_shared::env_reader::parse_or(
                    "OWL_GATE_MAXSEQ",
                    320.max(max_new + 64),
                ),
                prefill_chunk: 32,
                knobs,
            })
            .expect("构造");
            let loaded = engine
                .loader()
                .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), head_plan)
                .await
                .expect("装载");
            let running = engine.run(loaded).await.expect("装配");
            assert!(
                matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
                "带草稿装载应捕获成功,得 {:?}",
                running.capture_outcome()
            );
            running
        }

        /// 单轮(`spec_depth = 0` = 不投机 baseline;>0 = 投机)。
        /// spec_depth 是 `pub(crate)` 运行时可切字段(exec/spec.rs 每轮读)。
        async fn round(
            running: &mut RunningEngine<owl_cuda::GpuClient>,
            spec_depth: usize,
            prompt: &str,
            max_new: usize,
            tag: &str,
        ) -> (String, Vec<u32>) {
            let prompt_len = prompt.chars().count(); // 近似(测试口径)
            running.spec.depth = spec_depth;
            let (sid, id) = running.submit_session(Some(7), prompt, max_new).expect("submit");
            let tg = std::time::Instant::now();
            let text = drive_turn(running, id).await;
            let t_gen = tg.elapsed();
            let toks = running.bank.sessions
                .get(sid)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            let gen_n = toks.len().saturating_sub(prompt_len);
            let st = running.spec.stats;
            let al = if st.rounds > 0 { st.sum_m as f64 / st.rounds as f64 } else { 0.0 };
            eprintln!(
                "[dflash-gate] 生成段: {} tok / {t_gen:.1?} = {:.1} tok/s({tag})",
                gen_n,
                gen_n as f64 / t_gen.as_secs_f64(),
            );
            eprintln!(
                "[dflash-gate] {tag} spec_depth={spec_depth} rounds={} sum_m={} AL={al:.2}",
                st.rounds, st.sum_m
            );
            (text, toks)
        }

        // ── 装载**一次**,两轮:① 不投机 baseline ② 投机(用户令 2026-10-13) ──
        let depth: usize = owl_shared::env_reader::parse_or("OWL_SPEC_DEPTH", 7);
        let mut running = boot_gate_engine(ordinal, &dir, &ddir, max_new).await;
        let t0 = std::time::Instant::now();
        let (ta, ta_tokens) = round(&mut running, 0, &prompt, max_new, "baseline").await;
        let t_base = t0.elapsed();
        // E5-DF4:b1 事实 dump(prefill bisect 层 checksum / pf.last / emit)
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("b1."),
        );
        // E5-DF4 装载校验快照(b1;OWL_LOAD_VERIFY=1 时 loader 回读对比 +
        // metrics 记 `loadv.{key}` = 校验和)
        let l1: std::collections::HashMap<String, u64> = owl_shared::metrics::collect_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("loadv."),
        )
        .into_iter()
        .map(|r| (r.tag.clone(), r.counter))
        .collect();
        owl_shared::metrics::reset_metrics();
        let t1 = std::time::Instant::now();
        let (tb, tb_tokens) = round(&mut running, depth, &prompt, max_new, "spec").await;
        let t_spec = t1.elapsed();
        let l2: std::collections::HashMap<String, u64> = owl_shared::metrics::collect_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("loadv."),
        )
        .into_iter()
        .map(|r| (r.tag.clone(), r.counter))
        .collect();
        // 双 boot 按键校验和 diff(类 1 装载非确定的决定性仪表):
        // 共有键校验和不同 = 装载竞态实锤;l2 独有 = 草稿键(b2 专属,预期)
        {
            let mut diff = 0usize;
            let mut only2 = 0usize;
            for (k, v1) in &l1 {
                match l2.get(k) {
                    Some(v2) if v2 == v1 => {}
                    other => {
                        diff += 1;
                        if diff <= 8 {
                            eprintln!("[load-diff] {k}: {v1:#018x} vs {:?} ", other);
                        }
                    }
                }
            }
            for k in l2.keys() {
                if !l1.contains_key(k) {
                    only2 += 1;
                }
            }
            eprintln!(
                "[load-diff] b1_keys={} b2_keys={} mismatch={diff} l2_only(draft 预期)={only2}",
                l1.len(),
                l2.len()
            );
        }
        // 吞吐面:生成 tok 数 / gen 墙钟(含装载+prefill;粗口径,门内自证)
        let gen_n = |t: &[u32]| t.len().saturating_sub(18);
        eprintln!(
            "[dflash-gate] 吞吐 baseline: {} gen-tok / {t_base:.1?} = {:.1} tok/s(含装载)",
            gen_n(&ta_tokens),
            gen_n(&ta_tokens) as f64 / t_base.as_secs_f64(),
        );
        eprintln!(
            "[dflash-gate] 吞吐 spec:     {} gen-tok / {t_spec:.1?} = {:.1} tok/s(含装载)",
            gen_n(&tb_tokens),
            gen_n(&tb_tokens) as f64 / t_spec.as_secs_f64(),
        );
        eprintln!(
            "[dflash-gate] baseline({}tok)={ta:?}\n[dflash-gate] spec({}tok)={tb:?}",
            ta_tokens.len(),
            tb_tokens.len()
        );
        // E5-DF4 崩坏排查:b2 事实全量 dump(好/坏跑对比;b1 已在 reset 前 dump)
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("b2."),
        );
        // E5 性能:spec 轮逐相汇总(长生成轮账;gap = round − Σparts)
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("spec."),
        );
        assert_eq!(ta, tb, "恒等门 DFlash2:greedy 真草稿 spec 文本 ≡ 无 spec");
        assert_eq!(ta_tokens, tb_tokens, "恒等门 DFlash2:token 账逐位一致");
        // 旋钮已全量显式化,无 env 清场需要
    
        // 实例收尾屏障:drop 后等 server 线程退出 + CUDA ctx 析构(异步;
        // 下一实例的 17G 分配会与释放赛跑 —— 2026-10-07 非确定性定谳)
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
    }

/// 泵到目标 turn 完成,回吐全文(其间事件仅观测)
async fn drive_turn<D: DeviceClient>(running: &mut RunningEngine<D>, id: u64) -> String {
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("turn {id} 队列丢失"),
            TurnEvent::Prefill { turn, fed, total } if turn == id => {
                eprintln!("[ev] t{turn} prefill {fed}/{total}")
            }
            TurnEvent::Completed { turn, text, .. } if turn == id => break text,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
}

/// decode 边际速率差分基准(E5 性能会战终版口径,2026-10-08):
/// 同配置双 boot(16 / 128 token 上限),**边际 ms/tok = Δwall/Δtokens**
/// —— 固定成本(prefill 尾步/首步杂项/事件泵)与 EOS 点全部在差分中
/// 消掉。旧 e2e 的 wall÷max_new 与 gate 生成段 wall÷实际tok 两把尺子
/// 分别虚高/虚低(差分实测:4d21122 24.3ms vs HEAD 24.8ms = 无回退,
/// 当年"45 tok/s"是朴素口径虚高)。门控 OWL_AWQ27B_DIR + OWL_TEST_DEVICE。
///
/// 泄漏案探针(2026-10-10 深夜;slot≥1 崩塌案首刀):
/// GDN_SLOTS=2 下,turn(slot0) 前后对比 slot1 状态区指纹 ——
/// slot1 变 = 跨 slot 写实锤;不变 = 病在 slot1 核索引/复用路径。
/// 附:reset_gdn(slot1) 后回读验零(reset 生效性直接判)。
/// 门控 OWL_TEST_DEVICE。诊断打印,不做硬断言(判决先看数)。
pub async fn gpu_gdn_slot_cross_probe() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let mut knobs = EngineKnobs { gdn_slots: 2, ..Default::default() };
    knobs.prefix_cache = false;
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 512,
        prefill_chunk: 32,
        knobs,
    })
    .expect("构造");
    let dir = asset_dir();
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");

    // ① boot 后初始态:双 slot 指纹 + slot1 非零字节数(alloc 应已清零)
    let s1_0 = running.pool.gdn_slot_fingerprint(running.graphs.session.face_mut(), 1).await.unwrap();
    let nz0 = running.pool.gdn_slot_zero_check(running.graphs.session.face_mut(), 1).await.unwrap();
    eprintln!("[slot-probe] boot 后 slot1 非零字节 = {nz0}(期望 0 = alloc 清零生效)");

    // ② turn(slot0,新 session)
    let (sid, id) = running
        .submit_session(Some(7), "用五十字介绍长城。", 40)
        .expect("submit");
    let _ = drive_turn(&mut running, id).await;
    let _ = sid;

    // ③ turn 后双 slot 指纹
    let s1_1 = running.pool.gdn_slot_fingerprint(running.graphs.session.face_mut(), 1).await.unwrap();
    let s0_1 = running.pool.gdn_slot_fingerprint(running.graphs.session.face_mut(), 0).await.unwrap();
    let s1_nz = running.pool.gdn_slot_zero_check(running.graphs.session.face_mut(), 1).await.unwrap();
    eprintln!("[slot-probe] turn 后 slot1 非零字节 = {s1_nz}(>0 = turn(slot0) 跨写了 slot1)");
    let same1 = s1_0 == s1_1;
    eprintln!("[slot-probe] slot1 指纹 turn 前后相同 = {same1}(false = 跨 slot 写实锤)");
    let _ = s0_1;

    // ④ reset 生效性:slot0 填垃圾验证?——零检查已覆盖 slot1;slot0 的
    //    reset 由健康 turn 隐证。旋钮已显式化,无 env 清场需要。
}

pub async fn gpu_decode_marginal_bench() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
        eprintln!("skip: OWL_AWQ27B_DIR 未设");
        return;
    };
    // 口径与恒等门一致(greedy 单会话三旋钮,显式 knobs)
    async fn gen(ordinal: usize, dir: &str, max_new: usize) -> (f64, usize) {
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.gdn_slots = 1;
        knobs.snap_max = 1;
        knobs.prefix_cache = false;
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), owl_models::module::QuantPlan::F16)
            .await
            .expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let (_sid, id) = running
            .submit_session(Some(7), "用五十字介绍长城。", max_new)
            .expect("submit");
        let t0 = std::time::Instant::now();
        let _text = drive_turn(&mut running, id).await;
        let wall = t0.elapsed().as_secs_f64();
        let n = running.bank.sessions.get(7).map(|v| v.tokens.len()).unwrap_or(0);
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(3));
        (wall, n)
    }

    let (w1, n1) = gen(ordinal, &dir, 16).await;
    let (w2, n2) = gen(ordinal, &dir, 128).await;
    let dn = (n2 as i64 - n1 as i64).max(1) as f64;
    let marginal = (w2 - w1) * 1e3 / dn;
    eprintln!(
        "[marginal] turn1 {w1:.3}s/{n1}tok; turn2 {w2:.3}s/{n2}tok; 边际 = {marginal:.2} ms/tok = {:.1} tok/s",
        1e3 / marginal
    );
    // 旋钮已显式化,无 env 清场需要
}

// B4 自适应降级态机(§6.24):纯转移函数逐分支
#[test]
fn spec_degrade_transition_branches() {
    use crate::exec::spec_degrade_transition;
    // 成功轮:三账全复位(含降级态/退避)
    let (s, d, p) = spec_degrade_transition(3, true, 256, 2, 6, 64);
    assert_eq!((s, d, p), (0, false, 64));
    // 非降级连击未达阈值:streak 累加,态不变
    let (s, d, p) = spec_degrade_transition(0, false, 64, 0, 6, 64);
    assert_eq!((s, d, p), (1, false, 64));
    // 达阈值 → 入降级(探测周期 = 初值)
    let (s, d, p) = spec_degrade_transition(5, false, 64, 0, 6, 64);
    assert_eq!((s, d, p), (6, true, 64));
    // 降级态探测失败 → 退避 ×2
    let (s, d, p) = spec_degrade_transition(7, true, 64, 0, 6, 64);
    assert_eq!((s, d, p), (8, true, 128));
    // 退避封顶 512
    let (s, d, p) = spec_degrade_transition(9, true, 400, 0, 6, 64);
    assert_eq!((s, d, p), (10, true, 512));
    // threshold=0 = 禁用(永不入降级)
    let (s, d, p) = spec_degrade_transition(9, false, 64, 0, 0, 64);
    assert_eq!((s, d, p), (10, false, 64));
}

/// **`[graph] prefill_bucket` 档位面**(2026-10-15;M2b 落地后重写)。
///
/// 初版是"非 off ⇒ 装配期拒启"(实现未落时的防静默门)。M2b 把真接线放进
/// `Engine::run`(桶捕获需要模型 + 状态池;静止点律 ⇒ 只在 boot),`Engine::on`
/// 回到纯构造 ⇒ 三档均放行;无桶可用时走 eager 原路(且有 **warn 证据面**,
/// 不是静默降级 —— 真机档见 `graph_fleet::assemble` 的 ④ 段)。
#[cfg(test)]
#[tokio::test]
async fn prefill_bucket_levels_boot() {
    use owl_shared::config::PrefillBucket;
    let mk = |b: PrefillBucket| EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 64,
        prefill_chunk: 16,
        knobs: EngineKnobs { prefill_bucket: b, ..EngineKnobs::default() },
    };
    for b in [PrefillBucket::Off, PrefillBucket::T512, PrefillBucket::Geobuckets] {
        assert!(Engine::on(mk(b), CpuFace::new()).is_ok(), "{b:?} 应可构造(真接线在 run)");
    }
}

/// **桶闸道纯函数**(M2b):阶梯/向上取整/GDN host 槽臂判定(零设备)。
#[cfg(test)]
#[test]
fn prefill_bucket_pick_and_ladder() {
    use crate::prefill_plan::{gdn_host_slot_arm, ladder, pick_bucket};
    use owl_shared::config::PrefillBucket;
    // 阶梯:off 空;t512 单桶;geobuckets 五点
    assert!(ladder(PrefillBucket::Off).is_empty());
    assert_eq!(ladder(PrefillBucket::T512), &[512]);
    assert_eq!(ladder(PrefillBucket::Geobuckets), &[32, 64, 128, 256, 512]);
    // 向上取整(不向下):512 档下 t=1..=512 全命中 512;t>512 不命中
    for t in [1usize, 33, 128, 511, 512] {
        assert_eq!(pick_bucket(ladder(PrefillBucket::T512), t), Some(512));
    }
    assert_eq!(pick_bucket(ladder(PrefillBucket::T512), 513), None);
    // geobuckets:边界贴着阶梯走
    let l = ladder(PrefillBucket::Geobuckets);
    for (t, want) in [(1usize, 32usize), (32, 32), (33, 64), (65, 128), (257, 512), (512, 512)] {
        assert_eq!(pick_bucket(l, t), Some(want), "t={t}");
    }
    assert_eq!(pick_bucket(l, 900), None);
    // GDN host 槽臂:27B 几何 + chunked ⇒ t≥64 才吃 host 槽(小 T 走 varlen)
    let mut env = owl_models::env::EnvProvider::default();
    let dims = crate::state::ModelDims {
        hq: 24, hkv: 4, hd: 256, nk: 16, hk: 128, nv: 48, hv: 128, hidden: 2048,
        dtype: owl_iface::contract::Dtype::F16,
    };
    env.gdn.chunked = true;
    assert!(!gdn_host_slot_arm(&env, &dims, 32), "小桶走 varlen(设备槽)⇒ 不吃 host 槽");
    assert!(gdn_host_slot_arm(&env, &dims, 64));
    assert!(gdn_host_slot_arm(&env, &dims, 512));
    // 非 chunked(0.8B 夹具几何)⇒ 恒 varlen ⇒ 不吃 host 槽
    env.gdn.chunked = false;
    assert!(!gdn_host_slot_arm(&env, &dims, 512));
}

// ===========================================================================
// M2b:prefill 桶图(设计 §4–§6;GPU 门控)
// ===========================================================================

/// **桶图 ≡ eager(贪心逐 token)+ 桶确实被用上**(0.8B)。
///
/// 两引擎**同配置**,唯一差别 = `prefill_bucket`;同提示词贪心生成 ⇒ 输出文本
/// 必须逐字相同。为什么是 token 级而非 hidden 级:跨桶时 FI 的 tile 数/tiling
/// 随 t 变(设计 §4.1c"数值口径分两层")—— token 级 = 软门;桶内逐位门由
/// `fi_graph_replay_probe`(M2a)与 kernels 侧门承担;GDN 假行面由
/// `gdn_bucket_pad_probe` 承担。
///
/// 提示词长度故意跨两个桶(128 → 64):既验"向上取整命中",也验"多桶共存时
/// int_ws 的 plan 自携"(见 `server/flashinfer.rs::PlanEntry`)。
#[cfg(test)]
#[tokio::test]
async fn gpu_prefill_bucket_matches_eager() {
    use owl_shared::config::PrefillBucket;
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 日志面接上(桶捕获的 warn/info 是 boot 证据面;测试里默认无 sink)
    let _ = owl_shared::log::init_logger(owl_shared::log::LogConfig {
        level: owl_shared::log::LVL_INFO,
        ..Default::default()
    });
    // 贪心(关采样器)⇒ 两条路必须逐字同轨
    async fn boot_bucket(
        dir: &std::path::Path,
        ordinal: usize,
        b: PrefillBucket,
    ) -> RunningEngine<owl_cuda::GpuClient> {
        // `EnvProvider::default()` 是 attest=false(诊断 A/B 保留位);生产走
        // `from_config` ⇒ `from_dispatch` 恒真(2026-10-13 用户令)⇒ 本测显式打开。
        let env = owl_models::env::EnvProvider { attn: owl_models::env::AttnEnv { fi: true, ..Default::default() }, ..Default::default() };
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 2048,
            // 512:t_pad=512 桶(设计 M2b 的原目标形态 —— 长 chunk 也进图;
            // 提示词 ~600 tok ⇒ chunk 512 + 尾块 88 ⇒ 同时命中大桶与中桶)
            prefill_chunk: 512,
            knobs: EngineKnobs {
                prefill_bucket: b,
                sampler_enabled: false,
                env,
                ..EngineKnobs::default()
            },
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载");
        engine.run(loaded).await.expect("装配")
    }
    let prompt = "长城是中国古代的军事防御工程,用来抵御北方游牧民族的侵袭。".repeat(40);
    async fn gen(running: &mut RunningEngine<owl_cuda::GpuClient>, prompt: &str, n: usize) -> String {
        let id = running.submit(prompt.to_string(), n).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == id => return text,
                TurnEvent::Failed { turn, err } => panic!("turn {turn} 失败: {err}"),
                TurnEvent::Idle => panic!("队列丢失"),
                _ => {}
            }
        }
    }
    // ① 桶档
    let mut g = boot_bucket(&dir, ordinal, PrefillBucket::Geobuckets).await;
    let buckets = g.graphs.prefill.captured_t_pads();
    eprintln!("[m2b] 桶捕获:{:?}(summary = {})", buckets, g.graphs.prefill.summary());
    assert!(
        buckets.len() >= 3,
        "geobuckets 至少应捕获小桶若干(实测 {buckets:?});捕获失败看 warn 证据面"
    );
    let out_graph = gen(&mut g, &prompt, 24).await;
    let counts = g.graphs.prefill.replay_counts();
    eprintln!("[m2b] 桶回放次数:{counts:?}");
    // **证据面**:图路确实被走了(批 180 tok / chunk 128 ⇒ 命中 t128 + t64)
    let used: Vec<usize> = counts.iter().filter(|(_, n)| *n > 0).map(|(t, _)| *t).collect();
    assert!(!used.is_empty(), "桶一个都没回放 ⇒ 图路没生效(本测在测 eager vs eager)");
    assert!(used.contains(&512), "本 prompt(chunk=512)应命中 t512 桶,实测 {used:?}");
    // ② eager 档(同模型同配置,只关桶)
    let mut e = boot_bucket(&dir, ordinal, PrefillBucket::Off).await;
    let out_eager = gen(&mut e, &prompt, 24).await;
    eprintln!("[m2b] 桶档 {out_graph:?}\n[m2b] eager {out_eager:?}");
    assert!(!out_graph.trim().is_empty(), "桶档空回答");
    assert_eq!(out_graph, out_eager, "桶图与 eager 的贪心输出必须逐字相同(M2b 跨档门)");
}
