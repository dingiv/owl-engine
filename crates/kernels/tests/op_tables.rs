//! 算子准入表 ⇔ `.cu` 签名**机器互证**(2026-10-13 P3 的核心门)。
//!
//! 设计稿 §八 P3 的完成判据:「表 ⇔ 各自 `.cu` 逐参一致」。这里把它做成
//! 不可绕过的门:任何一张 `impl KernelSpec` 的参数表,只要它的 `SOURCE`
//! 非空(原生 NVRTC 核),就必须与**同一份源码里那个入口**的形参序列逐位
//! 一致 —— 槽序串是从 `PARAMS` **派生**的,所以本门同时锁住:
//!
//! 1. 表的槽类型/顺序 == 内核 ABI(错位 = 红,不靠人眼);
//! 2. 派生逻辑本身正确(若派生写错,这里也会红);
//! 3. 输出槽 `O` 确实落在 `.cu` 的指针位上。
//!
//! 解析器 = [`owl_kernels::cu_sig::parse_cu_sig`](**唯一**解析器;登记表
//! 的 C2 门与本门共用它 —— 两道门各写一份解析 = 各说各话,见坑册)。

use owl_kernels::ops;
use owl_kernels::spec::{meta_of, KernelSpec};

/// 表 vs `.cu`:逐参一致(原生核)
#[test]
fn op_tables_match_cu_signatures() {
    let mut checked = 0usize;
    for m in ops::all() {
        if m.src.is_empty() {
            continue; // foreign(FFI):无 .cu 源,另由 face 常量对拍(见单测)
        }
        // E 轮(2026-10-13):`SRC` 已从"源文本"改为**资产 id** —— 文本按需
        // 从磁盘取(仅测试面;运行期不再需要任何 `.cu` 文本)。
        let text = owl_kernels::cu_sig::cu_text(m.src);
        assert!(
            text.contains("extern \"C\" __global__"),
            "{}: 资产 {} 不是核源码(拿错文件?)",
            m.entry, m.src
        );
        let actual = owl_kernels::cu_sig::parse_cu_sig(&text, m.entry)
            .unwrap_or_else(|| panic!("{}: 源码里找不到同名入口", m.entry));
        // 线槽序里 `O`(输出角色)在 `.cu` 里就是一个指针位 ⇒ 归一 T 后比对
        let normalized = m.args_str().replace('O', "T");
        assert_eq!(
            normalized, actual,
            "{}: 参数表派生的槽序与 .cu 实签名不符\n  表: {}\n  cu: {}",
            m.entry,
            m.args_str(),
            actual
        );
        // 输出槽必须是 `.cu` 的指针位(不是标量位)
        if let Some(pos) = m.out_pos() {
            let cu_toks: Vec<&str> = actual.split(',').collect();
            assert_eq!(cu_toks[pos], "T", "{}: O 槽必须是 .cu 指针位", m.entry);
        }
        checked += 1;
    }
    assert!(checked >= 30, "建表数异常(仅 {checked} 条参与互证)");
    eprintln!("[op-tables] {checked} 张表与 .cu 签名逐参一致 ✓(foreign 表另计)");
}

/// **R1 判据(F 轮换源:构建期编出的符号 ⊆ 声明集)**。
///
/// 旧判据 = "登记表差额"(110 − 已建表)。F 轮登记表退役后,唯一真源是声明面,
/// 判据换成**符号集包含关系** —— 更贴 R1 本意:凡编出来的核都必须有
/// `impl KernelSpec`;**新增 `.cu` 忘了建表**会被当场抓住(旧判据抓不到:
/// 那条路要求人先去登记表登记一条,漏登记就漏检)。
#[test]
fn every_compiled_symbol_is_declared() {
    let u = owl_kernels::ops::undeclared_symbols();
    for (id, sym) in &u {
        eprintln!("[R1]   {id} :: {sym}");
    }
    assert!(
        u.is_empty(),
        "编出但未声明的核符号 {} 条 —— 新核必须带 impl KernelSpec(R1)",
        u.len()
    );
    eprintln!(
        "[R1] 编出符号集 ⊆ 声明集 ✓(声明表 {} 张)",
        owl_kernels::ops::declared_count()
    );
}

#[test]
fn op_table_entries_are_unique() {
    let mut seen = std::collections::HashSet::new();
    for m in ops::all() {
        assert!(seen.insert(m.entry), "{} 重复建表", m.entry);
    }
}

/// foreign 表(无 .cu 源)也必须声明槽序 —— 它是 server parse 与
/// `lower_kernel` 的**唯一权威**,空表 = 静默失去契约
#[test]
fn foreign_tables_declare_slots_without_source() {
    let mut n = 0usize;
    for m in ops::all() {
        if !m.src.is_empty() {
            continue;
        }
        assert!(!m.params.is_empty(), "{}: foreign 表无参数声明", m.entry);
        assert!(m.args_str().contains('O'), "{}: foreign 表未标输出槽", m.entry);
        n += 1;
    }
    assert!(n >= 3, "foreign 建表数异常(仅 {n})");
}

/// 参数表自身一致:张量槽必须带 dtype 语义,标量槽不许声明对齐
#[test]
fn param_table_is_self_consistent() {
    for m in ops::all() {
        for p in m.params {
            match p.ty {
                owl_kernels::spec::ParamType::Tensor { .. } => {}
                owl_kernels::spec::ParamType::Scalar(_) => {
                    assert!(p.align.is_none(), "{}: 标量槽 {} 不应有对齐要求", m.entry, p.name);
                }
            }
            assert!(!p.name.is_empty(), "{}: 参数名缺失", m.entry);
        }
    }
}

/// 派生形态与 trait 擦除一致(同一份数据,两条取法必须相等)
#[test]
fn meta_of_matches_direct_trait_const() {
    struct Probe;
    impl KernelSpec for Probe {
        const ENTRY: &'static str = "probe_op";
        const SRC: &'static str = "";
        const PARAMS: &'static [owl_kernels::spec::Param] = &[
            owl_kernels::spec::Param::t_in("x", owl_kernels::contract::Dtype::F16, 1),
            owl_kernels::spec::Param::sz("n"),
            owl_kernels::spec::Param::t_out("y", owl_kernels::contract::Dtype::F16, 1),
        ];
    }
    let m = meta_of::<Probe>();
    assert_eq!(m.entry, <Probe as KernelSpec>::ENTRY);
    assert_eq!(m.params.len(), <Probe as KernelSpec>::PARAMS.len());
    assert_eq!(m.args_str(), "T,sz,O");
}

/// **生产臂必须携带准入声明**(2026-10-13 P3 的"启用"判据):
/// driver 的生产拾取函数产出的 pick,其 `meta()` 必须非空 —— 否则
/// "已建表"只是摆设(表在册、发射端没接上)。
///
/// 覆盖 `docs/arch/kernel-准入门槛-施工设计.md` §14.4 生产集的**接入点**
/// (每个生产算子的 driver 臂各一条;新增生产算子时同步加行)。
#[test]
fn production_picks_carry_admission_meta() {
    use owl_kernels::driver::{self, DType, Hw, OpEnv};

    let env = OpEnv { hw: Hw { arch: owl_kernels::Arch::Sm86 }, page: 32, head_dim: 256, grp: 6 };
    // NC 注意力:shapes[0][0] = T,scalars = [hq, ?, hd]
    let nc_shapes = vec![vec![8usize, 256]];
    let nc_scalars = vec![24i64, 4, 256];
    // marlin AWQ:6 个父块 ⇒ AWQ 臂;5 个 ⇒ 对称臂
    let m6 = vec![vec![8usize, 2048]; 6];
    let m5 = vec![vec![8usize, 2048]; 5];
    let rms_shapes = vec![vec![8usize, 256], vec![256usize]];

    let picks: Vec<(&str, _)> = vec![
        // ---- attention / 池 ----
        ("xqa verify", driver::attn::xqa_verify()),
        ("xqa decode", driver::attn::xqa_decode()),
        ("K0 fp8 knhd", driver::attn::k0_write_fp8_knhd(16)),
        ("K0 fp8 classic", driver::attn::k0_write_fp8(16)),
        ("K0 bf16", driver::attn::k0_write(DType::BF16, 16)),
        (
            "qknorm insert fp8 knhd",
            driver::attn::qkv_norm_rope_insert_fp8kv_knhd(16, 24, 4, 256, 128),
        ),
        (
            "qknorm insert fp8 classic",
            driver::attn::qkv_norm_rope_insert_fp8kv(16, 24, 4, 256, 128),
        ),
        (
            "paged v2 fp8 knhd bs32",
            driver::attn::paged_decode_v2_fp8_knhd(&env, DType::F16, 256, 24, 4, 32, 4),
        ),
        (
            "paged v2 reduce",
            driver::attn::paged_v2_reduce(DType::F16, 256, 24, 4, 1),
        ),
        (
            "chunked prefill fp8 knhd",
            driver::attn::paged_prefill_fp8_knhd(&env, DType::F16, 256, 4, 24, 2540),
        ),
        ("fi prefill fp8", driver::attn::fi_prefill(true)),
        ("norm rope f16", driver::attn::norm_rope(DType::F16, 8, 24, 256)),
        ("norm rope bf16", driver::attn::norm_rope(DType::BF16, 8, 24, 256)),
        (
            "naive nc v2",
            driver::attn::naive_nc(DType::BF16, false, &nc_shapes, &nc_scalars),
        ),
        (
            "naive nc fp8kv",
            driver::attn::naive_nc(DType::BF16, true, &nc_shapes, &nc_scalars),
        ),
        // ---- GDN ----
        ("gdn chunked", driver::gdn::chunked_delta()),
        ("gdn gating_g", driver::gdn::gating_g(DType::F16)),
        ("gdn l2norm", driver::gdn::l2norm(DType::F16, 8)),
        ("gdn norm_act", driver::gdn::norm_act(DType::F16, 8, 128, 128)),
        ("gdn conv_fwd", driver::gdn::conv_fwd(DType::F16, 6144)),
        (
            "gdn conv_upd_dual",
            driver::gdn::conv_upd_dual(DType::F16),
        ),
        (
            "gdn decode_step_v2",
            driver::gdn::decode_step_v2(DType::F16, 1, 16, 128, 128),
        ),
        (
            "gdn recurrence",
            driver::gdn::recurrence_varlen_gqa(DType::F16, 128, 16, 128),
        ),
        // ---- elems / norm / mlp / dflash / load ----
        ("narrow", driver::elems::narrow(DType::F16)),
        (
            "gemv_dual",
            driver::elems::gemv_dual(DType::F16, 8, 8, 256, 8),
        ),
        ("add f16", driver::ops::add(DType::F16)),
        ("add bf16", driver::ops::add(DType::BF16)),
        ("sigmoid", driver::ops::sigmoid(DType::F16)),
        ("gate_mul", driver::attn::gate_mul(DType::F16)),
        ("cast f16->bf16", driver::elems::cast_dir(false)),
        ("cast bf16->f16", driver::elems::cast_dir(true)),
        ("concat f16", driver::elems::concat(DType::F16)),
        ("concat f32", driver::elems::concat(DType::F32)),
        ("embed", driver::elems::embed(DType::F16, 8)),
        ("argmax", driver::ops::argmax()),
        ("topk16", driver::ops::topk16(8)),
        ("topk16 split", driver::ops::topk16_split(8, 16)),
        ("topk16 merge", driver::ops::topk16_merge(8)),
        ("rmsnorm", driver::ops::rmsnorm(DType::F16, &rms_shapes)),
        (
            "fused_add_rmsnorm",
            driver::ln::fused_add_rmsnorm(DType::F16, 8, 256),
        ),
        ("silu_and_mul", driver::mlp::silu_and_mul(DType::F16, 8)),
        ("dflash conv", driver::dflash::conv(DType::BF16)),
        ("dflash select", driver::dflash::select(true)),
        ("ct_repack", driver::load::ct_repack(8, 8)),
        // ---- foreign ----
        ("marlin awq", driver::marlin::w4a16(&m6, DType::F16)),
        ("marlin f16", driver::marlin::w4a16(&m5, DType::F16)),
        ("marlin bf16", driver::marlin::w4a16(&m5, DType::BF16)),
    ];
    let n = picks.len();
    for (label, p) in picks {
        let m = p.meta().unwrap_or_else(|| panic!("{label}: 拾取名 {} 无准入声明", p.name));
        assert_eq!(m.entry, p.name, "{label}: 拾取名与元数据入口名不一致");
    }
    eprintln!("[op-tables] {n} 个生产臂均携带准入声明 ✓");
}

/// **名字级互证**(2026-10-15):声明表形参名 == `.cu` 形参名(逐位、含序)。
///
/// 为什么 token 级不够:token 串只编码 T/sz/i32/f32 的**形状**,看不见
/// "同形状、不同语义槽位"的漂移 —— 实测活体
/// `owl_qknorm_rope_kv_insert_f16{,_knhd}`:声明是旧签名、`.cu` 是新签名,
/// token 串逐字相同 ⇒ 旧门全绿,而调用点按新序喂 = 语义全错位。
#[test]
fn op_tables_match_cu_param_names() {
    let mut checked = 0usize;
    for m in ops::all() {
        if m.src.is_empty() {
            continue; // foreign(FFI):无 .cu 源(名字面由 face 常量/锁表承担)
        }
        let text = owl_kernels::cu_sig::cu_text(m.src);
        let Some(cu) = owl_kernels::cu_sig::parse_cu_param_names(&text, m.entry) else {
            panic!("{}: 源码里找不到同名入口", m.entry);
        };
        let mine = m.param_names();
        assert_eq!(
            mine,
            cu.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "{}: 声明表形参名与 .cu 不符(语义槽位错位)\n  表: {:?}\n  cu: {:?}",
            m.entry,
            mine,
            cu
        );
        checked += 1;
    }
    assert!(checked >= 30, "建表数异常(仅 {checked} 条参与名字互证)");
    eprintln!("[op-tables] {checked} 张表形参名与 .cu 逐位一致 ✓");
}

/// **非族算子:`out_pos()` 必须 = 最后一个张量槽**(现役回传 ABI;F1a)。
///
/// 判据来源:`server/launch.rs` 的回传口径 —— "**最后一个 Block/BlockSlice = 输出**"
/// (`issue_launch` 按此回传句柄,发射账 `out_id` 同口径)。家族算子**豁免**:
/// 它们 out 可非末位(marlin AWQ 的 out 在槽 2),其 runtime 按 `parse` 的**名**
/// 取 out,不依赖"末位"约定。
#[test]
fn non_family_out_slot_is_last_tensor() {
    for m in ops::all() {
        if m.is_family() {
            continue;
        }
        let pos = m.out_pos().unwrap_or_else(|| panic!("{}: 非族算子无 Out 槽", m.entry));
        let last = m
            .params
            .iter()
            .rposition(|p| p.kind() == owl_kernels::spec::ParamKind::Tensor)
            .unwrap_or_else(|| panic!("{}: 无张量槽", m.entry));
        assert_eq!(
            pos, last,
            "{}: out_pos={} 但最后一个张量槽={} —— 回传句柄按\"末位 Block\"取,声明必须末位",
            m.entry, pos, last
        );
    }
}

/// **池写集 ↔ 槽角色互证**(F1b;审查"声明自洽"缺门)。
///
/// 两条律(可机器判定,且不误伤激活域):
/// 1. 声明 `write`/`rw` 池 ⇒ **必须有承载写入的槽**(`InOut` 或 `Out`);
/// 2. 声明 `rw` 池 ⇒ **必须有 `InOut` 槽**(原地读写语义)。
///
/// **反向不查**:`InOut` 槽**不要求**池声明 —— 激活域的原地槽合法
/// (`fused_add_rmsnorm` 的 `residual`、gdn 的 conv 态),它们不是"池"。
#[test]
fn pool_writers_have_a_carrying_slot() {
    use owl_kernels::spec::{AccessMode, ParamRole};
    for m in ops::all() {
        let mut has_inout = false;
        let mut has_out = false;
        for p in m.params {
            match p.role {
                ParamRole::InOut => has_inout = true,
                ParamRole::Out => has_out = true,
                // 其余角色(Scratch/Const 等)不承载池写
                _ => {}
            }
        }
        for pa in m.pool {
            match pa.mode {
                AccessMode::Write => assert!(
                    has_inout || has_out,
                    "{}: 声明写池 {} 但无 InOut/Out 槽承载(写入落点不可见)",
                    m.entry,
                    pa.pool
                ),
                AccessMode::ReadWrite => assert!(
                    has_inout,
                    "{}: 声明读写池 {} 但无 InOut 槽(原地语义缺槽)",
                    m.entry,
                    pa.pool
                ),
                AccessMode::Read => {}
            }
        }
    }
}

