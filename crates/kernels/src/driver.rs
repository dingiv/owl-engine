//! driver —— 硬件感知算子拾取(REQ-HW-01「算子层 arch 分发表」落地点;
//! 2026-10-01 立项,用户设计:上层描述「要什么算子」,driver 决定
//! 「用哪个、怎么发射」,解释器拿到 pick 后要求 server 发射)。
//!
//! # 职责三件
//!
//! ① **名字单源**:核名的 dtype 后缀、变体选择(bs16/bs32 之类)只住
//!    本模块 —— 上层(models)不得拼写核名字符串、不得 `Box::leak` 拼名
//!    (gname/leak_name 由此绝迹);
//! ② **拾取**:按 (硬件档, dtype, 形状) 产出 [`KernelPick`]——名 +
//!    grid/block/smem;上层描述语义,参数推导住这里;
//! ③ **参数推导**:契约公式进代码(paged v1 的 smem 契约、delta_dec 的
//!    kd 项……)——硬编码参数事故面(smem 手抄越界、页配对错配,三次
//!    立案)在此结构性封死:调用点传形状,pick 出参数,错配 = 单测红,
//!    不再有「难以察觉」。
//!
//! # 渐进纪律(不一口吃成胖子)
//!
//! - 登记表(models `kernel::REGISTRY`,名字→源+签名,执行面验证)暂留
//!   models;models 侧耦合测试保证本模块产出的每个名字 ∈ 登记表 ——
//!   漂移在 CI 红灯,不在 GPU 上胡言;
//! - 硬件感知分阶段:arch 现由引擎 boot 装配占位 Sm86,CC 查询接通后
//!   升级;Sm89(40 系预留)只加枚举臂,调用点零改动;
//! - 迁移按族推进(gdn → attention → rope/embedding),每族对拍锚 +
//!   `OWL_GPU_PROF` 分账验收(算子集合不变,只换拾取方式)。
//!
//! # 被动律(charter 级,2026-10-01 用户立规)
//!
//! **kernels 无 I/O、无探测、无环境自取** —— 硬件感知发生在引擎侧
//! (boot 时一次),经 [`OpEnv`] 必传参进入;driver 收到「三零系」就按
//! 三零系选。`Hw::detect()` 之类探测入口永不进入本 crate:上层必须
//! 叫得动、决定得了。环境类型随 iface `DeviceClient::op_env` 下发。
//!
//! # 依赖立场
//!
//! 本模块零依赖(models 引 `sources` 不拖链接的同律):`DType` 是本模块
//! 自有最小枚举,与 owl-iface `contract::Dtype` 的转换在 models 侧完成
//! —— 契约类型不过 kernels,零依赖纪律不破。
//!
//! `Arch` 复用 crate 根的 REQ-HW-01 枚举(单一分发表键,不另立)。

// ============================================================================
// §1 硬件档
// ============================================================================

/// 硬件画像(拾取的全部依据;随感知手段升级扩充字段)。
/// `Arch` 复用 crate 根枚举(REQ-HW-01 单一分发表键)。
/// 构造权在引擎侧(boot 感知;kernels 零探测 —— 被动律)。
#[derive(Clone, Copy, Debug)]
pub struct Hw {
    pub arch: crate::Arch,
}



// ============================================================================
// §2 值类型
// ============================================================================

/// dtype(driver 自有最小集;按需加臂)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    /// BF16(E5-DF3 同日十四;DFlash2 草稿路径 BF16 化分派臂)
    BF16,
    /// 装载域 ct packed(U32;owl_ct_repack_u32)
    U32,
    /// **字节承载**(池块 / 量化输出;S1c 激活 int8 量化是首个非激活域输出)。
    /// 与 [`U32`](Self::U32) 同律:只有**声明里就是 U8** 的算子能收到它;
    /// 其余算子的 `match` 一律 [`unimplemented!`](unimplemented)(响亮拒)。
    U8,
}

/// pick 三件的中间形态(const 构造友好)
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
}

/// 语义算子身份证(model 层语义词表的值类型;内容 = 语义名空间,
/// 非 kernel 实现名 —— 实现名由 [`resolve`] 按环境推导)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpId(pub &'static str);

/// 执行环境(引擎/解释器装配,**必传参**;被动律:kernels 零探测)
#[derive(Clone, Copy, Debug)]
pub struct OpEnv {
    pub hw: Hw,
    /// KV 页大小(tokens/block;0 = 非 paged/legacy)。页配对律的 env 侧
    /// 单源 —— wrapper 变体选择在此吃页,调用点无页可错配
    pub page: usize,
    /// 注意力头维(0 = 未知 ⇒ 几何门跳过该项)。模型级几何事实,
    /// 由引擎 boot 从 `ModelDims` 合入(与 hw 同律:被动律)
    pub head_dim: usize,
    /// GQA 分组比 hq/hkv(0 = 未知/不适用)
    pub grp: usize,
}

/// 一次拾取请求(解释器 → driver;全部只读引用,零 I/O)
pub struct OpReq<'a> {
    pub op: OpId,
    pub env: &'a OpEnv,
    pub dt: DType,
    /// 输入张量形状(按 args 序;部分 op 的 grid/smem 从形状推导)
    pub shapes: &'a [Vec<usize>],
    /// 拾取推导常数(层语义几何:kd/vd/batch…;**非核参数**,不进签名,
    /// 序 = 各族函数文档)
    pub aux: &'a [usize],
    /// 标量参数槽(含于核签名的那些;U64/I32 折 i64)
    pub scalars: &'a [i64],
}

/// 拾取唯一入口(解释器的算子服务;纯函数,被动律)
///
/// 内部 = 实现分派表:新算子 = 新臂(+ 族函数 + 登记表条目),解释器
/// 与 model 层零改动。未知 OpId = 编程错误,panic 带词表提示
/// (与登记表 `source()` 同纪律)。
pub fn resolve(req: OpReq) -> KernelPick {
    let dt = req.dt;
    // 探针例外:lib 零依赖律(见 Cargo.toml)优先于 env 统一读取面 ——
    // 本 crate 不依赖 owl-shared,直呼 std::env(测试域已收编 env_reader)
    if std::env::var_os("OWL_RESOLVE_TRACE").is_some() {
        let _ = &req;
    }
    let ax = |i: usize| -> usize {
        *req.aux.get(i).unwrap_or_else(|| {
            panic!(
                "driver::resolve(\"{}\"): aux[{i}] 越界(aux len = {})",
                req.op.0,
                req.aux.len()
            )
        })
    };
    let pick = match req.op.0 {
        "gdn.gating_g" => gdn::gating_g(dt),
        "gdn.l2norm" => gdn::l2norm(dt, ax(0)),
        "gdn.conv_upd" => gdn::conv_upd(dt),
        "gdn.conv_upd_dual" => gdn::conv_upd_dual(dt),
        "gdn.delta_dec" => gdn::delta_dec(dt, ax(3), ax(0), ax(1), ax(2)), // aux = [batch, nv, kd, vd]
        "gdn.decode_step" => gdn::decode_step(dt, ax(0), ax(1), ax(2), ax(3)), // aux = [batch, nv, kd, vd]
        "gdn.decode_step_v2" => gdn::decode_step_v2(dt, ax(0), ax(1), ax(2), ax(3)),
        "gdn.conv_fwd" => gdn::conv_fwd(
            dt,
            ax(0),
            // N1(2026-10-14):t 维并行 ⇒ picker 需要 tokens(x 实到行数 = shapes[0][0])
            // 与 batch(标量槽 0);旧版只吃 d ⇒ 核内对 t 顺序扫描、批 1 时仅 8/8/24 CTA。
            req.shapes.first().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0),
            req.scalars.first().map(|b| (*b).max(0) as usize).unwrap_or(1),
        ),
        "gdn.recurrence_varlen_gqa" => gdn::recurrence_varlen_gqa(dt, ax(2), ax(0), ax(1)), // aux = [nv, kd, vd]
        "gdn.norm_act" => gdn::norm_act(dt, ax(0), ax(1), ax(2)), // aux = [rows, value_dim, group_size]
        // foreign 家族臂(2026-10-12 用户律:gdn 两臂入词表,与 native 同一
        // Call 通道;名字 = family 单源,发射配置 = FOREIGN 哨兵直通)
        "gdn.chunked_delta" => gdn::chunked_delta(),
        "gdn.scalar_delta" => gdn::scalar_delta(),
        "ops.sigmoid" => ops::sigmoid(dt),
        "ops.add" => ops::add(dt),
        "ops.mul" => ops::mul(dt),
        "ops.silu" => ops::silu(dt),
        "ops.rmsnorm" => ops::rmsnorm(dt, req.shapes),
        // 视觉塔族(2026-10-15;网格 = 一行一块 ⇒ 需要 x 的行数)
        "vis.layernorm" => vis::layernorm(
            dt,
            req.shapes.first().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0),
        ),
        "vis.gelu_tanh" => vis::gelu(dt, true),
        "vis.gelu_erf" => vis::gelu(dt, false),
        "vis.bias_add" => vis::bias_add(
            dt,
            req.shapes.first().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0),
        ),
        // 行数取 **idx 张量**(父序 = table, idx, wts)的 shapes[1][0]
        "vis.pos_embed" => vis::pos_embed(
            dt,
            req.shapes.get(1).map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0),
        ),
        // q 的 shapes[0] = [rows, heads, head_dim];标量槽 = [rows, heads, head_dim, n_seq]
        // **两臂分流(2026-10-15 用户裁决:别自写算子)**:f16 + 单图 ⇒ 走
        // vendored FlashInfer 的 ragged 非因果 prefill(FA2 家族,vLLM ViT 同路算法);
        // 其余(f32 锚/多图)⇒ native 臂。
        // 行数取 h 的 shapes[0] = [L, D](VLM 注入)
        "vis.image_scatter" => vis::image_scatter(
            dt,
            req.shapes.first().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0),
        ),
        "vis.attn_full" => {
            let q = req.shapes.first().cloned().unwrap_or_default();
            let n_seq = req.scalars.get(3).copied().unwrap_or(1);
            vis::attn_full(
                dt,
                q.first().copied().unwrap_or(0),
                q.get(1).copied().unwrap_or(0),
                q.get(2).copied().unwrap_or(0),
                n_seq,
            )
        }
        "attn.k0_write" => attn::k0_write(dt, ax(0)),
        "attn.k0_write_fp8" => attn::k0_write_fp8(ax(0)), // B6.2:f16 入 → e4m3 池(ax=tokens)
        "attn.k0_write_knhd" => attn::k0_write_knhd(dt, ax(0)),
        "attn.k0_write_fp8_knhd" => attn::k0_write_fp8_knhd(ax(0)),
        "attn.k0_write_fp8_bf16" => attn::k0_write_fp8_bf16(ax(0)), // B6:bf16 入 → e4m3 池
        "attn.paged_decode" => attn::paged_decode_v1(req.env, dt, ax(0), ax(1), ax(2), ax(3)), // aux = [hd, hq, hkv, nb]
        "attn.paged_decode_v2" => {
            // aux = [hd, hq, hkv, nb, nparts];页配对律同 v1(谓词单源)
            attn::paged_decode_v2(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4))
        }
        "attn.paged_decode_v2_fp8" => {
            // B6.1:fp8 e4m3 KV 读变体(形状契约同 f16;页配对律同源)
            attn::paged_decode_v2_fp8(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4))
        }
        "attn.paged_decode_v2_fp8_seq" => {
            // §三十一:verify 8 伪序列(grid.y = seqs;aux 末位 = seqs)
            attn::paged_decode_v2_fp8_seq(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4), ax(5))
        }
        "attn.paged_decode_v2_seq" => {
            // f16 兑底臂(seqs 参数化;同上)
            attn::paged_decode_v2_seq(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4), ax(5))
        }
        "attn.paged_decode_v2_knhd" => {
            attn::paged_decode_v2_knhd(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4))
        }
        "attn.paged_decode_v2_fp8_knhd" => {
            attn::paged_decode_v2_fp8_knhd(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4))
        }
        "attn.paged_decode_v2_fp8_seq_knhd" => {
            attn::paged_decode_v2_fp8_seq_knhd(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4), ax(5))
        }
        "attn.paged_decode_v2_seq_knhd" => {
            attn::paged_decode_v2_seq_knhd(req.env, dt, ax(0), ax(1), ax(2), ax(3), ax(4), ax(5))
        }
        "attn.paged_v2_reduce" => {
            // aux = [hd, hq, nparts, seqs](E16:核 seq_idx=blockIdx.y,
            // grid.y = seqs;decode 单序 = 1)
            attn::paged_v2_reduce(dt, ax(0), ax(1), ax(2), ax(3))
        }
        "attn.paged_prefill" => attn::paged_prefill(req.env, dt, ax(0), ax(1), ax(2), ax(3)), // aux = [hd, hkv, hq, tokens]
        "attn.paged_prefill_fp8" => attn::paged_prefill_fp8(req.env, dt, ax(0), ax(1), ax(2), ax(3)), // B6.3 fp8 读变体
        "attn.chunked_prefill_knhd" => attn::paged_prefill_knhd(req.env, dt, ax(0), ax(1), ax(2), ax(3)),
        "attn.chunked_prefill_fp8_knhd" => attn::paged_prefill_fp8_knhd(req.env, dt, ax(0), ax(1), ax(2), ax(3)),
        "attn.prefill_split" => {
            // aux = [hd, hkv, hq, tokens, nparts](ctx_base 走核参数槽,层侧传入)
            let (hd, hkv, hq, tokens, nparts) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::prefill_split(req.env, dt, hd, hkv, hq, tokens, nparts)
        }
        "attn.prefill_split_fp8kv" => {
            // aux 同 prefill_split;fp8 e4m3 KV 直读(2026-10-12 修雷 + 读量减半)
            let (hd, hkv, hq, tokens, nparts) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::prefill_split_fp8kv(req.env, dt, hd, hkv, hq, tokens, nparts)
        }
        "attn.prefill_split_knhd" => {
            let (hd, hkv, hq, tokens, nparts) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::prefill_split_knhd(req.env, dt, hd, hkv, hq, tokens, nparts)
        }
        "attn.prefill_split_fp8_knhd" => {
            let (hd, hkv, hq, tokens, nparts) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::prefill_split_fp8kv_knhd(req.env, dt, hd, hkv, hq, tokens, nparts)
        }
        "attn.prefill_split_reduce" => {
            // aux = [tokens, hq]
            attn::prefill_split_reduce(dt, ax(0), ax(1))
        }
        "attn.naive_decode" => attn::naive_decode(dt),
        "attn.gate_mul" => attn::gate_mul(dt),
        "elems.gemv_dual" => {
            // aux = [rows_b, rows_a, cols, tokens]
            elems::gemv_dual(dt, ax(0), ax(1), ax(2), ax(3))
        }
        "ops.narrow" => elems::narrow(dt),
        "ops.ts_stamp" => ops::ts_stamp(),
        "ops.argmax" => ops::argmax(),
        "ops.topk16" => ops::topk16(ax(0)),
        "ops.topk16_split" => ops::topk16_split(ax(0), ax(1)), // aux = [rows, nparts]
        "ops.topk16_merge" => ops::topk16_merge(ax(0)),        // aux = [rows]
        "ops.sigmoid_gate_mul" => attn::gate_mul(dt),
        "elems.cast_f16_bf16" => elems::cast_dir(false),
        "elems.cast_bf16_f16" => elems::cast_dir(true),
        "dflash.select" => dflash::select(ax(0) != 0), // aux = [bf16](输入侧;输出恒 f32)
        // aux = [rows];名字出自声明表单源(两档 = 两个 op_id)
        "elems.quant_int8_row_f16" => {
            elems::quant_int8_row(crate::ops::elems::QUANT_INT8_ROW_F16.entry, ax(0))
        }
        "elems.quant_int8_row_bf16" => {
            elems::quant_int8_row(crate::ops::elems::QUANT_INT8_ROW_BF16.entry, ax(0))
        }
        "dflash.conv" => dflash::conv(dt),
        "attn.naive_nc" => attn::naive_nc(dt, false, req.shapes, req.scalars),
        "attn.naive_nc_fp8kv" => attn::naive_nc(dt, true, req.shapes, req.scalars),
        "marlin.w4a16" => marlin::w4a16(req.shapes, dt),
        "marlin.w4a8_awq" => marlin::w4a8_awq(),
        "attn.fi_prefill" => attn::fi_prefill(false),
        "attn.fi_prefill_fp8kv" => attn::fi_prefill(true),
        "attn.xqa_verify" => attn::xqa_verify(),
        "attn.xqa_decode" => attn::xqa_decode(),
        "attn.xqa_prefill" => attn::xqa_prefill(),
        "elems.cast_f16_f32" => elems::cast_f16_f32(dt),
        "ops.concat" => elems::concat(dt),
        "ops.rope" => elems::rope(dt, ax(0)),
        "ops.embed" => elems::embed(dt, ax(0)),
        "attn.norm_rope" => attn::norm_rope(dt, ax(0), ax(1), ax(2)), // aux = [tokens, heads, hd]
        "attn.qkv_norm_rope_insert" => {
            let (tokens, hq, hkv, hd, half) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::qkv_norm_rope_insert(dt, tokens, hq, hkv, hd, half) // aux = [tokens, hq, hkv, hd, half]
        }
        "attn.qkv_norm_rope_insert_fp8kv" => {
            // B6.3:fp8 主池变体(几何同 f16;池写 e4m3 1B)
            let (tokens, hq, hkv, hd, half) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::qkv_norm_rope_insert_fp8kv(tokens, hq, hkv, hd, half)
        }
        "attn.qkv_norm_rope_insert_knhd" => {
            let (tokens, hq, hkv, hd, half) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::qkv_norm_rope_insert_knhd(dt, tokens, hq, hkv, hd, half)
        }
        "attn.qkv_norm_rope_insert_fp8kv_knhd" => {
            let (tokens, hq, hkv, hd, half) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::qkv_norm_rope_insert_fp8kv_knhd(tokens, hq, hkv, hd, half)
        }
                "ln.fused_add_rmsnorm" => {
            let (rows, n) = (ax(0), ax(1));
            ln::fused_add_rmsnorm(dt, rows, n)
        }
        "mlp.silu_and_mul" => mlp::silu_and_mul(dt, ax(0)),
        "load.ct_repack" => {
            let rows = req.shapes.first().map(|s| s[0]).unwrap_or(0);
            let cols = req.shapes.first().map(|s| s.get(1).copied().unwrap_or(0)).unwrap_or(0);
            load::ct_repack(rows, cols)
        }
        other => panic!(
            "driver::resolve(\"{other}\"): 未登记的语义算子命名空间 —— 动作词表住 models::ops::SemanticKernel(op_id 落本表),实现住本模块分派表(两侧须同票)"
        ),
    };
    // **几何门**(2026-10-13 P4;拾取即验):已建表算子的 `ShapeFamily`
    // 声明必须接受当前模型几何。错配 = compose 期编程/配置错(与
    // `aux 越界`/`无配对 wrapper` 同类)—— 在 boot warmup 阶段即暴露,
    // 等价于拒启。未建表算子(仅引入)无声明 ⇒ 放行(白名单门管它)。
    if let Some(m) = pick.meta() {
        let q = crate::spec::ShapeQuery {
            head_dim: req.env.head_dim,
            page: req.env.page,
            grp: req.env.grp,
            tile: 0,
        };
        if let Err(why) = m.shape.accepts(&q) {
            panic!(
                "driver::resolve(\"{}\") -> {}: 几何不接受 —— {why}(算子 ShapeFamily 声明;模型几何 hd={} page={} grp={})",
                req.op.0, m.entry, req.env.head_dim, req.env.page, req.env.grp
            );
        }
    }
    pick
}

/// 一次拾取的产物:名字 + 发射配置(与声明面 `KernelMeta` 对应
/// 的发射全部输入;grid 哨兵 (0,0,0) = 解释层按输出元素数自动 1D
/// ceil/256,同登记表约)
///
/// # 准入声明(2026-10-13 P3 立项;**F2 轮 2026-10-13 收口**)
///
/// **名字一律不手写**,两条来源:
/// - native 核 ⇒ [`KernelPick::of::<T>()`]:名字来自算子的 `ENTRY` 关联常量;
/// - foreign 家族臂 ⇒ 家族 face 常量(`family::*::…_FWD` / `client::*::SIG` 同源)。
///
/// **F2 轮前**还有 18 处 `KernelPick { name: "owl_…" }` 字面量(`probe_ts` /
/// `argmax` / `topk16` / `cast` / `qknorm_insert` ×3 / `gemv_dual` /
/// `ct_repack` / `decode_step{,_v2}` / `conv_{upd_dual,fwd}` / `recurrence` /
/// `reshape_and_cache` ×2 / `prefill_split_reduce` / `sigmoid_gate_mul`),
/// 依据是"未建表算子只能字面量构造"—— 而 F 轮把"未声明"清零后该前提消失,
/// 已全部升格为 `of::<T>()` ⇒ **driver 里零核名字符串**。
///
/// 这不是省字:`of::<T>()` 让"名字在 picker、声明面却漏了"这类不一致
/// **物理上写不出来**(F 轮抓到的 `vllm_paged_attention_v2_f16_hd128/hd256`
/// 活路径正是该病)。
///
/// 为什么 `meta` 不做成字段:构造点上百处,加字段 = 大面积机械改写;
/// 改成**按名查表**(`meta()`)后 `of::<T>()` 拿到类型耦合的名字,语义等价而 diff 最小。
#[derive(Clone, Debug)]
pub struct KernelPick {
    pub name: &'static str,
    pub shape: Shape,
}

impl KernelPick {
    /// **按 `KernelSpec` 拾取**(生产臂用):名字 = 算子关联常量,
    /// 元数据/槽序/检查函数随之而来(编译期耦合,名字不可能写错)。
    pub const fn of<T: crate::spec::KernelSpec>(shape: Shape) -> Self {
        Self { name: T::ENTRY, shape }
    }

    /// 本算子的准入声明(未建表 = `None`,即"仅引入")
    pub fn meta(&self) -> Option<&'static crate::spec::KernelMeta> {
        crate::ops::meta(self.name)
    }
}

/// 哨兵 1D 发射形态(逐元素/批量核的公共形态)
pub const SENTINEL_1D: Shape = Shape { grid: (0, 0, 0), block: (256, 1, 1), smem: 0 };

/// foreign 家族臂发射形态(2026-10-12 用户律:具名算子入词表):grid/
/// block 由 server 家族 runtime 单源,发射配置层/解释层零感知 ——
/// (0,0,0) 直通,server 按名分派后自算网格
pub const FOREIGN: Shape = Shape { grid: (0, 0, 0), block: (0, 0, 0), smem: 0 };

// ============================================================================
// §3 算子族(GDN;F5 批 1-5 核 + 工单 G 批核,2026-09-25 语义)
// ============================================================================

pub mod gdn {
    use super::{DType, KernelPick, Shape, FOREIGN, SENTINEL_1D};

    /// gdn_chunked 六核编排(FLA fork-bf16;foreign 家族臂):名字 =
    /// family 单源,发射配置 = FOREIGN 哨兵(server 家族 runtime 自算网格)
    pub fn chunked_delta() -> KernelPick {
        KernelPick { name: crate::family::gdn_chunked::GDN_CHUNKED_FWD, shape: FOREIGN }
    }

    /// gdn_scalar 单核臂(lmdeploy pre_sm90 port;foreign 家族臂同上)
    pub fn scalar_delta() -> KernelPick {
        KernelPick { name: crate::family::gdn_scalar::GDN_SCALAR_FWD, shape: FOREIGN }
    }

    /// gating_g:g = softplus(a_log) + dt_bias 门(批 1;哨兵 1D)
    pub fn gating_g(dt: DType) -> KernelPick {
        KernelPick { name: match dt {
            DType::F16 => "owl_gdn_gating_g_f16",
            DType::F32 => "owl_gdn_gating_g_f32",
            DType::BF16 | DType::U32 | DType::U8 => unimplemented!("GDN 族无 U32 变体"),
        }, shape: SENTINEL_1D }
    }

    /// l2norm:行末维归一(批 2;一 block 一行,eps 走参数槽)
    pub fn l2norm(dt: DType, rows: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_gdn_l2norm_f16",
                DType::F32 => "owl_gdn_l2norm_f32",
                DType::BF16 | DType::U32 | DType::U8 => unimplemented!("GDN 族无 U32 变体"),
            },
            shape: Shape { grid: (rows as u32, 1, 1), ..SENTINEL_1D },
        }
    }

    /// conv_upd:decode 槽更新(批 3;state 原地副作用,单流保序;哨兵 1D)
    pub fn conv_upd(dt: DType) -> KernelPick {
        KernelPick { name: match dt {
            DType::F16 => "owl_gdn_conv_upd_f16",
            DType::F32 => "owl_gdn_conv_upd_f32",
            DType::BF16 | DType::U32 | DType::U8 => unimplemented!("GDN 族无 U32 变体"),
        }, shape: SENTINEL_1D }
    }

    /// 刀3b(2026-10-04):q/k 双段 conv 槽更新单发(哨兵 1D,平面 =
    /// batch·(dq+dk);输出单块 [batch, dq+dk],消费侧列切 = 免费视图)
    pub fn conv_upd_dual(dt: DType) -> KernelPick {
        assert!(matches!(dt, DType::F16), "conv_upd_dual 仅有 f16 变体");
        KernelPick::of::<crate::ops::gdn::ConvUpdDualF16>(SENTINEL_1D)
    }

    /// delta_dec:单步递推(批 4)。shared 契约 = `(2·kd + 2)·4B`(kd 项
    /// 进公式 —— 曾硬编码 128,非 128 kd 即静默越界;收编后公式单源)。
    /// grid = (ceil(vd/64), batch·nv, 1),block (64,1,1)。
    pub fn delta_dec(dt: DType, vd: usize, batch: usize, nv: usize, kd: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_gdn_delta_dec_f16",
                DType::F32 => "owl_gdn_delta_dec_f32",
                DType::BF16 | DType::U32 | DType::U8 => unimplemented!("GDN 族无 U32 变体"),
            },
            shape: Shape {
                grid: (vd.div_ceil(64) as u32, (batch * nv) as u32, 1),
                block: (64, 1, 1),
                smem: ((2 * kd + 2) * 4) as u32,
            },
        }
    }

    /// decode_step:D1 整链融合(**仅 f16 变体**)。aux = [batch, nv, kd, vd];
    /// grid = (batch·nv,1,1),block = (kd,1,1)(kd==vd 守卫),动态 smem = 0
    /// (静态 warp_sums + q/k smem 在核内)。
    pub fn decode_step(dt: DType, batch: usize, nv: usize, kd: usize, vd: usize) -> KernelPick {
        assert!(
            matches!(dt, DType::F16),
            "owl_gdn_decode_step 仅有 f16 变体(dt={dt:?})"
        );
        assert_eq!(kd, vd, "decode_step 融合核要求 kd==vd(头方阵);得 {kd}/{vd}");
        assert!(kd <= 128 && kd.is_multiple_of(32), "decode_step 融合核要求 kd≤128 且 32 对齐,得 {kd}");
        KernelPick::of::<crate::ops::gdn::GdnDecodeStepF16>(Shape { grid: ((batch * nv) as u32, 1, 1), block: (kd as u32, 1, 1), smem: 0 })
    }

    /// D1-v2(2026-10-04 sglang 刺探):delta 相 float4 行组重写(同契约;
    /// 4 warp = 4 行组 × lane float4 列组,跨 warp 部分和经 shared 归并;
    /// sglang 同形实测 7.5µs = 844 GB/s,owl v1 28.4µs → 靶 ≤10µs)
    pub fn decode_step_v2(dt: DType, batch: usize, nv: usize, kd: usize, vd: usize) -> KernelPick {
        assert!(
            matches!(dt, DType::F16),
            "owl_gdn_decode_step_v2 仅有 f16 变体(dt={dt:?})"
        );
        assert_eq!(kd, vd, "decode_step v2 要求 kd==vd(头方阵);得 {kd}/{vd}");
        assert!(kd <= 128 && kd.is_multiple_of(32) && vd.is_multiple_of(4), "decode_step v2 要求 kd≤128 32 对齐且 vd/4,得 {kd}/{vd}");
        KernelPick::of::<crate::ops::gdn::DecodeStepV2F16>(Shape { grid: ((batch * nv) as u32, 1, 1), block: (vd as u32, 1, 1), smem: 0 })
    }

    /// conv_fwd:**prefill 卷前向**(工单 G;**仅 f16 变体**,f32 请求 =
    /// 编程错误,decl 构造期 panic —— 与登记表 `source()` 同纪律)。
    ///
    /// **发射几何(N1,2026-10-14;自 feat/prefill-graph 合入)**:
    /// `grid = (ceil(d/256), tiles, batch)`,`tiles = ceil(tokens / TILE_ROWS)`。
    /// 旧版是 `(1, ceil(d/256), 1)` + 核内对 t **顺序扫描** ⇒ 批 1 时 q/k/v 各只
    /// 8/8/24 个 CTA,延迟绑死(122-170 µs/发;nsys 定谳 = prefill 的 4.6%);
    /// 新核把 t 维切块并行 ⇒ 实测 **17.5 → 2.0 ms/chunk(−88.6%)**,
    /// prefill 相位 **−3.6~3.8%**(两把尺子互证,见
    /// `docs/arch/prefill-graph-施工.md` §13.3)。
    ///
    /// `tokens` 取自 `shapes[0][0]`(x 的实到行数);`batch` 取自标量槽 0
    /// (生产 prefill 恒 1;多层折叠路同)。**TILE_ROWS 只是网格提示** ——
    /// 核内自算 `ts = ceil(len/gridDim.y)`,picker 与核零常量耦合。
    /// **前置律**:`TILE_ROWS >= 3`(保证 tiles>1 ⇒ ts>=3 ⇒ 只有 tile 0 碰 state);
    /// 取 32。
    pub fn conv_fwd(dt: DType, d: usize, tokens: usize, batch: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_gdn_conv_fwd 仅有 f16 变体(dt={dt:?})");
        const TILE_ROWS: usize = 32;
        let tiles = tokens.div_ceil(TILE_ROWS).max(1);
        KernelPick::of::<crate::ops::gdn::ConvFwdF16>(Shape {
            grid: (d.div_ceil(256) as u32, tiles as u32, batch.max(1) as u32),
            block: (256, 1, 1),
            smem: 0,
        })
    }

    /// recurrence_varlen_gqa:批前向递推(工单 G;**仅 f16 变体**)。
    /// shared 契约 = `(4·kd + 4)·4B`;grid = (ceil(vd/8), nv, 1),
    /// block (32,8,1) = 8 头一组。
    pub fn recurrence_varlen_gqa(dt: DType, vd: usize, nv: usize, kd: usize) -> KernelPick {
        assert!(
            matches!(dt, DType::F16),
            "owl_gdn_recurrence_varlen_gqa 仅有 f16 变体(dt={dt:?})"
        );
        KernelPick::of::<crate::ops::gdn::RecurrenceVarlenGqaF16>(Shape {
            grid: (vd.div_ceil(8) as u32, nv as u32, 1),
            block: (32, 8, 1),
            smem: ((4 * kd + 4) * 4) as u32,
        })
    }

    /// norm_act:×w(非零中心)+ silu(批 5;一 block 一 (row, group))。
    /// grid = (rows·value_dim/group_size, 1, 1)。
    pub fn norm_act(dt: DType, rows: usize, value_dim: usize, group_size: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_gdn_norm_act_f16",
                DType::F32 => "owl_gdn_norm_act_f32",
                DType::BF16 | DType::U32 | DType::U8 => unimplemented!("GDN 族无 U32 变体"),
            },
            shape: Shape {
                grid: ((rows * value_dim / group_size) as u32, 1, 1),
                ..SENTINEL_1D
            },
        }
    }
}

// ============================================================================
// §4 逐元素共用族(跨算子消费;beta 臂 = sigmoid(b) 同款律)
// ============================================================================

pub mod ops {
    use super::{DType, KernelPick, Shape, SENTINEL_1D};

    /// sigmoid:逐元素(beta 臂 = sigmoid(b);M-b 同款律;哨兵 1D)
    pub fn sigmoid(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_sigmoid_f16",
                DType::F32 => "owl_sigmoid_f32",
                DType::BF16 | DType::U32 | DType::U8 => unimplemented!("sigmoid 无 BF16/U32 变体"),
            },
            shape: SENTINEL_1D,
        }
    }

    /// add:同形逐元素加(残差;哨兵 1D)
    pub fn add(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F32 => "owl_add_f32",
                DType::F16 => "owl_add_f16",
                DType::BF16 => "owl_add_bf16",
                DType::U32 | DType::U8 => unimplemented!("add 无 U32 变体"),
            },
            shape: SENTINEL_1D,
        }
    }

    /// mul:同形逐元素乘(门控;哨兵 1D)
    pub fn mul(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F32 => "owl_mul_f32",
                DType::F16 => "owl_mul_f16",
                DType::BF16 => "owl_mul_bf16",
                DType::U32 | DType::U8 => unimplemented!("mul 无 U32 变体"),
            },
            shape: SENTINEL_1D,
        }
    }

    /// silu:一元激活(哨兵 1D;BF16 无独立变体 —— MLP 走 silu_and_mul 融合)
    pub fn silu(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F32 => "owl_silu_f32",
                DType::F16 => "owl_silu_f16",
                DType::BF16 | DType::U32 | DType::U8 => unimplemented!("silu 无 BF16/U32 变体(MLP 走 silu_and_mul)"),
            },
            shape: SENTINEL_1D,
        }
    }

    /// 刀D 层间时间戳探针(链式拷贝 + clock64;哨兵 1D)
    pub fn ts_stamp() -> KernelPick {
        KernelPick::of::<crate::ops::elems::ProbeTsF16>(SENTINEL_1D)
    }

    /// 设备侧贪心 argmax(单块两段规约;fixed (1,1,1) × 256)
    pub fn argmax() -> KernelPick {
        KernelPick::of::<crate::ops::elems::ArgmaxF32IdxF16>(Shape { grid: (1, 1, 1), block: (256, 1, 1), smem: 0 })
    }

    /// 逐行 top-16(单缓冲 [rows, 32]:值半区降序 + 索引半区;aux = [rows])
    pub fn topk16(rows: usize) -> KernelPick {
        KernelPick::of::<crate::ops::elems::Topk16F16>(Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: 0 })
    }

    /// split-K 段一(2026-10-14 D2;grid = rows×nparts,dyn 0)
    pub fn topk16_split(rows: usize, nparts: usize) -> KernelPick {
        KernelPick::of::<crate::ops::elems::Topk16SplitF16>(Shape {
            grid: ((rows * nparts) as u32, 1, 1),
            block: (256, 1, 1),
            smem: 0,
        })
    }

    /// split-K 段二(一行一 warp;grid = rows,block = 32)
    pub fn topk16_merge(rows: usize) -> KernelPick {
        KernelPick::of::<crate::ops::elems::Topk16MergeF16>(Shape {
            grid: (rows as u32, 1, 1),
            block: (32, 1, 1),
            smem: 0,
        })
    }

    /// rmsnorm:一 block 一行(×(1+w) 同核,w_off 入参;rows 从形状推导:
    /// x_total / cols,alpha 末维 = cols)
    pub fn rmsnorm(dt: DType, shapes: &[Vec<usize>]) -> KernelPick {
        let name = match dt {
            DType::F32 => "owl_rmsnorm_f32",
            DType::F16 => "owl_rmsnorm_f16",
            DType::BF16 => "owl_rmsnorm_bf16",
            DType::U32 | DType::U8 => unimplemented!("rmsnorm 无 U32 变体"),
        };
        let x_total: usize = shapes.first().map(|s| s.iter().product()).unwrap_or(0);
        let cols: usize = shapes.get(1).map(|s| s.iter().product()).unwrap_or(1);
        let rows = x_total.checked_div(cols).unwrap_or(0);
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: 256 * 4 },
        }
    }
}

// ============================================================================
// §3.x DFlash2 草稿族(E5-DF1;语义锚 = sglang dflash.py)
// ============================================================================

pub mod dflash {
    use super::{DType, KernelPick, Shape, SENTINEL_1D};

    /// 格打分 + 贪心 walk 融合(单块;fixed (1,1,1) × 256)。
    /// ⚠️ 输入侧 f16/bf16(aux = [bf16]),输出恒 f32 —— 节点 dt 是输出,
    /// 路由不吃 req.dt
    pub fn select(bf16: bool) -> KernelPick {
        KernelPick {
            name: if bf16 { "owl_dflash_select_bf16" } else { "owl_dflash_select_f16" },
            shape: Shape { grid: (1, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// 分组动态深度 2-tap 卷积(哨兵 1D)
    pub fn conv(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_dflash_conv_f16",
                DType::BF16 => "owl_dflash_conv_bf16",
                DType::F32 | DType::U32 | DType::U8 => unimplemented!("dflash_conv 仅 f16/bf16"),
            },
            shape: SENTINEL_1D,
        }
    }
}

// ============================================================================
// §4.1 attention 族(vLLM port 家族;页配对律的 env 侧宿主)
// ============================================================================

pub mod attn {
    use super::{DType, Hw, KernelPick, OpEnv, Shape, FOREIGN, SENTINEL_1D};

    /// K0 批量写池(reshape_and_cache;grid (tokens,1,1),block 256)。
    /// dt 分派(E5-DF3 同日十四):F16 → f16 核;BF16 → bf16 核(草稿池)。
    pub fn k0_write(dt: DType, tokens: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "vllm_reshape_and_cache_f16",
                DType::BF16 => "vllm_reshape_and_cache_bf16",
                other => unimplemented!("k0_write 无 {other:?} 变体"),
            },
            shape: Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// B6.2:K0 写池 fp8 变体(f16/bf16 输入 → e4m3 池;形状无关单核)
    pub fn k0_write_fp8(tokens: usize) -> KernelPick {
        KernelPick::of::<crate::ops::attn::K0WriteFp8Classic>(Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 })
    }

    /// B6:K0 写池 fp8 变体(bf16 输入;草稿侧)
    pub fn k0_write_fp8_bf16(tokens: usize) -> KernelPick {
        KernelPick::of::<crate::ops::attn::ReshapeAndCacheFp8KvBf16>(Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 })
    }

    /// B6.1:decode v2 fp8 e4m3 KV 读(形状契约同 f16;页配对律同源)
    pub fn paged_decode_v2_fp8(
        env: &OpEnv,
        _dt: DType,
        hd: usize,
        hq: usize,
        hkv: usize,
        nb: usize,
        nparts: usize,
    ) -> KernelPick {
        paged_decode_v2_fp8_seq(env, _dt, hd, hq, hkv, nb, nparts, 1)
    }

    /// §三十一:伪序列变体(grid.y = seqs;decode = 1,verify = depth1)
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_v2_fp8_seq(
        env: &OpEnv,
        _dt: DType,
        hd: usize,
        hq: usize,
        _hkv: usize,
        _nb: usize,
        nparts: usize,
        seqs: usize,
    ) -> KernelPick {
        let page = env.page;
        let name = match (hd, page) {
            (128, 32) => "vllm_paged_attention_v2_fp8_hd128bs32",
            (256, 32) => "vllm_paged_attention_v2_fp8_hd256bs32",
            (128, 16) => "vllm_paged_attention_v2_fp8_hd128",
            (256, 16) => "vllm_paged_attention_v2_fp8_hd256",
            other => panic!("paged v2 fp8: (hd,page) {other:?} 无配对 wrapper"),
        };
        let floor = (128u32 / 32 / 2) * hd as u32 * 4;
        KernelPick {
            name,
            shape: Shape {
                grid: (hq as u32, seqs as u32, nparts as u32),
                block: (128, 1, 1),
                smem: 2048u32.max(floor),
            },
        }
    }

    /// B6.3:chunked prefill fp8 e4m3 KV 读(形状/网格契约同 f16)
    pub fn paged_prefill_fp8(
        env: &OpEnv,
        _dt: DType,
        hd: usize,
        hkv: usize,
        hq: usize,
        tokens: usize,
    ) -> KernelPick {
        assert!(
            env.page == 32,
            "chunked prefill bs32 契约:页 {} 非法",
            env.page
        );
        // TG=4 变体(2026-10-10 性能刀):hd256 走 64-token×4 线程组分摊核
        // (原核 acc[256]/线程 = 寄存器 spill,~2% 峰值);hd128 走旧核
        let (name, tokens_per_block) = match hd {
            128 => ("vllm_chunked_prefill_paged_attn_opt_fp8_hd128", 256usize),
            256 => ("vllm_chunked_prefill_paged_attn_opt_fp8_hd256", 256usize),
            other => panic!("chunked prefill fp8 仅 hd∈{{128,256}},得 {other}"),
        };
        KernelPick {
            name,
            shape: Shape {
                grid: ((hq / hkv) as u32, hkv as u32, tokens.div_ceil(tokens_per_block) as u32),
                block: (256, 1, 1),
                smem: (64 + 2 * hd * env.page * 2) as u32,
            },
        }
    }

    /// kNHD 变体(统一契约 P2;签名/网格同 classic,kernel 名 _knhd)
    pub fn paged_prefill_knhd(
        env: &OpEnv, dt: DType, hd: usize, hkv: usize, hq: usize, tokens: usize,
    ) -> KernelPick {
        let mut p = paged_prefill(env, dt, hd, hkv, hq, tokens);
        p.name = match hd {
            128 => "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd128",
            256 => "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256",
            other => panic!("chunked knhd 仅 hd∈{{128,256}},得 {other}"),
        };
        p
    }

    pub fn paged_prefill_fp8_knhd(
        env: &OpEnv, dt: DType, hd: usize, hkv: usize, hq: usize, tokens: usize,
    ) -> KernelPick {
        let mut p = paged_prefill(env, dt, hd, hkv, hq, tokens);
        p.name = match hd {
            128 => "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128",
            256 => "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
            other => panic!("chunked fp8 knhd 仅 hd∈{{128,256}},得 {other}"),
        };
        p
    }

    /// 页配对谓词(decode;声明期门控用 —— 执行期 resolve 同律 panic,
    /// 谓词与分派单源于此,调用点不可能配错页)。**(hd,page) 联合裁决**
    /// —— legacy 小头测试(hd∉{128,256})由 None 回退 naive 的历史行为
    /// 由此保持(v1_name 时代同款)
    pub fn paged_decode_ok(hd: usize, page: usize) -> bool {
        matches!((hd, page), (128, 16) | (128, 32) | (256, 16) | (256, 32))
    }

    /// 页配对谓词(prefill;bs32 契约 = vendor BLOCK∈{32,64} × hd∈{128,256})
    pub fn paged_prefill_ok(hd: usize, page: usize) -> bool {
        matches!(hd, 128 | 256) && page == 32
    }

    /// v1 分页打分:**页配对律在此单源**(wrapper 按 (hd,page) 选择;
    /// 2026-10-01 三轮立案的终结 —— 调用点再也拿不到「页不配对的核」)。
    /// smem 契约公式 = `max(nb·page·4B, (NUM_WARPS/2)·HD·4B)`(原
    /// models::paged_v1_smem 收编);grid (hq,1,1),block (128,1,1)。
    pub fn paged_decode_v1(
        env: &OpEnv,
        dt: DType,
        hd: usize,
        hq: usize,
        _hkv: usize,
        nb: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "paged v1 仅有 f16 变体(dt={dt:?})");
        let page = env.page;
        let name = match (hd, page) {
            (128, 32) => "vllm_paged_attention_v1_f16_hd128bs32",
            (256, 32) => "vllm_paged_attention_v1_f16_hd256bs32",
            (128, 16) => "vllm_paged_attention_v1_f16_hd128",
            (256, 16) => "vllm_paged_attention_v1_f16_hd256",
            other => panic!(
                "paged v1: (hd,page) {other:?} 无配对 wrapper(谓词 paged_decode_ok 应已拦截)"
            ),
        };
        let floor = (128u32 / 32 / 2) * hd as u32 * 4; // (NUM_WARPS/2)·HD·4B
        KernelPick {
            name,
            shape: Shape {
                grid: (hq as u32, 1, 1),
                block: (128, 1, 1),
                smem: ((nb * page * 4) as u32).max(floor),
            },
        }
    }

    /// v2 分页打分(在线 softmax 分块;E-decode 长 ctx 衰减主修
    /// 2026-10-04)。PARTITION=512:每 CTA ≤512 token,logits smem 恒
    /// 2KB(与 out_smem 复用同块,公式 = max(512·4B, floor));grid
    /// (hq, 1, nparts) z = nparts = ceil(nb·page/512)(核从 gridDim.z 读
    /// max_num_partitions;设备侧 seq_len 守卫早退,图安全)。页配对律
    /// 同 v1 单源(bs32 wrapper = <hd,32,128,512>)。输出 = 未归一化
    /// partials,归并交 reduce(两发;ctx ≤512 时归并退化为直拷)。
    pub fn paged_decode_v2(
        env: &OpEnv,
        dt: DType,
        hd: usize,
        hq: usize,
        hkv: usize,
        nb: usize,
        nparts: usize,
    ) -> KernelPick {
        paged_decode_v2_seq(env, dt, hd, hq, hkv, nb, nparts, 1)
    }

    /// §三十一:伪序列变体(grid.y = seqs)
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_v2_seq(
        env: &OpEnv,
        dt: DType,
        hd: usize,
        hq: usize,
        _hkv: usize,
        _nb: usize,
        nparts: usize,
        seqs: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "paged v2 仅有 f16 变体(dt={dt:?})");
        let page = env.page;
        let name = match (hd, page) {
            (128, 32) => "vllm_paged_attention_v2_f16_hd128bs32",
            (256, 32) => "vllm_paged_attention_v2_f16_hd256bs32",
            (128, 16) => "vllm_paged_attention_v2_f16_hd128",
            (256, 16) => "vllm_paged_attention_v2_f16_hd256",
            other => panic!(
                "paged v2: (hd,page) {other:?} 无配对 wrapper(谓词 paged_decode_ok 应已拦截)"
            ),
        };
        let floor = (128u32 / 32 / 2) * hd as u32 * 4; // (NUM_WARPS/2)·HD·4B
        KernelPick {
            name,
            shape: Shape {
                grid: (hq as u32, seqs as u32, nparts as u32),
                block: (128, 1, 1),
                smem: 2048u32.max(floor), // PARTITION_SIZE·4B vs out_smem floor
            },
        }
    }

    /// kNHD 变体(统一契约 P2;签名同 classic,kernel 名 _knhd)
    pub fn paged_decode_v2_knhd(
        env: &OpEnv, dt: DType, hd: usize, hq: usize, hkv: usize, nb: usize, nparts: usize,
    ) -> KernelPick {
        paged_decode_v2_seq_knhd(env, dt, hd, hq, hkv, nb, nparts, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_v2_seq_knhd(
        env: &OpEnv, dt: DType, hd: usize, hq: usize, _hkv: usize, _nb: usize,
        nparts: usize, seqs: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "paged v2 knhd 仅有 f16 变体(dt={dt:?})");
        let page = env.page;
        let name = match (hd, page) {
            (128, 32) => "vllm_paged_attention_v2_f16_knhd_hd128bs32",
            (256, 32) => "vllm_paged_attention_v2_f16_knhd_hd256bs32",
            (128, 16) => "vllm_paged_attention_v2_f16_knhd_hd128",
            (256, 16) => "vllm_paged_attention_v2_f16_knhd_hd256",
            other => panic!("paged v2 knhd: (hd,page) {other:?} 无配对 wrapper"),
        };
        let floor = (128u32 / 32 / 2) * hd as u32 * 4;
        KernelPick {
            name,
            shape: Shape {
                grid: (hq as u32, seqs as u32, nparts as u32),
                block: (128, 1, 1),
                smem: 2048u32.max(floor),
            },
        }
    }

    pub fn paged_decode_v2_fp8_knhd(
        env: &OpEnv, dt: DType, hd: usize, hq: usize, hkv: usize, nb: usize, nparts: usize,
    ) -> KernelPick {
        paged_decode_v2_fp8_seq_knhd(env, dt, hd, hq, hkv, nb, nparts, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_v2_fp8_seq_knhd(
        env: &OpEnv, _dt: DType, hd: usize, hq: usize, _hkv: usize, _nb: usize,
        nparts: usize, seqs: usize,
    ) -> KernelPick {
        let page = env.page;
        let name = match (hd, page) {
            (128, 32) => "vllm_paged_attention_v2_fp8_knhd_hd128bs32",
            (256, 32) => "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
            (128, 16) => "vllm_paged_attention_v2_fp8_knhd_hd128",
            (256, 16) => "vllm_paged_attention_v2_fp8_knhd_hd256",
            other => panic!("paged v2 fp8 knhd: (hd,page) {other:?} 无配对 wrapper"),
        };
        let floor = (128u32 / 32 / 2) * hd as u32 * 4;
        KernelPick {
            name,
            shape: Shape {
                grid: (hq as u32, seqs as u32, nparts as u32),
                block: (128, 1, 1),
                smem: 2048u32.max(floor),
            },
        }
    }

    /// v2 LSE 归并(模板无 BLOCK/页参数 —— 单 wrapper 通用两页;reduce
    /// smem = 2·nparts·4B(shared_max_logits + shared_exp_sums);
    /// **grid (hq, seqs, 1)**(E16 根治:核 seq_idx=blockIdx.y —— 原硬发
    /// y=1 只归并 seq0,verify 多序行 1.. 从未写出;ctx ≤512 时核内退化
    /// 为 tmp_out 直拷)。
    pub fn paged_v2_reduce(dt: DType, hd: usize, hq: usize, nparts: usize, seqs: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "paged v2 reduce 仅有 f16 变体");
        let name = match hd {
            128 => "vllm_paged_attention_v2_reduce_f16_hd128",
            256 => "vllm_paged_attention_v2_reduce_f16_hd256",
            other => panic!("paged v2 reduce 仅 hd∈{{128,256}},得 {other}"),
        };
        KernelPick {
            name,
            shape: Shape {
                grid: (hq as u32, seqs as u32, 1),
                block: (128, 1, 1),
                smem: (2 * nparts * 4) as u32,
            },
        }
    }

    /// chunked prefill 批核(**bs32 契约** = vendor BLOCK∈{32,64})。
    /// smem = 64 + 2·HD·BLOCK·2B;grid (Hq/Hkv, Hkv, ceil(tokens/256))。
    pub fn paged_prefill(
        env: &OpEnv,
        dt: DType,
        hd: usize,
        hkv: usize,
        hq: usize,
        tokens: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "chunked prefill 仅有 f16 变体(dt={dt:?})");
        assert!(
            env.page == 32,
            "chunked prefill bs32 契约:页 {} 非法(谓词 paged_prefill_ok 应已拦截)",
            env.page
        );
        let name = match hd {
            128 => "vllm_chunked_prefill_paged_attn_opt_f16_hd128",
            256 => "vllm_chunked_prefill_paged_attn_opt_f16_hd256",
            other => panic!("chunked prefill 仅 hd∈{{128,256}},得 {other}"),
        };
        KernelPick {
            name,
            shape: Shape {
                grid: ((hq / hkv) as u32, hkv as u32, tokens.div_ceil(256) as u32),
                block: (256, 1, 1),
                smem: (64 + 2 * hd * env.page * 2) as u32,
            },
        }
    }

    /// prefill split(flash-decoding;长 ctx 主案 2026-10-02):
    /// grid (Hq/Hkv, Hkv, qchunks × nparts),qchunks = ceil(tokens/64);
    /// block 256(TG=4,64 query/block);smem = 64 tok × hd × 2B × 2(K+V)
    /// = 64KB(hd256;>48KB 走发射器 opt-in 通道)。nparts 由层侧
    /// max_ctx = ctx_base + tokens 推得(≥4 才走 split)。
    pub fn prefill_split(
        env: &OpEnv,
        dt: DType,
        hd: usize,
        hkv: usize,
        hq: usize,
        tokens: usize,
        nparts: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "prefill split 仅有 f16 变体(dt={dt:?})");
        assert!(env.page == 32, "prefill split 页 32 契约,得 {}", env.page);
        // 变体由 models ops 层按 KV 池档位分派(fp8 池必须走 fp8kv 直读:
        // f16 核读 e4m3 字节池 = 字节错位,§二十八埋雷实录);此处名字由
        // 调用点 op 字符串决定(见 resolve 分支,chunked 同款)
        let name = match hd {
            256 => "owl_prefill_split_f16_hd256",
            other => panic!("prefill split 仅 hd256,得 {other}"),
        };
        let qchunks = tokens.div_ceil(64);
        KernelPick {
            name,
            shape: Shape {
                grid: ((hq / hkv) as u32, hkv as u32, (qchunks * nparts) as u32),
                block: (256, 1, 1),
                smem: (64 * hd * 2 * 2) as u32,
            },
        }
    }

    /// fp8 KV 直读变体(e4m3 字节池;pick 名 = owl_prefill_split_fp8kv_hd256;
    /// 几何与 f16 完全同式 —— stride index 同数学,单位 1B)
    pub fn prefill_split_fp8kv(
        env: &OpEnv,
        dt: DType,
        hd: usize,
        hkv: usize,
        hq: usize,
        tokens: usize,
        nparts: usize,
    ) -> KernelPick {
        let mut p = prefill_split(env, dt, hd, hkv, hq, tokens, nparts);
        p.name = "owl_prefill_split_fp8kv_hd256";
        p
    }

    /// kNHD 变体(统一契约 P2;签名/class 同,kernel 名 _knhd;
    /// host stride = page·hkv·hd / hd 由调用侧 op-env 投影,res 元数据不变)
    pub fn prefill_split_knhd(
        env: &OpEnv, dt: DType, hd: usize, hkv: usize, hq: usize, tokens: usize, nparts: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "prefill split knhd 仅有 f16 变体");
        assert!(env.page == 32, "prefill split 页 32 契约,得 {}", env.page);
        let name = match hd {
            256 => "owl_prefill_split_f16_knhd_hd256",
            other => panic!("prefill split knhd 仅 hd256,得 {other}"),
        };
        let qchunks = tokens.div_ceil(64);
        KernelPick {
            name,
            shape: Shape {
                grid: ((hq / hkv) as u32, hkv as u32, (qchunks * nparts) as u32),
                block: (256, 1, 1),
                smem: (64 * hd * 2 * 2) as u32,
            },
        }
    }

    pub fn prefill_split_fp8kv_knhd(
        env: &OpEnv, dt: DType, hd: usize, hkv: usize, hq: usize, tokens: usize, nparts: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "prefill split fp8 knhd 输入 f16(dt={dt:?})");
        assert!(env.page == 32, "prefill split 页 32 契约,得 {}", env.page);
        let name = match hd {
            256 => "owl_prefill_split_fp8kv_knhd_hd256",
            other => panic!("prefill split fp8 knhd 仅 hd256,得 {other}"),
        };
        let qchunks = tokens.div_ceil(64);
        KernelPick {
            name,
            shape: Shape {
                grid: ((hq / hkv) as u32, hkv as u32, (qchunks * nparts) as u32),
                block: (256, 1, 1),
                smem: (64 * hd * 2 * 2) as u32,
            },
        }
    }

    /// prefill split reduce(每 thread 一 (token, head) 合并 nparts;
    /// hd 仅 hd256 核名 —— hd 校验由核名/登记表承担)
    pub fn prefill_split_reduce(dt: DType, tokens: usize, hq: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "prefill split reduce 仅有 f16 变体");
        KernelPick::of::<crate::ops::attn::PrefillSplitReduceF16Hd256>(Shape {
            grid: ((tokens * hq).div_ceil(256) as u32, 1, 1),
            block: (256, 1, 1),
            smem: 0,
        })
    }

    /// naive 逐 token 解码(回退/对照臂;dt 双变体;哨兵 1D)
    pub fn naive_decode(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_naive_decode_attn_f16",
                DType::F32 => "owl_naive_decode_attn_f32",
                DType::BF16 | DType::U32 | DType::U8 => unimplemented!("naive attn 无 U32 变体"),
            },
            shape: SENTINEL_1D,
        }
    }

    /// 输出门融合(out × sigmoid(gate);**仅 f16**;哨兵 1D)
    pub fn gate_mul(dt: DType) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_sigmoid_gate_mul 仅有 f16 变体");
        KernelPick::of::<crate::ops::elems::SigmoidGateMulF16>(SENTINEL_1D)
    }

    /// 非因果块 attention(DFlash2 草稿;自块直读 + 前缀池;v2 签名:
    /// grid (T, Hq) × block (hd);t 自 shapes[0][0],hq/hd 自标量槽 0/2)
    pub fn naive_nc(
        dt: DType,
        fp8kv: bool,
        shapes: &[Vec<usize>],
        scalars: &[i64],
    ) -> KernelPick {
        let t = shapes.first().and_then(|s| s.first().copied()).unwrap_or(1) as u32;
        let hq = scalars.first().copied().unwrap_or(1) as u32;
        let hd = scalars.get(2).copied().unwrap_or(128) as u32;
        // v2 臂(2026-10-13 ③):dpl = hd/32 ∈ {4,8} 时走重写核(1.86×);其余旧核。
        // fp8kv 面 v2 未做 ⇒ 一律旧核。
        let v2 = !fp8kv && matches!(dt, DType::BF16) && matches!(hd, 128 | 256);
        let name = match (dt, fp8kv) {
            (DType::F16, false) => "owl_naive_attn_nc_f16",
            (DType::BF16, false) if v2 => "owl_naive_attn_nc_bf16_v2",
            (DType::BF16, false) => "owl_naive_attn_nc_bf16",
            (DType::F16, true) => "owl_naive_attn_nc_fp8kv_f16",
            (DType::BF16, true) => "owl_naive_attn_nc_fp8kv_bf16",
            (DType::F32, _) | (DType::U32, _) | (DType::U8, _) => {
                unimplemented!("naive_nc 仅 f16/bf16")
            }
        };
        KernelPick {
            name,
            shape: Shape { grid: (t, hq, 1), block: (hd.max(1), 1, 1), smem: 0 },
        }
    }

    /// XQA verify 注意力(foreign 家族臂;host 函数发射,workspace 在 runtime)
    pub fn xqa_verify() -> KernelPick {
        KernelPick::of::<crate::ops::xqa::XqaVerify>(FOREIGN)
    }

    /// XQA 裸 decode 注意力(q=1;foreign 家族臂;第二份实例化 `M_TILESIZE=16`)
    pub fn xqa_decode() -> KernelPick {
        KernelPick::of::<crate::ops::xqa::XqaDecode>(FOREIGN)
    }
    /// XQA prefill 臂(与 verify 同资产;几何 = 单块,住 server emit)
    pub fn xqa_prefill() -> KernelPick {
        KernelPick::of::<crate::ops::xqa::XqaPrefill>(FOREIGN)
    }

    /// FlashInfer paged prefill(foreign 家族臂;plan 缓存 + run 在家族 runtime)
    pub fn fi_prefill(fp8kv: bool) -> KernelPick {
        KernelPick {
            name: if fp8kv {
                crate::family::flashinfer::PREFILL_FI_FP8KV
            } else {
                crate::family::flashinfer::PREFILL_FI
            },
            shape: FOREIGN,
        }
    }

    /// norm_rope 融合:qk-norm(×(1+w)^{w_off})+ rotate-half partial rope
    /// (narrow+norm+rope 三发合一;strided 读 q_raw 的 per-head 半段)。
    /// grid (tokens, heads, 1);block (hd,1,1);smem = hd·4B(行内归约)。
    /// aux = [tokens, heads, hd];w_off/eps/stride/half 走核参数槽(层语义)。
    /// BF16 臂(E5-DF3 同日十四):x/w/out bf16,cos/sin 表 f16 共享。
    pub fn norm_rope(dt: DType, tokens: usize, heads: usize, hd: usize) -> KernelPick {
        let name = match dt {
            DType::F16 => "owl_norm_rope_f16",
            DType::BF16 => "owl_norm_rope_bf16",
            other => panic!("owl_norm_rope 无 {other:?} 变体"),
        };
        KernelPick {
            name,
            shape: Shape {
                grid: (tokens as u32, heads as u32, 1),
                block: (hd as u32, 1, 1),
                smem: (hd * 4) as u32,
            },
        }
    }

    /// qkv norm+rope+KV 插入(Wave-2 头号;minimax_m3 (token,head-slot)
    /// 结构 port,适配 gated 布局与 classic cache 寻址):
    /// q 头 → q_out(norm+rope);k 头 → norm+rope → key_cache 散写;
    /// 同块捎带 v → value_cache。替 norm_rope×2 + K0 三发。
    /// grid (T, Hq+Hkv, 1);block (hd,1,1);smem = hd·4B。
    /// aux = [tokens, hq, hkv, hd, half];page/hkv/half 为核运行参数(寻址)。
    pub fn qkv_norm_rope_insert(
        dt: DType,
        tokens: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        _half: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_qknorm_rope_kv_insert 仅有 f16 变体");
        KernelPick::of::<crate::ops::fused::QknormRopeKvInsertF16>(Shape {
            grid: (tokens as u32, (hq + hkv) as u32, 1),
            block: (hd as u32, 1, 1),
            smem: (hd * 4) as u32,
        })
    }

    /// kNHD 变体(统一契约 P1):q 出/w 写面与 classic 版逐式一致
    pub fn qkv_norm_rope_insert_knhd(
        dt: DType,
        tokens: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        _half: usize,
    ) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_qknorm_rope_kv_insert_knhd 仅有 f16 变体");
        KernelPick::of::<crate::ops::fused::QknormRopeKvInsertF16Knhd>(Shape {
            grid: (tokens as u32, (hq + hkv) as u32, 1),
            block: (hd as u32, 1, 1),
            smem: (hd * 4) as u32,
        })
    }

    pub fn qkv_norm_rope_insert_fp8kv_knhd(
        tokens: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        _half: usize,
    ) -> KernelPick {
        KernelPick::of::<crate::ops::fused::QknormRopeKvInsertF16Fp8Knhd>(Shape {
            grid: (tokens as u32, (hq + hkv) as u32, 1),
            block: (hd as u32, 1, 1),
            smem: (hd * 4) as u32,
        })
    }

    /// B6.3:fp8 e4m3 主池变体(几何逐字同 f16;cache 形参 u8 寻址)
    pub fn qkv_norm_rope_insert_fp8kv(
        tokens: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        _half: usize,
    ) -> KernelPick {
        KernelPick::of::<crate::ops::fused::QknormRopeKvInsertF16Fp8>(Shape {
            grid: (tokens as u32, (hq + hkv) as u32, 1),
            block: (hd as u32, 1, 1),
            smem: (hd * 4) as u32,
        })
    }

    /// kNHD 变体(kv布局统一契约 P1):K/V 同构 [nb,page,hkv,hd],x 形参占位不参与寻址
    pub fn k0_write_knhd(dt: DType, tokens: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "vllm_reshape_and_cache_f16_knhd",
                DType::BF16 => "vllm_reshape_and_cache_bf16_knhd",
                other => unimplemented!("k0_write_knhd 无 {other:?} 变体"),
            },
            shape: Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    pub fn k0_write_fp8_knhd(tokens: usize) -> KernelPick {
        KernelPick::of::<crate::ops::attn::K0WriteFp8Knhd>(Shape {
            grid: (tokens as u32, 1, 1),
            block: (256, 1, 1),
            smem: 0,
        })
    }

    /// paged prefill 的 Hw 占位(谓词用;env 之外不可得时)
    pub fn hw_placeholder() -> Hw {
        Hw { arch: crate::Arch::Sm86 }
    }
}

// ============================================================================
// §4.4 layernorm 融合族(C1;vLLM fused_add_rms_norm port)
// ============================================================================

pub mod ln {
    use super::{DType, KernelPick, Shape};

    /// fused_add_rmsnorm:residual 原地 += mixed;out = rmsnorm(residual)·w。
    /// grid (rows,1,1);block 256;smem 256·4B(行分段归约)。
    /// aux = [rows, n]。**副作用律**:residual 块原地写(conv_upd 同款)。
    /// BF16 臂(E5-DF3 同日十四;DFlash2 草稿层残差流)。
    pub fn fused_add_rmsnorm(dt: DType, rows: usize, _n: usize) -> KernelPick {
        let name = match dt {
            DType::F16 => "owl_fused_add_rmsnorm_f16",
            DType::BF16 => "owl_fused_add_rmsnorm_bf16",
            other => panic!("owl_fused_add_rmsnorm 无 {other:?} 变体"),
        };
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: (256 * 4) as u32 },
        }
    }
}

// ============================================================================
// §4.5 MLP 门控族(C1)
// ============================================================================

pub mod mlp {
    use super::{DType, KernelPick, Shape};

    /// silu_and_mul:out = silu(g) ⊙ u(双输入单输出;f16 half2 向量化,
    /// bf16 标量)。aux = [n](g/u 同形 [·, n])。
    pub fn silu_and_mul(dt: DType, n: usize) -> KernelPick {
        let name = match dt {
            DType::F16 => "owl_silu_and_mul_f16",
            DType::BF16 => "owl_silu_and_mul_bf16",
            other => panic!("owl_silu_and_mul 无 {other:?} 变体(dt={dt:?})"),
        };
        KernelPick {
            name,
            shape: Shape {
                grid: if matches!(dt, DType::F16) { (n.div_ceil(2).div_ceil(256) as u32, 1, 1) } else { (n.div_ceil(256) as u32, 1, 1) },
                block: (256, 1, 1),
                smem: 0,
            },
        }
    }
}

// ============================================================================
// §4.6 视觉塔族(Qwen3.5-VL ViT;2026-10-15 vision-tower 立项)
// ============================================================================

pub mod vis {
    use super::{DType, KernelPick, Shape, SENTINEL_1D};

    /// `layernorm`:x [rows, n] / weight [n] / bias [n] → out [rows, n]。
    /// grid (rows,1,1);block 256;smem 256·4B(行内**两趟**归约:mean → var;
    /// 与 torch.nn.LayerNorm 同序,不做 E[x²]−mean²)。
    /// 标量槽序 = n(i32) / eps(f32)。
    /// 设计:docs/arch/vision-tower-设计与CPU对拍.md §1.2 #5。
    pub fn layernorm(dt: DType, rows: usize) -> KernelPick {
        let name = match dt {
            DType::F32 => "owl_layernorm_f32",
            DType::F16 => "owl_layernorm_f16",
            DType::BF16 => "owl_layernorm_bf16",
            other => panic!("owl_layernorm 无 {other:?} 变体"),
        };
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: (256 * 4) as u32 },
        }
    }

    /// `gelu`:逐元素一元(tanh 近似 / 精确 erf 两变体;哨兵 1D)
    pub fn gelu(dt: DType, tanh: bool) -> KernelPick {
        let name = match (tanh, dt) {
            (true, DType::F32) => "owl_gelu_tanh_f32",
            (true, DType::F16) => "owl_gelu_tanh_f16",
            (true, DType::BF16) => "owl_gelu_tanh_bf16",
            (false, DType::F32) => "owl_gelu_erf_f32",
            (false, DType::F16) => "owl_gelu_erf_f16",
            (false, DType::BF16) => "owl_gelu_erf_bf16",
            (_, other) => panic!("owl_gelu 无 {other:?} 变体"),
        };
        KernelPick { name, shape: SENTINEL_1D }
    }

    /// `bias_add`:x [rows, cols] + bias [cols]。grid (rows,1,1);block 256。
    pub fn bias_add(dt: DType, rows: usize) -> KernelPick {
        let name = match dt {
            DType::F32 => "owl_bias_add_f32",
            DType::F16 => "owl_bias_add_f16",
            DType::BF16 => "owl_bias_add_bf16",
            other => panic!("owl_bias_add 无 {other:?} 变体"),
        };
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// `pos_embed`:4-tap 重采样 [n, dim]。grid (n,1,1);block 256。
    pub fn pos_embed(dt: DType, rows: usize) -> KernelPick {
        let name = match dt {
            DType::F32 => "owl_pos_embed_bilinear_f32",
            DType::F16 => "owl_pos_embed_bilinear_f16",
            other => panic!("owl_pos_embed_bilinear 无 {other:?} 变体(F32/F16 两臂)"),
        };
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// `image_scatter`:VLM 图像 token 注入(out[t] = idx[t] ≥ 0 ? merged[idx[t]] : h[t])。
    /// grid (rows,1,1);block 256(rows 取 **h** 的行数)。
    pub fn image_scatter(dt: DType, rows: usize) -> KernelPick {
        let name = match dt {
            DType::F32 => "owl_image_scatter_f32",
            DType::F16 => "owl_image_scatter_f16",
            other => panic!("owl_image_scatter 无 {other:?} 变体(F32/F16 两臂)"),
        };
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// `attn_full`:非因果 varlen(视觉塔)。
    ///
    /// **两臂(按 dtype + 图数分流;见 `resolve` 的 `vis.attn_full` 臂)**:
    /// - f16 + 单图 ⇒ **FlashInfer ragged 非因果 prefill**(foreign 家族臂;
    ///   grid/smem 由家族 runtime 自算 ⇒ `FOREIGN` 哨兵);
    /// - 其余 ⇒ native(一 block 8 行 × 一头,块内共享 smem K/V tile;
    ///   grid = (ceil(rows/8), heads),smem = 2·TILE(64)·head_dim·sizeof(T))。
    pub fn attn_full(dt: DType, rows: usize, heads: usize, head_dim: usize, n_seq: i64) -> KernelPick {
        // FI ragged 适配器只实例化 hd=64 ⇒ 条件里必须带上(否则静默按 64 维算)
        if dt == DType::F16 && n_seq == 1 && head_dim == 64 {
            return KernelPick {
                name: crate::family::flashinfer::ATTN_RAGGED_F16,
                shape: super::FOREIGN,
            };
        }
        let elem = match dt {
            DType::F32 => 4usize,
            DType::F16 => 2usize,
            other => panic!("owl_vis_attn_full 无 {other:?} 变体(F32/F16 两臂)"),
        };
        let name = match dt {
            DType::F32 => "owl_vis_attn_full_f32",
            DType::F16 => "owl_vis_attn_full_f16",
            other => unreachable!("上面已拒"),
        };
        const TILE: usize = 64;
        let smem = (2 * TILE * head_dim.max(1) * elem) as u32;
        let row_blocks = rows.div_ceil(8).max(1) as u32;
        KernelPick {
            name,
            shape: Shape { grid: (row_blocks, heads.max(1) as u32, 1), block: (256, 1, 1), smem },
        }
    }
}

// ============================================================================
// §4.2 共享逐元素/行核族(narrow/concat/rope/embed;dt 双变体)
// ============================================================================

pub mod elems {
    /// f16→f32 设备 cast(GDN chunked 编排配套;哨兵 1D;节点 dtype = F32 出)
    /// f16↔bf16 铸边界(哨兵 1D;to_bf16 = false → f16→bf16)
    pub fn cast_dir(to_f16: bool) -> KernelPick {
        KernelPick {
            name: if to_f16 { "owl_cast_bf16_f16" } else { "owl_cast_f16_bf16" },
            shape: SENTINEL_1D,
        }
    }

    pub fn cast_f16_f32(dt: DType) -> KernelPick {
        assert!(matches!(dt, DType::F32), "cast_f16_f32 出 f32,得 {dt:?}");
        KernelPick::of::<crate::ops::elems::CastF16F32>(SENTINEL_1D)
    }

    use super::{DType, KernelPick, Shape, SENTINEL_1D};

    fn dt_name(base: &str, dt: DType) -> &'static str {
        match dt {
            DType::F16 => match base {
                "narrow" => "owl_narrow_strided_f16",
                "concat" => "owl_concat_rows_f16",
                "rope" => "owl_rope_half_partial_f16",
                "embed" => "owl_embed_f16",
                _ => unimplemented!(),
            },
            DType::F32 => match base {
                "narrow" => "owl_narrow_strided_f32",
                "concat" => "owl_concat_rows_f32",
                "rope" => "owl_rope_half_partial_f32",
                "embed" => "owl_embed_f32",
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }

    /// 跨步窄切视图(q/gate 切分等;哨兵 1D)
    pub fn narrow(dt: DType) -> KernelPick {
        KernelPick { name: dt_name("narrow", dt), shape: SENTINEL_1D }
    }

    /// 刀3a'(2026-10-04):双权 GEMV 单发(b/a 投影;warp-per-row,
    /// half2 主路 + fp32 累加;decode m=1 专用)。grid (ceil(rows/4), m)
    /// × block 128(4 warp/块 = 4 行/块)。
    pub fn gemv_dual(dt: DType, rows_b: usize, rows_a: usize, _cols: usize, tokens: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "gemv_dual 仅有 f16 变体");
        let rows = (rows_b + rows_a) as u32;
        KernelPick::of::<crate::ops::elems::GemvDualF16>(Shape {
            grid: (rows.div_ceil(4), tokens as u32, 1),
            block: (128, 1, 1),
            smem: 0,
        })
    }

    /// 行栈 concat(arity ≤ 8;哨兵 1D)
    pub fn concat(dt: DType) -> KernelPick {
        KernelPick { name: dt_name("concat", dt), shape: SENTINEL_1D }
    }

    /// half 部分旋转(rope;一 token 一 block,block 128)
    pub fn rope(dt: DType, tokens: usize) -> KernelPick {
        KernelPick {
            name: dt_name("rope", dt),
            shape: Shape { grid: (tokens as u32, 1, 1), block: (128, 1, 1), smem: 0 },
        }
    }

    /// 激活 per-token int8 量化(W4A8 喂料手;S1c)。
    ///
    /// 几何 = **一 token 一 block**(grid = rows,block = 256 —— 上游同款:
    /// 行内 absmax 要块内归约);`cols` 走核内 strided 循环兜底。
    ///
    /// ⚠️ **两档变体走两个 op_id**(`elems.quant_int8_row_{f16,bf16}`),不靠
    /// `dt` 分流:本算子的**输出恒 U8**(字节承载),`dt` 载的是输出域,
    /// 输入档由层自己知道 ⇒ 名字由 op_id 定,名字字面量出自声明表
    /// (`ops::elems::QUANT_INT8_ROW_*`,单源)。与 cast 对同律。
    pub fn quant_int8_row(name: &'static str, rows: usize) -> KernelPick {
        KernelPick {
            name,
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// embedding 查表(一 token 一 block,block 1)
    pub fn embed(dt: DType, tokens: usize) -> KernelPick {
        KernelPick {
            name: dt_name("embed", dt),
            shape: Shape { grid: (tokens as u32, 1, 1), block: (1, 1, 1), smem: 0 },
        }
    }
}

// ============================================================================
// §4.3 装载域(ct packed → marlin B 设备重排)
// ============================================================================

pub mod load {
    use super::{KernelPick, Shape};

    /// ct packed → marlin B 重排(U32;**核签名形参即 rows/cols**,无 aux)。
    /// grid = (out/64, (k/8)/2),block 32(契约见 .cu 头注)。
    pub fn ct_repack(rows: usize, cols: usize) -> KernelPick {
        KernelPick::of::<crate::ops::load::CtRepackU32>(Shape {
            grid: ((rows / 64) as u32, (cols / 2) as u32, 1),
            block: (32, 1, 1),
            smem: 0,
        })
    }

    pub const CT_REPACK: super::OpId = super::OpId("load.ct_repack"); // OpId 全路径引用,免 import
}

// ============================================================================
// §5 单测:pick 形状契约(公式逐项对账)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_dec_smem_contract() {
        // kd=128 现役档 = 1032B(旧手抄字面量逐位);kd 变档公式跟随
        let p = gdn::delta_dec(DType::F16, 128, 2, 16, 128);
        assert_eq!(p.name, "owl_gdn_delta_dec_f16");
        assert_eq!(p.shape.grid, (2, 32, 1));
        assert_eq!(p.shape.block, (64, 1, 1));
        assert_eq!(p.shape.smem, 1032);
        assert_eq!(gdn::delta_dec(DType::F32, 64, 1, 8, 96).shape.smem, (2 * 96 + 2) * 4);
    }

    #[test]
    fn recurrence_smem_contract() {
        let p = gdn::recurrence_varlen_gqa(DType::F16, 128, 16, 128);
        assert_eq!(p.name, "owl_gdn_recurrence_varlen_gqa_f16");
        assert_eq!(p.shape.grid, (16, 16, 1));
        assert_eq!(p.shape.block, (32, 8, 1));
        assert_eq!(p.shape.smem, (4 * 128 + 4) * 4);
    }

    #[test]
    fn row_shapes_and_suffixes() {
        assert_eq!(gdn::l2norm(DType::F32, 7).name, "owl_gdn_l2norm_f32");
        assert_eq!(gdn::l2norm(DType::F32, 7).shape.grid, (7, 1, 1));
        assert_eq!(gdn::norm_act(DType::F16, 4, 2048, 128).shape.grid, (64, 1, 1));
        // N1(2026-10-14):grid = (ceil(d/256), tiles=ceil(tokens/32), batch);
        // tiles 只由 tokens 决定 ⇒ 几何自证(含 0/32/33 边界与 batch 维)
        assert_eq!(gdn::conv_fwd(DType::F16, 6144, 0, 1).shape.grid, (24, 1, 1));
        assert_eq!(gdn::conv_fwd(DType::F16, 6144, 32, 1).shape.grid, (24, 1, 1));
        assert_eq!(gdn::conv_fwd(DType::F16, 6144, 33, 1).shape.grid, (24, 2, 1));
        assert_eq!(gdn::conv_fwd(DType::F16, 2048, 461, 1).shape.grid, (8, 15, 1));
        assert_eq!(gdn::conv_fwd(DType::F16, 6144, 461, 3).shape.grid, (24, 15, 3));
        assert_eq!(ops::sigmoid(DType::F16).name, "owl_sigmoid_f16");
        assert_eq!(ops::sigmoid(DType::F32).name, "owl_sigmoid_f32");
    }

    #[test]
    #[should_panic(expected = "仅有 f16 变体")]
    fn f16_only_kernels_reject_f32() {
        let _ = gdn::conv_fwd(DType::F32, 6144, 128, 1);
    }

    #[test]
    fn attn_paged_decode_pairing_and_smem() {
        use super::attn;
        let env32 = OpEnv { hw: Hw { arch: crate::Arch::Sm86 }, page: 32, head_dim: 256, grp: 6 };
        let env16 = OpEnv { hw: Hw { arch: crate::Arch::Sm86 }, page: 16, head_dim: 256, grp: 6 };
        // 页配对律:wrapper 随页(32→bs32/TG1,16→bs16/TG2)
        assert_eq!(
            attn::paged_decode_v1(&env32, DType::F16, 256, 24, 4, 2).name,
            "vllm_paged_attention_v1_f16_hd256bs32"
        );
        assert_eq!(
            attn::paged_decode_v1(&env16, DType::F16, 256, 24, 4, 2).name,
            "vllm_paged_attention_v1_f16_hd256"
        );
        // smem 契约:上下文项 vs 地板(nb·page·4B vs 2·HD·4B)
        let p = attn::paged_decode_v1(&env32, DType::F16, 256, 24, 4, 2);
        assert_eq!(p.shape.smem, (128 / 32 / 2) * 256 * 4);
        assert_eq!(p.shape.grid, (24, 1, 1));
        assert_eq!(p.shape.block, (128, 1, 1));
    }

    #[test]
    fn attn_predicates_and_prefill_contract() {
        use super::attn;
        assert!(attn::paged_decode_ok(128, 32) && attn::paged_decode_ok(256, 16));
        assert!(!attn::paged_decode_ok(4, 32)); // legacy 小头回退 naive(v1_name 时代行为)
        assert!(!attn::paged_decode_ok(256, 8));
        assert!(attn::paged_prefill_ok(256, 32) && !attn::paged_prefill_ok(256, 16));
        assert!(!attn::paged_prefill_ok(64, 32));
        let env = OpEnv { hw: Hw { arch: crate::Arch::Sm86 }, page: 32, head_dim: 256, grp: 6 };
        let p = attn::paged_prefill(&env, DType::F16, 256, 4, 24, 300);
        assert_eq!(p.name, "vllm_chunked_prefill_paged_attn_opt_f16_hd256");
        assert_eq!(p.shape.grid, (6, 4, 2)); // ceil(300/256)=2
        assert_eq!(p.shape.smem, (64 + 2 * 256 * 32 * 2) as u32);
    }

    #[test]
    fn fused_norm_rope_and_silu_contract() {
        use super::{attn, mlp};
        // q 链(27B 档):grid (1, 24, 1) × block 256,smem 1KB
        let p = attn::norm_rope(DType::F16, 1, 24, 256);
        assert_eq!(p.name, "owl_norm_rope_f16");
        assert_eq!(p.shape.grid, (1, 24, 1));
        assert_eq!(p.shape.block, (256, 1, 1));
        assert_eq!(p.shape.smem, 1024);
        // silu_and_mul:inter 17408 → 34 对 block
        let p = mlp::silu_and_mul(DType::F16, 17408);
        assert_eq!(p.name, "owl_silu_and_mul_f16");
        assert_eq!(p.shape.grid, ((((17408 + 1) / 2 + 255) / 256) as u32, 1, 1));
        assert_eq!(p.shape.block, (256, 1, 1));
    }

    #[test]
    fn elems_and_load_picks() {
        use super::{elems, load};
        assert_eq!(elems::narrow(DType::F16).name, "owl_narrow_strided_f16");
        assert_eq!(elems::concat(DType::F32).name, "owl_concat_rows_f32");
        assert_eq!(elems::rope(DType::F16, 40).shape.grid, (40, 1, 1));
        assert_eq!(elems::rope(DType::F16, 40).shape.block, (128, 1, 1));
        assert_eq!(elems::embed(DType::F32, 5).shape.block, (1, 1, 1));
        let p = load::ct_repack(5120 / 8 * 4, 64);
        assert_eq!(p.name, "owl_ct_repack_u32");
        assert_eq!(p.shape.grid, (5120 / 8 * 4 / 64, 32, 1));
        assert_eq!(p.shape.block, (32, 1, 1));
        assert_eq!(load::CT_REPACK.0, "load.ct_repack");
    }
}

// ============================================================================
// §6 marlin 家族臂(foreign;2026-10-12 Kernel 节点消灭收编)
// ============================================================================

pub mod marlin {
    use super::{DType, KernelPick, FOREIGN};

    /// W4A16 GEMM(四名同 runtime;shapes.len = **父数(out 不算)**:
    /// AWQ 6 父(xs,qw,sc,zs,ws,c_tmp)/ f16·bf16 5 父;bf16 由 dt 路由)
    pub fn w4a16(shapes: &[Vec<usize>], dt: DType) -> KernelPick {
        let name = if shapes.len() == 6 {
            crate::family::marlin::GEMM_W4A16_AWQ
        } else if dt == DType::BF16 {
            crate::family::marlin::GEMM_W4A16_BF16
        } else {
            crate::family::marlin::GEMM_W4A16
        };
        KernelPick { name, shape: FOREIGN }
    }

    /// W4A8 × AWQ GEMM(int8 激活 × kU4 零偏;W4A8 线 2026-10-16)。
    ///
    /// 与 [`w4a16`] 的区别**不在 dtype/父数**而在**语义**(激活已量化 +
    /// per-token 尺度)⇒ 由层侧显式声明新动作(`marlin.w4a8_awq`),
    /// 不在此按形状猜(猜测 = 两臂静默互换面)。
    pub fn w4a8_awq() -> KernelPick {
        KernelPick {
            name: crate::family::marlin::GEMM_W4A8_AWQ,
            shape: FOREIGN,
        }
    }
}
