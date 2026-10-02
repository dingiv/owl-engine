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
    /// 装载域 ct packed(U32;owl_ct_repack_u32)
    U32,
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
    match req.op.0 {
        "gdn.gating_g" => gdn::gating_g(dt),
        "gdn.l2norm" => gdn::l2norm(dt, ax(0)),
        "gdn.conv_upd" => gdn::conv_upd(dt),
        "gdn.delta_dec" => gdn::delta_dec(dt, ax(3), ax(0), ax(1), ax(2)), // aux = [batch, nv, kd, vd]
        "gdn.conv_fwd" => gdn::conv_fwd(dt, ax(0)),
        "gdn.recurrence_varlen_gqa" => gdn::recurrence_varlen_gqa(dt, ax(2), ax(0), ax(1)), // aux = [nv, kd, vd]
        "gdn.norm_act" => gdn::norm_act(dt, ax(0), ax(1), ax(2)), // aux = [rows, value_dim, group_size]
        "ops.sigmoid" => ops::sigmoid(dt),
        "attn.k0_write" => attn::k0_write(ax(0)),
        "attn.k0_dual" => attn::k0_dual(ax(0)),
        "attn.k0_dual_fp8kv" => attn::k0_dual_fp8kv(ax(0)),
        "attn.paged_decode" => attn::paged_decode_v1(req.env, dt, ax(0), ax(1), ax(2), ax(3)), // aux = [hd, hq, hkv, nb]
        "attn.paged_prefill" => attn::paged_prefill(req.env, dt, ax(0), ax(1), ax(2), ax(3)), // aux = [hd, hkv, hq, tokens]
        "attn.prefill_split" => {
            // aux = [hd, hkv, hq, tokens, nparts](ctx_base 走核参数槽,层侧传入)
            let (hd, hkv, hq, tokens, nparts) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::prefill_split(req.env, dt, hd, hkv, hq, tokens, nparts)
        }
        "attn.prefill_split_reduce" => {
            // aux = [tokens, hq]
            attn::prefill_split_reduce(dt, ax(0), ax(1))
        }
        "attn.naive_decode" => attn::naive_decode(dt),
        "attn.gate_mul" => attn::gate_mul(dt),
        "ops.narrow" => elems::narrow(dt),
        "elems.cast_f16_f32" => elems::cast_f16_f32(dt),
        "ops.concat" => elems::concat(dt),
        "ops.rope" => elems::rope(dt, ax(0)),
        "ops.embed" => elems::embed(dt, ax(0)),
        "attn.norm_rope" => attn::norm_rope(dt, ax(0), ax(1), ax(2)), // aux = [tokens, heads, hd]
        "attn.qkv_norm_rope_insert" => {
            let (tokens, hq, hkv, hd, half) = (ax(0), ax(1), ax(2), ax(3), ax(4));
            attn::qkv_norm_rope_insert(dt, tokens, hq, hkv, hd, half) // aux = [tokens, hq, hkv, hd, half]
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
            "driver::resolve(\"{other}\"): 未登记的语义算子 —— 词表住              models::ops::ids,实现住本模块分派表(两侧须同票)"
        ),
    }
}

const fn aux1(a: &[usize]) -> usize {
    a[0]
}


/// 一次拾取的产物:名字 + 发射配置(构造 [`Kernel`](crate::sources) 对应
/// 的发射全部输入;grid 哨兵 (0,0,0) = 解释层按输出元素数自动 1D
/// ceil/256,同登记表约)
#[derive(Clone, Debug)]
pub struct KernelPick {
    pub name: &'static str,
    pub shape: Shape,
}

/// 哨兵 1D 发射形态(逐元素/批量核的公共形态)
pub const SENTINEL_1D: Shape = Shape { grid: (0, 0, 0), block: (256, 1, 1), smem: 0 };

// ============================================================================
// §3 算子族(GDN;F5 批 1-5 核 + 工单 G 批核,2026-09-25 语义)
// ============================================================================

pub mod gdn {
    use super::{DType, KernelPick, Shape, SENTINEL_1D};

    /// gating_g:g = softplus(a_log) + dt_bias 门(批 1;哨兵 1D)
    pub fn gating_g(dt: DType) -> KernelPick {
        KernelPick { name: match dt {
            DType::F16 => "owl_gdn_gating_g_f16",
            DType::F32 => "owl_gdn_gating_g_f32",
            DType::U32 => unimplemented!("GDN 族无 U32 变体"),
        }, shape: SENTINEL_1D }
    }

    /// l2norm:行末维归一(批 2;一 block 一行,eps 走参数槽)
    pub fn l2norm(dt: DType, rows: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_gdn_l2norm_f16",
                DType::F32 => "owl_gdn_l2norm_f32",
                DType::U32 => unimplemented!("GDN 族无 U32 变体"),
            },
            shape: Shape { grid: (rows as u32, 1, 1), ..SENTINEL_1D },
        }
    }

    /// conv_upd:decode 槽更新(批 3;state 原地副作用,单流保序;哨兵 1D)
    pub fn conv_upd(dt: DType) -> KernelPick {
        KernelPick { name: match dt {
            DType::F16 => "owl_gdn_conv_upd_f16",
            DType::F32 => "owl_gdn_conv_upd_f32",
            DType::U32 => unimplemented!("GDN 族无 U32 变体"),
        }, shape: SENTINEL_1D }
    }

    /// delta_dec:单步递推(批 4)。shared 契约 = `(2·kd + 2)·4B`(kd 项
    /// 进公式 —— 曾硬编码 128,非 128 kd 即静默越界;收编后公式单源)。
    /// grid = (ceil(vd/64), batch·nv, 1),block (64,1,1)。
    pub fn delta_dec(dt: DType, vd: usize, batch: usize, nv: usize, kd: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_gdn_delta_dec_f16",
                DType::F32 => "owl_gdn_delta_dec_f32",
                DType::U32 => unimplemented!("GDN 族无 U32 变体"),
            },
            shape: Shape {
                grid: (((vd + 63) / 64) as u32, (batch * nv) as u32, 1),
                block: (64, 1, 1),
                smem: ((2 * kd + 2) * 4) as u32,
            },
        }
    }

    /// conv_fwd:decode 卷前向(工单 G;**仅 f16 变体**,f32 请求 = 编程
    /// 错误,decl 构造期 panic —— 与登记表 `source()` 同纪律)。
    /// grid = (1, ceil(d/256), 1)。
    pub fn conv_fwd(dt: DType, d: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_gdn_conv_fwd 仅有 f16 变体(dt={dt:?})");
        KernelPick {
            name: "owl_gdn_conv_fwd_f16",
            shape: Shape { grid: (1, ((d + 255) / 256) as u32, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// recurrence_varlen_gqa:批前向递推(工单 G;**仅 f16 变体**)。
    /// shared 契约 = `(4·kd + 4)·4B`;grid = (ceil(vd/8), nv, 1),
    /// block (32,8,1) = 8 头一组。
    pub fn recurrence_varlen_gqa(dt: DType, vd: usize, nv: usize, kd: usize) -> KernelPick {
        assert!(
            matches!(dt, DType::F16),
            "owl_gdn_recurrence_varlen_gqa 仅有 f16 变体(dt={dt:?})"
        );
        KernelPick {
            name: "owl_gdn_recurrence_varlen_gqa_f16",
            shape: Shape {
                grid: (((vd + 7) / 8) as u32, nv as u32, 1),
                block: (32, 8, 1),
                smem: ((4 * kd + 4) * 4) as u32,
            },
        }
    }

    /// norm_act:×w(非零中心)+ silu(批 5;一 block 一 (row, group))。
    /// grid = (rows·value_dim/group_size, 1, 1)。
    pub fn norm_act(dt: DType, rows: usize, value_dim: usize, group_size: usize) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_gdn_norm_act_f16",
                DType::F32 => "owl_gdn_norm_act_f32",
                DType::U32 => unimplemented!("GDN 族无 U32 变体"),
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
    use super::{DType, KernelPick, SENTINEL_1D};

    /// sigmoid:逐元素(beta 臂 = sigmoid(b);M-b 同款律;哨兵 1D)
    pub fn sigmoid(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_sigmoid_f16",
                DType::F32 => "owl_sigmoid_f32",
                DType::U32 => unimplemented!("sigmoid 无 U32 变体"),
            },
            shape: SENTINEL_1D,
        }
    }
}

// ============================================================================
// §4.1 attention 族(vLLM port 家族;页配对律的 env 侧宿主)
// ============================================================================

pub mod attn {
    use super::{DType, Hw, KernelPick, OpEnv, Shape, SENTINEL_1D};

    /// K0 批量写池(reshape_and_cache;grid (tokens,1,1),block 256)
    pub fn k0_write(tokens: usize) -> KernelPick {
        KernelPick {
            name: "vllm_reshape_and_cache_f16",
            shape: Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// K0-dual(FlashInfer 配套;classic K/V + kNHD K/V 影子一次发射)
    pub fn k0_dual(tokens: usize) -> KernelPick {
        KernelPick {
            name: "owl_reshape_and_cache_dual_f16",
            shape: Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
    }

    /// K0-dual fp8 KV(K/V 影子写 e4m3;classic f16 照写)
    pub fn k0_dual_fp8kv(tokens: usize) -> KernelPick {
        KernelPick {
            name: "owl_reshape_and_cache_dual_f16_fp8kv",
            shape: Shape { grid: (tokens as u32, 1, 1), block: (256, 1, 1), smem: 0 },
        }
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
                grid: ((hq / hkv) as u32, hkv as u32, ((tokens + 255) / 256) as u32),
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
        let name = match hd {
            256 => "owl_prefill_split_f16_hd256",
            other => panic!("prefill split 仅 hd256,得 {other}"),
        };
        let qchunks = (tokens + 63) / 64;
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
        KernelPick {
            name: "owl_prefill_split_reduce_f16_hd256",
            shape: Shape {
                grid: ((((tokens * hq) + 255) / 256) as u32, 1, 1),
                block: (256, 1, 1),
                smem: 0,
            },
        }
    }

    /// naive 逐 token 解码(回退/对照臂;dt 双变体;哨兵 1D)
    pub fn naive_decode(dt: DType) -> KernelPick {
        KernelPick {
            name: match dt {
                DType::F16 => "owl_naive_decode_attn_f16",
                DType::F32 => "owl_naive_decode_attn_f32",
                DType::U32 => unimplemented!("naive attn 无 U32 变体"),
            },
            shape: SENTINEL_1D,
        }
    }

    /// 输出门融合(out × sigmoid(gate);**仅 f16**;哨兵 1D)
    pub fn gate_mul(dt: DType) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_sigmoid_gate_mul 仅有 f16 变体");
        KernelPick { name: "owl_sigmoid_gate_mul_f16", shape: SENTINEL_1D }
    }

    /// norm_rope 融合:qk-norm(×(1+w)^{w_off})+ rotate-half partial rope
    /// (narrow+norm+rope 三发合一;strided 读 q_raw 的 per-head 半段)。
    /// grid (tokens, heads, 1);block (hd,1,1);smem = hd·4B(行内归约)。
    /// aux = [tokens, heads, hd];w_off/eps/stride/half 走核参数槽(层语义)。
    pub fn norm_rope(dt: DType, tokens: usize, heads: usize, hd: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_norm_rope 仅有 f16 变体(dt={dt:?})");
        KernelPick {
            name: "owl_norm_rope_f16",
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
        KernelPick {
            name: "owl_qknorm_rope_kv_insert_f16",
            shape: Shape {
                grid: (tokens as u32, (hq + hkv) as u32, 1),
                block: (hd as u32, 1, 1),
                smem: (hd * 4) as u32,
            },
        }
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
    pub fn fused_add_rmsnorm(dt: DType, rows: usize, _n: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_fused_add_rmsnorm 仅有 f16 变体");
        KernelPick {
            name: "owl_fused_add_rmsnorm_f16",
            shape: Shape { grid: (rows as u32, 1, 1), block: (256, 1, 1), smem: (256 * 4) as u32 },
        }
    }
}

// ============================================================================
// §4.5 MLP 门控族(C1)
// ============================================================================

pub mod mlp {
    use super::{DType, KernelPick, Shape};

    /// silu_and_mul:out = silu(g) ⊙ u(双输入单输出;half2 向量化)。
    /// grid = ceil(n/512);block 256(每线程 2 元素)。
    /// aux = [n](g/u 同形 [·, n])。
    pub fn silu_and_mul(dt: DType, n: usize) -> KernelPick {
        assert!(matches!(dt, DType::F16), "owl_silu_and_mul 仅有 f16 变体(dt={dt:?})");
        KernelPick {
            name: "owl_silu_and_mul_f16",
            shape: Shape {
                grid: ((((n + 1) / 2 + 255) / 256) as u32, 1, 1),
                block: (256, 1, 1),
                smem: 0,
            },
        }
    }
}

// ============================================================================
// §4.2 共享逐元素/行核族(narrow/concat/rope/embed;dt 双变体)
// ============================================================================

pub mod elems {
    /// f16→f32 设备 cast(GDN chunked 编排配套;哨兵 1D;节点 dtype = F32 出)
    pub fn cast_f16_f32(dt: DType) -> KernelPick {
        assert!(matches!(dt, DType::F32), "cast_f16_f32 出 f32,得 {dt:?}");
        KernelPick { name: "owl_cast_f16_f32", shape: SENTINEL_1D }
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
        KernelPick {
            name: "owl_ct_repack_u32",
            shape: Shape {
                grid: ((rows / 64) as u32, (cols / 2) as u32, 1),
                block: (32, 1, 1),
                smem: 0,
            },
        }
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
        assert_eq!(gdn::conv_fwd(DType::F16, 6144).shape.grid, (1, 24, 1));
        assert_eq!(ops::sigmoid(DType::F16).name, "owl_sigmoid_f16");
        assert_eq!(ops::sigmoid(DType::F32).name, "owl_sigmoid_f32");
    }

    #[test]
    #[should_panic(expected = "仅有 f16 变体")]
    fn f16_only_kernels_reject_f32() {
        let _ = gdn::conv_fwd(DType::F32, 6144);
    }

    #[test]
    fn attn_paged_decode_pairing_and_smem() {
        use super::attn;
        let env32 = OpEnv { hw: Hw { arch: crate::Arch::Sm86 }, page: 32 };
        let env16 = OpEnv { hw: Hw { arch: crate::Arch::Sm86 }, page: 16 };
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
        assert_eq!(p.shape.smem, (2 * 32 * 4).max((128 / 32 / 2) * 256 * 4));
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
        let env = OpEnv { hw: Hw { arch: crate::Arch::Sm86 }, page: 32 };
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
