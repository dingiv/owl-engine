//! 设备状态块池(StatePool;2026-10-01 engine.rs 拆分立)。
//!
//! 引擎全部设备侧状态块的**治理对象**:KV paged 池(kvs)/ GDN 状态格
//! (gdns)/ GDN 快照池(snaps)/ 持久块表(bt)+ 池几何(page/nb/
//! paged/x/dims)。职责 = 分配([`StatePool::alloc`])+ 叶子句柄发放
//! (kv_caches/gdn_caches/bt_leaf)+ 状态治理(reset_gdn/capture_snap/
//! restore_snap/write_bt)。块账房(host 侧记账)不在此 —— 那是
//! BlockManager(blocks.rs);本池只管设备缓冲与池内快照 LRU。
//!
//! 布局策略表驱动(REQ-HW-01):KV 布局由 `kv_paged_policy(dtype)` 分发
//! —— Some = paged classic(kc [nb,Hkv,hd/x,page,x] / vc [nb,Hkv,hd,page];
//! 恒等块表:物理块 = 逻辑块,物理槽 = pos);None = legacy token-major
//! 直排(page = 容量,nb = 1)。
//!
//! dtype 定盘(F5 整模切换):KV cache 随 spec.dtype;GDN state 恒 f32
//! (三方先例 + 旧世界律);slots/kv_lens/bt f32(契约 5)。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops;
use owl_models::module::kv_paged_policy;
use owl_models::TensorOps;

type Result<T> = std::result::Result<T, ModelError>;

pub(crate) type BlockN = (Bytes, usize); // (块句柄, 元素数;重置分块写用)



/// GDN 状态格容量(**与会话数绑定,与 max_seq_tokens 解耦**)。
/// 格语义 = 每会话一格(SessionTable 分配/释放);按位分配是历史包袱:
/// s=4096 时 rec(1 MiB/格/层 × 18 层)将达 72 GB,而真实需求 = 格数。
/// 8 = 覆盖需求并发上限(§二 并发 ≤8);会话 close 即释放格号。
/// 槽数 = EngineKnobs.gdn_slots(显式传入;原 OWL_GDN_SLOTS,缺省 8)。

/// GDN 快照槽(E2c):key = 链尾物理块 id(内容身份);bufs = 逐层逐块
/// 行宽副本(与 gdns 同序同构,行宽 = elems / GDN_SLOTS)。池静态
/// 预分配,驱逐 = 换 key 覆盖写(零 alloc/free 往返)。
struct GdnSnapSlot {
    key: Option<u32>,
    last_use: u64,
    bufs: Vec<BlockN>,
}

pub(crate) struct KvBlocks {
    pub(crate) k_cache: BlockN,
    pub(crate) v_cache: BlockN,
}

struct GdnBlocks {
    conv_q: BlockN,
    conv_k: BlockN,
    conv_v: BlockN,
    rec: BlockN,
}

/// prefill 块构造所需维度账(run 期从 spec 提取;Copy 免 Clone 传播)
#[derive(Clone, Copy)]
pub(crate) struct ModelDims {
    pub(crate) hq: usize,
    pub(crate) hkv: usize,
    pub(crate) hd: usize,
    pub(crate) nk: usize,
    pub(crate) hk: usize,
    pub(crate) nv: usize,
    pub(crate) hv: usize,
    pub(crate) hidden: usize,
    pub(crate) dtype: Dtype,
}

/// GDN 状态叶子四件套(slots 由调用方按步注入;见 gdn_caches)
pub(crate) struct GdnLeaf {
    pub(crate) conv_q: TensorOps,
    pub(crate) conv_k: TensorOps,
    pub(crate) conv_v: TensorOps,
    pub(crate) rec: TensorOps,
}

/// 设备状态块池(KV paged 池 / GDN 状态格 / 快照池 / 持久块表)。
/// 治理对象(charter A1 味):S2 图档与执行器只经方法面触碰状态块,
/// 不直摸块句柄。
/// 池装配计划(engine run 从 knobs/spec_mode 派生;显式依赖律的
/// 参数对象 —— 原散列布尔/数值参数群收编:量化 = KvQuant 枚举,
/// 投机形态 = SpecMode 枚举,互斥语义由派生保证,调用点零自洽负担)
#[derive(Clone, Debug)]
pub(crate) struct PoolPlan {
    /// 单会话容量(= max_seq_tokens)
    pub seq_tokens: usize,
    /// 池容量(治理/钉值定谳后的 token 数)
    pub pool_tokens: usize,
    /// 主池 KV 量化
    pub kv: owl_models::env::KvQuant,
    /// FlashInfer K/V 影子池量化(None = 无 FI 面)
    pub fi: Option<owl_models::env::KvQuant>,
    /// 草稿池量化
    pub draft_kv: owl_models::env::KvQuant,
    /// 投机形态(快照/MTP 链/草稿池三件的唯一事实来源)
    pub spec_mode: crate::running::SpecMode,
    /// GDN 状态格容量
    pub gdn_slots: usize,
    /// spec 快照池深度
    pub snap_max: usize,
    /// VRAM 预算目标占比(0 = 治理禁用)
    pub vram_target: f64,
    /// VRAM 预留 MiB
    pub vram_reserve_mb: u64,
    /// GDN 画像 dump 全槽
    pub dump_all: bool,
}

pub(crate) struct StatePool {
    /// GDN 状态格容量(knobs.gdn_slots,boot 定谳;原自由函数 gdn_slots())
    pub(crate) gdn_slots: usize,
    /// GDN 画像 dump 全槽(knobs.gdn_dump_all)
    pub(crate) dump_all: bool,
    pub(crate) kvs: Vec<KvBlocks>,
    /// FlashInfer K/V 影子池(owl_reshape_and_cache_dual_f16(_fp8kv) 写;
    /// OWL_FLASHINFER=1 时分配,否则空)
    pub(crate) k_fis: Vec<BlockN>,
    pub(crate) v_fis: Vec<BlockN>,
    /// 影子量化档(None = f16 双宽;Fp8E4M3 = e4m3 单宽)
    pub(crate) fi_quant: Option<owl_models::env::KvQuant>,
    /// B6.2:主 KV 池 fp8 e4m3 承载(true = 池块 U32 字节承载 1B/elem;
    /// 读核走 *_fp8 变体,写核 K0 转换写)
    pub(crate) kv_fp8: bool,
    pub(crate) dflash_fp8: bool,
    gdns: Vec<GdnBlocks>,
    snaps: Vec<GdnSnapSlot>,
    snap_tick: u64,
    bt: Bytes,
    /// v2 分页 decode scratch(E-decode 2026-10-04;paged 时恒分配,
    /// None = legacy/无页策略;引擎注入 ctx → 层走 v2 + LSE 归并)
    pub(crate) attn_v2: Option<owl_models::module::AttnV2Scratch>,
    /// spec 快照(E5-M2:投机轮回滚缓冲;OWL_SPEC_DEPTH>0 时分配,
    /// 内容 = 轮首 GDN 态 state@B-1;restore 后仍有效,全接受后失效重拍)
    pub(crate) spec_snap: Option<GdnSnapSlot>,
    /// MTP 草稿链 KV 池(E5-M2b;独立页池,1 层;mtp 模式分配)
    pub(crate) mtp_kvs: Option<KvBlocks>,
    /// DFlash2 草稿 KV 池(E5-DF2;独立页池,5 层草稿几何 8/128)
    pub(crate) dflash_kvs: Option<Vec<KvBlocks>>,
    /// MTP 链持久块表槽(E5-M5 propose 图烘焙;[1, nb] f32,每轮重写)
    pub(crate) bt_mtp: Option<Bytes>,
    /// paged 池几何(页;legacy 模式 page = 容量,nb = 1)
    pub(crate) page: usize,
    pub(crate) nb: usize,
    pub(crate) paged: bool,
    x: usize,
    /// 池几何输入(spec 提取;prefill 叶子构造/执行器 dtype 分支用)
    pub(crate) dims: ModelDims,
}

impl StatePool {
    /// 分配全部状态块(零初始化;GDN 段每 turn 开始按格重置)。
    /// `pool_tokens` = 块池 token 口径容量(E2b:默认 2 × 单会话容量,
    /// OWL_POOL_TOKENS 可覆写;调用方读 env,本函数只管取整到页)。
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn alloc<D: DeviceClient>(
        face: &mut D,
        dims: ModelDims,
        layer_types: &[bool],
        plan: &PoolPlan,
    ) -> Result<StatePool> {
        let seq_tokens = plan.seq_tokens;
        let pool_tokens = plan.pool_tokens;
        // 布尔和稀泥废除(2026-10-10):量化 = KvQuant 枚举,投机形态 =
        // SpecMode 枚举(快照/MTP 链/草稿池三件由 mode 派生,互斥不再靠
        // 调用点自洽)
        let kv_fp8 = plan.kv == owl_models::env::KvQuant::Fp8E4M3;
        let dflash_fp8 = plan.draft_kv == owl_models::env::KvQuant::Fp8E4M3;
        let spec = plan.spec_mode != crate::running::SpecMode::Off;
        let mtp = plan.spec_mode == crate::running::SpecMode::Mtp;
        let dflash = plan.spec_mode == crate::running::SpecMode::DFlash2;
        let gdn_slots = plan.gdn_slots;
        let snap_max = plan.snap_max;
        let dump_all = plan.dump_all;
        let t = std::time::Instant::now();
        let n_full = layer_types.iter().filter(|&&f| f).count();
        let n_gdn = layer_types.len() - n_full;
        eprintln!("[boot] StatePool::alloc 进入(n_full={n_full} kv_fp8={kv_fp8})");

        // B6.5 显存硬预算治理(2026-10-10):池容量 =
        // min(手工上限, (target×total − live − reserve)/每 token 字节)。
        // target 默认 0.97;reserve 默认 1024MB(盖 CUDA ctx ~430MB + 逐 turn 瞬态 +
        // async 池惰性保留;实测 300MB 在连续 turn 下瞬态 OOM)
        // + 捕获瞬态 + 非追踪惰性保留。OWL_VRAM_TARGET/OWL_VRAM_RESERVE_MB
        // 可调;TOTAL 未落账(=0)或 OWL_VRAM_TARGET=0 → 治理禁用。
        // FIXME: 大量的硬编码
        let per_tok_bytes = (n_full * 2 * dims.hkv * dims.hd) * if kv_fp8 { 1 } else { 2 }
            + if dflash { 5 * 2 * 8 * 128 * 2 } else { 0 }; // 草稿池按 token 线性(页取整后略同)
        // 治理器之后的固定分配(GDN 格/快照 f32 同式;图 slab + ctx + 杂项 flat)
        let gdn_slot_elems = dims.nk * dims.hk * 3 + dims.nv * dims.hv * 3 + dims.nv * dims.hk * dims.hv;
        let fixed_after = n_gdn * gdn_slots * gdn_slot_elems * 4
            + snap_max * n_gdn * gdn_slot_elems * 4
            + (400 << 20); // 图 slab + CUDA ctx + warmup 瞬态 + 杂项
        let vram_reserve = plan.vram_reserve_mb << 20;
        let vram_total = owl_shared::vram::total_bytes();
        let pool_cap = if plan.vram_target > 0.0 && plan.vram_target < 1.0 && vram_total > 0 {
            let budget = (vram_total as f64 * plan.vram_target) as i64
                - vram_reserve as i64
                - fixed_after as i64
                - owl_shared::vram::live_bytes() as i64;
            let cap = (budget.max(0) as u64 / per_tok_bytes.max(1) as u64) as usize;
            eprintln!(
                "[boot] vram 治理:total={:.1}G free={:.2}G live={:.1}G target={:.0}% reserve={}MB fixed_after={:.2}G budget={:.2}G per_tok={}B → 池上限 {} tok(manual={})",
                vram_total as f64 / 1073741824.0,
                owl_shared::vram::free_bytes() as f64 / 1073741824.0,
                owl_shared::vram::live_bytes() as f64 / 1073741824.0,
                plan.vram_target * 100.0,
                vram_reserve >> 20,
                fixed_after as f64 / 1073741824.0,
                budget as f64 / 1073741824.0,
                per_tok_bytes,
                cap,
                pool_tokens
            );
            Some(cap)
        } else {
            None
        };
        let pool_tokens = match pool_cap {
            Some(cap) => pool_tokens.min(cap),
            None => pool_tokens,
        };

        let pol = kv_paged_policy(dims.dtype);
        let (page, x, nb, paged) = match &pol {
            Some(p) => (p.page, p.x, (seq_tokens + p.page - 1) / p.page, true),
            None => (seq_tokens, 1usize, 1usize, false),
        };
        // E2b 块池容量:pool_tokens 向上取整到页;不低于单会话容量
        let nb = paged.then(|| pool_tokens.div_ceil(page)).unwrap_or(nb).max(nb);

        let mut kvs: Vec<KvBlocks> = Vec::new();
        let mut k_fis: Vec<BlockN> = Vec::new();
        let mut v_fis: Vec<BlockN> = Vec::new();
        for _ in 0..n_full {
            // B6.2:fp8 池 = U32 字节承载 1B/elem(元素数不变,块大小减半);
            // 读核 *_fp8 变体读入即转 half,写核 K0 转换写
            let (k_elems, k_dt) = if kv_fp8 {
                (nb * dims.hkv * dims.hd * page / 4, Dtype::U32)
            } else {
                (nb * dims.hkv * dims.hd * page, dims.dtype)
            };
            let (v_elems, v_dt) = if kv_fp8 {
                (nb * dims.hkv * dims.hd * page / 4, Dtype::U32)
            } else {
                (nb * dims.hkv * dims.hd * page, dims.dtype)
            };
            let k_cache = zero_block_dt(face, k_elems, k_dt).await?;
            let v_cache = zero_block_dt(face, v_elems, v_dt).await?;
            kvs.push(KvBlocks { k_cache, v_cache });
        }
        // MTP 草稿链 KV(E5-M2b):独立页池(1 层,几何 = full 层同款,
        // 容量同 pool_tokens);不接前缀缓存 —— mtp KV 依赖 hidden,
        // 不可 token 哈希。仅 paged(链吃块表)。
        let mtp_kvs = if mtp && paged {
            Some(KvBlocks {
                k_cache: zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?,
                v_cache: zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?,
            })
        } else {
            None
        };
        // DFlash2 草稿池(E5-DF2):5 层 × 草稿几何(8 KV 头 × hd128;
        // z-lab 检查点家族定形),容量同 pool_tokens。仅 paged。
        // dtype = BF16(E5-DF3 同日十四;草稿路径 BF16 全程,独立于目标池
        // dims.dtype —— 同 2B/elem,几何 page/x 不变)
        let dflash_kvs = if dflash && paged {
            let (dkv, dhd, dlayers) = (8usize, 128usize, 5usize);
            // B6 偷显存:草稿池 e4m3(OWL_DFLASH_KV_FP8;1B/elem,U32 字节承载)
            let (df_elems, df_dt) = if dflash_fp8 {
                (nb * dkv * dhd * page / 4, Dtype::U32)
            } else {
                (nb * dkv * dhd * page, Dtype::BF16)
            };
            let mut ks = Vec::with_capacity(dlayers);
            for _ in 0..dlayers {
                ks.push(KvBlocks {
                    k_cache: zero_block_dt(face, df_elems, df_dt).await?,
                    v_cache: zero_block_dt(face, df_elems, df_dt).await?,
                });
            }
            Some(ks)
        } else {
            None
        };
        if let Some(kq) = plan.fi {
            // 影子池 [nb, page, Hkv, hd]:f16 = 2B/elem;fp8 = 1B/elem
            // (U32 块承载字节:elems = bytes/4,page 32 因子保证 4 整除)
            match kq {
                owl_models::env::KvQuant::None => {
                    for _ in 0..n_full {
                        k_fis.push(zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?);
                        v_fis.push(zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?);
                    }
                }
                owl_models::env::KvQuant::Fp8E4M3 => {
                    for _ in 0..n_full {
                        let n_u32 = nb * dims.hkv * dims.hd * page / 4;
                        k_fis.push(zero_block_dt(face, n_u32, Dtype::U32).await?);
                        v_fis.push(zero_block_dt(face, n_u32, Dtype::U32).await?);
                    }
                }
            }
        }
        // 块表持久块(E2b):内容 = 活跃会话块链,turn 切换/增长时重写
        // (write_bt;图烘焙 bt 指针,指针稳定图不重捕);legacy = 哑表
        let bt = if paged {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?
        } else {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1]).step(), face).await?
        };
        // MTP 链持久块表槽(E5-M5;propose 图烘焙;mtp 模式)
        let bt_mtp = if mtp && paged {
            Some(eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?)
        } else {
            None
        };
        // v2 decode scratch(E-decode):exp_sums/max_logits [1,hq,nparts]
        // f32 + tmp_out [hq·nparts·hd] f16;nparts = ceil(nb·page/512)
        // (PARTITION=512;boot 一次性持久块,不进竞技场收割面,图安全同 bt)
        let attn_v2 = if paged {
            let nparts = (nb * page + 511) / 512;
            let es = zero_block_dt(face, dims.hq * nparts, Dtype::F32).await?;
            let ml = zero_block_dt(face, dims.hq * nparts, Dtype::F32).await?;
            let to = zero_block_dt(face, dims.hq * nparts * dims.hd, dims.dtype).await?;
            Some(owl_models::module::AttnV2Scratch {
                exp_sums: block_leaf(&es.0, vec![1, dims.hq * nparts]),
                max_logits: block_leaf(&ml.0, vec![1, dims.hq * nparts]),
                tmp_out: block_leaf_dt(&to.0, vec![dims.hq * nparts * dims.hd], dims.dtype),
                nparts,
            })
        } else {
            None
        };
        let mut gdns: Vec<GdnBlocks> = Vec::new();
        for _ in 0..n_gdn {
            let conv_q = zero_block(face, gdn_slots * dims.nk * dims.hk * 3).await?;
            let conv_k = zero_block(face, gdn_slots * dims.nv * dims.hv * 3).await?;
            let conv_v = zero_block(face, gdn_slots * dims.nv * dims.hv * 3).await?;
            let rec = zero_block(face, gdn_slots * dims.nv * dims.hk * dims.hv).await?;
            gdns.push(GdnBlocks { conv_q, conv_k, conv_v, rec });
        }
        // GDN 快照池(E2c):行宽副本 × SNAP_MAX 份;仅 paged 形态
        let mut snaps: Vec<GdnSnapSlot> = Vec::new();
        if paged {
            for _ in 0..snap_max {
                let mut bufs: Vec<BlockN> = Vec::new();
                for _ in 0..n_gdn {
                    bufs.push(zero_block(face, dims.nk * dims.hk * 3).await?);
                    bufs.push(zero_block(face, dims.nv * dims.hv * 3).await?);
                    bufs.push(zero_block(face, dims.nv * dims.hv * 3).await?);
                    bufs.push(zero_block(face, dims.nv * dims.hk * dims.hv).await?);
                }
                snaps.push(GdnSnapSlot { key: None, last_use: 0, bufs });
            }
        }
        eprintln!(
            "[boot] 状态块分配 {:.2}s(kv f16 ×{} / gdn ×{},会话容量 {},池 {} 页 × {page} tok,页表 nb = {nb},dflash = {df})",
            t.elapsed().as_secs_f32(),
            kvs.len(),
            gdns.len(),
            seq_tokens,
            nb = nb,
            page = page,
            df = dflash_kvs.is_some()
        );
        // spec 快照(E5-M2):专用缓冲(独立于 E2c 前缀快照池,免 LRU 逐出)
        let spec_snap = if spec && paged {
            let mut bufs = Vec::new();
            for _ in 0..n_gdn {
                bufs.push(zero_block(face, dims.nk * dims.hk * 3).await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3).await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3).await?);
                bufs.push(zero_block(face, dims.nv * dims.hk * dims.hv).await?);
            }
            Some(GdnSnapSlot { key: None, last_use: 0, bufs })
        } else {
            None
        };
        Ok(StatePool { gdn_slots, dump_all, kvs, k_fis, v_fis, fi_quant: plan.fi,
            kv_fp8, dflash_fp8, gdns, snaps, snap_tick: 0, bt, bt_mtp, spec_snap, mtp_kvs, dflash_kvs, attn_v2, page, nb, paged, x, dims })
    }

    /// MTP 链块表叶子(propose 图烘焙;None = 未启用)
    pub(crate) fn bt_mtp_leaf(&self) -> Option<TensorOps> {
        self.bt_mtp
            .as_ref()
            .map(|b| TensorOps::of_block(b.id, Dtype::F32, vec![1, self.nb]))
    }

    /// MTP 链块表槽重写(每轮 propose 前;内容 = 会话 mtp 链)
    pub(crate) async fn write_bt_mtp<D: DeviceClient>(
        &self,
        face: &mut D,
        table: &[u32],
    ) -> Result<()> {
        let b = self.bt_mtp.as_ref().expect("bt_mtp 未分配(mtp 模式)");
        let mut v = vec![0f32; self.nb];
        for (i, x) in table.iter().enumerate() {
            v[i] = *x as f32;
        }
        face.write_block_f32(b, 0, &v).await
    }

    /// spec 快照缓冲句柄组(E5-M5 fold 图烘焙:状态 args = 快照 buf
    /// 就地,slot 恒 0;fold 后 restore 回写会话格)。序 = 层 × 4
    /// (conv_q/k/v/rec),与 capture/restore 同构。
    pub(crate) fn spec_snap_bufs(&self) -> Vec<BlockN> {
        self.spec_snap
            .as_ref()
            .map(|s| s.bufs.clone())
            .unwrap_or_default()
    }

    /// GDN 层数(fold 记录槽计数用)
    pub(crate) fn gdn_count(&self) -> usize {
        self.gdns.len()
    }

    /// DFlash2 草稿 KV 叶子组(dflash 模式;5 层 × (k, v);None = 未启用)
    // ── DFlash2 草稿池寻址权威(唯一出口;slot≥1 案配套,2026-10-10)──
    // 草稿池 = **线性逻辑寻址**:NC 核签名无块表参数,前缀读 [0, kv_len)
    // 从池基址直排;encode 写槽 = 逻辑位。写者(exec encode)与读者
    // (propose 图/NC)禁止各自推槽 —— 两套语言曾分叉(bt 物理写 vs 线性
    // 读),首 session 巧合重合,后续 session 全崩(slot≥1 案)。
    // 并发多 turn(B5 batch)共享线性空间会互踩 → B5 需 per-session 分区。

    /// encode 行 i(基址 base)的草稿 K/V 槽(K0 写入位)
    pub(crate) fn draft_encode_slot(base: usize, i: usize) -> f32 {
        (base + i) as f32
    }

    /// propose 噪声块自槽(块内行 i;K/V 核输入直读,不入池)
    pub(crate) fn draft_self_slot(pos: usize, i: usize) -> f32 {
        (pos + i) as f32
    }

    /// propose 前缀读窗:逻辑位 p(NC 线性直排)
    pub(crate) fn draft_prefix_slot(p: usize) -> f32 {
        p as f32
    }

    /// propose 读窗长(fp = 下轮 fed 位)
    pub(crate) fn draft_kv_len(fp: usize) -> f32 {
        (fp + 8) as f32
    }

    pub(crate) fn dflash_kv_leaves(&self) -> Option<Vec<(TensorOps, TensorOps)>> {
        let ks = self.dflash_kvs.as_ref()?;
        let (dkv, dhd) = (8usize, 128usize);
        // B6:fp8 草稿池 = U32 扁平字节账(主池同款)
        if self.dflash_fp8 {
            let shape = vec![self.nb * dkv * dhd * self.page / 4];
            return Some(ks
                .iter()
                .map(|kb| {
                    (
                        block_leaf_dt(&kb.k_cache.0, shape.clone(), Dtype::U32),
                        block_leaf_dt(&kb.v_cache.0, shape.clone(), Dtype::U32),
                    )
                })
                .collect());
        }
        Some(ks
            .iter()
            .map(|kb| {
                (
                    block_leaf_dt(
                        &kb.k_cache.0,
                        vec![self.nb, dkv, dhd / self.x, self.page, self.x],
                        self.dims.dtype,
                    ),
                    block_leaf_dt(
                        &kb.v_cache.0,
                        vec![self.nb, dkv, dhd, self.page],
                        self.dims.dtype,
                    ),
                )
            })
            .collect())
    }

    /// MTP 草稿链 KV 叶子对(mtp 模式;None = 未启用)
    pub(crate) fn mtp_kv_leaf(&self) -> Option<(TensorOps, TensorOps)> {
        let mk = self.mtp_kvs.as_ref()?;
        Some((
            block_leaf_dt(&mk.k_cache.0, self.k_shape(), self.dims.dtype),
            block_leaf_dt(&mk.v_cache.0, self.v_shape(), self.dims.dtype),
        ))
    }

    /// v2 scratch 叶子(None = legacy;引擎按步注入 ctx.attn_v2)
    pub(crate) fn attn_v2_scratch(&self) -> Option<owl_models::module::AttnV2Scratch> {
        self.attn_v2.clone()
    }

    /// KV 池形状(模式相关;与层分派同源策略)
    pub(crate) fn k_shape(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb, d.hkv, d.hd / self.x, self.page, self.x]
        } else {
            vec![self.nb, d.hkv, d.hd] // legacy:nb=1,page=容量
        }
    }
    pub(crate) fn v_shape(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb, d.hkv, d.hd, self.page]
        } else {
            vec![self.nb, d.hkv, d.hd]
        }
    }

    /// 逐层 KV cache 叶子句柄((k, v);slots/kv_lens/block_tables 由
    /// 调用方按步注入 —— 图闭包注输入槽,prefill 注 from_host 标量行)
    pub(crate) fn kv_caches(&self) -> Vec<(TensorOps, TensorOps)> {
        // B6.2:fp8 池 = U32 扁平字节账叶(指针面;读核按元素序自寻址,
        // FI 影子池 fp8 档同款)
        if self.kv_fp8 {
            let shape = vec![self.nb * self.dims.hkv * self.dims.hd * self.page / 4];
            return self
                .kvs
                .iter()
                .map(|b| {
                    (
                        block_leaf_dt(&b.k_cache.0, shape.clone(), Dtype::U32),
                        block_leaf_dt(&b.v_cache.0, shape.clone(), Dtype::U32),
                    )
                })
                .collect();
        }
        let k_shape = self.k_shape();
        let v_shape = self.v_shape();
        self.kvs
            .iter()
            .map(|b| {
                (
                    block_leaf_dt(&b.k_cache.0, k_shape.clone(), self.dims.dtype),
                    block_leaf_dt(&b.v_cache.0, v_shape.clone(), self.dims.dtype),
                )
            })
            .collect()
    }

    /// FlashInfer K/V 影子叶子(kNHD;OWL_FLASHINFER=1 时非空)。
    /// fp8 档:块为 U32 字节承载,leaf = 扁平 [bytes/4](FI 只吃指针,
    /// leaf shape 仅为账长;f16 档:真实几何 [nb, page, Hkv, hd])
    pub(crate) fn kv_fi_leaves(&self) -> Vec<(TensorOps, TensorOps)> {
        match self.fi_quant {
            Some(owl_models::env::KvQuant::Fp8E4M3) => {
                let shape = vec![self.nb * self.page * self.dims.hkv * self.dims.hd / 4];
                self.k_fis
                    .iter()
                    .zip(self.v_fis.iter())
                    .map(|(k, v)| {
                        (
                            block_leaf_dt(&k.0, shape.clone(), Dtype::U32),
                            block_leaf_dt(&v.0, shape.clone(), Dtype::U32),
                        )
                    })
                    .collect()
            }
            _ => {
                let shape = vec![self.nb, self.page, self.dims.hkv, self.dims.hd];
                self.k_fis
                    .iter()
                    .zip(self.v_fis.iter())
                    .map(|(k, v)| {
                        (
                            block_leaf_dt(&k.0, shape.clone(), self.dims.dtype),
                            block_leaf_dt(&v.0, shape.clone(), self.dims.dtype),
                        )
                    })
                    .collect()
            }
        }
    }

    /// D1 取证:会话格 GDN 状态画像(整块回读 → 活跃格行 maxabs/sum/首值。
    /// OWL_GDN_DUMP_ALL=1 → 全层 cv 扫描(定位首个分歧层);否则首末层
    /// cv+rec)。调试仪表,正常路径零调用。27B rec 整块 ~25MB/层,dtoh
    /// 有感 —— 仅取证轮启用。
    pub(crate) async fn gdn_slot_profile<D: DeviceClient>(
        &self,
        face: &mut D,
        gdn_slot: usize,
        step: u64,
        all: bool,
    ) -> Result<Vec<String>> {
        let nl = self.gdns.len();
        let mut out = Vec::new();
        for (li, g) in self.gdns.iter().enumerate() {
            let last = li + 1 == nl;
            // 全层扫 = 仅 cv;默认 = 首末层 cv+rec
            if all && !last {
                let line = self.profile_block(face, li, "cv", &g.conv_v, gdn_slot, step, false).await?;
                out.push(line);
                continue;
            }
            if !all && li != 0 && !last {
                continue;
            }
            for (name, bn) in [("cv", &g.conv_v), ("rec", &g.rec)] {
                let line = self.profile_block(face, li, name, bn, gdn_slot, step, true).await?;
                out.push(line);
            }
        }
        Ok(out)
    }

    async fn profile_block<D: DeviceClient>(
        &self,
        face: &mut D,
        li: usize,
        name: &str,
        bn: &BlockN,
        gdn_slot: usize,
        step: u64,
        verbose: bool,
    ) -> Result<String> {
        let row = bn.1 / self.gdn_slots;
        let mut buf = vec![0u8; bn.1 * 4];
        face.dtoh(&bn.0, &mut buf).await?;
        let floats: Vec<f32> = buf[gdn_slot * row * 4..(gdn_slot + 1) * row * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let maxabs = floats.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let sum: f32 = floats.iter().take(4096).sum();
        let head: Vec<String> = floats.iter().take(3).map(|v| format!("{v:.4}")).collect();
        Ok(format!(
            "[gdn] s{step} L{li}/{name} n={} max|.|={maxabs:.4e} sum4k={sum:.4}{}",
            floats.len(),
            if verbose { format!(" head={}", head.join(",")) } else { String::new() },
        ))
    }

    /// 逐层 GDN 状态叶子四件套(conv_q/conv_k/conv_v/rec;slots 由
    /// 调用方按步注入:图闭包注输入槽,prefill 注 from_host)
    pub(crate) fn gdn_caches(&self) -> Vec<GdnLeaf> {
        let d = &self.dims;
        self.gdns
            .iter()
            .map(|g| GdnLeaf {
                conv_q: block_leaf(&g.conv_q.0, vec![self.gdn_slots, d.nk * d.hk, 3]),
                conv_k: block_leaf(&g.conv_k.0, vec![self.gdn_slots, d.nv * d.hv, 3]),
                conv_v: block_leaf(&g.conv_v.0, vec![self.gdn_slots, d.nv * d.hv, 3]),
                rec: block_leaf(&g.rec.0, vec![self.gdn_slots, d.nv, d.hk, d.hv]),
            })
            .collect()
    }

    /// 块表叶子(模式相关:paged [1,nb] / legacy [1];run 图闭包用)
    pub(crate) fn bt_leaf(&self) -> TensorOps {
        if self.paged {
            TensorOps::of_block(self.bt.id, Dtype::F32, vec![1, self.nb])
        } else {
            TensorOps::of_block(self.bt.id, Dtype::F32, vec![1])
        }
    }

    /// 块表叶子(恒 [1,nb];prefill_chunk 既有形状,legacy nb=1 → [1,1])
    pub(crate) fn bt_leaf_flat(&self) -> TensorOps {
        TensorOps::of_block(self.bt.id, Dtype::F32, vec![1, self.nb])
    }

    /// 快照存在性(调度面复用边界判定:无快照的边界不可复用,GDN 态
    /// 对不上)
    pub(crate) fn has_snap(&self, key: u32) -> bool {
        self.snaps.iter().any(|s| s.key == Some(key))
    }

    /// GDN 状态重置(turn-open;新序列零态语义)。按格重置(offset =
    /// 格号 × 行字节)—— 多会话各清各格,其余格的其他会话状态不受扰。
    /// 设备侧 memset(2026-09-26 定谳:host 往返清零 1.2GB = 14s/turn,
    /// 黑洞实测)。KV 不重置:kv_len 窗口 + 先写后打分语义下,本 turn
    /// 打分的槽位全部由本 turn 写过。
    pub(crate) async fn reset_gdn<D: DeviceClient>(
        &self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        for g in &self.gdns {
            let blocks = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (bn, elems) in blocks {
                let row = elems / self.gdn_slots;
                face.memset_zero_at(bn, gdn_slot * row * 4, row * 4).await?;
            }
        }
        face.sync().await?;
        Ok(())
    }

    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲
    /// (D2D;每层 4 块按行宽拷贝)。同 key 覆盖写;池满 LRU 换 key。
    /// (key 解析 = 边界覆盖块,由调用方从会话块链取;见 exec.rs)
    /// 泄漏案探针(2026-10-10):指定 slot 的 GDN 状态区逐块 FNV。
    /// turn(slot0) 前后对比 slot1 指纹 —— 变了 = 跨 slot 写实锤。
    pub(crate) async fn gdn_slot_fingerprint<D: DeviceClient>(
        &self,
        face: &mut D,
        slot: usize,
    ) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        for g in &self.gdns {
            for (bn, elems) in [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ] {
                let row = elems / self.gdn_slots;
                let mut buf = vec![0u8; elems * 4]; // 整块回读(f32)
                face.dtoh(bn, &mut buf).await?;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in &buf[slot * row * 4..(slot + 1) * row * 4] {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
                out.push(h);
            }
        }
        Ok(out)
    }

    /// 泄漏案探针第二步:reset_gdn(slot) 后回读该 slot 是否全零。
    /// 返回 slot 区非零字节数(0 = reset 生效)。
    pub(crate) async fn gdn_slot_zero_check<D: DeviceClient>(
        &mut self,
        face: &mut D,
        slot: usize,
    ) -> Result<usize> {
        self.reset_gdn(face, slot).await?;
        let mut nonzero = 0usize;
        for g in &self.gdns {
            for (bn, elems) in [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ] {
                let row = elems / self.gdn_slots;
                let mut buf = vec![0u8; elems * 4]; // 整块回读(f32)
                face.dtoh(bn, &mut buf).await?;
                nonzero += buf[slot * row * 4..(slot + 1) * row * 4]
                    .iter()
                    .filter(|&&b| b != 0)
                    .count();
            }
        }
        Ok(nonzero)
    }

    pub(crate) async fn capture_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
        key: u32,
    ) -> Result<()> {
        let idx = match self.snaps.iter().position(|s| s.key == Some(key)) {
            Some(i) => i,
            None => self
                .snaps
                .iter()
                .position(|s| s.key.is_none())
                .unwrap_or_else(|| {
                    self.snaps
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, s)| s.last_use)
                        .map(|(i, _)| i)
                        .expect("池非空")
                }),
        };
        self.snap_tick += 1;
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let dst = &self.snaps[idx].bufs[li * 4 + j];
                face.copy_block_at(src, gdn_slot * row * 4, &dst.0, 0, row * 4).await?;
            }
        }
        let slot = &mut self.snaps[idx];
        slot.key = Some(key);
        slot.last_use = self.snap_tick;
        Ok(())
    }

    /// GDN 快照恢复(E2c;前缀命中时调用):槽位缓冲 → 会话格(D2D
    /// 反向)。同流保序,后续 prefill 天然在恢复态之后。
    pub(crate) async fn restore_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
        key: u32,
    ) -> Result<()> {
        let idx = self
            .snaps
            .iter()
            .position(|s| s.key == Some(key))
            .ok_or_else(|| ModelError::Msg(format!("快照 {key} 已被逐出")))?;
        self.snap_tick += 1;
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let src = &self.snaps[idx].bufs[li * 4 + j];
                face.copy_block_at(&src.0, 0, dst, gdn_slot * row * 4, row * 4).await?;
            }
        }
        Ok(())
    }

    /// spec 快照拍摄(E5-M2;轮首/全接受后):会话格 → 专用缓冲。
    /// 与 E2c 前缀快照分池(spec 缓冲不参与 LRU,恢复后仍有效)。
    pub(crate) async fn capture_spec_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        let n = self.gdns.len();
        let snap = self
            .spec_snap
            .as_mut()
            .ok_or_else(|| ModelError::Msg("spec 快照缓冲未分配(OWL_SPEC_DEPTH 未开)".into()))?;
        // E5-M4:批量通道(192 次 actor 往返 → 1;单 ack 保序)
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> = Vec::with_capacity(n * 4);
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let dst = &snap.bufs[li * 4 + j];
                copies.push(((*src).clone(), gdn_slot * row * 4, dst.0.clone(), 0usize, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        let _ = n;
        Ok(())
    }

    /// spec 快照恢复(E5-M2;部分接受后):专用缓冲 → 会话格(反向 D2D)。
    pub(crate) async fn restore_spec_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        let n = self.gdns.len();
        let snap = self
            .spec_snap
            .as_ref()
            .ok_or_else(|| ModelError::Msg("spec 快照缓冲未分配".into()))?;
        // E5-M4:批量通道(同 capture)
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> = Vec::with_capacity(n * 4);
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let src = &snap.bufs[li * 4 + j];
                copies.push((src.0.clone(), 0usize, (*dst).clone(), gdn_slot * row * 4, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        let _ = n;
        Ok(())
    }

    /// 活跃会话块表 → 持久 bt 块(设备)。图烘焙的是 bt 指针,内容随
    /// 会话切换/块链增长重写(指针稳定 = 图不重捕);f32 过线(契约 5)
    pub(crate) async fn write_bt<D: DeviceClient>(
        &self,
        face: &mut D,
        table: &[u32],
    ) -> Result<()> {
        let mut v = vec![0f32; self.nb];
        for (i, b) in table.iter().enumerate() {
            v[i] = *b as f32;
        }
        face.write_block_f32(&self.bt, 0, &v).await
    }
}

/// 按 dtype 清零分配(KV/GDN 状态块)。**设备侧 alloc + memset**,不走
/// host 零向量 —— 曾用 `vec![0f32; n]` + to_le_bytes 逐元素造 64MB(rec
/// 单层 16M 元素),debug 循环烧掉 ~90s(F5-4 同款 debug 转换税);池
/// alloc 无零保证,必须显式 memset(F5-4 已去 fill(0),不能省)
async fn zero_block_dt<D: DeviceClient>(face: &mut D, n: usize, dt: Dtype) -> Result<BlockN> {
    // 2B 激活域显式臂(BF16 曾落 else 按 4B 误解释 —— dtype 表驱动律)
    let esz = if matches!(dt, Dtype::F16 | Dtype::BF16) { 2 } else { 4 };
    let b = face.alloc(dt, n).await?;
    face.memset_zero(&b, n * esz).await?;
    Ok((b, n))
}

fn block_leaf_dt(b: &Bytes, shape: Vec<usize>, dt: Dtype) -> TensorOps {
    match dt {
        Dtype::F16 | Dtype::BF16 => TensorOps::of_block(b.id, dt, shape),
        _ => TensorOps::of_block(b.id, Dtype::F32, shape),
    }
}

async fn zero_block<D: DeviceClient>(face: &mut D, n: usize) -> Result<BlockN> {
    zero_block_dt(face, n, Dtype::F32).await
}

fn block_leaf(b: &Bytes, shape: Vec<usize>) -> TensorOps {
    TensorOps::of_block(b.id, Dtype::F32, shape)
}
