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

/// GDN 快照池深度(E2c;每份 ~19.4MB 设备侧,LRU 覆盖写)。
/// 复用边界 = 有快照的最深块边界;短于最近边界的匹配回退全量(一档取舍)。
/// OWL_SNAP_MAX 可调(27B 单卡贴顶场景降到 1 省 ~300MB;默认 4)。
fn snap_max() -> usize {
    std::env::var("OWL_SNAP_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}

/// GDN 状态格容量(**与会话数绑定,与 max_seq_tokens 解耦**)。
/// 格语义 = 每会话一格(SessionTable 分配/释放);按位分配是历史包袱:
/// s=4096 时 rec(1 MiB/格/层 × 18 层)将达 72 GB,而真实需求 = 格数。
/// 8 = 覆盖需求并发上限(§二 并发 ≤8);会话 close 即释放格号。
/// OWL_GDN_SLOTS 可调(27B 单卡贴顶场景降到 2 省 ~300MB;默认 8)。
pub(crate) fn gdn_slots() -> usize {
    std::env::var("OWL_GDN_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

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
pub(crate) struct StatePool {
    pub(crate) kvs: Vec<KvBlocks>,
    /// FlashInfer K/V 影子池(owl_reshape_and_cache_dual_f16 写;kNHD
    /// [nb,page,Hkv,hd];OWL_FLASHINFER=1 时分配,否则空)
    pub(crate) k_fis: Vec<BlockN>,
    pub(crate) v_fis: Vec<BlockN>,
    gdns: Vec<GdnBlocks>,
    snaps: Vec<GdnSnapSlot>,
    snap_tick: u64,
    bt: Bytes,
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
    pub(crate) async fn alloc<D: DeviceClient>(
        face: &mut D,
        dims: ModelDims,
        layer_types: &[bool],
        seq_tokens: usize,
        pool_tokens: usize,
        fi_enabled: bool,
    ) -> Result<StatePool> {
        let t = std::time::Instant::now();
        let n_full = layer_types.iter().filter(|&&f| f).count();
        let n_gdn = layer_types.len() - n_full;

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
            let k_cache = zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?;
            let v_cache = zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?;
            kvs.push(KvBlocks { k_cache, v_cache });
        }
        if fi_enabled {
            for _ in 0..n_full {
                // kNHD 影子 [nb, page, Hkv, hd](K/V 各一;与 classic 同尺寸)
                k_fis.push(zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?);
                v_fis.push(zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype).await?);
            }
        }
        // 块表持久块(E2b):内容 = 活跃会话块链,turn 切换/增长时重写
        // (write_bt;图烘焙 bt 指针,指针稳定图不重捕);legacy = 哑表
        let bt = if paged {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?
        } else {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1]).step(), face).await?
        };
        let mut gdns: Vec<GdnBlocks> = Vec::new();
        for _ in 0..n_gdn {
            let conv_q = zero_block(face, gdn_slots() * dims.nk * dims.hk * 3).await?;
            let conv_k = zero_block(face, gdn_slots() * dims.nv * dims.hv * 3).await?;
            let conv_v = zero_block(face, gdn_slots() * dims.nv * dims.hv * 3).await?;
            let rec = zero_block(face, gdn_slots() * dims.nv * dims.hk * dims.hv).await?;
            gdns.push(GdnBlocks { conv_q, conv_k, conv_v, rec });
        }
        // GDN 快照池(E2c):行宽副本 × SNAP_MAX 份;仅 paged 形态
        let mut snaps: Vec<GdnSnapSlot> = Vec::new();
        if paged {
            for _ in 0..snap_max() {
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
            "[boot] 状态块分配 {:.2}s(kv f16 ×{} / gdn ×{},槽位 {})",
            t.elapsed().as_secs_f32(),
            kvs.len(),
            gdns.len(),
            seq_tokens
        );
        Ok(StatePool { kvs, k_fis, v_fis, gdns, snaps, snap_tick: 0, bt, page, nb, paged, x, dims })
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

    /// FlashInfer K/V 影子叶子(kNHD;OWL_FLASHINFER=1 时非空)
    pub(crate) fn kv_fi_leaves(&self) -> Vec<(TensorOps, TensorOps)> {
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

    /// 逐层 GDN 状态叶子四件套(conv_q/conv_k/conv_v/rec;slots 由
    /// 调用方按步注入:图闭包注输入槽,prefill 注 from_host)
    pub(crate) fn gdn_caches(&self) -> Vec<GdnLeaf> {
        let d = &self.dims;
        self.gdns
            .iter()
            .map(|g| GdnLeaf {
                conv_q: block_leaf(&g.conv_q.0, vec![gdn_slots(), d.nk * d.hk, 3]),
                conv_k: block_leaf(&g.conv_k.0, vec![gdn_slots(), d.nv * d.hv, 3]),
                conv_v: block_leaf(&g.conv_v.0, vec![gdn_slots(), d.nv * d.hv, 3]),
                rec: block_leaf(&g.rec.0, vec![gdn_slots(), d.nv, d.hk, d.hv]),
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
                let row = elems / gdn_slots();
                face.memset_zero_at(bn, gdn_slot * row * 4, row * 4).await?;
            }
        }
        face.sync().await?;
        Ok(())
    }

    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲
    /// (D2D;每层 4 块按行宽拷贝)。同 key 覆盖写;池满 LRU 换 key。
    /// (key 解析 = 边界覆盖块,由调用方从会话块链取;见 exec.rs)
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
                let row = elems / gdn_slots();
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
                let row = elems / gdn_slots();
                let src = &self.snaps[idx].bufs[li * 4 + j];
                face.copy_block_at(&src.0, 0, dst, gdn_slot * row * 4, row * 4).await?;
            }
        }
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
