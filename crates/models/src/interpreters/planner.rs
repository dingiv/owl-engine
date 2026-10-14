//! 块池:块粒度的 slab 复用(liveness 驱动的 best-fit 空闲表)。P1″ 2026-10-14。
//!
//! # 设计来源(社区 port,不是自研)
//!
//! llama.cpp `ggml_gallocr` 的动态分配器 `ggml_dyn_tallocr`
//! (`packages/llama.cpp/ggml/src/ggml-alloc.c:202 ggml_dyn_tallocr_alloc`):
//! **best-fit 选空闲块 → 命中即零分配调用;释放即插回空闲表;一次 reserve
//! 之后运行期不再 malloc/free**。本模块把同一策略下沉到 owl 的「块」粒度。
//!
//! # 粒度差异(必须知道的边界)
//!
//! ggml 在**一整块 buffer 内按 offset** 分配 ⇒ 可合并相邻空闲段、可把大块
//! 切给小块请求。owl 的块 = server 签发的独立 CUDA allocation(`id → 显存`),
//! 解释器**拿不到 offset**(`Arg::Block` 只带 id;`Arg::BlockSlice` 的
//! byte_offset 由声明期 SliceView 合成,且**输出槽恒为全块** ——
//! `crates/kernels/src/contract.rs`)。
//! ⇒ 本模块移植的是**策略**(best-fit + 活死驱动复用 + 预算上界),
//! **不能**移植字节级合并/切分(那需要动 Arg/Bytes 契约 = 本 lane 文件面之外,
//! 见报告「留给父会话」)。碎片浪费由 best-fit + 「层循环同形分配」的周期性
//! 顶住:同一层内形状重复出现,前一位使用者的死块往往**尺寸完全相等**。
//!
//! # 为什么有效(父会话定谳,全录见 perf-roadmap 补录Ⅵ/Ⅶ)
//!
//! eager prefill 每 512-token chunk 有 ~1460 次 `face.free` + ~1465 次
//! `face.alloc`,每次 = 一条「客户端↔server」命令(往返 + server 侧
//! `cuMemAllocAsync`/`cuMemFreeAsync`;nsys 里 `cuMemAllocAsync` max 19.7 ms)。
//! chunk 内节点块是**周期性活死**的(逐层同形)⇒ 池化后稳态 alloc/free
//! **零命令**,超预算才真还给 server(一条批量 free 收尾)。
//!
//! # 语义边界(调用方必须守)
//!
//! 1. **入池前提 = 该块在 DAG 上已死**(所有消费者 kernel 已入队)。
//!    流序保证「后发的写排在前发的读之后」—— 与本解释器 reclaim 面
//!    (`block_users` 归零)同一法律;本模块**不做**活性判断,只做池化。
//! 2. **命中块内容 = 脏**(上一位使用者的旧值)⇒ 零化语义(`Op::Zeros`)
//!    必须由调用方补设备侧 memset(见 `EvalCtx::alloc_out`)。
//! 3. 池只装「已登记容量」的块(`register` 来自 alloc 路径);未登记的块
//!    (如 `Htod` 内部一路分配的块)**不池化**,`give_back` 返回 false ⇒
//!    调用方走真 free(保守,零风险)。

use std::collections::{BTreeMap, HashMap};

/// 池运行态统计(诊断面;调用方负责落 metrics/日志)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// 命中(零命令复用)
    pub hit: u64,
    /// 未命中(落 server 真分配)
    pub miss: u64,
    /// 入池成功(零命令归还)
    pub in_pool: u64,
    /// 预算满被挤出(真 free)
    pub spilled: u64,
    /// 收尾抽干时真 free 的块数
    pub flush_blocks: u64,
    /// 缓存字节高水位(显存开销上界观测面)
    pub peak_cached_bytes: usize,
}

impl PoolStats {
    /// 命中率(0 == 无请求;诊断口径)
    pub fn hit_rate(&self) -> f64 {
        let req = self.hit + self.miss;
        if req == 0 {
            0.0
        } else {
            self.hit as f64 / req as f64
        }
    }
}

/// 块池(单次求值生命周期;`drain` 收尾)。
pub struct BlockPool {
    /// 缓存字节上界(超过即挤出;显存开销 = 池内字节,有界)
    budget_bytes: usize,
    /// 当前缓存字节(仅空闲表内)
    cached_bytes: usize,
    /// 容量(字节)→ 空闲块 id(LIFO:先还的先用,贴合层循环的复用顺序)
    free: BTreeMap<usize, Vec<u64>>,
    /// 已借出块 id → 容量字节(`register` 登记,`give_back` 摘除)
    live: HashMap<u64, usize>,
    stats: PoolStats,
}

impl BlockPool {
    /// `budget_bytes` = 缓存字节上界(0 = 池退化,等价关)。
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            cached_bytes: 0,
            free: BTreeMap::new(),
            live: HashMap::new(),
            stats: PoolStats::default(),
        }
    }

    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// 取块:**best-fit**(最小容量 ≥ `need_bytes`;同 `ggml_dyn_tallocr_alloc`
    /// 的「best fit besides the last block」)。命中从空闲表摘除并回到 live 表。
    pub fn take(&mut self, need_bytes: usize) -> Option<u64> {
        let cap = *self.free.range(need_bytes..).next().map(|(k, _)| k)?;
        let id = self.free.get_mut(&cap).and_then(|v| v.pop())?;
        if self.free.get(&cap).is_some_and(|v| v.is_empty()) {
            self.free.remove(&cap);
        }
        self.cached_bytes -= cap;
        self.live.insert(id, cap);
        self.stats.hit += 1;
        Some(id)
    }

    /// 登记一次**真分配**的块(server 签发)。未登记的块不可池化。
    pub fn register(&mut self, id: u64, bytes: usize) {
        self.live.insert(id, bytes);
        self.stats.miss += 1;
    }

    /// 归还:`true` = 入池(调用方**不要**发 free 命令);`false` = 未登记或
    /// 预算满 ⇒ 调用方走真 free。
    pub fn give_back(&mut self, id: u64) -> bool {
        let Some(bytes) = self.live.remove(&id) else {
            return false; // 未登记(Htod 内部块等)⇒ 保守走真 free
        };
        if self.cached_bytes + bytes > self.budget_bytes {
            self.stats.spilled += 1;
            return false;
        }
        self.free.entry(bytes).or_default().push(id);
        self.cached_bytes += bytes;
        self.stats.in_pool += 1;
        self.stats.peak_cached_bytes = self.stats.peak_cached_bytes.max(self.cached_bytes);
        true
    }

    /// 抽干(收尾):返回全部缓存块 id,调用方**一条批量 free** 还给 server。
    pub fn drain(&mut self) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::with_capacity(self.free.values().map(Vec::len).sum());
        for (_, mut v) in std::mem::take(&mut self.free) {
            out.append(&mut v);
        }
        self.cached_bytes = 0;
        self.stats.flush_blocks += out.len() as u64;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// best-fit:要 100 时挑最小的够用容量(200),不是 1000;且空桶被清掉。
    #[test]
    fn best_fit_picks_smallest_fitting_capacity() {
        let mut p = BlockPool::new(1 << 20);
        p.register(1, 200);
        p.register(2, 1000);
        p.register(3, 400);
        assert!(p.give_back(1));
        assert!(p.give_back(2));
        assert!(p.give_back(3));
        assert_eq!(p.cached_bytes(), 1600);
        assert_eq!(p.take(100), Some(1), "best-fit 应挑 200 而非 400/1000");
        assert_eq!(p.take(100), Some(3), "下一个最小够用 = 400");
        assert_eq!(p.take(100), Some(2));
        assert_eq!(p.take(100), None, "空池应未命中");
        assert_eq!(p.cached_bytes(), 0);
    }

    /// 容量不足不命中(不做切分 —— 粒度边界),且未登记块一律 false(保守)。
    #[test]
    fn no_split_and_unknown_id_is_spilled() {
        let mut p = BlockPool::new(1 << 20);
        p.register(1, 64);
        assert!(p.give_back(1));
        assert_eq!(p.take(128), None, "块粒度不可切分:64 不满足 128");
        assert!(!p.give_back(999), "未登记块 ⇒ 调用方真 free");
        assert_eq!(p.stats().spilled, 0, "未登记 ≠ 预算挤出");
        assert_eq!(p.take(64), Some(1));
    }

    /// 预算:超上界挤出(**挤出的 id 从此失联** —— 调用方会真 free 它),
    /// 不超则入池;高水位可观测。
    #[test]
    fn budget_spills_and_records_peak() {
        let mut p = BlockPool::new(1000);
        p.register(1, 600);
        p.register(2, 600);
        assert!(p.give_back(1));
        assert!(!p.give_back(2), "600+600 > 1000 ⇒ 挤出");
        assert_eq!(p.stats().spilled, 1);
        assert_eq!(p.cached_bytes(), 600);
        assert_eq!(p.stats().peak_cached_bytes, 600);
        // 命中后缓存下降;挤出过的 id 已交还 server,池不得再认领
        assert_eq!(p.take(600), Some(1));
        assert!(!p.give_back(2), "挤出过的 id 已交还 server,不可再入池");
        assert_eq!(p.cached_bytes(), 0);
        // 新块在预算内正常入池
        p.register(3, 900);
        assert!(p.give_back(3));
        assert_eq!(p.cached_bytes(), 900);
    }

    /// **层循环同形复用**:48 层 × 每层同形 3 块 ⇒ 第 2 层起全部命中
    /// (这正是 prefill 的分配形状;池预算只需 ≈ 一层的活性集)。
    #[test]
    fn layer_loop_reuse_hits_after_first_layer() {
        const LAYERS: usize = 48;
        const SHAPES: [usize; 3] = [4096, 16384, 65536];
        let mut p = BlockPool::new(128 << 20);
        let mut next_id = 1u64;
        for layer in 0..LAYERS {
            for &s in &SHAPES {
                let id = match p.take(s) {
                    Some(id) => id,
                    None => {
                        let id = next_id;
                        next_id += 1;
                        p.register(id, s);
                        id
                    }
                };
                // 用完即还(模拟 liveness 归零)
                assert!(p.give_back(id), "层内归还必入池(预算充足)");
            }
            if layer == 0 {
                assert_eq!(p.stats().miss, 3, "首层 3 次真分配");
            }
        }
        let st = p.stats();
        assert_eq!(st.miss, 3, "只有首层落 server:得 {}", st.miss);
        assert_eq!(st.hit, ((LAYERS - 1) * SHAPES.len()) as u64);
        assert!(st.hit_rate() > 0.97, "命中率 {:.4}(只有首层落 server)", st.hit_rate());
        assert!(p.cached_bytes() <= 128 << 20);
        let drained = p.drain();
        assert_eq!(drained.len(), SHAPES.len(), "收尾只 1 条批量 free 的 3 个块");
        assert_eq!(p.cached_bytes(), 0);
    }
}
