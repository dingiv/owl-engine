//! **选择层(B 轴;KVMem 式)纯逻辑核心** —— 索引 / 打分 / 选择 / 装配计划。
//!
//! 依据:`docs/arch/kvmem-选择层-设计与施工.md`(B 轴卷)+ 设计卷 §29(论文精读)。
//! 本模块**只做纯 host 逻辑**(零 I/O、零设备面)—— 与 `store` 的既有纪律一致:
//! 「账 + 身份 + 策略」;设备面的三件(索引采集的 K 读、tile 打分、re-RoPE/scatter)
//! 分别落在 engine / kernels,接口由本模块的类型定死。
//!
//! # 论文对应(照抄,勿自创)
//!
//! | 论文 | 本模块 |
//! |---|---|
//! | Mean-K 索引:每块 × 每层 × 每 KV head 一个**去 RoPE** 的 K 均值 | [`MeanKIndex`] / [`MeanKEntry`] |
//! | 块相关度 `R_b = Σ_m mean_{l,h} softmax_{b'∈C}(q·k̄/√d)` | [`score_blocks`] |
//! | sink + recent 保底、按分 top-k、选中后**按时间序重排** | [`select`] |
//! | 两视图分离:`R_t/L_t/E_t` 分解、`L_physical = W_t \ G_{t-1}`、页表别名装配 | [`assemble_view`] |
//! | 位置一致性:raw K 真源 + **有界 delta re-RoPE**(超限重建) | [`PosAction`] / [`PosFrame`] |
//!
//! # 与 A 轴的关系(裁决⑩:不改 A 的接口)
//!
//! 本模块**不碰** `Rel`/`BlockLeg`/槽池:它输出的是**计划**([`ViewPlan`]),
//! 执行(页表写、re-RoPE、gather/scatter 搬运)仍归 engine —— 与 store 卷 §21
//! 的「store 不做 I/O,engine 编排」同一条律。


use crate::error::{StoreError, StoreResult};
use crate::id::ChainHash;

// ───────────────────────────── 索引(账) ─────────────────────────────

/// 一个 (块, 层, KV 头) 的检索表示:**去 RoPE** 的 K 均值(f16 位型)。
///
/// 为什么 f16:`k̄` 只用于**排序**,不参与注意力数值 ⇒ 半精度足够,索引体量减半
/// (论文同款:索引 0.25GiB@256K → 9.5GiB@10M)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeanKEntry {
    /// 层号。
    pub layer: u16,
    /// KV 头号。
    pub kv_head: u16,
    /// 去 RoPE 的 K 均值(f16 位型;长度 = `head_dim`)。
    pub k_bar: Vec<u16>,
}

/// 单块的索引面(**扁平存储**:`[layer][kv_head][dim]` 连续 f16 位型)。
///
/// 为什么扁平(2026-10-18 B-1 实测):原实现每 (层, KV 头) 一条
/// `MeanKEntry`(各自一个 `Vec<u16>`)⇒ 打分逐 (块, 层, KV 头) 取行时为
/// 每行一次堆指针追逐;27B 几何 8192 块 = **52 万次**分散读,实测把 host
/// 打分从线性推到超线性(2048 块 97ms → 8192 块 550ms)。扁平后整块一段,
/// 行 = 段内定长切片(零指针追逐、对预取下友好,也是将来 GPU staged tile
/// 上传的天然单位)。
#[derive(Clone, Debug)]
struct BlockIndex {
    /// `[layer][kv_head][dim]` 连续(f16 位型);长 = `entries_per_block × dim`。
    k_bar: Vec<u16>,
    /// 触达时钟(host 准入/淘汰序;论文:近期 + 累积频率,不是纯 LRU)。
    last_use: u64,
    /// 累积检索命中次数(论文的"累积检索频率";准入用它而非纯 LRU)。
    hits: u32,
}

/// 索引账(块 → Mean-K 面)。
///
/// **host 驻留**(论文同款:全量索引在 host,GPU 只 staged bounded tile 打分)。
/// 容量**不在这里限**:淘汰序由调用方按策略对象出票(与 `TierLedger`/`EvictPolicy` 同律)。
#[derive(Debug)]
pub struct MeanKIndex {
    dim: usize,
    layers: usize,
    kv_heads: usize,
    blocks: std::collections::HashMap<ChainHash, BlockIndex>,
    tick: u64,
}

impl MeanKIndex {
    /// 构造(`dim` = head_dim;`layers` × `kv_heads` = 每块条目数)。
    pub fn new(dim: usize, layers: usize, kv_heads: usize) -> StoreResult<Self> {
        if dim == 0 || layers == 0 || kv_heads == 0 {
            return Err(StoreError::Invariant(format!(
                "索引几何非法:dim={dim} layers={layers} kv_heads={kv_heads}"
            )));
        }
        Ok(Self { dim, layers, kv_heads, blocks: std::collections::HashMap::new(), tick: 0 })
    }

    /// 几何(`(dim, layers, kv_heads)`)。
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.dim, self.layers, self.kv_heads)
    }

    /// 每块条目数。
    pub fn entries_per_block(&self) -> usize {
        self.layers * self.kv_heads
    }

    /// 单块索引字节(报告/预算用;f16 = 2B/元素)。
    pub fn block_bytes(&self) -> usize {
        self.entries_per_block() * self.dim * 2
    }

    /// 索引总字节。
    pub fn bytes(&self) -> usize {
        self.blocks.len() * self.block_bytes()
    }

    /// 登记一块的索引面(**形状校验**:条目数 = 层×头,每向量长 = dim;
    /// 重复登记 = 覆盖 + 触达)。
    pub fn register(&mut self, block: ChainHash, entries: Vec<MeanKEntry>) -> StoreResult<()> {
        if entries.len() != self.entries_per_block() {
            return Err(StoreError::Invariant(format!(
                "索引条目数不符:{} ≠ 层×头 {}",
                entries.len(),
                self.entries_per_block()
            )));
        }
        for (pos, e) in entries.iter().enumerate() {
            if e.k_bar.len() != self.dim {
                return Err(StoreError::Invariant(format!(
                    "Mean-K 向量长不符:{} ≠ dim {}",
                    e.k_bar.len(),
                    self.dim
                )));
            }
            if e.layer as usize >= self.layers || e.kv_head as usize >= self.kv_heads {
                return Err(StoreError::Invariant(format!(
                    "索引越界:layer={} kv_head={}(共 {}×{})",
                    e.layer, e.kv_head, self.layers, self.kv_heads
                )));
            }
            // **规范序**(2026-10-18 B-1):`entries[l·kv_heads + g]`,与
            // [`Self::entry_at`]/`score_blocks` 的直下标同源。
            // 为什么不收散序再查表:打分热路径里 `find` 是 O(层×头) 线性扫
            // —— 实测 8192 块×48 层×8 头下占 ~60% 墙钟(27B 几何);
            // 乱序登记 = 响亮拒(不静默变慢/错取)。
            if pos != e.layer as usize * self.kv_heads + e.kv_head as usize {
                return Err(StoreError::Invariant(format!(
                    "索引条目非规范序:位置 {pos} 处是 (layer={}, kv_head={});\
                     应为 entries[l·kv_heads + g] 序",
                    e.layer, e.kv_head
                )));
            }
        }
        // 扁平化(校验已在上面做完 ⇒ 只做拼装)
        let mut flat: Vec<u16> = Vec::with_capacity(self.entries_per_block() * self.dim);
        for e in entries {
            flat.extend_from_slice(&e.k_bar);
        }
        self.tick += 1;
        let tick = self.tick;
        match self.blocks.get_mut(&block) {
            Some(b) => {
                b.k_bar = flat;
                b.last_use = tick;
            }
            None => {
                self.blocks.insert(block, BlockIndex { k_bar: flat, last_use: tick, hits: 0 });
            }
        }
        Ok(())
    }

    /// **规范序直取**(打分热路径):`(块, 层, KV 头)` → **f16 位型行切片**。
    ///
    /// 返回 `None` = 未登记 / 索引越界(调用方当作“无数据”,不是错)。
    /// 行 = 扁平段内定长切片 ⇒ 零指针追逐、零线性扫描。
    #[inline]
    pub fn entry_at(&self, block: ChainHash, layer: usize, kv_head: usize) -> Option<&[u16]> {
        if layer >= self.layers || kv_head >= self.kv_heads {
            return None;
        }
        let b = self.blocks.get(&block)?;
        let off = (layer * self.kv_heads + kv_head) * self.dim;
        b.k_bar.get(off..off + self.dim)
    }

    /// 整块的**扁平 k̄ 段**(`[layer][kv_head][dim]`;诊断/搬运面用)。
    pub fn block_k_bar(&self, block: ChainHash) -> Option<&[u16]> {
        self.blocks.get(&block).map(|b| b.k_bar.as_slice())
    }

    /// 是否已登记。
    pub fn contains(&self, block: ChainHash) -> bool {
        self.blocks.contains_key(&block)
    }

    /// 触达(命中新鲜度 + 累积频率)。
    pub fn touch(&mut self, block: ChainHash) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(b) = self.blocks.get_mut(&block) {
            b.last_use = tick;
            b.hits = b.hits.saturating_add(1);
        }
    }

    /// 块数。
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// 空?
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// **淘汰候选**(按"近期 + 累积频率"出序:先最冷、再最少命中;
    /// 论文 §4.4:host 准入看这两者,不是纯 LRU)。
    pub fn evict_candidates(&self, protect: &[ChainHash]) -> Vec<(ChainHash, u64, u32)> {
        let mut v: Vec<(ChainHash, u64, u32)> = self
            .blocks
            .iter()
            .filter(|(k, _)| !protect.contains(k))
            .map(|(k, b)| (*k, b.last_use, b.hits))
            .collect();
        v.sort_by_key(|(_, last_use, hits)| (*last_use, *hits));
        v
    }

    /// 丢一块的索引(调用方负责出账)。
    pub fn drop_block(&mut self, block: ChainHash) -> StoreResult<()> {
        self.blocks
            .remove(&block)
            .map(|_| ())
            .ok_or_else(|| StoreError::Unknown(format!("块 {block:?} 无索引")))
    }

    /// 报告。
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            blocks: self.blocks.len(),
            entries_per_block: self.entries_per_block(),
            bytes: self.bytes(),
        }
    }
}

/// 索引报告(体量/形状)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexStats {
    /// 已登记块数。
    pub blocks: usize,
    /// 每块条目数(层 × KV 头)。
    pub entries_per_block: usize,
    /// 索引总字节。
    pub bytes: usize,
}

// ─────────────────────── 位置去除(逆 RoPE;唯一产地) ───────────────────────

/// **rotate-half 逆变换的单对**(核内约定;论文 §4.3「构造索引前先去位置编码」)。
///
/// 正变换(核内):`x' = x·c − y·s`,`y' = x·s + y·c`(`y` = `x[i+half]`);
/// 逆变换 = `(x, y)` 如上式用 `−s`——两条调用方([`MeanKCollector`]
/// 与 B 轴 Q 面)**共用这一对**,禁第二套写法(约定错 = 静默错值)。
#[inline]
pub fn derope_pair(x: f32, y: f32, c: f32, s: f32) -> (f32, f32) {
    (x * c + y * s, -x * s + y * c)
}

/// **整行的半维配对逆 RoPE**(原地;rotate-half + **partial 直通**)。
///
/// `cos`/`sin` 长 = `rotary/2`;只反旋 `[0, 2·rotary/2)`,尾维原样(核内同律)。
/// 形状不符 = **响亮拒**(不静默截断)。
pub fn derope_halves_inplace(row: &mut [f32], cos: &[f32], sin: &[f32]) -> StoreResult<()> {
    let half = cos.len();
    if sin.len() != half || half == 0 || 2 * half > row.len() {
        return Err(StoreError::Invariant(format!(
            "逆 RoPE 形状不符:row={} cos={} sin={}(须 cos=sin=rotary/2 ≤ row/2)",
            row.len(),
            cos.len(),
            sin.len()
        )));
    }
    for i in 0..half {
        let (x, y) = (row[i], row[i + half]);
        let (nx, ny) = derope_pair(x, y, cos[i], sin[i]);
        row[i] = nx;
        row[i + half] = ny;
    }
    Ok(())
}

// ───────────────────────── 采集(B-0:索引怎么来) ─────────────────────────

/// **Mean-K 采集器**(B-0;纯 host 数学):喂一个块的 K 行 → 逐 (层, KV 头) 求均值。
///
/// 两种喂法(对应两条 K 来源;§8 的"K 源决策"):
/// - [`Self::push_pre_rope`]:**RoPE 前**的 K(模型面 norm 输出直接给出)⇒ 天然位置无关;
/// - [`Self::push_post_rope`]:**KV 缓存里的 K**(已打 RoPE)+ 该位置的 cos/sin
///   ⇒ 就地**逆 RoPE** 再累加(论文 §4.3:构造索引前先去位置编码)。
///
/// 均值在 f64 累加(f16 输出只用于**排序**,累加精度不省),`finish` 出 f16 位型。
#[derive(Debug)]
pub struct MeanKCollector {
    dim: usize,
    layers: usize,
    kv_heads: usize,
    /// `[layer][kv_head][dim]` 累加器。
    sums: Vec<f64>,
    /// 已喂 token 数(每 (层,头) 同一计数 ⇒ 一个标量)。
    count: usize,
}

impl MeanKCollector {
    /// 构造(几何须与 [`MeanKIndex`] 一致)。
    pub fn new(dim: usize, layers: usize, kv_heads: usize) -> StoreResult<Self> {
        if dim == 0 || layers == 0 || kv_heads == 0 {
            return Err(StoreError::Invariant(format!(
                "采集器几何非法:dim={dim} layers={layers} kv_heads={kv_heads}"
            )));
        }
        Ok(Self {
            dim,
            layers,
            kv_heads,
            sums: vec![0.0; layers * kv_heads * dim],
            count: 0,
        })
    }

    /// 喂一行 **RoPE 前** K(位置无关)。
    pub fn push_pre_rope(&mut self, layer: usize, kv_head: usize, k: &[f32]) -> StoreResult<()> {
        let base = self.offset(layer, kv_head)?;
        if k.len() != self.dim {
            return Err(StoreError::Invariant(format!("K 行长不符:{} ≠ {}", k.len(), self.dim)));
        }
        for (i, v) in k.iter().enumerate() {
            self.sums[base + i] += *v as f64;
        }
        self.count += 1;
        Ok(())
    }

    /// 喂一行 **RoPE 后** K + 该位置的 cos/sin ⇒ 就地**逆 RoPE**(旋转 −θ)后累加。
    ///
    /// 旋转半维形式(与本引擎 RoPE 同族):`k' = k·cos ± rot(k)·sin`;
    /// 逆变换 = 用 `−sin`。`cos`/`sin` 长度 = `dim/2`。
    pub fn push_post_rope(
        &mut self,
        layer: usize,
        kv_head: usize,
        k: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> StoreResult<()> {
        let half = self.dim / 2;
        if !self.dim.is_multiple_of(2) {
            return Err(StoreError::Invariant(format!("dim {} 必须为偶数(RoPE 半维)", self.dim)));
        }
        if k.len() != self.dim || cos.len() != half || sin.len() != half {
            return Err(StoreError::Invariant(format!(
                "逆 RoPE 形状不符:k={} cos={} sin={}(须 {} / {} / {})",
                k.len(),
                cos.len(),
                sin.len(),
                self.dim,
                half,
                half
            )));
        }
        let base = self.offset(layer, kv_head)?;
        for i in 0..half {
            // 逆旋转:x' = x·cos + y·sin;y' = −x·sin + y·cos
            let x = k[2 * i];
            let y = k[2 * i + 1];
            let (c, s) = (cos[i], sin[i]);
            self.sums[base + 2 * i] += (x * c + y * s) as f64;
            self.sums[base + 2 * i + 1] += (-x * s + y * c) as f64;
        }
        self.count += 1;
        Ok(())
    }

    /// **半维配对**的逆 RoPE(另一种业界约定:`(i, i+half)` 成对,而非 `(2i, 2i+1)`)。
    ///
    /// 为什么要两种:RoPE 的配对约定随实现而变(interleaved `(2i,2i+1)` vs
    /// **rotate-half** `(i, i+d/2)`),而**去 RoPE 必须与写入侧的约定严格一致**
    /// (不一致 ⇒ 索引位置相关 = 静默错值)。接线时按 `qknorm_rope_kv_insert` 的
    /// 实际约定二选一;两者都在此列明,免得接线现场凭记忆猜。
    pub fn push_post_rope_halves(
        &mut self,
        layer: usize,
        kv_head: usize,
        k: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> StoreResult<()> {
        if !self.dim.is_multiple_of(2) {
            return Err(StoreError::Invariant(format!("dim {} 必须为偶数(RoPE 半维)", self.dim)));
        }
        // **partial 感知**(核内同律:`rotary_dim ≤ head_dim`,尾维直通):
        // cos/sin 长 = `rotary_dim/2`,只旋转 `[0, rotary_dim)`,其余原样累加。
        if k.len() != self.dim {
            return Err(StoreError::Invariant(format!(
                "逆 RoPE 行长不符:k={} dim={}",
                k.len(),
                self.dim
            )));
        }
        if cos.len() * 2 > self.dim || cos.is_empty() {
            return Err(StoreError::Invariant(format!(
                "逆 RoPE 旋转维不符:cos={} dim={}(须 0 < 2·len ≤ dim)",
                cos.len(),
                self.dim
            )));
        }
        let base = self.offset(layer, kv_head)?;
        // 逆旋**只此一份实现**(`derope_pair`;约定单源)——
        // 直接算进累加器,**不经过 scratch 双拷**(2026-10-19 A3 热路修正:
        // 每行省一次 1 KiB 拷贝 + 一次 256 元素回读;采集是逐行热路)。
        let half = cos.len();
        let acc = &mut self.sums[base..base + self.dim];
        for i in 0..half {
            let (nx, ny) = derope_pair(k[i], k[i + half], cos[i], sin[i]);
            acc[i] += nx as f64;
            acc[i + half] += ny as f64;
        }
        for i in 2 * half..self.dim {
            acc[i] += k[i] as f64;
        }
        self.count += 1;
        Ok(())
    }

    /// 已喂 token 数。
    pub fn len(&self) -> usize {
        self.count
    }

    /// 空?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// **收口**:出该块的索引面(逐 (层, KV 头) 一条 f16 均值)。
    ///
    /// 空块(一个 token 都没喂)= 结构性拒(空索引没有意义,静默给零向量 = 假账)。
    pub fn finish(self) -> StoreResult<Vec<MeanKEntry>> {
        if self.count == 0 {
            return Err(StoreError::Invariant("采集器为空:该块没有 K 可索引".into()));
        }
        let n = self.count as f64;
        let mut out = Vec::with_capacity(self.layers * self.kv_heads);
        for l in 0..self.layers {
            for h in 0..self.kv_heads {
                let base = (l * self.kv_heads + h) * self.dim;
                let k_bar: Vec<u16> = self.sums[base..base + self.dim]
                    .iter()
                    .map(|v| f32_to_f16((*v / n) as f32))
                    .collect();
                out.push(MeanKEntry { layer: l as u16, kv_head: h as u16, k_bar });
            }
        }
        Ok(out)
    }

    fn offset(&self, layer: usize, kv_head: usize) -> StoreResult<usize> {
        if layer >= self.layers || kv_head >= self.kv_heads {
            return Err(StoreError::Invariant(format!(
                "采集越界:layer={layer} kv_head={kv_head}(共 {}×{})",
                self.layers, self.kv_heads
            )));
        }
        Ok((layer * self.kv_heads + kv_head) * self.dim)
    }
}

// ───────────────────────────── 打分 ─────────────────────────────

/// 当前 query 的**去 RoPE** 向量面(逐 token × 逐层 × 逐查询头)。
///
/// 论文 Eq.10 的外层 `Σ_m` 对 **query token** 求和 ⇒ 本结构按 token 保存,
/// 由 [`score_blocks`] 逐 token 做 softmax(softmax 非线性,不能先对 token 求和)。
#[derive(Clone, Debug, Default)]
pub struct QueryFace {
    layers: usize,
    q_heads: usize,
    dim: usize,
    /// token → 扁平 `[layer][q_head][dim]`(去 RoPE)。
    tokens: Vec<Vec<f32>>,
}

impl QueryFace {
    /// 构造(几何须与索引一致)。
    pub fn new(dim: usize, layers: usize, q_heads: usize) -> Self {
        Self { layers, q_heads, dim, tokens: Vec::new() }
    }

    /// 追加一个 query token 的面(长度必须 = 层×头×dim)。
    pub fn push_token(&mut self, flat: Vec<f32>) -> StoreResult<()> {
        let want = self.layers * self.q_heads * self.dim;
        if flat.len() != want {
            return Err(StoreError::Invariant(format!(
                "query 面长度不符:{} ≠ {}",
                flat.len(),
                want
            )));
        }
        self.tokens.push(flat);
        Ok(())
    }

    /// token 数。
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 空?
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// 几何(层, 查询头, dim)。
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.layers, self.q_heads, self.dim)
    }

    /// **窗合并**(B-1 快路;`ScoreMode::LogitMean` 专用):
    /// 把窗口内全部 query token 的 `q̃` **逐元素相加** ⇒ 单向量。
    ///
    /// 为何合法:logit 对 query 是**线性**的
    /// (`Σ_m q_m·k̄ = (Σ_m q_m)·k̄`)⇒ 合并后的分数与逐 token 求和**逐位等价**,
    /// 但索引只流一遍、f16 解码只做一次(**索引重复读从 W×降到 1×**)。
    ///
    /// ∇ softmax-均值口径**不可**用此合并(softmax 非线性);那条路继续逐 token。
    pub fn collapsed_sum(&self) -> StoreResult<Vec<f32>> {
        if self.is_empty() {
            return Err(StoreError::Invariant("query 面为空,无可合并".into()));
        }
        let want = self.layers * self.q_heads * self.dim;
        let mut out = vec![0.0f32; want];
        for t in &self.tokens {
            if t.len() != want {
                return Err(StoreError::Invariant(format!(
                    "query 面行长不符:{} ≠ {want}",
                    t.len()
                )));
            }
            for (o, v) in out.iter_mut().zip(t) {
                *o += *v;
            }
        }
        Ok(out)
    }
}

/// 打分调参。
#[derive(Clone, Copy, Debug)]
pub struct ScoreTuning {
    /// 缩放(`1/√d`;0 = 自动按 dim 推)。
    pub scale: f32,
    /// 聚合口径(见 [`ScoreMode`];缺省 = 论文原样)。
    pub mode: ScoreMode,
}

impl Default for ScoreTuning {
    fn default() -> Self {
        Self { scale: 0.0, mode: ScoreMode::default() }
    }
}

/// **(层,头) 聚合口径** —— 论文 Eq.10 与“排序用”的分歧点。
///
/// 两者对**排序**等价(softmax 对同一 (层,头) 内单调 ⇒ 序 = logit 序),
/// 但对**分值幅度**差很多(2026-10-18 实测,96 块)∶
///
/// | 口径 | 语义 | 分值范围 |
/// |---|---|---|
/// | [`Self::SoftmaxMean`] | **论文原样**(推荐;偏差不偏) | 全在 `0.073~0.132`(**≈1/块数**,平局成片) |
/// | [`Self::LogitMean`] | 排序/阈值/预算用(logit 面取均) | 随 logits 自然展开 |
///
/// ∇ 选哪一:凡“只要 top-k 名单”的调用方两者等价(建议留 SoftmaxMean);
/// 凡“要拿分值做阈值/预算/权重”的(B-4 在线预算、按分排序的报表)
/// 必须用 [[Self::LogitMean]] —— softmax 均值在候选块多时**动态范围被压到 1/块数**。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScoreMode {
    /// 论文 Eq.10:逐 (层,头) 先对**候选块** softmax,再对 (层,头) 与 query token 取均。
    #[default]
    SoftmaxMean,
    /// 逐 (层,头) 的**原始 logit**(`q̃·k̄/√d`)直接对 (层,头) 与 token 取均;
    /// 未登记候选 = `-inf`(不静默当 0)。
    LogitMean,
}

/// **8 路展开点积**(B-1 热路):8 个独立标量累加器 + `chunks_exact(8)`
/// ⇒ 打破 f32 归约依赖、让 LLVM 发 SIMD FMA(严格 FP 下单个累加器不得重排:
/// 这是唯一合法的向量化写法)。
/// 注:求和序与朴素版不同 ⇒ 末位可能差 1ulp(快路与论文口径路不追求逐位等)。
#[inline]
fn dot8(row: &[f32], q: &[f32]) -> f32 {
    let n = row.len().min(q.len());
    let (row, q) = (&row[..n], &q[..n]);
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut a4, mut a5, mut a6, mut a7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let full = n / 8 * 8;
    let (rf, qf) = (&row[..full], &q[..full]);
    for (r, x) in rf.chunks_exact(8).zip(qf.chunks_exact(8)) {
        a0 += r[0] * x[0];
        a1 += r[1] * x[1];
        a2 += r[2] * x[2];
        a3 += r[3] * x[3];
        a4 += r[4] * x[4];
        a5 += r[5] * x[5];
        a6 += r[6] * x[6];
        a7 += r[7] * x[7];
    }
    let mut s = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    for (r, x) in row[full..].iter().zip(&q[full..]) {
        s += r * x;
    }
    s
}

/// 打分分块行数(B-1 热路:f16→f32 暂存区行数;256 行 × dim128 = 128KB 栈外机)
const TILE_ROWS: usize = 256;

/// **块相关度打分**(论文 Eq.10):
///
/// ```text
/// R_b = Σ_m  mean_{l,h}  softmax_{b'∈C}( q_{l,m,h} · k̄_{l,b',g(h)} / √d )
/// ```
///
/// `gqa`:查询头 → KV 头的映射(`len = q_heads`;MHA 时 = 恒等)。
/// 返回**与 `cands` 同序**的相关度(未登记的块 = 0)。
pub fn score_blocks(
    index: &MeanKIndex,
    face: &QueryFace,
    cands: &[ChainHash],
    gqa: &[u16],
    tuning: ScoreTuning,
) -> StoreResult<Vec<f32>> {
    let (dim, layers, kv_heads) = index.geometry();
    if face.dim != dim || face.layers != layers {
        return Err(StoreError::Invariant(format!(
            "打分几何不符:query (dim={}, layers={}) ≠ index ({dim}, {layers})",
            face.dim, face.layers
        )));
    }
    if gqa.len() != face.q_heads {
        return Err(StoreError::Invariant(format!(
            "GQA 映射长不符:{} ≠ 查询头 {}",
            gqa.len(),
            face.q_heads
        )));
    }
    if cands.is_empty() || face.is_empty() {
        return Ok(vec![0.0; cands.len()]);
    }
    let scale = if tuning.scale > 0.0 { tuning.scale } else { 1.0 / (dim as f32).sqrt() };
    let mut r = vec![0.0f32; cands.len()];
    // **热路(LogitMean)**:**两层线性折叠** —— 论文口径的分数
    // `mean_{l,h} logit` 对 query 与头都是线性的,故
    //
    // ```text
    // mean_{l,h} dot(q_{l,h}, k̄_{l,g(h)})  =  Σ_l Σ_g (|G(g)|/hq) · dot(q̄_{l,g}, k̄_{l,g})
    //     q̄_{l,g} = mean_{h∈G(g)} q_{l,h}      (① GQA 头折叠)
    //     q 本身先经窗合并 Σ_m                     (② query 窗折叠 ⇒ 单向量)
    // ```
    //
    // ⇒ 打分从「窗 W × 头 hq × 层 × KV 头」次行扫描降到 **「层 × KV 头」一次**(本档 6×;
    // 与窗折叠叠加 = 48×),且两者都**不是近似**(仅 FP 重排;逐位同参考式)。
    if tuning.mode == ScoreMode::LogitMean {
        let q = face.collapsed_sum()?;
        let hq = face.q_heads;
        // ① 按 KV 头分组(头的归属表)
        let mut q_heads_of: Vec<Vec<usize>> = vec![Vec::new(); kv_heads];
        for (h, g_raw) in gqa.iter().enumerate() {
            let g = *g_raw as usize;
            if g >= kv_heads {
                return Err(StoreError::Invariant(format!("GQA 映射越界:{g} ≥ {kv_heads}")));
            }
            q_heads_of[g].push(h);
        }
        // ② q̄_{l,g} = 该 g 名下查询头的均值(与 |G|/hq 权重等价)
        let mut qbar = vec![0.0f32; layers * kv_heads * dim];
        for l in 0..layers {
            for (g, hs) in q_heads_of.iter().enumerate() {
                if hs.is_empty() {
                    continue;
                }
                let dst = &mut qbar[(l * kv_heads + g) * dim..(l * kv_heads + g + 1) * dim];
                for h in hs {
                    let qo = (l * hq + *h) * dim;
                    for (o, v) in dst.iter_mut().zip(&q[qo..qo + dim]) {
                        *o += *v;
                    }
                }
                let inv = 1.0 / hs.len() as f32;
                for o in dst.iter_mut() {
                    *o *= inv;
                }
            }
        }
        // ③ 单累加器(逐候选);权重 |G|/hq 与 1/层数 最后统一乘
        //
        // **并行口径(2026-10-19 262k 冲刺)**:按**候选分片**,每片自带
        // `acc/seen/tile` ⇒ 每个候选的 (层, KV 头) 累加**顺序与串行逐位相同**
        // (只并行不同候选,不动任何求和次序)⇒ 数值**逐位不变**,只买吞吐。
        // 为什么必须并行:召回点打分是 O(块数) —— 262k(8192 块)/ 每 512 tok 一次
        // 完整召回的档上,串行 host 打分实测 ~32s/51k tok ⇒ 262k 要十几分钟。
        // 分片数 = 可用并行度(上限 16;候选少于一页时直接串行,零线程开销)。
        let nthreads = if cands.len() < TILE_ROWS * 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(16)
        };
        let acc: Vec<f32>;
        let seen: Vec<usize>;
        if nthreads <= 1 {
            let mut a = vec![0.0f32; cands.len()];
            let mut sn = vec![0usize; cands.len()];
            let mut tile = vec![0.0f32; TILE_ROWS * dim];
            let mut have = vec![false; TILE_ROWS];
            for l in 0..layers {
                for (g, hs) in q_heads_of.iter().enumerate() {
                    if hs.is_empty() {
                        continue;
                    }
                    let w = hs.len() as f32 / hq as f32;
                    let ql = &qbar[(l * kv_heads + g) * dim..(l * kv_heads + g + 1) * dim];
                    for (ci, chunk) in cands.chunks(TILE_ROWS).enumerate() {
                        for (ri, b) in chunk.iter().enumerate() {
                            match index.entry_at(*b, l, g) {
                                Some(row) => {
                                    let dst = &mut tile[ri * dim..(ri + 1) * dim];
                                    for (o, bits) in dst.iter_mut().zip(row) {
                                        *o = f16_to_f32(*bits);
                                    }
                                    have[ri] = true;
                                }
                                None => have[ri] = false,
                            }
                        }
                        for (ri, row) in tile.chunks_exact(dim).take(chunk.len()).enumerate() {
                            if !have[ri] {
                                continue;
                            }
                            a[ci * TILE_ROWS + ri] += dot8(row, ql) * (scale * w);
                            sn[ci * TILE_ROWS + ri] += 1;
                        }
                    }
                }
            }
            acc = a;
            seen = sn;
        } else {
            let mut a = vec![0.0f32; cands.len()];
            let mut sn = vec![0usize; cands.len()];
            let per = cands.len().div_ceil(nthreads);
            let parallel_err = std::cell::Cell::new(false);
            // 共享引用在闭包外绑定(`&Vec` 是 Copy ⇒ 每片各持一份引用,不搬走本体)
            let q_heads_ref = &q_heads_of;
            let qbar_ref = &qbar;
            let index_ref = index;
            std::thread::scope(|scope| {
                let mut hs = Vec::with_capacity(nthreads);
                for (ti, (a_out, sn_out)) in a
                    .chunks_mut(per)
                    .zip(sn.chunks_mut(per))
                    .enumerate()
                {
                    let lo = ti * per;
                    let sub = &cands[lo..lo + a_out.len()];
                    let err = &parallel_err;
                    hs.push(scope.spawn(move || {
                        let mut tile = vec![0.0f32; TILE_ROWS * dim];
                        let mut have = vec![false; TILE_ROWS];
                        for l in 0..layers {
                            for (g, hsv) in q_heads_ref.iter().enumerate() {
                                if hsv.is_empty() {
                                    continue;
                                }
                                let w = hsv.len() as f32 / hq as f32;
                                let ql = &qbar_ref[(l * kv_heads + g) * dim
                                    ..(l * kv_heads + g + 1) * dim];
                                for (ci, chunk) in sub.chunks(TILE_ROWS).enumerate() {
                                    for (ri, b) in chunk.iter().enumerate() {
                                        match index_ref.entry_at(*b, l, g) {
                                            Some(row) => {
                                                let dst = &mut tile[ri * dim..(ri + 1) * dim];
                                                for (o, bits) in dst.iter_mut().zip(row) {
                                                    *o = f16_to_f32(*bits);
                                                }
                                                have[ri] = true;
                                            }
                                            None => have[ri] = false,
                                        }
                                    }
                                    for (ri, row) in
                                        tile.chunks_exact(dim).take(chunk.len()).enumerate()
                                    {
                                        if !have[ri] {
                                            continue;
                                        }
                                        a_out[ci * TILE_ROWS + ri] += dot8(row, ql) * (scale * w);
                                        sn_out[ci * TILE_ROWS + ri] += 1;
                                    }
                                }
                            }
                        }
                        let _ = err;
                    }));
                }
                for h in hs {
                    if h.join().is_err() {
                        parallel_err.set(true);
                    }
                }
            });
            if parallel_err.get() {
                return Err(StoreError::Invariant("打分并行线程 panic(结果不可信)".into()));
            }
            acc = a;
            seen = sn;
        }
        // mean over (层, KV 头) —— 缺失行不计入分母(与慢路“缺块不当 0”同律)
        let wl = 1.0 / layers.max(1) as f32;
        for (i, v) in acc.iter().enumerate() {
            let s = seen[i];
            if s == 0 {
                // 一个 (层,KV 头) 都没有 ⇒ 未登记/空块
                r[i] = f32::NEG_INFINITY;
            } else {
                r[i] += *v * wl * (layers as f32 / s as f32);
            }
        }
        return Ok(r);
    }
    // Σ_m(逐 query token;softmax 非线性 ⇒ 不可先求和)
    for t in &face.tokens {
        // mean_{l,h}:先逐 (层,查询头) 聚合,再对所有 (层,头) 取均值
        let pairs = layers * face.q_heads;
        let mut acc = vec![0.0f32; cands.len()];
        // LogitMean:逐候选的“有效 (层,头) 计数”(未登记 = -inf,不入均值;
        // 与 SoftmaxMean 的“价全量分母”同语义:两者都不得把缺块当 0)
        let mut seen = vec![0usize; cands.len()];
        for l in 0..layers {
            for (h, g_raw) in gqa.iter().enumerate() {
                let g = *g_raw as usize;
                if g >= kv_heads {
                    return Err(StoreError::Invariant(format!("GQA 映射越界:{g} ≥ {kv_heads}")));
                }
                let q_off = (l * face.q_heads + h) * dim;
                // ① 逐候选块的点积(**规范序直取**:`entry_at` 零扫描;
                // 未登记块 = -inf ⇒ 任何口径都不会把它当正常块)
                let mut logits = vec![f32::NEG_INFINITY; cands.len()];
                for (i, b) in cands.iter().enumerate() {
                    let Some(row) = index.entry_at(*b, l, g) else { continue };
                    let mut dot = 0.0f32;
                    for (d, bits) in row.iter().enumerate() {
                        dot += t[q_off + d] * f16_to_f32(*bits);
                    }
                    logits[i] = dot * scale;
                }
                if tuning.mode == ScoreMode::LogitMean {
                    // ② 直取 logit(无 softmax):只在**本 (层,头) 有有效值**的候选上累加
                    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if !max.is_finite() {
                        continue;
                    }
                    for (i, x) in logits.iter().enumerate() {
                        if x.is_finite() {
                            acc[i] += *x;
                            seen[i] += 1;
                        }
                    }
                    continue;
                }
                // ② softmax over 候选块(全 -inf ⇒ 跳过该 (层,头))
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if !max.is_finite() {
                    continue;
                }
                let mut sum = 0.0f32;
                for x in logits.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                if sum <= 0.0 {
                    continue;
                }
                for (i, x) in logits.iter().enumerate() {
                    acc[i] += *x / sum;
                }
            }
        }
        // mean_{l,h} + Σ_m
        if tuning.mode == ScoreMode::LogitMean {
            // 逐候选自己的有效 (层,头) 数作分母;一个都没 = -inf(永不入选)
            for (i, v) in acc.iter().enumerate() {
                r[i] += if seen[i] == 0 { f32::NEG_INFINITY } else { *v / seen[i] as f32 };
            }
            continue;
        }
        let denom = pairs.max(1) as f32;
        for (i, v) in acc.iter().enumerate() {
            r[i] += *v / denom;
        }
    }
    Ok(r)
}

/// f16 位型 → f32 的**公开口**(引擎侧解包用;与内部同一实现,不留第二套)。
pub fn f16_to_f32_pub(bits: u16) -> f32 {
    f16_to_f32(bits)
}

/// f16 位型 → f32(索引侧半精度;`half` 依赖不入 store 的依赖面,自算)。
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // 次正规数:规格化
            let mut e = -1i32;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = (m & 0x3ff) << 13;
            let exp32 = (127 - 15 + 1 + e) as u32;
            (sign << 31) | (exp32 << 23) | m
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// f32 → f16 位型(索引登记侧用;RNE 舍入)。
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let frac = b & 0x7f_ffff;
    if exp <= 0 {
        // 下溢:次正规或零(索引只用于排序,极值损失可接受)
        if exp < -10 {
            return sign;
        }
        let m = (frac | 0x80_0000) >> (1 - exp + 13);
        return sign | m as u16;
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // 饱和(inf)
    }
    // RNE:看被截掉的 13 位
    let round = 0x1000u32;
    let mut f = frac >> 13;
    let rem = frac & 0x1fff;
    if rem > round || (rem == round && (f & 1) == 1) {
        f += 1;
        if f == 0x400 {
            // 进位溢出到指数
            return sign | (((exp + 1) as u16) << 10);
        }
    }
    sign | ((exp as u16) << 10) | f as u16
}

// ─────────────────── 覆盖率(量具;B-0 硬门 / 约定实证) ───────────────────

/// **逐块注意力质量**(量具的一半):给一个 query 对全部历史 token 的 logits
/// (或任意非负权重)与"token → 块号"映射,返回每块的质量占比(Σ=1)。
///
/// 来源由调用方给(将来 = engine 的 attention 读出口);本函数**纯 host**,
/// 与 `score_blocks` 的排序面**同序可比**。
pub fn block_mass(logits: &[f32], block_of_token: &[usize], n_blocks: usize) -> Vec<f32> {
    if logits.len() != block_of_token.len() || n_blocks == 0 {
        return vec![0.0; n_blocks];
    }
    // softmax over token(数值稳定;全 -inf ⇒ 全零)
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; n_blocks];
    }
    let mut mass = vec![0.0f32; n_blocks];
    let mut sum = 0.0f32;
    for (l, b) in logits.iter().zip(block_of_token) {
        let w = (*l - max).exp();
        if *b < n_blocks {
            mass[*b] += w;
        }
        sum += w;
    }
    if sum > 0.0 {
        for m in mass.iter_mut() {
            *m /= sum;
        }
    }
    mass
}

/// **top-k 覆盖率曲线**(量具的另一半;论文 G5③:top-8 ≈ 66.5% / top-16 ≈ 77%)。
///
/// `attn_mass` = 真实逐块质量([`block_mass`] 的输出或任何同序权重);
/// `scores` = 索引打分([`score_blocks`] 的输出,**同序**);
/// `ks` = 要报的 k 值。返回 `(k, 覆盖率)`;覆盖率 = 按分排序前 k 块的质量之和。
///
/// ⚠️ **这条曲线同时是"去 RoPE 配对约定"的实证**:约定选错 ⇒ 索引打分退化为
/// 噪声 ⇒ 覆盖率掉到 ≈ k/块数 的随机水平(一眼可辨)。
pub fn coverage_curve(attn_mass: &[f32], scores: &[f32], ks: &[usize]) -> Vec<(usize, f64)> {
    let n = attn_mass.len().min(scores.len());
    if n == 0 {
        return ks.iter().map(|k| (*k, 0.0)).collect();
    }
    let total: f64 = attn_mass[..n].iter().map(|m| *m as f64).sum();
    if total <= 0.0 {
        return ks.iter().map(|k| (*k, 0.0)).collect();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        scores[*b].partial_cmp(&scores[*a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::with_capacity(ks.len());
    for k in ks {
        let kk = (*k).min(n);
        let hit: f64 = order[..kk].iter().map(|i| attn_mass[*i] as f64).sum();
        out.push((*k, hit / total));
    }
    out
}

// ───────────────────────────── 选择 ─────────────────────────────

/// 选择预算(论文:`N(Q_τ) + B_blk·|R_τ| ≤ B_a`)。
#[derive(Clone, Copy, Debug)]
pub struct SelectBudget {
    /// 历史块预算(块数)。
    pub blocks: usize,
    /// **保底 sink**(时间序最前的若干块;论文:always retained)。
    pub sink: usize,
    /// **保底 recent**(时间序最后的若干块)。
    pub recent: usize,
}

impl SelectBudget {
    /// 构造(预算为 0 = 选择层关;调用方据此走 dense 路径)。
    pub fn new(blocks: usize, sink: usize, recent: usize) -> Self {
        Self { blocks, sink, recent }
    }
}

/// 选中的历史工作集(**时间序**;论文:选中后按时间序重排再物化)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockSet {
    /// 选中的块(时间序 = `order` 的顺序)。
    pub blocks: Vec<ChainHash>,
    /// 因预算被挤掉的候选数(报告面)。
    pub dropped: usize,
    /// 保底 sink/recent 占的位数(报告面)。
    pub reserved: usize,
}

/// **纯近期选择**(无打分、无排序):sink 保底 + recent 保底 + **从尾往前填满**。
///
/// 用途:**两次完整召回之间的“廉价边界”**(见 `OWL_RECALL_EVERY`)。
/// 口径与参考实现的 prefill 压力路径(`pick_prefill_ungrouped`)同构 ——
/// 那里也只做「sink + mandatory + 从最新往前填」,**不打分不排序**。
///
/// 为什么不复用 `select` 传空分:那会让 `score_of` 全 `-inf`,sink/recent 之外的名额
/// **填不满**(优先级不明确);这里的语义是**明确的近期优先**。
pub fn select_recency(order: &[ChainHash], budget: SelectBudget) -> BlockSet {
    let n = order.len();
    if budget.blocks == 0 || n == 0 {
        return BlockSet::default();
    }
    let sink_n = budget.sink.min(n);
    let recent_n = budget.recent.min(n.saturating_sub(sink_n));
    let mut picked: Vec<bool> = vec![false; n];
    for p in picked.iter_mut().take(sink_n) {
        *p = true;
    }
    for p in picked.iter_mut().skip(n.saturating_sub(recent_n)) {
        *p = true;
    }
    let reserved = picked.iter().filter(|x| **x).count();
    // 从尾往前填(越近越先留);已被保底占位的不重复计数
    let mut count = reserved;
    for i in (0..n).rev() {
        if count >= budget.blocks {
            break;
        }
        if !picked[i] {
            picked[i] = true;
            count += 1;
        }
    }
    let dropped = n - count;
    let blocks = order
        .iter()
        .enumerate()
        .filter(|(i, _)| picked[*i])
        .map(|(_, b)| *b)
        .collect();
    BlockSet { blocks, dropped, reserved }
}

/// **选择**(论文 §4.3):sink + recent 保底 → 按分补足 → **按时间序重排**。
///
/// `order` = 候选块的**时间序**(论文的 chronology;不在 `order` 里的候选被忽略)。
/// `scored` = [`score_blocks`] 的输出(与 `cands` 同序,本函数内部转 map)。
pub fn select(
    scored: &[(ChainHash, f32)],
    order: &[ChainHash],
    budget: SelectBudget,
) -> BlockSet {
    if budget.blocks == 0 {
        return BlockSet::default();
    }
    // **分数表一次建完**(2026-10-19 性能修正):原实现把 `score_of` 写成对 `scored`
    // 的**线性查找**,而建 `rest` 时逐元素调一次 ⇒ **O(B²)**。B=8192 时 =
    // 3300 万次比对/边界(纯 latent 开销,与打分无关),在“每块边界召回一次”下
    // 会随 ctx 平方增长。一次建表 ⇒ O(B)。
    let mut table: std::collections::HashMap<ChainHash, f32> =
        std::collections::HashMap::with_capacity(scored.len());
    for (b, s) in scored {
        table.insert(*b, *s);
    }
    let score_of =
        |b: &ChainHash| -> f32 { table.get(b).copied().unwrap_or(f32::NEG_INFINITY) };
    // ① 保底:sink(时间序最前) + recent(时间序最后)
    let n = order.len();
    let sink_n = budget.sink.min(n);
    let recent_n = budget.recent.min(n.saturating_sub(sink_n));
    let mut picked: Vec<bool> = vec![false; n];
    for p in picked.iter_mut().take(sink_n) {
        *p = true;
    }
    for p in picked.iter_mut().skip(n.saturating_sub(recent_n)) {
        *p = true;
    }
    let reserved = picked.iter().filter(|x| **x).count();
    // ② 按分补足(分高者优先;同分 = 时间序靠后优先 —— 论文的"越近越可能有用")
    let mut rest: Vec<(usize, f32)> = (0..n)
        .filter(|i| !picked[*i])
        .map(|i| (i, score_of(&order[i])))
        .collect();
    rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)));
    let room = budget.blocks.saturating_sub(reserved);
    for (i, _) in rest.iter().take(room) {
        picked[*i] = true;
    }
    let dropped = n - picked.iter().filter(|x| **x).count();
    // ③ 时间序重排(论文:restore chronological order before materializing)
    let blocks = order
        .iter()
        .enumerate()
        .filter(|(i, _)| picked[*i])
        .map(|(_, b)| *b)
        .collect();
    BlockSet { blocks, dropped, reserved }
}

// ───────────────────────────── 装配计划 ─────────────────────────────

/// 单块的位置动作(论文 §4.4 的三态 + 搬入)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosAction {
    /// 已在设备 + 紧凑位置未变 ⇒ **零动作**(复用物理页)。
    Keep,
    /// 已在设备但位置变了 ⇒ **有界 delta re-RoPE**(在页内原地重打)。
    DeltaReRope,
    /// delta 次数超限 ⇒ 从 **raw K 重建**一次(论文:防低精度累加误差)。
    Rebuild,
    /// 不在设备(host/盘)⇒ **搬入 + 按新位置重打 RoPE**。
    StageIn,
}

/// 一块在**新执行视图**里的位置帧。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosFrame {
    /// 块身份(内容身份;与 A 轴同源)。
    pub block: ChainHash,
    /// 紧凑位置(视图内 token 起点)。
    pub pos: u32,
    /// 该块要做的位置动作。
    pub action: PosAction,
}

/// **装配计划**(纯 host;执行归 engine:页表写 + re-RoPE + gather/scatter 搬运)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewPlan {
    /// 逐块位置帧(**时间序**;页表按此序构造 ⇒ 零拷贝别名)。
    pub frames: Vec<PosFrame>,
    /// 需 stage-in 的块(物理搬入)。
    pub stage_in: Vec<ChainHash>,
    /// 需从 raw K 重建的块。
    pub rebuild: Vec<ChainHash>,
    /// 视图位置跨度(紧凑窗口的 token 数 = 块数 × 块长)。
    pub pos_span: u32,
}

impl ViewPlan {
    /// 需要 re-RoPE 的块数(delta + 重建 + 搬入)。
    pub fn re_rope_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.action != PosAction::Keep)
            .count()
    }
}

/// **装配视图**(论文 §4.4):`W_t` 选中集 → 紧凑位置 + 逐块动作。
///
/// - `prev` = **上一步视图**的 `(块, 位置, 已 delta 次数)`(论文 `G_{t-1}`);
/// - `delta_limit` = 有界 delta re-RoPE 的上限(超限 ⇒ [`PosAction::Rebuild`]);
/// - 位置从 `base_pos` 起,按 `set.blocks` 的时间序**连续**分配(紧凑窗口)。
///
/// 三态判定即论文的 `R_t / L_t / E_t` 分解 + `L_physical = W_t \ G_{t-1}`:
/// 在 `prev` 里且位置相同 = `Keep`;在 `prev` 里但位置变了 = delta/重建;
/// 不在 `prev` 里 = `StageIn`(出视图的块由调用方 stage-out)。
pub fn assemble_view(
    set: &BlockSet,
    prev: &[(ChainHash, u32, u32)],
    base_pos: u32,
    block_tokens: u32,
    delta_limit: u32,
) -> ViewPlan {
    let mut plan = ViewPlan::default();
    let mut pos = base_pos;
    for b in &set.blocks {
        let action = match prev.iter().find(|(k, _, _)| k == b) {
            None => {
                plan.stage_in.push(*b);
                PosAction::StageIn
            }
            Some((_, old_pos, deltas)) => {
                if *old_pos == pos {
                    PosAction::Keep
                } else if *deltas >= delta_limit {
                    plan.rebuild.push(*b);
                    PosAction::Rebuild
                } else {
                    PosAction::DeltaReRope
                }
            }
        };
        plan.frames.push(PosFrame { block: *b, pos, action });
        pos = pos.saturating_add(block_tokens);
    }
    plan.pos_span = pos.saturating_sub(base_pos);
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx() -> MeanKIndex {
        // 1 层 × 2 KV 头 × dim 2 —— 手算得动
        MeanKIndex::new(2, 1, 2).expect("索引")
    }

    fn entry(layer: u16, kv_head: u16, v: [f32; 2]) -> MeanKEntry {
        MeanKEntry { layer, kv_head, k_bar: v.iter().map(|x| f32_to_f16(*x)).collect() }
    }

    #[test]
    fn f16_round_trip_is_close() {
        for x in [0.0f32, 1.0, -1.0, 0.5, 2.0, -3.25, 0.001] {
            let y = f16_to_f32(f32_to_f16(x));
            assert!((y - x).abs() <= x.abs() * 1e-3 + 1e-6, "{x} → {y}");
        }
    }

    /// **B-0 门**:逆 RoPE 正确性(打 RoPE 再逆 ⇒ 回原值)+ 均值 + 形状。
    #[test]
    fn collector_derope_round_trip_and_mean() {
        let dim = 4usize;
        let mut c = MeanKCollector::new(dim, 1, 1).expect("采集器");
        // 两个 token:位置 0(θ=0 ⇒ cos=1,sin=0)与位置 1(θ=0.5)
        let k0 = [1.0f32, 2.0, 3.0, 4.0];
        let k1 = [0.5f32, -1.0, 2.0, 0.25];
        c.push_pre_rope(0, 0, &k0).expect("pre");
        // 位置 1:先打 RoPE(x' = x·c − y·s;y' = x·s + y·c),再喂"RoPE 后"⇒ 逆回
        let (th0, th1) = (0.3f32, 0.5f32);
        let (c0, s0) = (th0.cos(), th0.sin());
        let (c1, s1) = (th1.cos(), th1.sin());
        let rotated = [
            k1[0] * c0 - k1[1] * s0,
            k1[0] * s0 + k1[1] * c0,
            k1[2] * c1 - k1[3] * s1,
            k1[2] * s1 + k1[3] * c1,
        ];
        c.push_post_rope(0, 0, &rotated, &[c0, c1], &[s0, s1]).expect("post");
        assert_eq!(c.len(), 2);
        let entries = c.finish().expect("收口");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].k_bar.len(), dim);
        let mean = [0.75f32, 0.5, 2.5, 2.125]; // (k0 + k1) / 2
        for (i, m) in mean.iter().enumerate() {
            let got = f16_to_f32(entries[0].k_bar[i]);
            assert!((got - m).abs() < 1e-2, "第 {i} 维:逆 RoPE 后应回原值 {m},得 {got}");
        }
        // 空采集 = 结构性拒
        assert!(MeanKCollector::new(dim, 1, 1).expect("采集器").finish().is_err());
        // 形状错 = 响亮拒
        let mut c2 = MeanKCollector::new(dim, 1, 1).expect("采集器");
        assert!(c2.push_pre_rope(0, 0, &[1.0, 2.0]).is_err());
        assert!(c2.push_post_rope(0, 0, &[0.0; 4], &[1.0], &[0.0]).is_err());
        assert!(c2.push_pre_rope(0, 3, &[0.0; 4]).is_err());
    }

    /// 半维配对(rotate-half)的逆 RoPE 也要对:打 RoPE 再逆 ⇒ 回原值。
    #[test]
    fn collector_derope_halves_round_trip() {
        let dim = 4usize;
        let mut c = MeanKCollector::new(dim, 1, 1).expect("采集器");
        let k = [1.0f32, 2.0, 3.0, 4.0];
        let (c0, s0) = (0.3f32.cos(), 0.3f32.sin());
        let (c1, s1) = (0.5f32.cos(), 0.5f32.sin());
        // rotate-half 正变换:x'[i] = x[i]·c − x[i+half]·s;x'[i+half] = x[i]·s + x[i+half]·c
        let rot = [
            k[0] * c0 - k[2] * s0,
            k[1] * c1 - k[3] * s1,
            k[0] * s0 + k[2] * c0,
            k[1] * s1 + k[3] * c1,
        ];
        c.push_post_rope_halves(0, 0, &rot, &[c0, c1], &[s0, s1]).expect("逆");
        let e = c.finish().expect("收口");
        for (i, want) in k.iter().enumerate() {
            let got = f16_to_f32(e[0].k_bar[i]);
            assert!((got - want).abs() < 1e-2, "第 {i} 维应回 {want},得 {got}");
        }
    }

    #[test]
    fn index_registers_and_validates_shape() {
        let mut ix = idx();
        let a = ChainHash::new(1);
        ix.register(a, vec![entry(0, 0, [1.0, 0.0]), entry(0, 1, [0.0, 1.0])]).expect("登记");
        assert!(ix.contains(a));
        assert_eq!(ix.len(), 1);
        assert_eq!(ix.stats().entries_per_block, 2);
        assert_eq!(ix.block_bytes(), 2 * 2 * 2);
        // 形状错 = 响亮拒
        assert!(ix.register(ChainHash::new(2), vec![entry(0, 0, [1.0, 0.0])]).is_err());
        assert!(ix.register(ChainHash::new(3), vec![entry(1, 0, [1.0, 0.0]), entry(0, 0, [1.0, 0.0])]).is_err());
    }

    #[test]
    fn evict_order_is_recency_then_hits() {
        let mut ix = idx();
        for k in 1..=3u64 {
            ix.register(ChainHash::new(k), vec![entry(0, 0, [1.0, 0.0]), entry(0, 1, [0.0, 1.0])])
                .expect("登记");
        }
        ix.touch(ChainHash::new(1)); // 1 变最新 + 1 次命中
        let cands = ix.evict_candidates(&[ChainHash::new(2)]);
        assert_eq!(cands.len(), 2, "protect 生效");
        assert_eq!(cands[0].0, ChainHash::new(3), "3 最冷(无命中)");
    }

    /// 打分:**手算**一个 (1 层, 1 查询头, dim 2) × 2 块 的例子。
    #[test]
    fn score_matches_hand_computation() {
        let mut ix = idx();
        let a = ChainHash::new(1);
        let b = ChainHash::new(2);
        ix.register(a, vec![entry(0, 0, [1.0, 0.0]), entry(0, 1, [0.0, 0.0])]).expect("A");
        ix.register(b, vec![entry(0, 0, [0.0, 1.0]), entry(0, 1, [0.0, 0.0])]).expect("B");
        // query:1 token,1 查询头 → KV 头 0;q = [1, 0](与 A 同向)
        let mut face = QueryFace::new(2, 1, 1);
        face.push_token(vec![1.0, 0.0]).expect("面");
        let s = score_blocks(&ix, &face, &[a, b], &[0], ScoreTuning::default()).expect("打分");
        // 点积:A = 1.0/√2, B = 0.0 ⇒ softmax 后 A ≈ 0.668, B ≈ 0.332(mean_{l,h} = 恒等)
        assert!((s[0] - 0.668).abs() < 0.01, "A = {}", s[0]);
        assert!((s[1] - 0.332).abs() < 0.01, "B = {}", s[1]);
        assert!((s[0] + s[1] - 1.0).abs() < 1e-5, "softmax 归一到 1");
        // 未登记的块 = 0 分
        let s2 = score_blocks(&ix, &face, &[ChainHash::new(9)], &[0], ScoreTuning::default())
            .expect("打分");
        assert_eq!(s2[0], 0.0);
    }

    /// **B-1 打分之辩**:两种聚合口径 —— ①**排序等价**(同为单调变换内聚)
    /// ②**幅度差异**(softmax 均值被压到 ≈1/块数)。
    #[test]
    fn score_modes_rank_identically_but_differ_in_range() {
        // 8 块 × 2 层 × 1 查询头 × dim 2;k̄ 方向渐变 ⇒ 分值有分级
        let mut ix = MeanKIndex::new(2, 2, 1).expect("索引");
        let cands: Vec<ChainHash> = (0..8u64).map(|i| ChainHash::new(i + 1)).collect();
        for (i, b) in cands.iter().enumerate() {
            let ang = (i as f32) * 0.35;
            let v = vec![ang.cos(), ang.sin()];
            ix.register(*b, vec![entry(0, 0, [v[0], v[1]]), entry(1, 0, [v[0], v[1]])])
                .expect("登记");
        }
        let mut face = QueryFace::new(2, 2, 1);
        face.push_token(vec![1.0, 0.0, 1.0, 0.0]).expect("面");
        let soft = score_blocks(
            &ix, &face, &cands, &[0],
            ScoreTuning { scale: 0.0, mode: ScoreMode::SoftmaxMean },
        )
        .expect("softmax 面");
        let logit = score_blocks(
            &ix, &face, &cands, &[0],
            ScoreTuning { scale: 0.0, mode: ScoreMode::LogitMean },
        )
        .expect("logit 面");
        // ① 排序等价
        let rank = |v: &[f32]| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap());
            idx
        };
        assert_eq!(rank(&soft), rank(&logit), "softmax 单调 ⇒ 两种口径的排序必须一致");
        // ② 幅度:softmax 均值总和 ≈ 1/n_tokens(逐 token 归一)
        let ssum: f32 = soft.iter().sum();
        assert!((ssum - 1.0).abs() < 1e-4, "softmax 口径総和≈1,得 {ssum}");
        assert!(
            soft.iter().all(|v| *v <= 0.5),
            "softmax 均值在候选多时被压到 1/块数:top = {}",
            soft.iter().cloned().fold(f32::MIN, f32::max)
        );
        // logit 口径能够出负数(自然展开;阈值/预算面才是可信值)
        assert!(logit.iter().any(|v| *v < 0.0), "logit 口径应自然展开: {logit:?}");
        // 未登记候选:softmax = 0,logit = -inf(两者都不得当“正常块”)
        let miss = score_blocks(&ix, &face, &[ChainHash::new(77)], &[0], ScoreTuning::default())
            .expect("缺块");
        assert_eq!(miss[0], 0.0);
        let miss2 = score_blocks(
            &ix, &face, &[ChainHash::new(77)], &[0],
            ScoreTuning { scale: 0.0, mode: ScoreMode::LogitMean },
        )
        .expect("缺块");
        assert_eq!(miss2[0], f32::NEG_INFINITY, "logit 口径下缺块 = -inf ⇒ 永不入选");
    }

    /// **B-1 延迟规模门**(纯 host 算术,无设备):27B 几何(48 层 × 8 KV 头
    /// × dim128)× 候选块数 → 打分 + 选择的墙钟时间。
    ///
    /// 口径 = 论文 174ms@256K(8192 块)。**双档**:
    /// - `release` = 真门([512, 2048, 8192],8192 ≤ 300ms);
    /// - `debug` = 只跑 [256, 512](纯 f32 热点在 debug 慢 ~20×,跑 8192 要 ~1min,
    ///   白烧 CI 时间;真数看 release 档)。
    ///
    /// **量级预判**(带宽律):每 (层,查询头) 要流一遍自己的 `k̄` 行集
    /// ⇒ 全量索引读 ≈ `块数×层×头×dim×2B` = 805MB@8192 块,20GB/s 量级
    /// ⇒ **~40ms**(内存带宽) + 0.8GFLOP(算力) ⇒ host 够用,**不需要**上 GPU tile。
    #[test]
    fn bench_selector_latency_scales_with_blocks() {
        // 真实 27B(cf. `models/cyankiwi/Qwen3.8-27B-AWQ-INT4/config.json`):
        // 64 层、每 4 层一个全注意力 ⇒ **16**;hq=24 / hkv=4 / hd=256
        let (dim, layers, kv_heads, hq) = (256usize, 16usize, 4usize, 24usize);
        let gqa: Vec<u16> = (0..hq).map(|h| (h / (hq / kv_heads)) as u16).collect();
        let win = 8usize; // query 窗(与探针同口径)
        let sizes: &[usize] =
            if cfg!(debug_assertions) { &[256, 512] } else { &[512, 2048, 8192] };
        let mut out = Vec::new();
        for n_blocks in sizes.iter().copied() {
            let mut ix = MeanKIndex::new(dim, layers, kv_heads).expect("索引");
            let cands: Vec<ChainHash> = (0..n_blocks as u64).map(|i| ChainHash::new(i + 1)).collect();
            for b in &cands {
                let mut ent = Vec::with_capacity(layers * kv_heads);
                for l in 0..layers {
                    for h in 0..kv_heads {
                        let mut row = vec![0u16; dim];
                        for (d, v) in row.iter_mut().enumerate() {
                            let x = (((l * 31 + h * 7 + d) as f32) * 0.017).sin();
                            *v = f32_to_f16(x);
                        }
                        ent.push(MeanKEntry { layer: l as u16, kv_head: h as u16, k_bar: row });
                    }
                }
                ix.register(*b, ent).expect("登记");
            }
            let mut face = QueryFace::new(dim, layers, hq);
            for m in 0..win {
                let mut flat = vec![0.0f32; layers * hq * dim];
                for (i, v) in flat.iter_mut().enumerate() {
                    *v = (((i + m) as f32) * 0.011).cos();
                }
                face.push_token(flat).expect("面");
            }
            let t0 = std::time::Instant::now();
            let scores = score_blocks(&ix, &face, &cands, &gqa, ScoreTuning::default())
                .expect("打分");
            let set = select(
                &cands
                    .iter()
                    .copied()
                    .zip(scores.iter().copied())
                    .collect::<Vec<_>>(),
                &cands,
                SelectBudget::new(64, 2, 4),
            );
            let dt_paper = t0.elapsed();
            // 快路:窗合并 + logit 面(线性 ⇒ 与逐 token 求和逐位等价)
            let t1 = std::time::Instant::now();
            let fast = score_blocks(
                &ix,
                &face,
                &cands,
                &gqa,
                ScoreTuning { scale: 0.0, mode: ScoreMode::LogitMean },
            )
            .expect("快路打分");
            let _ = select(
                &cands.iter().copied().zip(fast.iter().copied()).collect::<Vec<_>>(),
                &cands,
                SelectBudget::new(64, 2, 4),
            );
            let dt_fast = t1.elapsed();
            out.push((
                n_blocks,
                dt_paper.as_secs_f64() * 1000.0,
                dt_fast.as_secs_f64() * 1000.0,
            ));
            assert_eq!(set.blocks.len(), 64, "预算应满额");
            assert_eq!(set.reserved, 6, "sink/recent 保底计数");
        }
        for (n, paper, fast) in &out {
            eprintln!(
                "[B-1 延迟] 27B 几何 候选 {n} 块 × 窗 {win} ⇒ 论文口径(softmax 均值,逐 token) {paper:.1} ms | 快路(窗合并 + logit 面) {fast:.1} ms"
            );
        }
        // **门分两档**:release 是真门 —— **快路** 8192 块 ≤ 400ms;
        // debug 只证不卡死。
        //
        // 基准对比(本机 EPYC 74F3,release):
        // - **基线 x86-64(SSE2)** = 323ms;
        // - **`-C target-cpu=native`(AVX2+FMA+f16c)= 164.9ms** ⇒ **与论文
        //   174ms@256K 同量级**。基线门留 400ms 是两个构建都能过;要不要全局
        //   开 native 另设(cf. 台账 §6 挂账)。
        // (优化史:7745 → 1183 → 473 → 323ms;三刀 = 窗折叠 + GQA 头折叠
        //  + 扁平索引/dot8;共 24×~47×)
        let fast_worst = out.iter().map(|(_, _, f)| *f).fold(0.0f64, f64::max);
        if cfg!(debug_assertions) {
            assert!(fast_worst < 20_000.0, "debug 档快路 512 块 >20s(复杂度或扫描错了?): {out:?}");
        } else {
            assert!(
                fast_worst <= 400.0,
                "release 门:快路 8192 块打分+选择应 ≤400ms(基线构建;native ≈165ms): {out:?}"
            );
        }
    }

    /// **B-0 量具门**:逐块质量(softmax 归一)+ top-k 覆盖率(含"排序错 ⇒ 掉到随机")。
    #[test]
    fn coverage_measures_ranking_quality() {
        // 4 块 × 1 token:logits 让块 0 最大
        let logits = [2.0f32, 1.0, 0.5, 0.0];
        let block_of = [0usize, 1, 2, 3];
        let mass = block_mass(&logits, &block_of, 4);
        let sum: f32 = mass.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "质量应归一到 1,得 {sum}");
        assert!(mass[0] > mass[1] && mass[1] > mass[2] && mass[2] > mass[3]);
        // 完美排序 ⇒ 覆盖率 = 前缀和
        let perfect = [mass[0], mass[1], mass[2], mass[3]];
        let curve = coverage_curve(&mass, &perfect, &[1, 2, 4]);
        assert!((curve[0].1 - mass[0] as f64).abs() < 1e-6, "top-1 = 块0 质量");
        assert!((curve[1].1 - (mass[0] + mass[1]) as f64).abs() < 1e-6);
        assert!((curve[2].1 - 1.0).abs() < 1e-6, "top-4 = 全量");
        // **排序错 ⇒ 掉到随机**:反序打分 ⇒ top-1 = 最差块
        let reversed = [mass[3], mass[2], mass[1], mass[0]];
        let bad = coverage_curve(&mass, &reversed, &[1]);
        assert!((bad[0].1 - mass[3] as f64).abs() < 1e-6, "反序 ⇒ 只捞到最差块");
        assert!(bad[0].1 < curve[0].1, "约定错/打分噪声一眼可辨");
        // 全 -inf / 空 = 零(不 panic)
        assert_eq!(block_mass(&[f32::NEG_INFINITY; 3], &[0, 1, 2], 3).iter().sum::<f32>(), 0.0);
        assert!(coverage_curve(&[], &[], &[1]).iter().all(|(_, c)| *c == 0.0));
    }

    #[test]
    fn select_reserves_sink_recent_then_topk_then_reorders() {
        let order: Vec<ChainHash> = (1..=8u64).map(ChainHash::new).collect();
        // 分数:越靠后的越高(便于观察"保底 + 补足"的组合)
        let scored: Vec<(ChainHash, f32)> =
            order.iter().enumerate().map(|(i, b)| (*b, i as f32)).collect();
        let set = select(&scored, &order, SelectBudget::new(4, 1, 1));
        // 保底 = 块1(sink) + 块8(recent);再按分补 2 个 ⇒ 7, 6
        assert_eq!(set.blocks, vec![order[0], order[5], order[6], order[7]], "时间序 + 保底 + 补足");
        assert_eq!(set.reserved, 2);
        assert_eq!(set.dropped, 4);
        // 预算 0 = 选择层关
        assert!(select(&scored, &order, SelectBudget::new(0, 1, 1)).blocks.is_empty());
    }

    #[test]
    fn view_plan_actions_and_span() {
        let a = ChainHash::new(1);
        let b = ChainHash::new(2);
        let c = ChainHash::new(3);
        let set = BlockSet { blocks: vec![a, b, c], dropped: 0, reserved: 0 };
        // 上一步:a 在同位置(Keep)、b 位置变了且已 delta 3 次(超限 ⇒ Rebuild)
        let prev = vec![(a, 0u32, 0u32), (b, 64u32, 3u32)];
        let plan = assemble_view(&set, &prev, 0, 32, 3);
        assert_eq!(plan.frames[0].action, PosAction::Keep);
        assert_eq!(plan.frames[1].action, PosAction::Rebuild);
        assert_eq!(plan.frames[2].action, PosAction::StageIn);
        assert_eq!(plan.pos_span, 96, "紧凑窗口 = 3 块 × 32");
        assert_eq!(plan.stage_in, vec![c]);
        assert_eq!(plan.rebuild, vec![b]);
        assert_eq!(plan.re_rope_count(), 2, "Rebuild + StageIn 都要重打");
        // 位置未变 = 零动作
        let plan2 = assemble_view(&set, &[(a, 0, 5), (b, 32, 5), (c, 64, 5)], 0, 32, 3);
        assert_eq!(plan2.re_rope_count(), 0);
        assert!(plan2.stage_in.is_empty());
    }
}
