//! CpuFace:CPU 后端句柄 —— `DeviceClient` 契约的进程内同步实现。
//!
//! 客户端直接持有(`CpuFace::new()` 即后端本体),无管道无 actor:
//! 每个契约方法 = 同步执行 + 直接返回(GPU 版对应物 = GpuClient 提交
//! 进 server 队列,Future pending 至回执)。
//!
//! 与 owl-cuda 的语义对照(契约同源 owl-iface::contract):
//! - `alloc/htod/dtoh/launch/sync`:同一语义,CPU 上"完成"即"返回";
//! - **图捕获/重放**(§18 mock 档):捕获窗 = **只录不发**,重放 = 依次重算
//!   (CPU 无指针烤入,重放语义天然安全);窗内禁搬运/同步 —— **与 GPU server 同律**
//!   (捕获契约是引擎侧的,不许 mock 放宽);
//! - **容量墙 + 故障注入**:mock 档把"显存"变成一个可配上限(记账按**声明 dtype**,
//!   与真卡口径一致),可注入 `PoolExhausted` —— 真卡上做不了的压力/错误路径测试。
//!
//! 默认档(`CpuFace::new()`)行为与改造前**逐条一致**:无限容量、零注入。

use crate::ops;
use crate::value::Value;
use owl_iface::contract::{
    Arg, BlockLedgerRow, Bytes, DeviceClient, Dtype, GraphId, KernelSource, LaunchMsg, ModelError,
    UpChunk,
};
use owl_iface::contract::Shape;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};



/// **mock 档调用/耗时统计**(诊断面;§18)。
///
/// 为什么用静态:face 所有权进了引擎(`Engine::on` 之后测试拿不到它),
/// 而这是**只读诊断账**(与 metrics 同类;`mock_stats_reset` 可复位)。
/// 索引:0=alloc 1=htod 2=upload_pinned 3=upload_and_launch 4=copy_block_at 5=launch
static MOCK_CALLS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static MOCK_NS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];


/// 统计快照(诊断面;`[...]` = 各索引的 (调用次数, 累计纳秒))。
pub fn mock_stats() -> [(u64, u64); 6] {
    let mut out = [(0u64, 0u64); 6];
    for i in 0..6 {
        out[i] = (MOCK_CALLS[i].load(Ordering::Relaxed), MOCK_NS[i].load(Ordering::Relaxed));
    }
    out
}

/// 复位统计(测试间隔离)。
pub fn mock_stats_reset() {
    for i in 0..6 {
        MOCK_CALLS[i].store(0, Ordering::Relaxed);
        MOCK_NS[i].store(0, Ordering::Relaxed);
    }
}

/// 同步段计时(记账到静态;`idx` 见 [`mock_stats`])。
macro_rules! mock_timed {
    ($idx:expr, $body:expr) => {{
        let __t = std::time::Instant::now();
        let __r = $body;
        MOCK_CALLS[$idx].fetch_add(1, Ordering::Relaxed);
        MOCK_NS[$idx].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        __r
    }};
}

/// **mock 档限制**(§18:把 CPU 面当"无卡 GPU server"用)——默认零值 = 现状行为。
#[derive(Clone, Copy, Debug, Default)]
pub struct MockLimits {
    /// "显存"上限字节(0 = 无限)。记账按**声明 dtype**(与真卡一致),
    /// 而不是 CPU 侧 f32 存储的实际占用。
    pub vram_bytes: usize,
    /// 故障注入:第 N 次分配(`alloc`/`htod`/复合上传)返回 `PoolExhausted`(0 = 关)。
    pub fail_alloc_nth: u32,
    /// **计算空转**:`launch` 不路由到朴素算子,只把输出块清零后返回。
    /// 用途:让引擎的 warmup / 全前向在无卡机上"跑得过" ——
    /// **账与搬运等价、数值不等价**(这是 mock 的界线,见 §18)。
    pub null_kernels: bool,
    /// **图捕获能力**(默认 **关**):关闭时 `graph_*` 结构化拒 ——
    /// **默认档与改造前逐条一致**(引擎的"无图能力 ⇒ 回 eager"降级路径
    /// 仍由默认档守着);mock 档置 true 才启用捕获/重放。
    pub graph_capture: bool,
    /// **未初始化块填毒**(默认 **关** = 现状:块按零初始化)。
    ///
    /// 置 true 后 `alloc_uninit` 的新块填 **NaN** 位型 ⇒ "池复用忘 memset /
    /// 依赖 alloc 恒零"这一缺陷类别在 mock 上**当场显形**(真卡无零保证 ⇒
    /// 默认档的零初始化本是乐观的)。2026-10-17 审查 §三轮-C1 落地。
    /// 默认 false ⇒ 存量用例行为逐条不变。
    pub poison_uninit: bool,
}

/// 捕获窗里录下的一条发射(重放所需的最小集 —— `LaunchMsg` 本无 `Clone`)。
#[derive(Clone, Debug)]
struct Recorded {
    kernel: String,
    args: Vec<Arg>,
    out_elems: usize,
}

impl Recorded {
    /// 重建发射消息(`source` 不参与 CPU 路由:只按名字选朴素算子)。
    fn to_msg(&self) -> LaunchMsg {
        LaunchMsg {
            kernel: KernelSource { name: self.kernel.clone(), source: String::new() },
            args: self.args.clone(),
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: self.out_elems,
        }
    }
}

/// CPU 后端(与 GPU server 同一契约;测试/对拍/CPU 推理路径)
pub struct CpuFace {
    blocks: HashMap<u64, Value>,
    next: u64,
    /// **归还记账面**(2026-10-15 段求值 P0):`free` 的 id 顺序入账。
    /// 语义:CPU 面**不真焚数据**(存量用例依赖"free 后仍可读"),但把"谁被归还"
    /// 记下来 —— 段间存活(piecewise 的 keep 不变量)、活性回收覆盖面这些
    /// **记账类判据**必须能观测,否则测不出来。
    freed: Vec<u64>,
    /// 每块的**声明事实**(dtype + 字节数):容量墙记账与上传解码的权威
    /// (真卡口径;`free` 时回冲)。
    declared: HashMap<u64, (Dtype, usize)>,
    /// “显存”已用字节(**增量维护**:alloc/htod 加、free 减 —— 不开着
    /// 遍历全表求和的 O(n²),装载数万块时差距是分钟级)。
    vram_used_bytes: usize,
    /// mock 档(默认 = 无限/无故障)
    mock: MockLimits,
    /// 分配调用计数(故障注入用)
    alloc_calls: u32,
    /// 图捕获态(`None` = 不在捕获窗)
    capture: Option<Vec<Recorded>>,
    /// **累积批的待提交队列**(`queue` 追加 / `flush` 一次提交;2026-10-21)。
    /// CPU 面没有信箱往返 ⇒ flush 逐条即序(segments 如实报 N)。
    pending: Vec<owl_iface::command::Command>,
    /// 已实例化的图(`GraphId` → 录制序列)
    graphs: HashMap<u64, Vec<Recorded>>,
    next_graph: u64,
    /// **pinned 租约池**(CPU 面与 GPU 同语义:租约所有权移入 `upload_pinned`,
    /// 用完**归还池**而不是 drop)。
    ///
    /// §18.4.3 定谳:旧的默认实现每次 `HeapRegion(vec![0u8; n])` 新建 33MB 缓冲
    /// ⇒ 输入侧 staging 每次都冷(装载实测解码 67ns/元素 vs 隔离 2.5ns)。
    /// 池化后同一批缓冲复用 —— 填充方(装载器)与解码方吃的是同一块**热**内存。
    pinned_free: Vec<Box<dyn owl_iface::contract::PinnedRegion + Send>>,
}

impl Default for CpuFace {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFace {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            next: 1,
            freed: Vec::new(),
            declared: HashMap::new(),
            vram_used_bytes: 0,
            mock: MockLimits::default(),
            alloc_calls: 0,
            capture: None,
            graphs: HashMap::new(),
            next_graph: 1,
            pinned_free: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// **mock 档构造**(§18):容量墙 + 故障注入 + 计算空转(无卡开发/压力测试用;
    /// `CpuFace::new()` 行为不变)。
    pub fn mock(limits: MockLimits) -> Self {
        let mut f = Self::new();
        f.mock = limits;
        f
    }

    /// 已归还块 id(记账面;见字段注释)
    pub fn freed_ids(&self) -> &[u64] {
        &self.freed
    }

    /// "显存"已用字节(**声明口径**;容量墙报告/断言面)。
    pub fn vram_used(&self) -> usize {
        self.vram_used_bytes
    }

    /// 分配记账(声明口径;与 [`Self::charge`] 配对)。
    fn note_alloc(&mut self, id: u64, dtype: Dtype, bytes: usize) {
        self.declared.insert(id, (dtype, bytes));
        self.vram_used_bytes += bytes;
    }

    /// 取一块 pinned 租约(池里找**够用且最小**的;没有则新建)。
    fn pinned_take(&mut self, bytes: usize) -> Box<dyn owl_iface::contract::PinnedRegion + Send> {
        let mut best: Option<usize> = None;
        for (i, b) in self.pinned_free.iter().enumerate() {
            if b.as_bytes().len() < bytes {
                continue;
            }
            match best {
                Some(j) if self.pinned_free[j].as_bytes().len() <= b.as_bytes().len() => {}
                _ => best = Some(i),
            }
        }
        match best {
            Some(i) => self.pinned_free.swap_remove(i),
            None => Box::new(owl_iface::contract::HeapRegion(vec![0u8; bytes])),
        }
    }

    /// 归还 pinned 租约(池有上限;过大的缓冲不囤 —— 避免锁页/内存夯高)。
    fn pinned_put(&mut self, buf: Box<dyn owl_iface::contract::PinnedRegion + Send>) {
        const POOL_CAP: usize = 64;
        const KEEP_MAX_BYTES: usize = 256 << 20;
        if self.pinned_free.len() < POOL_CAP && buf.as_bytes().len() <= KEEP_MAX_BYTES {
            self.pinned_free.push(buf);
        }
    }

    /// 池内缓冲数(诊断/测试面)。
    pub fn pinned_pool_len(&self) -> usize {
        self.pinned_free.len()
    }

    /// 声明 dtype(**唯一真源** = `declared` 表;缺失 = F32 兼容默认)。
    ///
    /// 返回句柄上的 `Bytes.dtype` 只是它的投影 —— 2026-10-17 审查定谳:
    /// 此前 `alloc` 把句柄钉成 F32、内部表记真 dtype,两个副本各说一套。
    fn decl_dtype(&self, id: u64) -> Dtype {
        self.declared.get(&id).map(|(d, _)| *d).unwrap_or(Dtype::F32)
    }

    /// **设备字节 ↔ 值平面槽位**的唯一折算口(声明 dtype 是权威)。
    ///
    /// 值平面恒 f32(一元素一槽),而设备字节数是**声明 dtype** 口径
    /// (`len × dsz`)。所有入参字节数必须过这里:非整除 / 越界 = **响亮拒**
    /// (与真卡 `GpuServer::check_range` 同一条语义)。
    ///
    /// 病史(2026-10-17 审查 §A1,实测):清零/拷贝/往返四处曾**硬除 4** +
    /// `.min()` 静默钳位 ⇒ F16/BF16 声明块**静默半做**(8 元素清零只落 4 槽;
    /// `swap` 字节往返把 [1..8] 腐蚀成 [0,1.875,0,2,…])。
    fn slots_at(
        &self,
        id: u64,
        off_bytes: usize,
        len_bytes: usize,
        what: &str,
    ) -> Result<std::ops::Range<usize>, ModelError> {
        let dtype = self.decl_dtype(id);
        let dsz = dtype.size_bytes();
        let n = self
            .blocks
            .get(&id)
            .ok_or(ModelError::DeadBlock { id })?
            .f32
            .len();
        if dsz == 0 || off_bytes % dsz != 0 || len_bytes % dsz != 0 {
            return Err(ModelError::Msg(format!(
                "CpuFace::{what}: 偏移/长度须按声明 dtype 对齐(块 {id} 声明 {dtype:?} dsz {dsz};off {off_bytes}B len {len_bytes}B)"
            )));
        }
        let (off, slots) = (off_bytes / dsz, len_bytes / dsz);
        if off + slots > n {
            return Err(ModelError::Msg(format!(
                "CpuFace::{what}: 块 {id} 越界(槽 {n};需 {off}..{})",
                off + slots
            )));
        }
        Ok(off..off + slots)
    }

    /// 分配前置:故障注入 + 容量墙(越界 = `PoolExhausted`,响亮 ——
    /// 真卡同一条语义,调用方的"容量不足"分支在 mock 上可测)。
    fn charge(&mut self, bytes: usize) -> Result<(), ModelError> {
        self.alloc_calls += 1;
        if self.mock.fail_alloc_nth > 0 && self.alloc_calls == self.mock.fail_alloc_nth {
            return Err(ModelError::PoolExhausted {
                pool: "cpu-mock(故障注入)".into(),
                needed: bytes as u64,
                available: 0,
            });
        }
        let used = self.vram_used_bytes;
        if self.mock.vram_bytes > 0 && used + bytes > self.mock.vram_bytes {
            return Err(ModelError::PoolExhausted {
                pool: "cpu-mock(容量墙)".into(),
                needed: bytes as u64,
                available: self.mock.vram_bytes.saturating_sub(used) as u64,
            });
        }
        Ok(())
    }

    /// 捕获窗守卫:窗内只放行 `alloc` + `launch`(与 GPU server 的 reject 表
    /// **同一条契约** —— mock 不放宽捕获语义,否则测不出引擎的窗内违规)。
    fn capture_guard(&self, what: &str) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(format!(
                "CpuFace::{what}: 捕获窗内禁止搬运/同步(GPU server 同律:窗内只有 alloc + launch)"
            )));
        }
        Ok(())
    }

    /// 捕获期登记(`Some` = 已入窗)—— 交给调用方决定语义。
    fn recording(&self) -> bool {
        self.capture.is_some()
    }
}

/// 发射的**输出块与事实面**(线格式:最后一个 `Block` 槽 = 输出;
/// 捕获期与执行期共用同一条推法)。
fn launch_out_facts(msg: &LaunchMsg) -> Result<(u64, (Dtype, u8)), ModelError> {
    let id = msg
        .args
        .iter()
        .rev()
        .find_map(|a| match a {
            Arg::Block { id, .. } => Some(*id),
            _ => None,
        })
        .ok_or_else(|| ModelError::Msg("launch: args 中无输出块".to_string()))?;
    let facts = msg.args.iter().rev().find_map(|a| a.facts()).unwrap_or((Dtype::F32, 0));
    Ok((id, facts))
}

/// **解码落块**(自由函数)。
///
/// §18.4.4 定谳:`src` 若在 `upload_pinned` 里直接来自 `Box<dyn PinnedRegion>`
/// 的虚调用,LLVM **无法证明 `src` 与 `dst` 不别名** ⇒ 每次写后重载源、禁向量化/预取
/// ⇒ 实测比同代码的独立函数慢 **20-70×**(装载 19s vs 0.95s)。
/// 抽成带 `&[u8]`/`&mut [f32]` 签名的函数 = 把别名保证交给编译器。
/// ⚠️ **必须 `inline(never)`**：内联会把"`src` 来自 dyn 虚调用 ⇒ 别名未知"
/// 的假设传染回调用者，循环重新失优化（实测 19s vs 0.95s）。
#[inline(never)]
fn decode_into(dtype: Dtype, src: &[u8], dst: &mut [f32]) -> Result<(), ModelError> {
    let dsz = dtype.size_bytes();
    // **长度必须精确**(2026-10-17 审查:原先 `.min()` 静默截断 ⇒ 短源被无声吃掉一半)
    if dsz == 0 || src.len() != dst.len() * dsz {
        return Err(ModelError::Msg(format!(
            "decode_into: 字节数 {} ≠ {} 槽 × {dsz}({:?})—— 长度须精确,不许截断",
            src.len(),
            dst.len(),
            dtype
        )));
    }
    match dtype {
        Dtype::F32 => {
            for (i, d) in dst.iter_mut().enumerate() {
                let b = 4 * i;
                *d = f32::from_le_bytes([src[b], src[b + 1], src[b + 2], src[b + 3]]);
            }
        }
        Dtype::F16 => {
            let lut = f16_lut();
            for (i, d) in dst.iter_mut().enumerate() {
                let b = 2 * i;
                *d = lut[u16::from_le_bytes([src[b], src[b + 1]]) as usize];
            }
        }
        Dtype::BF16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                let b = 2 * i;
                *d = f32::from_bits((u16::from_le_bytes([src[b], src[b + 1]]) as u32) << 16);
            }
        }
        other => {
            return Err(ModelError::Msg(format!(
                "decode_into: 不支持的声明 dtype {other:?}"
            )));
        }
    }
    Ok(())
}

/// **值平面 → 设备字节**(`decode_into` 的逆;同一张 dtype 表)。
///
/// 契约:槽数 × dsz == 目标字节数(不等 = 响亮拒)。BF16 与 `decode_into`
/// 同族:**高 16 位截断**(round-trip 逐位可逆;不用 `from_f32` 的舍入,
/// 否则 f32→bf16→f32 不再恒等,字节级往返门会假红)。
#[inline(never)]
fn encode_from(dtype: Dtype, src: &[f32], dst: &mut [u8]) -> Result<(), ModelError> {
    let dsz = dtype.size_bytes();
    if dsz == 0 || dst.len() != src.len() * dsz {
        return Err(ModelError::Msg(format!(
            "encode_from: 字节数 {} ≠ {} 槽 × {dsz}({:?})",
            dst.len(),
            src.len(),
            dtype
        )));
    }
    match dtype {
        Dtype::F32 => {
            for (i, v) in src.iter().enumerate() {
                dst[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        Dtype::F16 => {
            for (i, v) in src.iter().enumerate() {
                dst[2 * i..2 * i + 2].copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
            }
        }
        Dtype::BF16 => {
            for (i, v) in src.iter().enumerate() {
                dst[2 * i..2 * i + 2].copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        other => {
            return Err(ModelError::Msg(format!(
                "encode_from: 不支持的声明 dtype {other:?}"
            )));
        }
    }
    Ok(())
}

impl DeviceClient for CpuFace {
    /// **单条设备命令**(2026-10-21 契约收口):CPU 面 = 同步直调,
    /// "完成即返回";具体语义仍由下方既有覆写承担(不另起一套)。
    async fn exec(
        &mut self,
        cmd: owl_iface::command::Command,
    ) -> Result<owl_iface::command::Outcome, ModelError> {
        use owl_iface::command::{Command as C, Outcome};
        // 队列非空 ⇒ 响亮拒(与 GPU 面同律;CPU 面虽无往返,顺序承诺一致)
        if !self.pending.is_empty() {
            return Err(ModelError::Msg(format!(
                "命令面:还有 {} 条命令未提交(先 `flush()`)",
                self.pending.len()
            )));
        }
        match cmd {
            C::Alloc { dtype, elems, zero, name } => {
                let b = if zero {
                    self.alloc(dtype, elems, &name).await?
                } else {
                    self.alloc_uninit(dtype, elems, &name).await?
                };
                Ok(Outcome::Buf(b))
            }
            C::Free { ids } => {
                self.free(&ids).await?;
                Ok(Outcome::Done)
            }
            C::Htod { dtype, shape, src, stream, name } => {
                // CPU 面无流:承载点即"落地",流选择无语义(不是静默丢弃 ——
                // 上游两条路径最终都落到同一同步拷贝)
                let _ = stream;
                Ok(Outcome::Buf(self.htod(dtype, &shape, &src, &name).await?))
            }
            C::Dtoh { id, off_bytes, want_bytes } => {
                let b = Bytes::new(id, 0, Dtype::U8, 0);
                let mut out = vec![0u8; want_bytes];
                if off_bytes == 0 {
                    self.dtoh(&b, &mut out).await?;
                } else {
                    self.dtoh_at(&b, off_bytes, &mut out).await?;
                }
                Ok(Outcome::Bytes(out))
            }
            C::Launch { msg } => Ok(Outcome::Buf(self.launch(msg).await?)),
            C::Sync => {
                self.sync().await?;
                Ok(Outcome::Done)
            }
            C::GraphBegin => {
                self.graph_begin().await?;
                Ok(Outcome::Done)
            }
            C::GraphEnd => Ok(Outcome::Graph(self.graph_end().await?)),
            C::GraphLaunch { graph } => {
                self.graph_launch(graph).await?;
                Ok(Outcome::Done)
            }
            C::GraphAbort => {
                self.graph_abort().await?;
                Ok(Outcome::Done)
            }
            C::GraphDestroy { graph } => {
                self.graph_destroy(graph).await?;
                Ok(Outcome::Done)
            }
        }
    }

    /// **追加到待提交批**(不入队任何设备动作;CPU 面纯 Host 缓存)。
    fn queue(
        &mut self,
        cmd: owl_iface::command::Command,
    ) -> Result<usize, ModelError> {
        let i = self.pending.len();
        self.pending.push(cmd);
        Ok(i)
    }

    /// **提交并等待本批**:CPU 面逐条即序(segments = N,不冒功)。
    async fn flush(&mut self) -> Result<owl_iface::command::BatchOutcome, ModelError> {
        if self.pending.is_empty() {
            return Ok(owl_iface::command::BatchOutcome::empty());
        }
        let cmds = std::mem::take(&mut self.pending);
        let mut outcomes = Vec::with_capacity(cmds.len());
        for cmd in cmds {
            outcomes.push(self.exec(cmd).await);
        }
        let n = outcomes.len();
        Ok(owl_iface::command::BatchOutcome::collect(outcomes, n))
    }

    /// **批**:CPU 面同线程、零信箱往返 ⇒ 逐条即序。
    /// `segments` 如实报 N(不冒功报 1 —— 它衡量的是"投递段数",CPU 面就是 N)。
    async fn exec_batch(
        &mut self,
        cmds: Vec<owl_iface::command::Command>,
    ) -> Result<owl_iface::command::BatchOutcome, ModelError> {
        if !self.pending.is_empty() {
            return Err(ModelError::Msg(format!(
                "命令面:还有 {} 条命令未提交(先 `flush()`)",
                self.pending.len()
            )));
        }
        let mut outcomes = Vec::with_capacity(cmds.len());
        for cmd in cmds {
            outcomes.push(self.exec(cmd).await);
        }
        let segments = outcomes.len();
        Ok(owl_iface::command::BatchOutcome::collect(outcomes, segments))
    }

    /// 进入捕获窗(不可嵌套;窗内只录不发)。
    async fn graph_begin(&mut self) -> Result<(), ModelError> {
        if !self.mock.graph_capture {
            // **默认档**:无图能力(与改造前一致;引擎据此回 eager)
            return Err(ModelError::Msg(
                "CpuFace: 图捕获未启用(默认档无图能力;测试用 CpuFace::mock(MockLimits{{graph_capture: true, ..}}))"
                    .to_string(),
            ));
        }
        if self.capture.is_some() {
            return Err(ModelError::Msg("CpuFace::graph_begin: 不可嵌套捕获".into()));
        }
        self.capture = Some(Vec::new());
        Ok(())
    }

    /// 结束捕获:实例化图(空窗响亮拒 —— 与 GPU 同律)。
    async fn graph_end(&mut self) -> Result<GraphId, ModelError> {
        // 空窗拒 = **仍在捕获态**(调用方须 `graph_abort` 收尾 —— 与 GPU 同律:
        // 失败不得把窗悄悄关掉)
        {
            let rec = self
                .capture
                .as_ref()
                .ok_or_else(|| ModelError::Msg("CpuFace::graph_end: 未在捕获态".into()))?;
            if rec.is_empty() {
                return Err(ModelError::Msg(
                    "CpuFace::graph_end: 空捕获窗(发射数须 > 0)".into(),
                ));
            }
        }
        let rec = self.capture.take().expect("刚查过");
        let id = self.next_graph;
        self.next_graph += 1;
        self.graphs.insert(id, rec);
        Ok(id)
    }

    /// 重放(CPU 面 = 依次重算;写回捕获时的同一批块 —— 与 GPU 的池块语义一致)。
    async fn graph_launch(&mut self, graph: GraphId) -> Result<(), ModelError> {
        self.capture_guard("graph_launch")?;
        let rec = self
            .graphs
            .get(&graph)
            .cloned()
            .ok_or_else(|| ModelError::Msg(format!("CpuFace::graph_launch: 未知图 {graph}")))?;
        for r in rec {
            let _ = self.launch(r.to_msg()).await?;
        }
        Ok(())
    }

    /// 窗内出错收尾:退出捕获态 + 丢弃录制。
    /// **块不回收**(CPU 面 `free` 不焚数据,窗内 alloc 的块交给调用方后续 free);
    /// GPU 档的"窗内自产块整批作废"由引擎侧不再引用等义。
    async fn graph_abort(&mut self) -> Result<(), ModelError> {
        if self.capture.take().is_none() {
            return Err(ModelError::Msg("CpuFace::graph_abort: 未在捕获态".into()));
        }
        Ok(())
    }

    /// 销毁图(幂等)。
    async fn graph_destroy(&mut self, graph: GraphId) -> Result<(), ModelError> {
        self.graphs.remove(&graph);
        Ok(())
    }

    /// **未初始化分配**(2026-10-17 审查 §三轮-C1):默认 = 与 `alloc` 同(清零,
    /// 存量行为不变 —— 真卡的"未初始化"无零保证,而 mock 此前恒清零,把
    /// "忘 memset"这一类别整体掩盖);`MockLimits::poison_uninit` 置位 ⇒ 新块
    /// 填 **NaN** 位型(`0x7FC0_0000`,判据 = 与自身不相等)。
    async fn alloc_uninit(
        &mut self,
        dtype: Dtype,
        elems: usize,
        name: &str,
    ) -> Result<Bytes, ModelError> {
        let b = self.alloc(dtype, elems, name).await?;
        if self.mock.poison_uninit {
            let v = self
                .blocks
                .get_mut(&b.id)
                .ok_or(ModelError::DeadBlock { id: b.id })?;
            for x in v.f32.iter_mut() {
                *x = f32::from_bits(0x7FC0_0000);
            }
        }
        Ok(b)
    }

    /// 显存分配(清零;CPU = Vec;无流参数 —— 同步直调即序)。
    /// 声明 dtype 放行 F32/F16/BF16(**存储恒 f32**,声明只入记账/解码口径);
    /// U32/U8 拒(执行面无此值域)。
    ///
    /// `_name`(全量记名):CPU 面无显存账房(值块是 host Vec)⇒ 无 OOM
    /// 现场可归因,名字无处落 —— 有意不存(不是遗忘;GPU 面由
    /// `GpuClient`→`Command::Alloc`→块账房 `BlockEntry.name` 承担)。
    async fn alloc(&mut self, dtype: Dtype, elems: usize, _name: &str) -> Result<Bytes, ModelError> {
        match dtype {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => {}
            other => {
                return Err(ModelError::Msg(format!(
                    "CpuFace::alloc: 仅 F32/F16/BF16(值域 f32),得 {other:?}"
                )))
            }
        }
        // 容量墙按**声明 dtype**记账(真卡口径:alloc(F16,1024) 占 2KB,
        // 不因 mock 的 f32 存储而翻倍 —— 否则"卡能装多少"的账失真)
        let bytes = elems * dtype.size_bytes();
        self.charge(bytes)?;
        let v = mock_timed!(0, ops::zeros(Dtype::F32, &vec![elems])?);
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, v);
        self.note_alloc(id, dtype, bytes);
        // 值平面恒 f32(一槽一元素),但**句柄 dtype = 声明 dtype**
        // (2026-10-17 审查:原钉 F32 ⇒ 下游用 `len × dtype.size_bytes()`
        //  推字节的调用点(graph_plan 段/覆盖块)按 4B 算,与真卡 2B 分家)
        Ok(Bytes::new(id, elems, dtype, 0))
    }

    /// 装载(host 字节 → f32 值块);`_name` 同 [`CpuFace::alloc`](有意不存)。
    async fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
        _name: &str,
    ) -> Result<Bytes, ModelError> {
        self.capture_guard("htod")?; // 窗内禁搬运(GPU 同律)
        self.charge(src.len())?;
        let v = mock_timed!(1, ops::htod(dtype, shape, src)?);
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, v);
        self.note_alloc(id, dtype, src.len());
        // 句柄事实:len = **元素数**(契约 `Bytes.len` 元素口径;原先恒 0 ⇒
        // "len × dsz" 推字节的调用点全得 0)
        Ok(Bytes::new(id, owl_iface::contract::numel(shape), dtype, shape.len() as u8))
    }

    /// 归还(记账面覆写;见 [`CpuFace::freed_ids`])—— **不焚数据**:
    /// CPU 面是"记账真值",存量用例在 free 后仍读块属有意行为。
    /// 容量墙回冲:归还的块不再占"显存"(与真卡 free 同义)。
    async fn free(&mut self, ids: &[u64]) -> Result<(), ModelError> {
        self.freed.extend_from_slice(ids);
        for id in ids {
            if let Some((_, bytes)) = self.declared.remove(id) {
                self.vram_used_bytes = self.vram_used_bytes.saturating_sub(bytes);
            }
        }
        Ok(())
    }

    /// 带偏移读(`dtoh_at`;CPU 面 = 值块切片)—— store 的搬迁单测靠它跑真 face。
    async fn dtoh_at(
        &mut self,
        b: &Bytes,
        off_bytes: usize,
        out: &mut [u8],
    ) -> Result<(), ModelError> {
        self.capture_guard("dtoh_at")?;
        // 口径 = **声明 dtype**(与 GpuServer 的 `D2H` 同一条:字节数由调用方
        // 按声明 dtype 给;此前硬除 4 ⇒ F16/BF16 块读半段并当 f32 编码)
        let dtype = self.decl_dtype(b.id);
        let range = self.slots_at(b.id, off_bytes, out.len(), "dtoh_at")?;
        let v = self
            .blocks
            .get(&b.id)
            .ok_or(ModelError::DeadBlock { id: b.id })?;
        encode_from(dtype, &v.f32[range], out)
    }

    /// 收割(**声明 dtype 字节**;长度须与块一致)
    async fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError> {
        self.capture_guard("dtoh")?;
        let dtype = self.decl_dtype(b.id);
        let n = self
            .blocks
            .get(&b.id)
            .ok_or(ModelError::DeadBlock { id: b.id })?
            .f32
            .len();
        let want = n * dtype.size_bytes();
        if want != out.len() {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {} 声明 {dtype:?} 共 {want}B != 收割 {}",
                b.id,
                out.len()
            )));
        }
        let v = self
            .blocks
            .get(&b.id)
            .ok_or(ModelError::DeadBlock { id: b.id })?;
        encode_from(dtype, &v.f32, out)
    }

    /// 发射:按 kernel 名路由到朴素算子(动作表的 CPU 面;
    /// 同步执行 —— "提交即完成",无 sticky 延迟)。
    /// **捕获窗内 = 只录不发**(与 GPU server 同律:图内发射不执行)。
    async fn launch(&mut self, msg: LaunchMsg) -> Result<Bytes, ModelError> {
        if self.recording() {
            let (id, (dtype, rank)) = launch_out_facts(&msg)?;
            if !self.blocks.contains_key(&id) {
                return Err(ModelError::DeadBlock { id });
            }
            let rec = Recorded {
                kernel: msg.kernel.name.clone(),
                args: msg.args.clone(),
                out_elems: msg.out_elems,
            };
            self.capture.as_mut().expect("recording").push(rec);
            return Ok(Bytes::new(id, msg.out_elems, dtype, rank));
        }
        // **计算空转档**(§18):不路由算子,输出块清零后返回。
        // 账与搬运仍真:引擎能在无卡机跑完 warmup/全前向(数值无意义)。
        if self.mock.null_kernels {
            let (id, (dtype, rank)) = launch_out_facts(&msg)?;
            if let Some(v) = self.blocks.get_mut(&id) {
                for x in v.f32.iter_mut() {
                    *x = 0.0;
                }
            }
            return Ok(Bytes::new(id, msg.out_elems, dtype, rank));
        }
        // 参数槽解析:Block → 值块;类型化标量
        let mut vals: Vec<Value> = Vec::with_capacity(msg.args.len());
        let mut u64s: Vec<u64> = Vec::new();
        let mut i32s: Vec<i32> = Vec::new();
        let mut f32s: Vec<f32> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { id, .. } => {
                    vals.push(
                        self.blocks
                            .get(id)
                            .cloned()
                            .ok_or(ModelError::DeadBlock { id: *id })?,
                    )
                }
                Arg::BlockSlice { id, byte_offset, elems, .. } => {
                    // 刀1:连续切片视图 —— host 真值直接切片(CPU 面零指针)。
                    // 折算口径 = **声明 dtype**(与 `slots_at` 同一折算;F32 块 = /4,
                    // 与既有行为一致)。2026-10-17 审查:原硬除 4 + 越界**静默保留整块**
                    // ⇒ 非 F32 块偏移错位、且 `v.shape = vec![n]` 对着整块数据撒谎。
                    let (byte_offset, n) = (*byte_offset as usize, *elems as usize);
                    let dsz = self.decl_dtype(*id).size_bytes();
                    let full = self
                        .blocks
                        .get(id)
                        .cloned()
                        .ok_or(ModelError::DeadBlock { id: *id })?;
                    if dsz == 0 || byte_offset % dsz != 0 {
                        return Err(ModelError::Msg(format!(
                            "launch(BlockSlice): 字节偏移 {byte_offset} 未按声明 dtype 对齐(块 {id} dsz {dsz})"
                        )));
                    }
                    let off = byte_offset / dsz;
                    if off + n > full.f32.len() {
                        return Err(ModelError::Msg(format!(
                            "launch(BlockSlice): 块 {id} 越界(槽 {};需 {off}..{})",
                            full.f32.len(),
                            off + n
                        )));
                    }
                    let mut v = full;
                    v.f32 = v.f32[off..off + n].to_vec();
                    v.shape = vec![n];
                    vals.push(v);
                }
                Arg::U64(v) => u64s.push(*v),
                Arg::I32(v) => i32s.push(*v),
                Arg::F32(v) => f32s.push(*v),
            }
        }
        let out = match msg.kernel.name.as_str() {
            "owl_add_f32" => ops::add(&vals[0], &vals[1])?,
            "owl_mul_f32" => ops::mul(&vals[0], &vals[1])?,
            "owl_silu_f32" => ops::silu(&vals[0])?,
            "owl_sigmoid_f32" => ops::sigmoid(&vals[0])?,
            "owl_matmul_f32" => {
                // 标量与 lower_matmul 对位:m/k/n(标量权威;alloc 块无形状,
                // 原 k 自推 a.shape 被压成 1 —— 2026-09-26 engine M0 逮到)
                let (m, k, n) = (i32s[0] as usize, i32s[1] as usize, i32s[2] as usize);
                ops::matmul(&vals[0], &vals[1], m, k, n)?
            }
            "owl_matmul_nt_f32" => {
                // nt:B [n,k] 直读;标量与 lower_matmul_nt 对位:m/k/n
                let (m, k, n) = (i32s[0] as usize, i32s[1] as usize, i32s[2] as usize);
                ops::matmul_nt(&vals[0], &vals[1], m, k, n)?
            }
            "owl_rmsnorm_f32" => {
                // 标量与 lower_rmsnorm 对位:cols(I32)/eps(F32)/w_off(I32)
                ops::rmsnorm(&vals[0], &vals[1], i32s[0] as usize, f32s[0], i32s[1] != 0)?
            }
            "owl_layernorm_f32" => {
                // 标量与 ops/vis.rs 声明对位:n(I32)/eps(F32);父 = x/weight/bias
                ops::layernorm(&vals[0], &vals[1], &vals[2], i32s[0] as usize, f32s[0])?
            }
            "owl_rope_half_partial_f32" => ops::rope_half_partial(
                &vals[0],
                &vals[1],
                &vals[2],
                &vals[3],
                u64s[0] as usize,
                u64s[1] as usize,
                u64s[2] as usize,
            )?,
            "owl_narrow_strided_f32" => ops::narrow_strided(
                &vals[0],
                u64s[0] as usize,
                u64s[1] as usize,
                u64s[2] as usize,
                u64s[3] as usize,
            )?,
            "owl_gelu_tanh_f32" => ops::gelu_tanh(&vals[0])?,
            "owl_gelu_erf_f32" => ops::gelu_erf(&vals[0])?,
            "owl_bias_add_f32" => ops::bias_add(&vals[0], &vals[1], u64s[0] as usize)?,
            "owl_pos_embed_bilinear_f32" => ops::pos_embed_bilinear(
                &vals[0],
                &vals[1],
                &vals[2],
                u64s[0] as usize,
                u64s[1] as usize,
            )?,
            "owl_image_scatter_f32" => {
                ops::image_scatter(&vals[0], &vals[1], &vals[2], u64s[0] as usize)?
            }
            "owl_vis_attn_full_f32" => ops::attn_full(
                &vals[0],
                &vals[1],
                &vals[2],
                &vals[3],
                u64s[0] as usize,
                u64s[1] as usize,
                u64s[2] as usize,
                i32s[0] as usize,
                f32s[0],
            )?,
            other => {
                return Err(ModelError::Msg(format!(
                    "CpuFace::launch: 未注册的动作 {other}"
                )))
            }
        };
        // 写回 eval 预 alloc 的 out 块(与 GPU 语义一致:发射原位写输出)
        let (id, facts) = launch_out_facts(&msg)?;
        self.blocks.insert(id, out);
        // 出参句柄:事实面由线格式 out 槽承载(CPU 面同律)
        let (dtype, rank) = facts;
        Ok(Bytes::new(id, msg.out_elems, dtype, rank))
    }

    async fn upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        self.capture_guard("upload_pinned")?;
        let __t = std::time::Instant::now();
        // CPU 面 = f32 存储世界，但**解码按声明 dtype**（F32/F16/BF16）；
        // **索引式循环**（§18.4.3 实测：比 `chunks_exact+zip` 快约 2×）。
        // 长度/偏移的非整除与越界在 `slots_at` 里响亮拒(不许静默截尾)。
        let dtype = self.decl_dtype(dst.id);
        if len_bytes > buf.as_bytes().len() {
            return Err(ModelError::Msg(format!(
                "upload_pinned: 租约 {}B < 声明上传 {len_bytes}B(块 {})",
                buf.as_bytes().len(),
                dst.id
            )));
        }
        let bytes = &buf.as_bytes()[..len_bytes];
        let range = self.slots_at(dst.id, offset_bytes, len_bytes, "upload_pinned")?;
        let v = self
            .blocks
            .get_mut(&dst.id)
            .ok_or(ModelError::DeadBlock { id: dst.id })?;
        decode_into(dtype, bytes, &mut v.f32[range])?;
        MOCK_CALLS[2].fetch_add(1, Ordering::Relaxed);
        MOCK_NS[2].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // **租约归还池**(与 GPU 面"完成回调归还其池"同语义;§18.4.3)
        self.pinned_put(buf);
        Ok(())
    }

    /// pinned 租约**池化发放**(覆盖默认 `HeapRegion(vec![0u8; n])`:复用热缓冲)。
    async fn alloc_pinned(
        &mut self,
        bytes: usize,
        _name: &str,
    ) -> Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError> {
        Ok(self.pinned_take(bytes))
    }

    /// **复合口(E3):上传并发射**(装载 cast 路径的写口)—— CPU 面天然同步,
    /// 逐 chunk 写入后发射;与 GPU 的"设备端事件序"同义(无 host 闸栏语义差)。
    async fn upload_and_launch(
        &mut self,
        dst: &Bytes,
        chunks: Vec<UpChunk>,
        msg: LaunchMsg,
    ) -> Result<Bytes, ModelError> {
        self.capture_guard("upload_and_launch")?;
        let __t = std::time::Instant::now();
        for c in chunks {
            self.upload_pinned(c.lease, dst, c.offset_bytes, c.len_bytes).await?;
        }
        let r = self.launch(msg).await;
        MOCK_CALLS[3].fetch_add(1, Ordering::Relaxed);
        MOCK_NS[3].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    /// **块账本投影**(调试面):全活块 (id, 字节, 单调假地址, 账本 0)。
    /// 假地址按 id 派生(64KiB 步进 ⇒ 区间不重叠)—— 让"同 id 两地址 /
    /// 两活块区间相交"这类审计在 mock 上**可判定**。
    async fn block_ledger_dump(&mut self) -> Result<Vec<BlockLedgerRow>, ModelError> {
        let mut rows: Vec<BlockLedgerRow> = self
            .declared
            .iter()
            .map(|(id, (_dtype, bytes))| BlockLedgerRow {
                name: String::new(),
                id: *id,
                bytes: *bytes,
                addr: 0x1000_0000u64 + id * 0x1_0000,
                ledger: 0,
            })
            .collect();
        rows.sort_by_key(|r| r.id);
        Ok(rows)
    }

    /// 池回收:CPU 面无设备池 ⇒ 无可回收(0;"不刷告警"由调用方按
    /// `device_kernels()` 区分)。
    async fn mem_pool_trim(&mut self) -> Result<u64, ModelError> {
        Ok(0)
    }

    async fn write_block_f32(
        &mut self,
        dst: &Bytes,
        offset_elems: usize,
        data: &[f32],
    ) -> Result<(), ModelError> {
        self.capture_guard("write_block_f32")?;
        let v = self
            .blocks
            .get_mut(&dst.id)
            .ok_or(ModelError::DeadBlock { id: dst.id })?;
        let end = offset_elems + data.len();
        if v.f32.len() < end {
            return Err(ModelError::Msg(format!(
                "write_block_f32: 块 {} 长 {} < 写入终点 {end}",
                dst.id,
                v.f32.len()
            )));
        }
        v.f32[offset_elems..end].copy_from_slice(data);
        Ok(())
    }

    /// 块清零(按**声明 dtype** 口径折算元素;越界/非整除 = 响亮拒)
    async fn memset_zero(&mut self, dst: &Bytes, len_bytes: usize) -> Result<(), ModelError> {
        self.capture_guard("memset_zero")?;
        let range = self.slots_at(dst.id, 0, len_bytes, "memset_zero")?;
        let v = self
            .blocks
            .get_mut(&dst.id)
            .ok_or(ModelError::DeadBlock { id: dst.id })?;
        for x in &mut v.f32[range] {
            *x = 0.0;
        }
        Ok(())
    }

    /// 设备内块→块拷贝(CPU 面 = 声明 dtype 字节的编解码中转;E2c 快照通道)。
    ///
    /// 释义 = **字节级拷贝**(真卡 `cudaMemcpyAsync` 同义):源端编码 →
    /// 目标端解码 ⇒ src/dst 声明 dtype 不同也正确(槽数不必相等)。
    async fn copy_block_at(
        &mut self,
        src: &Bytes,
        src_off_bytes: usize,
        dst: &Bytes,
        dst_off_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        self.capture_guard("copy_block_at")?;
        let (src_dtype, src_range) =
            (self.decl_dtype(src.id), self.slots_at(src.id, src_off_bytes, len_bytes, "copy_block_at(src)")?);
        let dst_range = self.slots_at(dst.id, dst_off_bytes, len_bytes, "copy_block_at(dst)")?;
        // 源 -> 设备字节
        let bytes = {
            let v = self.blocks.get(&src.id).ok_or(ModelError::DeadBlock { id: src.id })?;
            let mut buf = vec![0u8; len_bytes];
            encode_from(src_dtype, &v.f32[src_range], &mut buf)?;
            buf
        };
        // 设备字节 -> 目标
        let dtype = self.decl_dtype(dst.id);
        let v = self
            .blocks
            .get_mut(&dst.id)
            .ok_or(ModelError::DeadBlock { id: dst.id })?;
        decode_into(dtype, &bytes, &mut v.f32[dst_range])
    }

    /// 带偏移的块清零(E2b:GDN 按格重置;声明 dtype 口径)
    async fn memset_zero_at(
        &mut self,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        self.capture_guard("memset_zero_at")?;
        let range = self.slots_at(dst.id, offset_bytes, len_bytes, "memset_zero_at")?;
        let v = self
            .blocks
            .get_mut(&dst.id)
            .ok_or(ModelError::DeadBlock { id: dst.id })?;
        for x in &mut v.f32[range] {
            *x = 0.0;
        }
        Ok(())
    }

    async fn sync(&mut self) -> Result<(), ModelError> {
        self.capture_guard("sync")?;
        Ok(()) // CPU:无在飞操作(同步直调,天然无积压)
    }
}

/// **f16 → f32 查找表**(65536 项,256KB —— 装进 L2)。
///
/// 为什么用表:`half::f16::to_f32()` 是**跨 crate 调用 + 软件位路径**,
/// dev profile 下实测 **~80ns/次**(769M 次 = 61s);查表后 ~1-2ns。
/// 表用 half 本体构建 ⇒ 与 half 语义**逐值一致**(`f16_lut_matches_half_bit_for_bit` 守)。
fn f16_lut() -> &'static [f32; 65536] {
    static LUT: std::sync::OnceLock<[f32; 65536]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 65536];
        for (i, v) in t.iter_mut().enumerate() {
            *v = half::f16::from_bits(i as u16).to_f32();
        }
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_iface::contract::{HeapRegion, PinnedRegion};

    /// `owl_add_f32` 的线格式迷你样例(输出块 = **最后一个** Block 槽 —— 契约)。
    fn msg_add(a: u64, b: u64, out: u64, n: usize) -> LaunchMsg {
        LaunchMsg {
            kernel: KernelSource { name: "owl_add_f32".into(), source: String::new() },
            args: vec![
                Arg::Block { id: a, dtype: Dtype::F32, rank: 1 },
                Arg::Block { id: b, dtype: Dtype::F32, rank: 1 },
                Arg::Block { id: out, dtype: Dtype::F32, rank: 1 },
            ],
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            out_elems: n,
        }
    }

    async fn read_f32(f: &mut CpuFace, b: &Bytes, n: usize) -> Vec<f32> {
        let mut buf = vec![0u8; 4 * n];
        f.dtoh(b, &mut buf).await.expect("dtoh");
        buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// **图捕获/重放(§18)**:窗内只录不发 → 重放 = 重算;空窗/嵌套/窗内搬运
    /// 全部**响亮拒**(与 GPU server 同一条捕获契约 —— mock 不放宽)。
    #[tokio::test]
    async fn graph_capture_records_and_replay_recomputes() {
        // 图能力 = mock 档显式开(默认档无图,与改造前一致)
        let mut f = CpuFace::mock(MockLimits { graph_capture: true, ..Default::default() });
        let a = f.alloc(Dtype::F32, 4, "t").await.expect("a");
        let b = f.alloc(Dtype::F32, 4, "t").await.expect("b");
        let out = f.alloc(Dtype::F32, 4, "t").await.expect("out");
        f.write_block_f32(&a, 0, &[1.0, 2.0, 3.0, 4.0]).await.expect("写 a");
        f.write_block_f32(&b, 0, &[10.0, 20.0, 30.0, 40.0]).await.expect("写 b");

        // 默认档拒图(回归哨兵:引擎的"无图能力"降级路径靠它守着)
        {
            let mut plain = CpuFace::new();
            assert!(plain.graph_begin().await.is_err(), "默认档必须拒图");
        }

        f.graph_begin().await.expect("进窗");
        f.launch(msg_add(a.id, b.id, out.id, 4)).await.expect("录一条");
        let g = f.graph_end().await.expect("实例化");

        assert_eq!(read_f32(&mut f, &out, 4).await[0], 0.0, "捕获窗内只录不发");
        f.graph_launch(g).await.expect("重放");
        let got = read_f32(&mut f, &out, 4).await;
        assert_eq!(got, vec![11.0, 22.0, 33.0, 44.0], "重放 = 重算");
        f.graph_launch(g).await.expect("幂等重放");
        f.graph_destroy(g).await.expect("销毁");
        assert!(f.graph_launch(g).await.is_err(), "已销毁图必须拒");

        // 空窗 / 嵌套 / 窗内搬运
        f.graph_begin().await.expect("进窗");
        assert!(f.graph_end().await.is_err(), "空捕获窗响亮拒");
        assert!(f.graph_begin().await.is_err(), "不可嵌套捕获");
        assert!(
            f.htod(Dtype::F32, &vec![1], &[0u8; 4], "t").await.is_err(),
            "窗内禁 htod(与 GPU 同律)"
        );
        assert!(f.sync().await.is_err(), "窗内禁 sync");
        assert!(
            f.copy_block_at(&a, 0, &out, 0, 4).await.is_err(),
            "窗内禁设备内拷贝"
        );
        f.graph_abort().await.expect("作废");
        assert!(f.sync().await.is_ok(), "出窗后同步恢复");
    }

    /// **容量墙 + 故障注入(§18)**:容量不足 = `PoolExhausted`(真卡同一条语义);
    /// `free` 回冲;注入命中第 N 次分配。
    #[tokio::test]
    async fn mock_vram_wall_and_fault_injection_are_loud() {
        // 墙:1024 字节 ⇒ 一个 256-elem f32 块正好占满
        let mut f = CpuFace::mock(MockLimits { vram_bytes: 1024, ..Default::default() });
        let a = f.alloc(Dtype::F32, 256, "t").await.expect("恰满");
        assert_eq!(f.vram_used(), 1024);
        let err = f.alloc(Dtype::F32, 1, "t").await.expect_err("越界必拒");
        assert!(matches!(err, ModelError::PoolExhausted { .. }), "得 {err}");
        assert!(format!("{err}").contains("容量墙"), "{err}");
        // free 回冲(与真卡口径一致)
        f.free(&[a.id]).await.expect("归还");
        assert_eq!(f.vram_used(), 0);
        let _b = f.alloc(Dtype::F32, 256, "t").await.expect("归还后可再分配");

        // 声明口径:alloc(F16, n) 只记 2n 字节(不按 mock 的 f32 存储翻倍)
        let mut w = CpuFace::mock(MockLimits { vram_bytes: 4096, ..Default::default() });
        let _h = w.alloc(Dtype::F16, 1024, "t").await.expect("f16 块");
        assert_eq!(w.vram_used(), 2048, "按声明 dtype 记账(真卡口径)");

        // 注入:第 2 次分配失败(一次性)
        let mut g = CpuFace::mock(MockLimits { fail_alloc_nth: 2, ..Default::default() });
        let _x = g.alloc(Dtype::F32, 1, "t").await.expect("第 1 次");
        let err = g.alloc(Dtype::F32, 1, "t").await.expect_err("第 2 次注入");
        assert!(format!("{err}").contains("故障注入"), "{err}");
        let _y = g.alloc(Dtype::F32, 1, "t").await.expect("注入是一次性的");
    }

    /// **复合上传口(§18)**:`upload_and_launch` 按**声明 dtype** 解码(BF16 源)
    /// + 就地发射 —— 装载 cast 路径在 CPU 档可跑的关键。
    #[tokio::test]
    async fn upload_and_launch_decodes_declared_dtype() {
        let mut f = CpuFace::new();
        let a = f.alloc(Dtype::BF16, 2, "t").await.expect("a(声明 BF16)");
        let b = f.alloc(Dtype::F32, 2, "t").await.expect("b");
        let out = f.alloc(Dtype::F32, 2, "t").await.expect("out");
        f.write_block_f32(&b, 0, &[1.0, 1.0]).await.expect("写 b");
        // BF16 源:1.0 = 0x3F80,2.0 = 0x4000
        let src: Vec<u8> = [0x3F80u16, 0x4000u16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let lease: Box<dyn PinnedRegion + Send> = Box::new(HeapRegion(src.clone()));
        f.upload_and_launch(
            &a,
            vec![UpChunk { offset_bytes: 0, len_bytes: src.len(), lease }],
            msg_add(a.id, b.id, out.id, 2),
        )
        .await
        .expect("复合上传 + 发射");
        let got = read_f32(&mut f, &out, 2).await;
        assert_eq!(got, vec![2.0, 3.0], "BF16 解码(1,2) + b(1,1) = (2,3)");
    }

    /// **计算空转档(§18)**:未注册的核(引擎真实前向会发的 attention/GDN 族)
    /// 不再报错 —— 输出清零返回(账跑得通、数值不算)。
    #[tokio::test]
    async fn null_kernels_turn_unregistered_ops_into_noops() {
        let mut f = CpuFace::mock(MockLimits { null_kernels: true, ..Default::default() });
        let a = f.alloc(Dtype::F32, 4, "t").await.expect("a");
        let out = f.alloc(Dtype::F32, 4, "t").await.expect("out");
        f.write_block_f32(&a, 0, &[7.0; 4]).await.expect("写");
        f.write_block_f32(&out, 0, &[9.0; 4]).await.expect("写 out");
        // 未注册核:空转档不报错
        let unsupported = LaunchMsg {
            kernel: KernelSource { name: "owl_attention_paged_f16".into(), source: String::new() },
            args: vec![Arg::Block { id: out.id, dtype: Dtype::F32, rank: 1 }],
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            out_elems: 4,
        };
        f.launch(unsupported).await.expect("空转档不拒未注册核");
        assert_eq!(read_f32(&mut f, &out, 4).await, vec![0.0; 4], "输出清零");
        // 默认档仍响亮拒(不放宽)
        let mut g = CpuFace::new();
        let o2 = g.alloc(Dtype::F32, 4, "t").await.expect("out2");
        let err = g
            .launch(LaunchMsg {
                kernel: KernelSource { name: "owl_attention_paged_f16".into(), source: String::new() },
                args: vec![Arg::Block { id: o2.id, dtype: Dtype::F32, rank: 1 }],
                grid: (1, 1, 1),
                block: (1, 1, 1),
                shared_mem: 0,
                out_elems: 4,
            })
            .await
            .expect_err("默认档必须拒");
        assert!(format!("{err}").contains("未注册"), "{err}");
    }

    /// 块账本投影:活块可见、按 id 排序、假地址区间不重叠(审计可判定)。
    #[tokio::test]
    async fn block_ledger_dump_exposes_live_blocks() {
        let mut f = CpuFace::new();
        let a = f.alloc(Dtype::F32, 8, "t").await.expect("a");
        let _b = f.alloc(Dtype::BF16, 8, "t").await.expect("b");
        let rows = f.block_ledger_dump().await.expect("账本");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, a.id);
        assert_eq!(rows[0].bytes, 32);
        assert_eq!(rows[1].bytes, 16, "BF16 声明口径");
        assert!(rows[0].addr < rows[1].addr, "假地址单调");
        assert!(rows[1].addr - rows[0].addr >= 0x1_0000, "区间不重叠");
        // 归还后从账本消失(free 回冲)
        f.free(&[a.id]).await.expect("归还");
        assert_eq!(f.block_ledger_dump().await.expect("账本").len(), 1);
    }
}

#[cfg(test)]
mod bench {
    /// 解码循环吞吐(诊断用:确认"逐元素路径"本身的速度)。
    #[test]
    fn decode_throughput_microbench() {
        let n = 10_000_000usize;
        let bytes = vec![0x3Fu8; n * 2];
        let mut out = vec![0f32; n];
        let t = std::time::Instant::now();
        for (d, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *d = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
        }
        let el = t.elapsed().as_secs_f64();
        eprintln!("[bench] {n} 元素 {el:.3}s ⇒ {:.0} MB/s", (n * 2) as f64 / el / 1e6);
    }
}

#[cfg(test)]
mod lut_tests {
    /// 查找表与 `half` 本体**逐位一致**(含 subnormal / ±0 / inf / nan)。
    #[test]
    fn f16_lut_matches_half_bit_for_bit() {
        let lut = super::f16_lut();
        for i in 0..65536usize {
            let want = half::f16::from_bits(i as u16).to_f32();
            assert_eq!(lut[i].to_bits(), want.to_bits(), "位级不一致 @ {i:#06x}");
        }
    }
}

#[cfg(test)]
mod bench2 {
    use std::hint::black_box;
    use std::time::Instant;

    /// **受控对照(§18.4.2)**:同数据、同写入目标,切分方式不同 + 纯写基线(black_box 防消除)。
    #[test]
    fn decode_variants_bench() {
        let n = 10_000_000usize;
        let bytes = vec![0x3Fu8; n * 2];
        let mut out = vec![0f32; n];
        let lut = super::f16_lut();

        // A:`chunks_exact(2)`(**变长 slice**)+ zip(当前实现)
        let t = Instant::now();
        for (d, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *d = lut[u16::from_le_bytes([c[0], c[1]]) as usize];
        }
        let a = t.elapsed().as_secs_f64();

        // B:索引访问(无 slice 构造)
        let t = Instant::now();
        for i in 0..n {
            let b = 2 * i;
            let h = u16::from_le_bytes([bytes[b], bytes[b + 1]]);
            out[i] = lut[h as usize];
        }
        let b = t.elapsed().as_secs_f64();

        // C:`iter().step_by` + 取值?改用 chunks_exact 单端(只读不 zip)
        let t = Instant::now();
        let mut acc = 0u32;
        for c in bytes.chunks_exact(2) {
            acc = acc.wrapping_add(u16::from_le_bytes([c[0], c[1]]) as u32);
        }
        let c = t.elapsed().as_secs_f64();
        black_box(acc);

        // D:纯写基线(black_box 防死写消除)
        let t = Instant::now();
        for d in out.iter_mut() {
            *d = black_box(1.0);
        }
        let d = t.elapsed().as_secs_f64();
        black_box(&out);

        eprintln!("[cmp] n={n}({:.0}MB→{:.0}MB)", (n * 2) as f64 / 1e6, (n * 4) as f64 / 1e6);
        eprintln!("[cmp] A chunks_exact+zip+lut : {a:.3}s ({:.1}ns/元素)", a / n as f64 * 1e9);
        eprintln!("[cmp] B 索引+lut             : {b:.3}s ({:.1}ns/元素)", b / n as f64 * 1e9);
        eprintln!("[cmp] C 只读 chunks_exact    : {c:.3}s ({:.1}ns/元素)", c / n as f64 * 1e9);
        eprintln!("[cmp] D 纯写(black_box)      : {d:.3}s ({:.1}ns/元素)", d / n as f64 * 1e9);
    }
}

#[cfg(test)]
mod bench3 {
    use std::hint::black_box;
    use std::time::Instant;

    /// **真实规模对照**(134MB 输入 / 268MB 输出 = embed_tokens 档,超 L3)。
    #[test]
    fn decode_variants_big() {
        let n = 67_108_864usize; // 与装载实测的块 1 同量级
        let bytes = vec![0x3Fu8; n * 2];
        let mut out = vec![0f32; n];
        let lut = super::f16_lut();

        let t = Instant::now();
        for (d, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *d = lut[u16::from_le_bytes([c[0], c[1]]) as usize];
        }
        let a = t.elapsed().as_secs_f64();
        black_box(&out);

        let t = Instant::now();
        for i in 0..n {
            let b = 2 * i;
            let h = u16::from_le_bytes([bytes[b], bytes[b + 1]]);
            out[i] = lut[h as usize];
        }
        let b = t.elapsed().as_secs_f64();
        black_box(&out);

        let t = Instant::now();
        for d in out.iter_mut() {
            *d = black_box(1.0);
        }
        let d = t.elapsed().as_secs_f64();
        black_box(&out);

        eprintln!("[big] n={:.0}M({:.0}MB→{:.0}MB)", n as f64 / 1e6, (n * 2) as f64 / 1e6, (n * 4) as f64 / 1e6);
        eprintln!("[big] A chunks_exact+zip : {a:.3}s ({:.1}ns/元素)", a / n as f64 * 1e9);
        eprintln!("[big] B 索引+lut          : {b:.3}s ({:.1}ns/元素)", b / n as f64 * 1e9);
        eprintln!("[big] D 纯写(black_box)   : {d:.3}s ({:.1}ns/元素)", d / n as f64 * 1e9);
    }
}

#[cfg(test)]
mod bench4 {
    use std::hint::black_box;
    use std::time::Instant;

    /// **首次触碰 vs 复写**(定谳用):同规模(≈6GB,与装载的块总量同量级)。
    #[test]
    fn first_touch_vs_hot() {
        // 与装载**同分布**:304 个 ~20MB 块(而不是 30 个 192MB 大块 —— 后者会被
        // THP 折叠成 2MB 页,掩盖小块分配的 TLB 行为)
        let blocks = 304usize;
        let elems = 5_000_000usize; // 20MB/块 × 304 = 6.08GB
        let t_all = Instant::now();
        let mut v: Vec<Vec<f32>> = Vec::with_capacity(blocks);
        let t = Instant::now();
        for _ in 0..blocks {
            let mut b = vec![0f32; elems];
            for x in b.iter_mut() {
                *x = 1.0;
            }
            v.push(b);
        }
        let first = t.elapsed().as_secs_f64();

        let t = Instant::now();
        for b in v.iter_mut() {
            for x in b.iter_mut() {
                *x = 2.0;
            }
        }
        let hot = t.elapsed().as_secs_f64();
        black_box(&v);
        let total = (blocks * elems) as f64;
        eprintln!(
            "[touch] 首次 {first:.2}s ({:.1}ns/元素) · 复写 {hot:.2}s ({:.1}ns/元素) · 分配墙钟 {:.2}s",
            first / total * 1e9,
            hot / total * 1e9,
            t_all.elapsed().as_secs_f64(),
        );
    }
}

/// **内存基准助手**(诊断;必须在**优化编译的 crate 内**定义 —— 写在
/// integration test 里的对照循环是未优化的,实测差 35 倍,§18.4.4)。
///
/// 返回:顺序复写 `elems` 个 f32 × `iters` 遍的 **ns/元素**。
pub fn bench_write(elems: usize, iters: usize) -> f64 {
    let mut v = vec![0f32; elems];
    let t = std::time::Instant::now();
    for _ in 0..iters {
        for x in v.iter_mut() {
            *x = std::hint::black_box(3.0);
        }
    }
    let ns = t.elapsed().as_nanos() as f64 / (elems * iters) as f64;
    std::hint::black_box(&v);
    ns
}

/// 同上:顺序读 `n` 字节(**首次读**,每次新建 —— 冷)的 ns/字节。
pub fn bench_read(n_bytes: usize) -> f64 {
    let v = vec![0x3Fu8; n_bytes];
    let t = std::time::Instant::now();
    let mut h = 0u32;
    for c in v.iter() {
        h = h.wrapping_add(*c as u32);
    }
    std::hint::black_box(h);
    t.elapsed().as_nanos() as f64 / n_bytes as f64
}

/// **复刻 `upload_pinned` 的 F16 解码循环**(优化编译;诊断对照用)。
pub fn bench_decode_f16(bytes: &[u8], dst: &mut [f32]) -> f64 {
    let lut = f16_lut();
    let n = dst.len().min(bytes.len() / 2);
    let t = std::time::Instant::now();
    for i in 0..n {
        let b = 2 * i;
        dst[i] = lut[u16::from_le_bytes([bytes[b], bytes[b + 1]]) as usize];
    }
    let ns = t.elapsed().as_nanos() as f64 / n as f64;
    std::hint::black_box(&dst[..n]);
    ns
}
