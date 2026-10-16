//! server 状态面:设备上下文 + 流注册表(三固定流)+ 池块账房
//! + kernel 编译缓存 + pinned 码头 + 图捕获状态机/图注册表。
//!
//! 只被 actor 线程触碰(单线程所有权;零锁)。

use crate::ffi::{device_get_count, free_host, malloc_host, CudaContext, CudaFunction, CudaSlice, CudaStream, CAPTURE_MODE_THREAD_LOCAL};
use owl_iface::contract::GraphId;
use owl_iface::contract::ModelError;
use owl_iface::contract::{numel, Shape};
use std::collections::HashMap;
use std::sync::Arc;

/// 设备上下文诊断/调优选项(**显式依赖律,2026-10-10**:env 派生数据由
/// 入口/测试经本结构显式传入,设备路径零 env 读取)。
/// [`DiagOpts::from_config`] 是入口侧组装器(server config / 测试用例)。
#[derive(Clone, Debug, Default)]
pub struct DiagOpts {
    /// 逐命令/图节点计时(原 OWL_SRV_TIMING)
    pub srv_timing: bool,
    /// 捕获期插桩(原 OWL_CAP_PROF)
    pub cap_prof: bool,
    // (原 `launch_sync` 字段已删,2026-10-16):**唯一读者**是 server 侧
    // `ServerProbes.launch_sync`——而那个字段自己也没人读(逐发射同步
    // 早已改成 `sync_at(sync_bit::LAUNCH)`,alloc 侧改
    // `sync_at(sync_bit::ALLOC)`)⇒ 只剩“写进结构体再搬一次”的空转。
    // 与 `legacy_sync`(2026-10-15)同一形态,故删；“全同步”仍由
    // 配置面 `cuda.launch_sync` ⇒ `sync_mask = ALL` 表达。
    /// 旁路事件钉序(原 OWL_FREE_LEGACY;崩坏案 A/B 判别)
    pub free_legacy: bool,
    // (原 `legacy_sync` 字段已删,2026-10-15):它的两个消费者早已各自改成
    // **按位判定** —— dtoh 走 `sync_at(DTOH)`(harvest.rs),回收栅栏走
    // `sync_at(FENCE)`(fence_targets)⇒ 字段只剩派生式没有读者。留着会让
    // 审查者以为 "单开 bit1 会连带改 fence 语义"(实测不成立),故删。
    /// **同步点分解掩码**(2026-10-15):6 处同步点逐位开关,见
    /// [`owl_shared::config::sync_bit`]。`launch_sync = true` ⇒ `ALL`。
    /// 归因用:先看哪一位单独就能把病臂治好,再谈修法(靶子 = +17%)。
    pub sync_mask: u32,
    /// server 提交计时(server probes 面;DiagOpts 一体携带免双参)
    pub launch_time: bool,
    /// dtoh 分相(server probes 面)
    pub d2h_prof: bool,
    /// 逐核归因(server probes 面)
    pub gpu_prof: bool,
    /// 图实例化旗标(原 OWL_GRAPH_FLAGS;UPLOAD=2 / DEVICE_LAUNCH=4;
    /// 类型化位面经 OwlConfig.cuda.graph_flags → to_bits())
    pub graph_flags: u64,
    /// 捕获 slab 固定档 MiB(None = warmup 计量定量;原 OWL_CAPTURE_SLAB_MB
    /// —— 24G 贴顶的 27B + spec 双图场景,verify 图按需调小)
    pub capture_slab_mb: Option<usize>,
    /// **发射账容量**(cuda.trace_ring;0 = 关)。>0 ⇒ server 记最近 N 条
    /// 命令(launch/htod/dtoh/alloc/free/memset/copy;见 `crate::trace`)。
    pub trace_ring: usize,
    /// **生产算子白名单**(dispatch.allowed_ops;None/空 = 门关)。
    /// `Arc` = 与 server 主循环共享,逐发射判定零克隆。
    pub allowed_ops: Option<std::sync::Arc<Vec<String>>>,
}

impl DiagOpts {
    /// 同步点查询(归因面):该位是否武装。热路径只是一次位与。
    pub fn sync_at(&self, bit: u32) -> bool {
        self.sync_mask & bit != 0
    }

    /// **入口侧组装器**:自全量配置对象显式组装(设备路径零 env 读取)
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        use owl_shared::config::sync_bit;
        let c = &cfg.cuda;
        // 同步点分解:launch_sync 仍是「全同步」的合称(向后兼容/回退),
        // sync_mask 逐点归因(生产缺省 0 = 全关)。
        let mask = if c.launch_sync { sync_bit::ALL } else { c.sync_mask };
        Self {
            srv_timing: c.srv_timing,
            cap_prof: c.cap_prof,
            free_legacy: c.free_legacy,
            sync_mask: mask,
            launch_time: c.launch_time,
            d2h_prof: c.d2h_prof,
            gpu_prof: c.gpu_prof,
            graph_flags: c.graph_flags.to_bits(),
            capture_slab_mb: c.capture_slab_mb,
            trace_ring: c.trace_ring,
            allowed_ops: cfg
                .dispatch
                .allowed_ops
                .as_ref()
                .filter(|v| !v.is_empty())
                .map(|v| std::sync::Arc::new(v.clone())),
        }
    }
}

// 流身份证(server 内部路由键;客户端不可见 —— 三固定流是 server 策略)
pub(super) type StreamId = u64;
/// **尾事件句柄的线程域包装**(与 [`GraphHolder`] 同律:句柄只在 server
/// 线程解引用/销毁 —— 该线程独占上下文;raw `CUevent` 非 Send 只是指针事实,
/// cudarc 自身对 `CudaStream` 亦 `unsafe impl Send/Sync`)
#[derive(Clone, Copy)]
pub(super) struct TailEvent(crate::ffi::sys::CUevent);

unsafe impl Send for TailEvent {}
unsafe impl Sync for TailEvent {}

/// E1 事件池容量(轮转;E2 起 = 在飞等待上界的记账口)
const EVENT_POOL_N: usize = 8;

pub(super) const STREAM_H2D: StreamId = 0;
pub(super) const STREAM_COMPUTE: StreamId = 1;
pub(super) const STREAM_D2H: StreamId = 2;

// ============================================================================
// 设备选择(UUID 钉卡;数字序事故免疫)
// ============================================================================

/// 设备选择器:优先 UUID(唯一稳定身份证);Ordinal 仅测试/单机便捷用。
#[derive(Debug, Clone)]
pub enum DeviceSelector {
    /// 16 字节设备 UUID(推荐;跨重启/跨机器枚举序稳定)
    Uuid([u8; 16]),
    /// CUDA ordinal(枚举序敏感;仅单卡/测试场景)
    Ordinal(usize),
}

impl DeviceSelector {
    /// 解析为 ordinal(UUID:遍历设备表比对;Ordinal:原样)
    pub(super) fn resolve(&self) -> Result<usize, String> {
        match self {
            Self::Ordinal(o) => Ok(*o),
            Self::Uuid(uuid) => {
                let count = device_get_count().map_err(|e| format!("device_get_count: {e:?}"))?;
                for o in 0..count as usize {
                    if crate::ffi::device_uuid(o).map_err(|e| format!("device_uuid({o}): {e:?}"))? == *uuid {
                        return Ok(o);
                    }
                }
                Err(format!(
                    "UUID {uuid:?} 不在设备表(count={count};检查 CUDA_VISIBLE_DEVICES)"
                ))
            }
        }
    }
}

// ============================================================================
// 池块账房
// ============================================================================

/// 块的两种形态:
/// - Owned:图外独立分配(cudaMallocAsync)
/// - Carved:捕获期从 capture slab 切出的切片(零分配;slab 随块保活)
///   `cap_n` = 该块**可用容量**(对齐后;S1.5 回收凭它 —— `bytes` 是账长,
///   两者可不同:切 1000B 得 cap_n=1024B)
pub(super) enum Block {
    Owned(CudaSlice<u8>),
    Carved { slab: Arc<CudaSlice<u8>>, off: usize, n: usize, cap_n: usize },
}

/// 账本内块 id 位编码(裁决 2,2026-10-13):单 u64 高位 ledger、低 40 位
/// 本账递增 local。线格式 `Bytes{id,len}` 不变 —— 只是 id 不再跨账本撞号
/// (pitfall §27 根因)。报错可同时打印两段。
const BLOCK_LEDGER_SHIFT: u32 = 40;
const BLOCK_LOCAL_MASK: u64 = (1u64 << BLOCK_LEDGER_SHIFT) - 1;

/// 打包块 id(分配点)。
fn pack_block_id(ledger: u32, local: u64) -> u64 {
    ((ledger as u64) << BLOCK_LEDGER_SHIFT) | (local & BLOCK_LOCAL_MASK)
}

/// 访问器:块 id → 所属账本(0 成本解码;诊断/断言用)。
pub(super) fn ledger_of(id: u64) -> u32 {
    (id >> BLOCK_LEDGER_SHIFT) as u32
}

/// 访问器:块 id → 账本内局部序号(测试/诊断)。
#[cfg(test)]
fn local_of(id: u64) -> u64 {
    id & BLOCK_LOCAL_MASK
}

/// 解析点断言:块 id 必须属本账本(跨账本 = §27 撞号案;六步压成一行)。
fn assert_ledger(id: u64, ledger: u32) -> Result<(), ModelError> {
    let owner = ledger_of(id);
    if owner != ledger {
        return Err(ModelError::CrossLedger { id, owner, reader: ledger });
    }
    Ok(())
}

/// 在册块 → **按名聚合**文本(纯函数:OOM dump 与单测共用;2026-10-16)。
///
/// 输入是 `(名字, 字节)` 流 —— **不碰 `BlockEntry` 本体**:OOM 现场渲染与
/// 显存账目本体解耦(单测零 CUDA 依赖:裁决面只考"名字怎么聚、怎么排、怎么
/// 截断、空名怎么显形"四件事)。
///
/// 排序=字节降序(谁占大头在前);`top` 之外折入"…另 N 类"。空名渲染成
/// `(未命名!)` —— 漏网信号必须显眼,不能静默混入正常名字里。
pub(crate) fn render_block_report<'a, I>(blocks: I, top: usize) -> String
where
    I: Iterator<Item = (&'a str, usize)>,
{
    let mut agg: std::collections::HashMap<&str, (u64, usize)> = std::collections::HashMap::new();
    for (name, bytes) in blocks {
        let g = agg.entry(name).or_insert((0, 0));
        g.0 += bytes as u64;
        g.1 += 1;
    }
    if agg.is_empty() {
        return "(块账空)".to_string();
    }
    let mut rows: Vec<(&str, (u64, usize))> = agg.into_iter().collect();
    rows.sort_unstable_by_key(|(_, v)| std::cmp::Reverse(v.0));
    let total: u64 = rows.iter().map(|(_, v)| v.0).sum();
    let nblk: usize = rows.iter().map(|(_, v)| v.1).sum();
    let mut s = format!(
        "在册 {} 块 / 合计 {}(按名 top{top}):",
        nblk,
        fmt_bytes(total)
    );
    for (name, (bytes, n)) in rows.iter().take(top) {
        s.push_str(&format!(
            " {}={}×{n}",
            if name.is_empty() { "(未命名!)" } else { name },
            fmt_bytes(*bytes)
        ));
    }
    if rows.len() > top {
        s.push_str(&format!(" …另 {} 类", rows.len() - top));
    }
    s
}

/// 字节 → 人读单位(KB/MiB;2026-10-16)。
///
/// 为什么不用"一律 MiB":OOM 现场两头都要看得清 —— 大块(GB 级权重/KV)与
/// 小块(几十 KB 的常量/表件)同在,一刀切 MiB 会把小块全渲染成 `0.0MiB`。
fn fmt_bytes(n: u64) -> String {
    if n >= 1 << 20 {
        format!("{:.1}MiB", n as f64 / 1048576.0)
    } else {
        format!("{:.1}KiB", n as f64 / 1024.0)
    }
}

/// 按名清单 → **JSON**(结构化落盘;2026-10-16)。
///
/// 为什么要有它:人读的一行够排障,但"跨时间比对/被脚本消费"(例如
/// 启动前期 vs 稳态、OOM 前 vs OOM 时)需要机器可读。格式故意做成
/// **行无关的扁平对象**:
///
/// ```json
/// {"blocks":[{"name":"kv.k.0","bytes":16777216,"blocks":1}],"total_bytes":123,"total_blocks":4}
/// ```
///
/// 与 [`render_block_report`] **同源**(同一聚合、同一降序、同一单位律不适用于
/// JSON —— 这里给**原始字节**),所以两者永不互相漂移。名字原样输出(JSON
/// 转义由本函数承担:引号/反斜杠/控制字符)。
pub(crate) fn block_report_json<'a, I>(blocks: I) -> String
where
    I: Iterator<Item = (&'a str, usize)>,
{
    let mut agg: std::collections::HashMap<&str, (u64, usize)> = std::collections::HashMap::new();
    for (name, bytes) in blocks {
        let g = agg.entry(name).or_insert((0, 0));
        g.0 += bytes as u64;
        g.1 += 1;
    }
    let mut rows: Vec<(&str, (u64, usize))> = agg.into_iter().collect();
    rows.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(b.0)));
    let total_bytes: u64 = rows.iter().map(|(_, v)| v.0).sum();
    let total_blocks: usize = rows.iter().map(|(_, v)| v.1).sum();
    // **显式拼**(不吃 format 的 `{{`/`}}` 转义):嵌套名字槽 + 花括号叠加时
    // 转义极易数错,而错了就是"不是合法 JSON"这种无声失效(实测踩过:
    // `"…{total_blocks}}}"` 少一个 `}` ⇒ 落盘现场不可解析)。逐段 push 之后
    // 每一处开合都肉眼可数;`gpu_server` 步骤 13 的真解析器是它的门。
    let mut out = String::with_capacity(256 + rows.len() * 64);
    out.push('{');
    out.push_str("\"blocks\":[");
    for (i, (name, (bytes, n))) in rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str("\"name\":");
        out.push_str(&json_string(name));
        out.push_str(&format!(",\"bytes\":{bytes},\"blocks\":{n}"));
        out.push('}');
    }
    out.push(']');
    out.push_str(&format!(",\"total_bytes\":{total_bytes},\"total_blocks\":{total_blocks}"));
    out.push('}');
    out
}

/// **峰值快照决策**(纯函数:门值/步进/上界三重判据 ⇒ 离线可测)。
///
/// 返回 `Some(新水位)` = 该写;`None` = 不写(未到门/未破步进/写满)。
/// 与 `GpuServer::maybe_dump_peak` 同源(那边只负责取 env + I/O)。
 pub(crate) fn peak_should_dump(
 live: u64,
 threshold_bytes: u64,
 watermark: u64,
 written: u32,
 max: u32,
 ) -> Option<u64> {
 if threshold_bytes == 0 || live < threshold_bytes || written >= max {
     return None;
 }
 // 步进 5%:首次(watermark==0)或显著更高才写
 if watermark > 0 && live < watermark.saturating_mul(105) / 100 {
     return None;
 }
 Some(live)
 }

/// **OOM 现场落盘路径解析**(纯函数;离线可测,2026-10-16)。
///
/// 门值语义(与 `prompt_dump` 家法一致):
/// - `"1"` / `""` ⇒ `/tmp/owl-vram-{ms}.json`(时间戳防覆盖);
/// - 其他值 ⇒ 当显式路径用(调用方指定,便于 CI/脚本固定落点)。
pub(crate) fn resolve_vram_dump_path(target: &str, ms_epoch: u128) -> String {
    match target {
        "" | "1" => format!("/tmp/owl-vram-{ms_epoch}.json"),
        p => p.to_string(),
    }
}

/// **现场文件名 → 毫秒戳**(自家两种形态;非自家 = `None`)。
///
/// 形态:`owl-vram-{ms}.json`(OOM 现场)与
/// `owl-vram-peak-{ms}-{MiB}MiB.json`(峰值阶梯)。
/// 保留裁剪**只认这两种名字** —— 显式路径(用户给的)与陌生文件一律不碰。
pub(crate) fn vram_dump_stamp(name: &str) -> Option<u128> {
    let rest = name.strip_prefix("owl-vram-")?.strip_suffix(".json")?;
    let rest = rest.strip_prefix("peak-").unwrap_or(rest);
    rest.split('-').next()?.parse::<u128>().ok()
}

/// **现场保留裁剪**(2026-10-16):把 `dir` 里自家现场裁到最多 `keep` 份(保新)。
///
/// 为什么需要:OOM 现场 + 峰值阶梯(`OWL_VRAM_PEAK_MAX` 份/次运行)在
/// `/tmp` 里**跨运行累积** —— 单进程上界管不住"跑了很多轮测试"的堆积。
///
/// 判据全在明处:①只认 [`vram_dump_stamp`] 认得出的名字(陌生文件不动);
/// ②排序键 = 名字里内嵌的毫秒戳(不依赖 mtime —— 拷贝/恢复会改写 mtime);
/// ③`keep == 0`(关)或目录读不到 ⇒ 一份不删。返回**实际**删除份数。
pub(crate) fn prune_vram_dumps(dir: &std::path::Path, keep: usize) -> usize {
    if keep == 0 {
        return 0;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut stamps: Vec<(u128, std::path::PathBuf)> = Vec::new();
    for e in rd.flatten() {
        let name_os = e.file_name();
        let Some(name) = name_os.to_str() else { continue };
        let Some(ms) = vram_dump_stamp(name) else { continue };
        stamps.push((ms, e.path()));
    }
    if stamps.len() <= keep {
        return 0;
    }
    stamps.sort_unstable_by_key(|(ms, _)| *ms);
    let drop_n = stamps.len() - keep;
    stamps
        .into_iter()
        .take(drop_n)
        .filter(|(_, p)| std::fs::remove_file(p).is_ok())
        .count()
}

/// JSON 字符串字面量(含转义;名字来自调用点,含路径/冒号/引号的可能)。
pub(crate) fn json_string(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// 未命名块计数(全量记名漏网信号;纯函数,与 [`render_block_report`] 同源)。
pub(crate) fn count_unnamed<'a, I>(blocks: I) -> usize
where
    I: Iterator<Item = (&'a str, usize)>,
{
    blocks.filter(|(name, _)| name.is_empty()).count()
}
/// 块账条目:块本体 + 账长 + 账本身份 + 布局/类型(§6.2 VramBlock 最小面)。
///
/// `dtype`/`shape` 当前从字节口径填充 —— cuda 池账是字节级分配
/// (`new_block` 只见 `slice.len()`),元素级 dtype/shape 待分配 API 携带
/// 元数据后细化;本轮先落 `ledger` 身份字段(撞号门)。
///
/// `BlockDtype` 是**本账局部** dtype token(不动线格式 `Dtype`;字节口径
/// 无元素宽语义,向共享 `Dtype` 加 U8 会连带 driver::DType / models 映射
/// 全链改,超出 P1 最小面)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlockDtype {
    /// 字节口径(池块/元数据 blob)
    Bytes,
}

impl BlockDtype {
    fn size_bytes(self) -> usize {
        match self {
            BlockDtype::Bytes => 1,
        }
    }
}

pub(super) struct BlockEntry {
    pub(super) block: Block,
    /// 账长(字节口径;f16 基线)
    pub(super) bytes: usize,
    /// 分配它的账本(actor 序号;解析点断言 == 本账本)
    pub(super) ledger: u32,
    pub(super) dtype: BlockDtype,
    pub(super) shape: Shape,
    /// **分配点点名**(全量记名;2026-10-16)。
    ///
    /// 为什么必须在这里:`[vram-ledger]` 的 OOM 现场只有三笔**总量**
    /// (池 used/reserved、驱动 free、引擎 live),追不到"谁占着"。名字是
    /// 分配**调用点**的事实(核名/层名/装载键名),由 `alloc` 面一路带到
    /// 账房 —— 池命中(复用)时名字保持首次分配的那次(块身份不变,
    /// 复用它的语义也一致:同一张图的同一个槽)。
    pub(super) name: String,
}

// ============================================================================
// 设备上下文 + 流注册表
// ============================================================================

// ============================================================================
// 家族面捕获入境纪律(2026-10-15;设计 §4.3e)
// ============================================================================
//
// **为什么需要**:G2 捕获窗白名单只管 client 来的 `Command`(Htod/Dtoh/Sync 拒),
// 而家族 `run()` 在同一 actor 线程里**直接调 `DeviceRes`** ⇒ 白名单对它无效。
// 旧口径是"家族 grave 律"(代码注释原话:"捕获窗内由家族 grave 律保证不再分配"),
// 即**靠自律**。实测定谳(2026-10-03 深夜,`代差分析与演进路线.md:222`):
// cuBLAS gemv splitK 在捕获流上内部 `malloc` ⇒ 图里烙下 **192 对
// MEM_ALLOC/MEM_FREE(3088B)** 节点(回放期重分配 + 派发税),修法是
// `cublasSetWorkspace` **预绑进程级驻留工作区** —— 即"三条正路"之第一条。
//
// **三条正路**(窗内材料的合法来源):
//   ① **装配期分配**(FI 的 int/float ws、XQA consts、cublas 工作区);
//   ② **不变盒**(`InvariantBox`:构造即冻结、生命周期 ≥ 图 —— gdn_scalar 的
//      soff,"捕获窗 async 节点,回放恒读盒");
//   ③ **引擎真块**(走 `Command::Alloc` ⇒ 捕获期落 slab、进账房 —— marlin ws /
//      XQA scratch/semaphores)。
//
// 本判定的射程 = 窗内**走第四条路**的动作(现建 scratch / 裸切片 H2D):
// 观察档(**缺省**)**放行 + 计数 + 告警**(零行为变化,先取证);
// `OWL_CAP_STRICT=1` 升为**结构化拒**(M2b 捕获桶前必须开的硬门)。

/// 家族面在捕获窗内的动作类别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FamAct {
    /// scratch 分配(`DeviceRes::alloc`):窗内 `cudaMallocAsync` 可被捕获
    /// ⇒ 烙 MEM_ALLOC/MEM_FREE 节点 + **完全在 slab/账房之外**
    ScratchAlloc,
    /// H2D 上传(`DeviceRes::upload`):`invariant` = 源是不变盒(合规范式)
    Upload { invariant: bool },
}

/// 严格档开关(进程级一次读取;诊断旋钮 —— 与 `OWL_GDN_FRESH_SCRATCH` 同族)
pub(super) fn cap_strict() -> bool {
    static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STRICT.get_or_init(|| {
        owl_shared::env_reader::parse_or::<u8>("OWL_CAP_STRICT", 0) != 0
    })
}

/// 入境判定(**纯函数**,零 CUDA ⇒ 可单测):
/// - 不变盒上传 ⇒ 恒放行(捕获窗的合规范式);
/// - 走第四条路的动作 ⇒ `strict` 拒 / 观察档放行(调用方负责 count + WARN)。
pub(super) fn fam_window_verdict(act: FamAct, strict: bool) -> Result<(), String> {
    let what = match act {
        FamAct::Upload { invariant: true } => return Ok(()),
        FamAct::ScratchAlloc => "家族 scratch 分配(alloc)",
        FamAct::Upload { invariant: false } => "家族裸切片 H2D 上传(upload)",
    };
    let why = "捕获窗内的家族面分配/上传会**被烙进图**:分配 ⇒ MEM_ALLOC/MEM_FREE \
               节点 + 不进 slab/账房;上传 ⇒ 源必须生命周期 ≥ 图(不变盒),否则回放 \
               读到已释放/已改写的 host 内存(静默错值)。三条正路 = 装配期分配 / \
               不变盒 / 引擎真块(设计 §4.3e)";
    if strict {
        Err(format!("{what}:{why}"))
    } else {
        Ok(())
    }
}

// ============================================================================
// 图引用租约(2026-10-15;把"指针稳定契约"从约定升级成机制)
// ============================================================================

/// **图引用租约** —— 把"被图烘过的块不得回收"变成可判定的簿记。
///
/// # 它挡的是哪一类错
///
/// 图里烘的是**裸设备指针**;而被捕进图的块有两类:
/// - `Carved`(捕获窗内 Alloc ⇒ slab 切片):`free_blocks` 一律 no-op ⇒ 天然安全;
/// - `Owned`(池件 = 权重/KV/GDN 态/bt 表;以及 `GraphPlan` 的输入槽):
///   **可被 free**。一旦某处把它 free 掉,设备池会把同址发给下一个 alloc
///   ⇒ 图变成"指向别人内存的陈旧指针"—— 表现是**静默错值**(不崩、不报错),
///   正是本仓最贵的病型。
///
/// # 怎么挡
///
/// 捕获窗内每一次指针解析(`block_ptr`)都是"这个块会被烘进图"的证据 ⇒
/// 记进窗口集;`graph_end` 收窗时登记进租约集(**只增不减** —— 当前没有
/// 单图销毁通道,图活到上下文结束,见 `GpuCtx.graphs`)。`free_blocks` 对命中者
/// **拒收 + 记账 + 告警**(响亮,但不致命:这一块本来就"还未到可回收时点")。
///
/// **没有它也能"靠约定活着"**(现役池件 boot 固定、输入槽无人 free)——
/// 但那正是 M1(SlabArena 复用)与 S-2(单图销毁)一动就会破的墙。先把它
/// 变成机制,后面两刀才有依据。
#[derive(Default, Debug)]
pub(super) struct GraphRefs {
    /// 捕获窗内解析过的**可回收类**块 id(窗口内累积;begin 时清)
    win: std::collections::HashSet<u64>,
    /// id → **引用它的存活图数**(计数而非集合:单图销毁后须能正确解钉)
    pinned: std::collections::HashMap<u64, u32>,
}

impl GraphRefs {
    /// 开新窗(清窗口集;**不动既有租约** —— 已登记的图还活着)
    pub(super) fn begin_window(&mut self) {
        self.win.clear();
    }

    /// 窗内指针解析登记(幂等;**只登记 `Owned` 类**,理由见 `block_ptr`)
    pub(super) fn record(&mut self, id: u64) {
        self.win.insert(id);
    }

    /// 收窗 → 入租约;返回本图引用集(`GraphHolder.refs`)
    pub(super) fn adopt(&mut self) -> Vec<u64> {
        let v: Vec<u64> = self.win.drain().collect();
        for id in &v {
            *self.pinned.entry(*id).or_insert(0) += 1;
        }
        v
    }

    /// 单图销毁:递减引用计数(归零则解钉)。返回解钉块数(诊断)
    pub(super) fn forget(&mut self, refs: &[u64]) -> usize {
        let mut unpinned = 0;
        for id in refs {
            match self.pinned.get_mut(id) {
                Some(c) if *c > 1 => *c -= 1,
                Some(_) => {
                    self.pinned.remove(id);
                    unpinned += 1;
                }
                None => {}
            }
        }
        unpinned
    }

    /// 该块是否被存活图引用(`free_blocks` 拒收判据)
    pub(super) fn is_pinned(&self, id: u64) -> bool {
        self.pinned.contains_key(&id)
    }

    #[cfg(test)]
    fn pinned_len(&self) -> usize {
        self.pinned.len()
    }

    #[cfg(test)]
    fn window_len(&self) -> usize {
        self.win.len()
    }
}

pub(super) struct GpuCtx {
    pub ctx: Arc<CudaContext>,
    /// 诊断/调优选项(入口显式传入;显式依赖律)
    pub(super) diag: DiagOpts,
    /// 流注册表:三固定流(H2D/COMPUTE/D2H;客户端不可自创)
    streams: HashMap<StreamId, Arc<CudaStream>>,
    /// **三流尾事件**(E1:纯记账 —— 每命令末尾 record 一次,**不等待**;
    /// 序消费在 E2/E3。设计:`docs/arch/三流事件序-设计与迁移.md`)
    tail_event: HashMap<StreamId, TailEvent>,
    /// **每块"清零完成"事件**(2026-10-16 定谳的填坑):`alloc{zero:true}` 的
    /// `memset_zeros` 是 **COMPUTE 流上的设备侧异步写**,而 host→device 搬运在
    /// **H2D 流** ⇒ 跨流无序,**晚到的 memset 会把刚落地的搬运抹成 0**
    /// (实测:A_log 32B 写完即读回 = 0)。本账在 memset 后记一个事件,
    /// 搬运该块时等它一次;块回收时清账。
    zero_ev: HashMap<u64, TailEvent>,
    /// 事件池(round-robin):"重录挂在等待中的事件"是 CUDA UB(不报错)
    /// ⇒ 池化是唯一正确的复用方式;容量 = 在飞等待上界(E2 起记账)
    event_pool: Vec<TailEvent>,
    ev_next: usize,
    /// 图捕获状态机(见 graph_begin/end 的护栏注释)
    ordinal: usize,
    capture: Option<CaptureState>,
    /// 图引用租约(捕获窗引用集 + 存活图并集;见 [`GraphRefs`])。`RefCell`
    /// 是因为 `block_ptr` 是 `&self`(launch 解析面共用);actor 单线程持有
    /// ⇒ 无跨线程竞争
    graph_refs: std::cell::RefCell<GraphRefs>,
    /// 图注册表:id → 实例化图(重放用)
    graphs: HashMap<GraphId, GraphHolder>,
    next_graph: GraphId,
    /// 块账房:id → 块条目(账长/账本身份/dtype/shape)
    blocks: HashMap<u64, BlockEntry>,
    next_block: u64,
    /// 本账本编号(= GpuServer actor 序号;块 id 高位编码此值)
    ledger: u32,
    /// 动态 smem 每块上限(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    /// _OPTIN;发射前硬顶守卫,超限结构化拒绝 —— 2026-09-27 v1 smem 事故)
    smem_optin: usize,
    /// 刀1.6:探针开关(boot 解析;热路径零 env 查询)
    pub(super) srv_timing: bool,
    pub(super) cap_prof: bool,
    /// 诊断下沉口(注入面):日志 + 分相计时都走它(cuda 不依赖日志 crate)
    pub(super) sink: owl_shared::log::LogSink,
}

/// **捕获窗内发射材料的登记表**(2026-10-15;A1.6 的落地,materials 账)。
///
/// 为什么需要:
/// ① **空窗哨兵漏判** —— 家族/外来核走 `Exec::launch`(不产生 `Command`),
///    原哨兵只数原生 `Command::Launch` ⇒ **纯家族发射的窗会被误判为空**
///    (2026-10-03 `graph_dispatch_probe2` 踩过:外来核探针须补一发原生哨兵);
/// ② **"图里烘了什么"无账** —— `GraphPlan` 的槽/输出是账内的,而家族窗内
///    自产的 scratch/meta 是账外的(设计 §4.3e)⇒ 这张表是唯一线索。
///
/// 只按**内核名**聚合(不存每个节点):窗内可有数千发,原样存会在 host 侧
/// 白白几百 KB;而"哪些核在图里"正是治理面要的信息。
#[derive(Default, Debug, Clone)]
pub(super) struct CaptureNodes {
    /// 发射总数(原生 + 家族;= 原 `launches` 哨兵值)
    pub total: usize,
    /// 内核名 → 次数(键 = `String`:原生路径给的是 `LaunchMsg.kernel.name`,
    /// 家族路径给 `&'static str`;窗口内一次性分摊,不值得为省几个 String 分叉)
    by_kernel: std::collections::HashMap<String, usize>,
}

impl CaptureNodes {
    pub(super) fn add(&mut self, kernel: &str) {
        self.total += 1;
        *self.by_kernel.entry(kernel.to_string()).or_insert(0) += 1;
    }

    /// Top-N 摘要(次数降序,同数按名序;日志/诊断用)
    pub(super) fn top(&self, n: usize) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.by_kernel.iter().map(|(k, c)| (k.clone(), *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }

    pub(super) fn distinct(&self) -> usize {
        self.by_kernel.len()
    }
}

/// 捕获会话:目标流 + 切块 slab + 发射计数(空窗哨兵用)
struct CaptureState {
    stream: Arc<CudaStream>,
    slab: Arc<CudaSlice<u8>>,
    /// slab 内分配器(bump + 尺寸类回收;S1.5 ⇒ 容量 = 峰值活跃集而非总和)
    arena: owl_shared::slab_hint::SlabArena,
    /// **本窗自产块表**(abort 时整批作废的依据;graph_end 时交给 GraphHolder)
    carved: Vec<u64>,
    /// **本窗发射材料登记表**(原生 + 家族;空窗哨兵 + 治理账)
    nodes: CaptureNodes,
    /// 本图实际 slab 容量(warmup 计量定量或 env 固定档;carve 校验
    /// 与 graph_begin 分配必须同口 —— 旧实现 carve 校验读 env,与
    /// hint 定量的 slab 脱口,2026-10-10)
    cap: usize,
}

/// CudaGraph 非 Send(cudarc 未标注);我们把它钉死在 actor 线程单一所有权
/// 下使用(所有图操作都在 dispatch 线程序列化),跨线程移动 GpuServer 携带
/// 它是安全的 —— 与 cudarc "须外部串行化" 的要求一致。
/// 图持有(手搓实例化面;exec + 捕获流成对,Drop 双销毁。
/// 为何不走 cudarc safe CudaGraph:见 ffi.rs graph 定谳注释 —— 其
/// end_capture 强绑 AUTO_FREE_ON_LAUNCH,每 launch +4.45ms)
struct GraphHolder {
    exec: crate::ffi::sys::CUgraphExec,
    /// 捕获期 CUgraph(exec 实例化后独立,但随 holder 销毁,保持
    /// cudarc CudaGraph 同等资源语义)
    cu_graph: crate::ffi::sys::CUgraph,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// **本图烘入指针的 `Owned` 块集**(租约;见 [`GraphRefs`])
    refs: Vec<u64>,
    /// **本图专属的 slab 块集**(捕获窗自产;`graph_destroy` 时整批移出
    /// 账房 ⇒ 最后一个引用走掉即物理释放)
    carved: Vec<u64>,
    /// **本图烘入的发射材料登记**(A1.6 落地;见 [`CaptureNodes`])——
    /// `graph_destroy` 的日志与将来的显存归因都读它
    nodes: CaptureNodes,
}
unsafe impl Send for GraphHolder {}

impl Drop for GraphHolder {
    fn drop(&mut self) {
        unsafe {
            let _ = crate::ffi::graph_exec_destroy(self.exec);
            let _ = crate::ffi::graph_destroy(self.cu_graph);
        }
    }
}

// ============================================================================
// 池账快照(诊断;2026-10-16 "看着有 free 却分不出来"的第一现场面)
// ============================================================================

/// 默认内存池的账: `(used, used_high, reserved, reserved_high)` 字节。
///
/// `used` = 引擎正在用的字节;`reserved` = 池已向驱动映射的字节 —— **两者之差
/// = 已 free 但池未归还的死页**(`RELEASE_THRESHOLD=u64::MAX` 下永不归还 OS)。
/// 分配失败时读它即可分辨"真占满了" vs "池把页抱着不放"。
/// 返回 None = 读不到(池未就绪/接口不可用;调用方降级为文本)。
pub(super) fn pool_ledger_snapshot() -> Option<(u64, u64, u64, u64)> {
    use cudarc::driver::sys as cs;
    unsafe {
        let mut dev: cs::CUdevice = 0;
        if cs::cuCtxGetDevice(&mut dev) != cs::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut pool = std::mem::MaybeUninit::<cs::CUmemoryPool>::uninit();
        if cs::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), dev) != cs::CUresult::CUDA_SUCCESS {
            return None;
        }
        let pool = pool.assume_init();
        let get = |attr: cs::CUmemPool_attribute| -> Option<u64> {
            let mut v: u64 = 0;
            let rc = cs::cuMemPoolGetAttribute(
                pool,
                attr,
                &mut v as *mut u64 as *mut core::ffi::c_void,
            );
            (rc == cs::CUresult::CUDA_SUCCESS).then_some(v)
        };
        Some((
            get(cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            get(cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
            get(cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            get(cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
        ))
    }
}

impl GpuCtx {
    pub(super) fn new(
        selector: &DeviceSelector,
        diag: DiagOpts,
        ledger: u32,
        sink: owl_shared::log::LogSink,
    ) -> Result<Self, String> {
        let ordinal = selector.resolve()?;
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e:?}"))?;
        ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        // B6.5:设备总显存落账(vram 治理面;一次)
        match cudarc::driver::result::device::get(ordinal as i32)
            .and_then(|dev| unsafe { cudarc::driver::result::device::total_mem(dev) })
        {
            Ok(bytes) => owl_shared::vram::total_set(bytes as u64),
            Err(e) => sink.logf(owl_shared::log::LVL_INFO, "boot", format_args!("vram total 查询失败(治理禁用): {e:?}")),
        }
        if let Ok((free, _)) = cudarc::driver::result::mem_get_info() {
            owl_shared::vram::free_set(free as u64);
        }
        // G0 护栏:关 event-tracking(坑 A)。多流 + event tracking 会让
        // safe 层在块读写上插事件,污染图捕获(CAPTURE_ISOLATION/图内事件节点)
        unsafe { ctx.disable_event_tracking() };
        // ── P1″(2026-10-14):**默认池属性调优**(NVIDIA CUDA Programming
        //    Guide §4.3.4.2 / §4.3.4.4;cudarc 缺省 = 驱动缺省,对
        //    "整块只跑不 sync" 的 eager 路径最不利):
        //    ① `RELEASE_THRESHOLD` 缺省 0 ⇒ 每次 stream/event/device
        //       synchronize 都把物理页还给 OS ⇒ 下一发 alloc 重新映射(昂贵 OS 调用);
        //    ② `REUSE_ALLOW_OPPORTUNISTIC` 缺省 0 ⇒ free 掉的块**必须等 CPU
        //       侧同步**才能复用 ⇒ 逐节点 alloc/free churn 只能不断长池子。
        //    实测依据:512-tok prefill 375 ms 中设备核仅 8%;`OWL_NO_RECLAIM=1`
        //    (干脆不 free)⇒ **77-117 ms(5×)**,但无界常驻 OOM ⇒ 要的是
        //    "能复用"而不是"不释放"。
        //    (池归属 = 设备默认池:server 用 `cuMemAllocAsync`,即默认池。
        //     预算纪律不变:`vram_target` 侧仍可随时 `cuMemPoolTrimTo` 回收。)
        // 裸 sys 层调用(cudarc `result::` 包装在 cuda-13020 档被 cfg 掉;
        // 原始绑定不门控 —— 本项目已有 sys 直用先例,见下方 MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        let pool_setup = || -> Result<(), String> {
            use cudarc::driver::sys as cs;
            let dev = cudarc::driver::result::device::get(ordinal as i32)
                .map_err(|e| format!("device::get: {e:?}"))?;
            let mut pool = std::mem::MaybeUninit::<cs::CUmemoryPool>::uninit();
            let rc = unsafe { cs::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), dev) };
            if rc != cs::CUresult::CUDA_SUCCESS {
                return Err(format!("cuDeviceGetDefaultMemPool: {rc:?}"));
            }
            let pool = unsafe { pool.assume_init() };
            let mut threshold: u64 = u64::MAX;
            let mut opportunistic: i32 = 1;
            let mut follow_events: i32 = 1;
            for (attr, val, what) in [
                (
                    cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut core::ffi::c_void,
                    "RELEASE_THRESHOLD=UINT64_MAX",
                ),
                (
                    cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                    &mut opportunistic as *mut i32 as *mut core::ffi::c_void,
                    "REUSE_ALLOW_OPPORTUNISTIC=1",
                ),
                (
                    cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                    &mut follow_events as *mut i32 as *mut core::ffi::c_void,
                    "REUSE_FOLLOW_EVENT_DEPENDENCIES=1",
                ),
            ] {
                let rc = unsafe { cs::cuMemPoolSetAttribute(pool, attr, val) };
                // 家规:可失败调用不得吞掉 —— 池属性是性能面,失败 = 可见降级
                if rc != cs::CUresult::CUDA_SUCCESS {
                    sink.logf(
                        owl_shared::log::LVL_WARN,
                        "pool",
                        format_args!("池属性 {what} 设置失败(性能面降级): {rc:?}"),
                    );
                }
            }
            Ok(())
        };
        match pool_setup() {
            Ok(()) => sink.logf(
                owl_shared::log::LVL_INFO,
                "pool",
                format_args!("默认池已调优:release=MAX / opportunistic=1 / follow_events=1"),
            ),
            Err(e) => sink.logf(
                owl_shared::log::LVL_WARN,
                "pool",
                format_args!("默认池调优失败(驱动缺省行为): {e}"),
            ),
        }
        // 动态 smem 每块上限(boot 查一次缓存;发射前硬顶守卫用,
        // 不逐发射查 CUDA —— 热路径零额外调用)
        let smem_optin = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map_err(|e| format!("查 MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: {e:?}"))?
            as usize;
        // ---- 装配期架构门(设计 §15.4)----
        // 设备 sm ∉ 构造期编出的代际集 ⇒ **现在**拒启,而不是等到某次发射
        // 报 "no kernel image is available for execution on the device"。
        // 资产集为空(`OWL_CUBIN_SKIP` 构建)= 门关(仅用于门的负例验证)。
        {
            let archs = owl_kernels::family::cubins::CUBIN_ARCHS;
            if !archs.is_empty() {
                let major = ctx
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                    .map_err(|e| format!("查 COMPUTE_CAPABILITY_MAJOR: {e:?}"))?;
                let minor = ctx
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                    .map_err(|e| format!("查 COMPUTE_CAPABILITY_MINOR: {e:?}"))?;
                let dev = format!("sm_{major}{minor}");
                if !archs.contains(&dev.as_str()) {
                    return Err(format!(
                        "装配期架构门:设备 {dev}(ordinal {ordinal} )不在构建期 cubin 代际集 {archs:?} 内 ——\
                         重建时请加 `OWL_CUBIN_ARCHS=...,{dev}`(设计 §15.4)"
                    ));
                }
            }
            // ---- 零运行期编译覆盖率(boot 报告 + require 硬门)----
            let (covered, total, missing) = owl_kernels::family::cubins::coverage();
            sink.logf(owl_shared::log::LVL_INFO, "aot", format_args!("构建期 cubin 代际 {archs:?} / 资产 {} 个(fp {}) —— 登记表覆盖 {covered}/{total}",
                owl_kernels::family::cubins::CUBIN_ASSETS.len(),
                owl_kernels::family::cubins::ASSET_FP
            ));
            if !missing.is_empty() {
                let show: Vec<&str> = missing.iter().take(8).copied().collect();
                let total_show = missing.len();
                return Err(format!(
                    "零运行期编译硬门:{total_show} 个登记表核无构建期 cubin 资产 —— \
                     这些入口的 .cu 不在 build.rs 的 include_str! 闭包内(见 [aot] 行);\
                     前几个: {show:?}"
                ));
            }
        }
        let mut streams = HashMap::new();
        for (id, _role) in [(STREAM_H2D, "h2d"), (STREAM_COMPUTE, "compute"), (STREAM_D2H, "d2h")] {
            let s = ctx.new_stream().map_err(|e| format!("建流({_role}): {e:?}"))?;
            streams.insert(id, s);
        }
        // E1:三流事件池(**CU_EVENT_DISABLE_TIMING** —— 计时的 event 会拖慢
        // 且扰动时序;要计时另开探针)
        let mut event_pool = Vec::with_capacity(EVENT_POOL_N);
        for i in 0..EVENT_POOL_N {
            let ev = crate::ffi::event_create(crate::ffi::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(|e| format!("建事件({i}): {e:?}"))?;
            event_pool.push(TailEvent(ev));
        }
        Ok(Self {
            ctx,
            diag: diag.clone(),
            streams,
            tail_event: HashMap::new(),
            event_pool,
            ev_next: 0,
            zero_ev: HashMap::new(),
            ordinal,
            srv_timing: diag.srv_timing,
            cap_prof: diag.cap_prof,
            capture: None,
            graph_refs: Default::default(),
            graphs: HashMap::new(),
            next_graph: 1,
            blocks: HashMap::new(),
            next_block: 1,
            ledger,
            sink,
            smem_optin,
        })
    }

    /// 动态 smem 每块上限(boot 缓存;issue_launch 硬顶守卫用)
    pub(super) fn smem_optin(&self) -> usize {
        self.smem_optin
    }

    /// 流反查(id → 流)
    /// **E1:把"本命令的完成点"记到该流的尾事件**(纯记账;不等待任何东西)。
    ///
    /// - **捕获期跳过**:图内 event 节点会污染捕获拓扑(cuda-pitfalls §26
    ///   "event-tracking 启动即关"同律),且捕获不执行 ⇒ 尾事件语义不受影响;
    /// - 失败不破坏命令语义(记账面),由调用方响亮记日志;
    /// - 事件取池轮转:因为"某事件被 A 流等待、同时 B 流重录它"= **未定义行为**
    ///   (不是报错),池化是唯一正确复用方式。
    pub(super) fn record_tail(&mut self, sid: StreamId) -> Result<(), ModelError> {
        if self.capture_stream() {
            return Ok(());
        }
        if self.event_pool.is_empty() {
            return Err(ModelError::Msg("事件池未装配".into()));
        }
        let ev = self.event_pool[self.ev_next % self.event_pool.len()];
        self.ev_next = self.ev_next.wrapping_add(1);
        let stream = self.stream(sid)?.clone();
        unsafe { crate::ffi::event_record(ev.0, stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("record_tail(stream {sid}): {e:?}")))?;
        self.tail_event.insert(sid, ev);
        Ok(())
    }

    /// **跨流等待**(E2b 起用;E3 的序消费口):把 `stream` 排到 `dep` 的
    /// **当前尾事件**之后 —— 设备端排队、**零 host 阻塞**(实测提交 p50 180ns,
    /// 对比 host 全排空 1.9ms/键)。
    ///
    /// - **捕获期禁止**(`cuStreamWaitEvent` 不可捕获)⇒ 防御性拒(正常路径由
    ///   server 白名单先拦,此处是第二道);
    /// - `dep` 尚无完成点 ⇒ 响亮拒(不静默跳过 —— 跳过就等于丢序)。
    /// **块账本快照**(调试面;`GET /debug/blocks`):迭代活块并解析设备地址。
    /// 捕获期禁调(解析会登记图引用租约 —— 那是执行语义,不是观测语义)。
    pub(super) fn block_ledger(&self) -> Result<Vec<owl_iface::contract::BlockLedgerRow>, ModelError> {
        if self.capture_stream() {
            return Err(ModelError::Msg(
                "block_ledger: 捕获期禁止(解析会登记图引用)".into(),
            ));
        }
        let stream = self.stream(STREAM_COMPUTE)?.clone();
        let mut rows = Vec::with_capacity(self.blocks.len());
        for (&id, e) in self.blocks.iter() {
            // 单块解析失败 ⇒ 响亮拒(调试面禁静默跳过:少一行就少一个证据)
            let (addr, bytes) = self.block_ptr(id, &stream)?;
            rows.push(owl_iface::contract::BlockLedgerRow {
                id,
                bytes,
                addr,
                ledger: e.ledger,
            });
        }
        rows.sort_by_key(|r| r.id);
        Ok(rows)
    }

    /// **默认池回收**(内存面;见 `DeviceClient::mem_pool_trim`):
    /// `cuMemPoolTrimTo(pool, 0)` 把池内**已 free 的缓存块**还给 OS,
    /// 回 `RESERVED_MEM_CURRENT` 的差值(字节;0 = 无可回收)。
    ///
    /// 为什么由本项目负责:池属性 `RELEASE_THRESHOLD=UINT64_MAX`(性能面,
    /// 见上方池调优注释)⇒ free 后**永不自动还 OS**,装载期那批暂存块
    /// (27B 档 156MB)会一直占着,把跑图期余量吃掉。装载收尾调一次即可。
    pub(super) fn mem_pool_trim(&mut self) -> Result<u64, ModelError> {
        use cudarc::driver::sys as cs;
        // 池句柄 = 设备默认池(server 的 cudaMallocAsync 都走它)
        let dev = cudarc::driver::result::device::get(self.ordinal as i32)
            .map_err(|e| ModelError::Msg(format!("mem_pool_trim: device::get: {e:?}")))?;
        let mut pool = std::mem::MaybeUninit::<cs::CUmemoryPool>::uninit();
        // SAFETY:dev 来自 device::get;MaybeUninit 按 CUDA 契约写满
        let rc = unsafe { cs::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), dev) };
        if rc != cs::CUresult::CUDA_SUCCESS {
            return Err(ModelError::Msg(format!("mem_pool_trim: cuDeviceGetDefaultMemPool: {rc:?}")));
        }
        let pool = unsafe { pool.assume_init() };
        let reserved = |pool: cs::CUmemoryPool| -> Result<u64, ModelError> {
            let mut v: u64 = 0;
            // SAFETY:attr 与出参类型按 CUDA 契约配对(RESERVED_MEM_CURRENT = u64)
            let rc = unsafe {
                cs::cuMemPoolGetAttribute(
                    pool,
                    cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                    &mut v as *mut u64 as *mut core::ffi::c_void,
                )
            };
            if rc != cs::CUresult::CUDA_SUCCESS {
                return Err(ModelError::Msg(format!("mem_pool_trim: get RESERVED: {rc:?}")));
            }
            Ok(v)
        };
        let before = reserved(pool)?;
        // minBytesToKeep = 0 ⇒ 未使用的缓存块全还
        let rc = unsafe { cs::cuMemPoolTrimTo(pool, 0) };
        if rc != cs::CUresult::CUDA_SUCCESS {
            return Err(ModelError::Msg(format!("mem_pool_trim: cuMemPoolTrimTo: {rc:?}")));
        }
        let after = reserved(pool)?;
        Ok(before.saturating_sub(after))
    }

    /// **登记"本块已清零"事件**(在 `alloc{zero:true}` 的 memset 之后由 server 调)
    pub(super) fn note_zeroing(&mut self, id: u64, stream: StreamId) -> Result<(), ModelError> {
        let st = self.stream(stream)?.clone();
        let ev =
            crate::ffi::event_create(crate::ffi::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(|e| ModelError::Msg(format!("note_zeroing({id}): 建事件 {e:?}")))?;
        unsafe { crate::ffi::event_record(ev, st.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("note_zeroing({id}): 记入 {e:?}")))?;
        if let Some(old) = self.zero_ev.insert(id, TailEvent(ev)) {
            unsafe { let _ = crate::ffi::event_destroy(old.0); }
        }
        Ok(())
    }

    /// **等待该块的清零落地**(写入该块的搬运前必调;无账 = 无需等待)。
    /// 语义 = "别让本块的 memset 晚到把搬运抹了";等完即清账(一次足矣)——
    /// 等待是**设备端**的(零 host 阻塞),事件在入队等待后即可销毁
    /// (cudarc 同款纪律:等待捕获的是调用当刻的事件状态)。
    pub(super) fn wait_zeroing(&mut self, id: u64, dst_stream: StreamId) -> Result<(), ModelError> {
        let Some(ev) = self.zero_ev.remove(&id) else {
            return Ok(());
        };
        let st = self.stream(dst_stream)?;
        let r = unsafe {
            crate::ffi::stream_wait_event(
                st.cu_stream(),
                ev.0,
                crate::ffi::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .map_err(|e| ModelError::Msg(format!("wait_zeroing({id}): {e:?}")));
        unsafe { let _ = crate::ffi::event_destroy(ev.0); }
        r
    }

    /// 块回收时清清零账(防 id 复用后误等陈旧事件)
    pub(super) fn clear_zeroing(&mut self, ids: &[u64]) {
        for id in ids {
            if let Some(ev) = self.zero_ev.remove(id) {
                unsafe { let _ = crate::ffi::event_destroy(ev.0); }
            }
        }
    }

    pub(super) fn wait_stream_on_tail(&self, stream: StreamId, dep: StreamId) -> Result<(), ModelError> {
        if self.capture_stream() {
            return Err(ModelError::Msg(
                "跨流等待不可捕获:捕获期禁止 wait_stream_on_tail".into(),
            ));
        }
        // **按需新事件,不池化**(与 `server/harvest.rs::handle_dtoh_inner` 的
        // 「塔零案」修法**同机制**):`record_event(None)` 每次新建 + `wait(&ev)`
        // 设备端等待。
        //
        // ⚠️ **池化复用是错的**(2026-10-16 实测定谳):CUDA 语义里"事件被某流
        // 等待、同时被重录"= **UB(不报错)** —— 池轮转 8 事件下 4B VLM 四图
        // **确定性错值**(bird top-1 47 vs 280)。故本口一律新建;
        // state 里那套 `tail_event/event_pool` 降为**纯记账**(只录不等人,
        // 不参与任何定序)。
        let dep_stream = self.stream(dep)?;
        let ev = dep_stream.record_event(None).map_err(|e| {
            ModelError::Msg(format!("wait_stream_on_tail({stream}←{dep}): 记入 {e:?}"))
        })?;
        let st = self.stream(stream)?;
        st.wait(&ev).map_err(|e| {
            ModelError::Msg(format!("wait_stream_on_tail({stream}←{dep}): 等待 {e:?}"))
        })
    }

    /// 该流的尾事件(仅 server 内部消费;`None` = 该流尚无已记账的完成点)
    #[allow(dead_code)] // E2 的依赖判序消费口(本刀只记账)
    pub(super) fn tail_event(&self, sid: StreamId) -> Option<TailEvent> {
        self.tail_event.get(&sid).copied()
    }

    /// 收摊:销毁事件池与尾事件句柄(在流栅栏之后调;进程生命周期末点)
    pub(super) fn destroy_events(&self) {
        for ev in self.event_pool.iter().copied() {
            unsafe {
                let _ = crate::ffi::event_destroy(ev.0);
            }
        }
    }

    pub(super) fn stream(&self, id: StreamId) -> Result<&Arc<CudaStream>, ModelError> {
        self.streams
            .get(&id)
            .ok_or_else(|| ModelError::Msg(format!("流 {id} 不存在(固定三流:H2D/COMPUTE/D2H)")))
    }

    // ======================================================================
    // 图捕获状态机(护栏所在)
    // ======================================================================

    /// 当前是否捕获中
        /// 设备 ordinal(foreign-kernel 需要:marlin SM 数查询的 dev 参数)
    pub(super) fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    /// 已登记图数(诊断:dump 的相位判别用 —— 有图则说明图态臂在跑)
    pub(super) fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    pub(super) fn capture_stream(&self) -> bool {
        self.capture.is_some()
    }

    /// 图捕获开始(G1 不可嵌套;slab 在捕获**前**分配 —— 捕获期零 cudaMalloc)。
    /// 捕获流恒为 COMPUTE(路由策略:Launch/Alloc 全在 COMPUTE)。
    pub(super) fn graph_begin(&mut self) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(
                "graph_begin: 已有捕获进行中(不可嵌套/并发;先 graph_end)".to_string(),
            ));
        }
        // 引用租约:开新窗(嵌套已在上面拒掉 ⇒ 这里清的是"上一张图已收窗"
        // 之后的窗口集;既有租约不动)
        self.graph_refs.get_mut().begin_window();
        let stream = self.stream(STREAM_COMPUTE)?.clone();
        // 捕获前必须排空该流(在飞任务会污染捕获窗)
        stream
            .synchronize()
            .map_err(|e| ModelError::Msg(format!("graph_begin: 预排空失败 {e:?}")))?;
        // slab 容量三档:diag 固定档(兼容);缺省 = warmup 计量
        // hint(见 owl_shared::slab_hint;满额 × 图数 = VRAM 爆的工程债);
        // hint 缺席 = 64MiB 兜底
        let env_mb = self.diag.capture_slab_mb.filter(|&v| v > 0);
        let slab_cap = match env_mb {
            Some(mb) => mb << 20,
            None => owl_shared::slab_hint::take_hint().unwrap_or(64 << 20),
        };
        let slab = unsafe { stream.alloc::<u8>(slab_cap) }
            .map_err(|e| ModelError::Msg(format!("graph_begin: 捕获 slab 分配失败({}MiB) {e:?}", slab_cap >> 20)))?;
        owl_shared::vram::live_add(slab_cap as i64); // B6.5(直配不走 handle_alloc)
        let slab = Arc::new(slab);
        stream
            .begin_capture(CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| ModelError::Msg(format!("graph_begin: {e:?}")))?;
        self.capture = Some(CaptureState {
            stream,
            slab,
            arena: owl_shared::slab_hint::SlabArena::default(),
            carved: Vec::new(),
            nodes: CaptureNodes::default(),
            cap: slab_cap,
        });
        Ok(())
    }

    /// 图捕获结束(实例化 + 登记)。空捕获窗拒绝(哨兵③:发射数须 > 0)。
    pub(super) fn graph_end(&mut self) -> Result<GraphId, ModelError> {
        let cap = self
            .capture
            .take()
            .ok_or_else(|| ModelError::Msg("graph_end: 当前没有进行中的捕获".to_string()))?;
        let mut cap = cap;
        // 哨兵③:空捕获窗(begin_capture 后一次发射都没有;原生与家族同计)
        if cap.nodes.total == 0 {
            // 仍须 end_capture 退出捕获态;空图被驱动以 NULL 返回
            let g = unsafe { crate::ffi::stream_end_capture(cap.stream.cu_stream()) }
                .map_err(|e| ModelError::Msg(format!("graph_end: {e:?}")))?;
            if !g.is_null() {
                let _ = unsafe { crate::ffi::graph_destroy(g) };
            }
            // 窗内自产块**整批作废**(不登记图 ⇒ 永不会被回放;移出账房
            // ⇒ slab 最后一个引用走掉即释放)。旧实现留尸 = 整条 slab 钉到
            // 进程结束(空窗虽罕见,但同一路径的"窗内出错"很常见)。
            let orphan = self.release_window_blocks(&cap);
            if orphan > 0 {
                self.sink.logf(
                    owl_shared::log::LVL_WARN,
                    "graph-abort",
                    format_args!("空捕获窗作废:回收 {orphan} 块 slab 引用(slab 随最后一个引用释放)"),
                );
            }
            return Err(ModelError::Msg(
                "graph_end: 空捕获窗(发射数 = 0;先在捕获期发射 ≥ 1 个 kernel)".to_string(),
            ));
        }
        // 裸端捕获 + 零旗标实例化(AUTO_FREE_ON_LAUNCH 定谳见 ffi.rs;
        // 图执行语义与 safe 版逐位一致 —— 同 cuGraphLaunch 入 COMPUTE 流)
        let cu_graph = unsafe { crate::ffi::stream_end_capture(cap.stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("graph_end: {e:?}")))?;
        if cu_graph.is_null() {
            return Err(ModelError::Msg(
                "graph_end: 空捕获窗(发射数 = 0;哨兵③)".to_string(),
            ));
        }
        let mut exec: crate::ffi::sys::CUgraphExec = std::ptr::null_mut();
        // 刀D:实例化旗标 A/B 门(UPLOAD=2 预热 / DEVICE_LAUNCH=4 设备侧
        // 派发;派发税 10.8µs/节点 vs vLLM ~1.4 立案的根因排查面)
        let flags: u64 = self.diag.graph_flags;
        unsafe { crate::ffi::sys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, flags) }
            .result()
            .map_err(|e| ModelError::Msg(format!("graph_end: 实例化(flags={flags}): {e:?}")))?;
        // 节点数/类型观测(S1 launch 税立案:探针判 0.33µs/节点,引擎图
        // 实测 1.84µs/节点 —— 按类型分解找贵节点)
        if self.srv_timing {
            let mut n: usize = 0;
            unsafe {
                crate::ffi::sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n);
            }
            let mut nodes = vec![std::ptr::null_mut(); n];
            let mut got: usize = n;
            unsafe {
                crate::ffi::sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut got);
            }
            let mut kinds: std::collections::HashMap<i32, usize> = Default::default();
            let mut alloc_sizes: std::collections::HashMap<usize, usize> = Default::default();
            let mut cpy_sizes: std::collections::HashMap<(&str, usize), usize> = Default::default();
            let mut kern_names: std::collections::HashMap<String, usize> = Default::default();
            for nd in &nodes[..got] {
                let mut t: crate::ffi::sys::CUgraphNodeType = unsafe { std::mem::zeroed() };
                unsafe {
                    crate::ffi::sys::cuGraphNodeGetType(*nd, &mut t);
                }
                *kinds.entry(t as i32).or_default() += 1;
                // 刀1 取证:MEM_ALLOC 节点尺寸分布(派发税定位;真枚举:
                // 0=EMPTY 1=KERNEL 2=MEMCPY 3=MEMSET 4=HOST 5=GRAPH
                // 10=MEM_ALLOC 11=MEM_FREE)
                if t as u32 == 0 {
                    // 刀D:内核名清单(图材料单;派发税定位)
                    let mut params: crate::ffi::sys::CUDA_KERNEL_NODE_PARAMS =
                        unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        crate::ffi::sys::cuGraphKernelNodeGetParams_v2(*nd, &mut params)
                    };
                    if r == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                        let mut name: *const std::ffi::c_char = std::ptr::null();
                        let rn = unsafe {
                            crate::ffi::sys::cuFuncGetName(&mut name, params.func)
                        };
                        if rn == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                            let cname = unsafe { std::ffi::CStr::from_ptr(name) };
                            let sname = cname.to_string_lossy().to_string();
                            *kern_names.entry(sname).or_default() += 1;
                        }
                    }
                }
                if t as u32 == 10 {
                    let mut params: crate::ffi::sys::CUDA_MEM_ALLOC_NODE_PARAMS =
                        unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        crate::ffi::sys::cuGraphMemAllocNodeGetParams(*nd, &mut params)
                    };
                    if r == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                        *alloc_sizes.entry(params.bytesize).or_default() += 1;
                    }
                }
                if t as u32 == 1 {
                    let mut params: crate::ffi::sys::CUDA_MEMCPY_NODE_PARAMS =
                        unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        crate::ffi::sys::cuGraphMemcpyNodeGetParams(*nd, &mut params.copyParams)
                    };
                    if r == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                        // 3D 拷贝参数:宽度字节 = WidthInBytes;srcMemoryType/
                        // dstMemoryType 1=host 2=device(判方向)
                        let cp = &params.copyParams;
                        let kind = match (cp.srcMemoryType as u32, cp.dstMemoryType as u32) {
                            (1, 2) => "H2D",
                            (2, 1) => "D2H",
                            (2, 2) => "D2D",
                            _ => "?",
                        };
                        *cpy_sizes
                            .entry((kind, cp.WidthInBytes as usize))
                            .or_default() += 1;
                    }
                }
            }
            self.sink.logf(owl_shared::log::LVL_DEBUG, "gl-prof", format_args!("图节点 = {got}(类型 {kinds:?};真枚举 0=EMPTY 1=KERNEL 2=MEMCPY 3=MEMSET 4=HOST 10=MEM_ALLOC 11=MEM_FREE)"));
            if !alloc_sizes.is_empty() {
                let mut rows: Vec<_> = alloc_sizes.into_iter().collect();
                rows.sort_by_key(|b| std::cmp::Reverse(b.0));
                for (sz, n) in rows.iter().take(12) {
                    self.sink.logf(owl_shared::log::LVL_DEBUG, "gl-prof", format_args!("  MEM_ALLOC {sz}B × {n}"));
                }
            }
            if !kern_names.is_empty() {
                let mut rows: Vec<_> = kern_names.into_iter().collect();
                rows.sort_by_key(|b| std::cmp::Reverse(b.1));
                for (name, n) in rows.iter().take(40) {
                    self.sink.logf(owl_shared::log::LVL_DEBUG, "gl-prof", format_args!("  K {name} × {n}"));
                }
            }
            if !cpy_sizes.is_empty() {
                let mut rows: Vec<_> = cpy_sizes.into_iter().collect();
                rows.sort_by(|a, b| b.0.cmp(&a.0));
                for (sz, n) in rows.iter().take(12) {
                    self.sink.logf(owl_shared::log::LVL_DEBUG, "gl-prof", format_args!("  MEMCPY {:?} × {n}", sz));
                }
            }
        }
        let id = self.next_graph;
        self.next_graph += 1;
        // **收窗 → 入租约**(§图引用租约):本窗内解析过的 `Owned` 块从此被
        // `free_blocks` 拒收 —— 图里的裸指针就此获得"活着"的保证。
        let refs = self.graph_refs.get_mut().adopt();
        self.sink.logf(
            owl_shared::log::LVL_DEBUG,
            "graph-refs",
            format_args!(
                "图 {id}:发射 {} 发 / 内核 {} 种 / `Owned` 引用 {} 块 / slab 自产 {} 块(已入租约)— top {:?}",
                cap.nodes.total,
                cap.nodes.distinct(),
                refs.len(),
                cap.carved.len(),
                cap.nodes.top(5)
            ),
        );
        self.graphs.insert(
            id,
            GraphHolder {
                exec,
                cu_graph,
                stream: cap.stream.clone(),
                refs,
                carved: std::mem::take(&mut cap.carved),
                nodes: cap.nodes.clone(),
            },
        );
        Ok(id)
    }

    /// **捕获窗作废**(abort;窗内出错/空窗的收尾)。
    ///
    /// 语义 = "这一窗从未存在":退出捕获态 + 销毁半成品图 + **窗内自产块
    /// 整批移出账房**(⇒ slab 物理释放)。不做它的话,失败捕获会把**整条 slab**
    /// (最坏 GB 级)永久钉在账房里到进程结束(旧实现的挂账 "graph abort")。
    pub(super) fn graph_abort(&mut self) -> Result<(), ModelError> {
        let cap = self
            .capture
            .take()
            .ok_or_else(|| ModelError::Msg("graph_abort: 当前没有进行中的捕获".to_string()))?;
        let g = unsafe { crate::ffi::stream_end_capture(cap.stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("graph_abort: {e:?}")))?;
        if !g.is_null() {
            let _ = unsafe { crate::ffi::graph_destroy(g) };
        }
        let orphan = self.release_window_blocks(&cap);
        // 窗集不入租约(这些块已消失;留着只会污染下一窗的 adopt)
        self.graph_refs.get_mut().begin_window();
        self.sink.logf(
            owl_shared::log::LVL_WARN,
            "graph-abort",
            format_args!("捕获窗已作废:销毁半成品图 + 回收 {orphan} 块 slab 引用"),
        );
        Ok(())
    }

    /// 把某捕获窗的自产块移出账房(⇒ slab 的 `Arc` 计数下降;归零即释放)。
    /// 返回回收块数。abort/空窗/单图销毁三处共用。
    #[must_use]
    fn release_window_blocks(&mut self, cap: &CaptureState) -> usize {
        let mut n = 0;
        for id in &cap.carved {
            if self.blocks.remove(id).is_some() {
                n += 1;
            }
        }
        n
    }

    /// **销毁已登记图**(单图退休;LRU 驱逐 / plan `close()` 的底层)。
    ///
    /// 三件事:①排队序(先排空 COMPUTE —— `cuGraphExecDestroy` 在有在飞
    /// launch 时未定义,且块释放须与回放排空同序)②解除本图租约(计数递减,
    /// 归零即解钉 ⇒ 客户端随后可 free 输入槽)③本图 slab 块整批移出账房。
    pub(super) fn graph_destroy(&mut self, gid: GraphId) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(
                "graph_destroy: 捕获进行中不可销毁图(先 graph_end/graph_abort)".to_string(),
            ));
        }
        let Some(g) = self.graphs.remove(&gid) else {
            return Err(ModelError::Msg(format!("graph_destroy: 未知图 {gid}")));
        };
        self.stream(STREAM_COMPUTE)?
            .synchronize()
            .map_err(|e| ModelError::Msg(format!("graph_destroy: 预排空失败 {e:?}")))?;
        let unpinned = self.graph_refs.get_mut().forget(&g.refs);
        let mut released = 0;
        for id in &g.carved {
            if self.blocks.remove(id).is_some() {
                released += 1;
            }
        }
        let (n_launch, n_kind) = (g.nodes.total, g.nodes.distinct());
        drop(g); // GraphHolder::drop ⇒ exec + cu_graph 双销毁
        self.sink.logf(
            owl_shared::log::LVL_INFO,
            "graph-destroy",
            format_args!(
                "图 {gid} 已销毁:解钉 {unpinned} 块 `Owned` + 回收 {released} 块 slab 引用(原发射 {n_launch} 发 / 内核 {n_kind} 种)"
            ),
        );
        Ok(())
    }

    /// 图重放(捕获期间拒绝;流序异步提交;路由 COMPUTE 流)
    pub(super) fn graph_launch(&mut self, gid: GraphId) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(
                "graph_launch: 捕获进行中,不可回放(先 graph_end)".to_string(),
            ));
        }
        let stream = self.stream(STREAM_COMPUTE)?.clone();
        let prof = self.srv_timing;
        let t0 = prof.then(std::time::Instant::now);
        let (exec, cu_stream) = {
            let g = self
                .graphs
                .get(&gid)
                .ok_or_else(|| ModelError::Msg(format!("graph_launch: 图 {gid} 不存在")))?;
            (g.exec, g.stream.cu_stream())
        };
        unsafe { crate::ffi::graph_launch(exec, cu_stream) }
            .map_err(|e| ModelError::Msg(format!("graph_launch({gid}): {e:?}")))?;
        if prof {
            self.sink.logf(owl_shared::log::LVL_DEBUG, "gl-prof", format_args!("gid={gid} launch={:?}", t0.unwrap().elapsed()));
        }
        // 诊断开关(diag.sync_at(G RAPH)):回放后同步归因(烘焙指针消费者)
        if self.diag.sync_at(owl_shared::config::sync_bit::GRAPH) {
            let t1 = std::time::Instant::now();
            stream
                .synchronize()
                .map_err(|e| ModelError::Msg(format!("graph-sync({gid}): {e:?}")))?;
            self.sink.logf(owl_shared::log::LVL_DEBUG, "gl-prof", format_args!("gid={gid} gpu-sync={:?}", t1.elapsed()));
        }
        Ok(())
    }

    // ======================================================================
    // 池块账房
    // ======================================================================

    /// 登记新块(图外:独立分配)
    ///
    /// `name` = 分配点点名(必填;见 [`BlockEntry::name`])。
    /// **空名 = 调用点违约**(全量记名纪律):debug 构建当场炸(带 stack),
    /// release 构建不拦但由 `block_report_by_name`/`unnamed_blocks` 计入
    /// `(未命名!)` —— OOM 现场一眼可见"谁没报名字"。
    pub(super) fn new_block(&mut self, slice: CudaSlice<u8>, name: &str) -> u64 {
        debug_assert!(
            !name.is_empty(),
            "new_block: 空名 = 分配点未记名(全量记名纪律;见 BlockEntry::name)"
        );
        let bytes = slice.len();
        let local = self.next_block;
        self.next_block += 1;
        let id = pack_block_id(self.ledger, local);
        self.blocks.insert(
            id,
            BlockEntry {
                block: Block::Owned(slice),
                bytes,
                ledger: self.ledger,
                dtype: BlockDtype::Bytes,
                shape: vec![bytes],
                name: name.to_string(),
            },
        );
        id
    }

    /// 捕获期切块(零 cudaMalloc;**slab 内 arena**:bump + 尺寸类回收,
    /// 对齐 256B)。
    ///
    /// S1.5(2026-10-15,M1):原为**纯 bump**(无回收)⇒ 容量下限 = 分配总和
    /// (27B/512-tok chunk 实测 9362 MiB ⇒ OOM,prefill 图根本捕不上)。现回收表
    /// 复用已死块空间 ⇒ 容量 = **峰值活跃集**。
    ///
    /// ⚠️ 对齐前提同旧:向量化核(uint4/float4 16B 访存、cublas、FlashInfer)
    /// 要求基底对齐;旧核全标量未暴露此缺陷(2026-09-27 v1 paged 核
    /// MISALIGNED_ADDRESS 定谳)。
    pub(super) fn carve_block(&mut self, n: usize, name: &str) -> Result<u64, ModelError> {
        debug_assert!(
            !name.is_empty(),
            "carve_block: 空名 = 分配点未记名(全量记名纪律;见 BlockEntry::name)"
        );
        let cap = self.capture.as_mut().ok_or_else(|| {
            ModelError::Msg("carve_block: 仅限捕获期(内部护栏违例)".to_string())
        })?;
        if self.cap_prof {
            self.sink.logf(owl_shared::log::LVL_DEBUG, "cap-prof", format_args!("carve {}B", n));
        }
        let cap_total = cap.cap;
        let (off, cap_n) = cap.arena.carve(n, cap_total).ok_or_else(|| {
            ModelError::Msg(format!(
                "捕获 slab 耗尽:需 {n}B(峰值 {cap_total} 已满)—— 探针定尺失准或 OWL_CAPTURE_SLAB_MB 固定档过小"
            ))
        })?;
        let local = self.next_block;
        self.next_block += 1;
        let id = pack_block_id(self.ledger, local);
        let slab = cap.slab.clone();
        cap.carved.push(id);
        self.blocks.insert(
            id,
            BlockEntry {
                block: Block::Carved { slab, off, n, cap_n },
                bytes: n,
                ledger: self.ledger,
                dtype: BlockDtype::Bytes,
                shape: vec![n],
                name: name.to_string(),
            },
        );
        Ok(id)
    }

    /// 块设备指针 + 元素数(Owned 直取;Carved 切视图)
    pub(super) fn block_ptr(
        &self,
        id: u64,
        stream: &CudaStream,
    ) -> Result<(u64, usize), ModelError> {
        assert_ledger(id, self.ledger)?;
        use crate::ffi::DevicePtr;
        let e = self.blocks.get(&id).ok_or(ModelError::DeadBlock { id })?;
        // 图引用租约登记:捕获窗内的每次解析 = "这个块会被烘进图"
        // (死块在上面已拒 ⇒ 不记野 id)。窗外的解析与图无关,零开销。
        //
        // ⚠️ **只登记 `Owned`**:`Carved` 的安全由**类型**保证(捕获后 free
        // 一律 no-op;窗内 free = 回 arena 复用 —— 正是 M1 峰值口径的来源,
        // 不能被租约拦下)。租约要补的恰是 `Owned`(池件/输入槽)那一半。
        if self.capture.is_some() && matches!(e.block, Block::Owned(_)) {
            self.graph_refs.borrow_mut().record(id);
        }
        // 块账自洽(哨兵;cfg! 展开 ⇒ 两档都编译,零热路径语义)
        debug_assert_eq!(e.ledger, self.ledger, "块条目账本与 id 账本不一致");
        debug_assert_eq!(numel(&e.shape), e.bytes, "块 shape/账长不一致");
        debug_assert!(e.dtype.size_bytes() >= 1, "块 dtype 非法");
        match &e.block {
            Block::Owned(slice) => {
                let (p, _) = slice.device_ptr(stream);
                Ok((p, slice.len()))
            }
            Block::Carved { slab, off, n, .. } => {
                let view = slab.slice(*off..*off + *n);
                let (p, _) = view.device_ptr(stream);
                Ok((p, *n))
            }
        }
    }

    /// 捕获期发射计数 + 登记(哨兵③:graph_end 空窗校验;Launch 恒在 COMPUTE
    /// = 捕获流)。原生路径(`Command::Launch`)与家族路径(`record_capture`)
    /// **共用**这一个入口 —— 家族发射同样算"窗内有发射"(原实现只数原生,
    /// 会把纯家族窗误判为空窗,见 [`CaptureNodes`])。
    pub(super) fn note_launch(&mut self, kernel: &str) {
        if let Some(cap) = &mut self.capture {
            cap.nodes.add(kernel);
        }
    }

    /// 块账长(字节口径;账长校验 + **准入哨兵**用 —— 哨兵侧
    /// `Result → Option` 降级:探测不该炸,真解析的拒绝仍在
    /// `block_ptr`/`assert_ledger`)
    pub(super) fn block_len(&self, id: u64) -> Result<usize, ModelError> {
        assert_ledger(id, self.ledger)?;
        Ok(self.blocks.get(&id).ok_or(ModelError::DeadBlock { id })?.bytes)
    }

    /// **块账按名清单**(OOM dump 的"谁占着"面;2026-10-16)。
    ///
    /// 三笔总量账([`crate::state::pool_ledger_snapshot`] 的池 used/reserved、
    /// 驱动 `mem_get_info`、引擎 `vram::live`)只能回答"多满",不能回答"谁"。
    /// 本方法把**在册块按名聚合**(字节降序),OOM 现场一行看清单。
    ///
    /// 口径:账房在册 = 尚未 free 的块(捕获 slab 块含在内;slab 本体另算);
    /// 名字为分配点所报(空名 = 调用点违约,单独计数上屏 —— 见
    /// [`Self::unnamed_blocks`],这是"全量记名"的漏网信号)。
    pub(super) fn block_report_by_name(&self, top: usize) -> String {
        render_block_report(self.blocks.values().map(|e| (e.name.as_str(), e.bytes)), top)
    }

    /// 块账结构化清单(JSON;查询/OOM 落盘共用)。
    pub(super) fn block_json(&self) -> String {
        block_report_json(self.blocks.values().map(|e| (e.name.as_str(), e.bytes)))
    }

    /// 未命名块计数(全量记名的漏网信号:>0 即"有分配点没报名字")。
    pub(super) fn unnamed_blocks(&self) -> usize {
        count_unnamed(self.blocks.values().map(|e| (e.name.as_str(), e.bytes)))
    }

    /// **搬运/清零/拷贝类命令的统一边界校验**(2026-10-15;A13 审计补口)。
    ///
    /// 判据:`offset + len ≤ block_len`(块账本 = 字节口径)。
    ///
    /// # 为什么必须在这一层收
    ///
    /// 设备内**无越界保护**:`ptr + offset` 一旦超尾,写就落到**邻块**——
    /// 而池复用使邻块身份随布局变 ⇒ 表现为"布局敏感的确定性错值"
    /// (A13 期间"邻块 clobber"曾是头号嫌疑,最后靠 compute-sanitizer 才排除)。
    /// 与其每次靠仪器挖,不如在**唯一咽喉**(server 命令面)把它变成结构化拒绝。
    /// 补口前实测:`Launch` 有 L3 全量校验,而 `HtodChunk(s)/UploadPinned/
    /// MemsetZero/CopyBlock/CopyBatch` **六条一律无边界校验**。
    pub(super) fn check_range(
        &self,
        id: u64,
        off: usize,
        len: usize,
        what: &str,
    ) -> Result<(), ModelError> {
        let n = self
            .block_len(id)
            .map_err(|e| ModelError::Msg(format!("[{what}-arg] id={id} :: {e}")))?;
        if !range_fits(n, off, len) {
            return Err(ModelError::Msg(format!(
                "[{what}-arg] id={id} 越界:offset={off} + len={len} > block_len={n}"
            )));
        }
        Ok(())
    }

    /// 中间块回收(E2a):移除 Owned 块(CudaSlice drop → 流序
    /// free_async 归还设备池);Carved 块(slab 切片)跳过不删;
    /// 未知 id 容错跳过(幂等)。返回实际回收数。
    ///
    /// **E5 跨流回收律(2026-10-08 崩坏案修正)+ 定向化(2026-10-14 P1′-laneC)**:
    /// free_async 落在**块自身分配流**。危险集合 = **分配流 ≠ COMPUTE 的块** ——
    /// 实测只有 htod dst(分配在 H2D、队列浅 ⇒ free 立即归池),而 COMPUTE 可能
    /// 还压着该块的在途读者(65 层深队列)⇒ 设备池把同址发给下一个 alloc 即成
    /// 跨流 UAF。旧注“流序安全”仅对同流复用成立。修法:drop 前 COMPUTE 记事件,
    /// **需要钉序的那些流** wait 之 —— free 被钉在读者排空后,任意流复用皆安全。
    /// COMPUTE 分配的块 free 落在 COMPUTE 自身(深队列),流序天然保序
    /// ⇒ **无需栅栏**(实测这类占 ~98%)。
    ///
    /// ⚠️ 旧实现按「任一 Owned 块」发栅栏 ⇒ 每 chunk ~2900 道栅栏,把 H2D 流
    /// 钉在 COMPUTE 尾上(nsys:H2D 5863 次等待 = 808.9 ms)⇒ 下一发 htod 卡住
    /// ⇒ 等回执的客户端卡住 ⇒ **管道永远跑不到前面**(设备 idle 85%)。
    /// 捕获期跳过(事件会被烤进图)。
    pub(super) fn free_blocks(&mut self, ids: &[u64]) -> usize {
        // **图引用租约:拒收被图引用过的块**(见 [`GraphRefs`])。
        //
        // 为什么必须在咽喉处收:图烘的是裸指针 —— 被 free 的 `Owned` 块
        // (池件/输入槽)会被设备池**原址重发**给下一个 alloc ⇒ 图变"陈旧
        // 指针"(静默错值,不崩)。`Carved` 由下面的类型判据天然免疫,这里
        // 补的正是另一半。行为 = **响亮拒收但不致命**:记账 + 告警,块留在
        // 账房与 slab 里 ⇒ 没有 UAF/别名窗口。
        // 快路径:无命中(稳态)零分配。
        let rest_storage: Vec<u64>;
        let ids: &[u64] = if ids.iter().any(|id| self.graph_refs.borrow().is_pinned(*id)) {
            let (pinned, rest): (Vec<u64>, Vec<u64>) = ids
                .iter()
                .partition(|id| self.graph_refs.borrow().is_pinned(**id));
            self.sink.count("srv.count.free_pinned_reject", pinned.len() as u64);
            // 告警带"是哪张图钉的"(refs 字段的实际用途:诊断可行动)
            let holder: Vec<GraphId> = self
                .graphs
                .iter()
                .filter(|(_, g)| g.refs.iter().any(|r| pinned.contains(r)))
                .map(|(gid, _)| *gid)
                .collect();
            self.sink.logf(
                owl_shared::log::LVL_WARN,
                "free-pinned",
                format_args!(
                    "{} 块被图引用租约拒收(首个 id={};持有图 {holder:?};图烘入该地址 ⇒ free 会让设备池原址重发)",
                    pinned.len(),
                    pinned[0]
                ),
            );
            rest_storage = rest;
            &rest_storage
        } else {
            ids
        };
        if self.capture.is_none() && !self.diag.free_legacy {
            let fence = self.fence_targets(ids);
            if !fence.is_empty() {
                // 可观测面(父会话收口验收用):实发栅栏数 / 等待流数 / 被定向化省掉数
                self.sink.count("srv.count.free_fence", 1);
                self.sink.count("srv.count.free_fence_streams", fence.len() as u64);
                let sink = self.sink;
                if let Ok(comp) = self.stream(STREAM_COMPUTE) {
                    if let Ok(ev) = comp.record_event(None) {
                        for s in &fence {
                            // 家规:可失败返回值不吞 —— 栅栏失败降级为**可见**事件
                            // (块回收照跑;后续复用由设备池的保守分配兜底)
                            if let Err(e) = s.wait(&ev) {
                                sink.logf(
                                    owl_shared::log::LVL_WARN,
                                    "free-fence",
                                    format_args!("跳流栅栏 wait 失败(已降级;若见跨流脏读请查此): {e:?}"),
                                );
                            }
                        }
                    } // 事件创建失败 = 回旧语义(不让回收路径崩),但记一笔
                    else {
                        sink.logf(
                            owl_shared::log::LVL_WARN,
                            "free-fence",
                            format_args!("跳流栅栏 event 创建失败(本次回收无钉序)"),
                        );
                    }
                }
            } else if ids
                .iter()
                .any(|id| self.blocks.get(id).is_some_and(|e| matches!(e.block, Block::Owned(_))))
            {
                // 旧实现(“任一 Owned 块 ⇒ 恒等 H2D+D2H 都等”)会在此发一道栅栏
                // —— 本计数 = 被定向化省掉的栅栏数(谓词与旧实现逐字同为 Owned;
                // 每 chunk 千级;应远大于上面)
                self.sink.count("srv.count.free_fence_skip", 1);
            }
        }
        let mut n = 0;
        let mut freed_bytes: i64 = 0;
        for id in ids {
            // M1:warmup 影子 arena 归位(探针关 = 无动作)
            owl_shared::slab_hint::probe_free(*id);
            let Some((kind, off, cap_n)) = self.blocks.get(id).map(|e| match &e.block {
                Block::Owned(_) => (0u8, 0usize, 0usize),
                Block::Carved { off, cap_n, .. } => (1u8, *off, *cap_n),
            }) else {
                continue;
            };
            match kind {
                // ── Owned:图外块 → 真释放(块自身流;已按上律钉序)──
                0 => {
                    // 捕获窗内 Owned 死块**不释放**:`free_async` 可被捕获 ⇒
                    // 会烙 MEM_FREE 节点 + 地址漂移;而窗内分配一律 carve ⇒
                    // 窗内的 Owned 块必是**输入/池件**(它们的地址已/将被烘
                    // 入图)—— 防御性跳过(租约的窗内盲区就在此封口)。
                    if self.capture.is_some() {
                        self.sink.count("srv.count.free_owned_in_window", 1);
                        continue;
                    }
                    if let Some(e) = self.blocks.remove(id) {
                        if let Block::Owned(slice) = &e.block {
                            freed_bytes += slice.len() as i64; // B6.5 live 记账
                        }
                        drop(e);
                        n += 1;
                    }
                }
                // ── Carved 且**捕获窗内**:回 slab 内回收表(纯 host 簿记 ⇒
                //    捕获安全)。安全依据:scoped 归约只归还**不可达**中间块,
                //    而捕获按求值序记录 ⇒ 复用点在所有消费者之后(设计 §4.3)。
                1 if self.capture.is_some() => {
                    self.blocks.remove(id);
                    if let Some(cap) = self.capture.as_mut() {
                        cap.arena.release(off, cap_n);
                        n += 1;
                    }
                }
                // ── Carved 且**捕获后**:地址已烘进图 ⇒ **保持 no-op**
                //    (不可再发放,否则图内已烘焙地址被无关张量覆盖 ——
                //    施工文档 §5.2)。真要回收只能走 `graph_destroy`(整图退休)。
                _ => {
                    self.sink.count("srv.count.free_carved_noop", 1);
                }
            }
        }
        owl_shared::vram::live_add(-freed_bytes);
        n
    }

    /// 回收栅栏的目标流(去重;2026-10-14 定向化):
    /// - 默认(新语义):只有**分配流 ≠ COMPUTE** 的 Owned 块需要钉序;
    /// - `sync_bit::FENCE`(旧行为):任一 Owned 块 ⇒ H2D + D2H 都等。
    fn fence_targets(&self, ids: &[u64]) -> Vec<Arc<CudaStream>> {
        let mut out: Vec<Arc<CudaStream>> = Vec::new();
        let compute = self.streams.get(&STREAM_COMPUTE);
        for id in ids {
            let Some(e) = self.blocks.get(id) else { continue };
            let Block::Owned(slice) = &e.block else { continue };
            if self.diag.sync_at(owl_shared::config::sync_bit::FENCE) {
                for sid in [STREAM_H2D, STREAM_D2H] {
                    if let Some(s) = self.streams.get(&sid) {
                        if !out.iter().any(|p| Arc::ptr_eq(p, s)) {
                            out.push(s.clone());
                        }
                    }
                }
                continue;
            }
            let st = slice.stream();
            let is_compute = compute.is_some_and(|c| Arc::ptr_eq(c, st));
            if !is_compute && !out.iter().any(|p| Arc::ptr_eq(p, st)) {
                out.push(st.clone());
            }
        }
        out
    }
}

// ============================================================================
// kernel 装载缓存(2026-10-13 载体切换:构建期 cubin 优先,零运行期编译)
// ============================================================================
//
// 两个平面:
// - **AOT 面**:`build.rs` 把每个 `.cu` 编成 fatbin(`$OUT_DIR/cubins/`),登记表
//   以**源文本哈希**为键(`owl_kernels::family::cubins`)。同一文件的核共享一个
//   模块(旧路径是**每核一个模块**,逐核 nvrtc + 重复装载)。
// - **无 NVRTC 面**(2026-10-13 裁决:零运行期编译 ⇒ NVRTC 已退役):
//   资产未命中 = **拒发**(不降级到运行期编译)。
//
// `aot_hits` / `misses` 是**机器判据面**:验收 = 整轮 e2e 日志 `aot-miss` 恒 0;
// 结构性判据 = `crates/kernels/tests/no_runtime_compile.rs`(生产路径零编译调用)。

pub(super) struct KernelCache {
    compiled: HashMap<String, CudaFunction>,
    /// AOT 资产名 → 已载模块(同 `.cu` 的核共享一个模块)
    modules: HashMap<&'static str, Arc<cudarc::driver::CudaModule>>,
    /// 资产的核数(诊断/验收判据)
    pub(super) aot_hits: usize,
    /// 无资产的核名(每名一次;= 拒启清单/机器判据面)
    pub(super) misses: Vec<String>,
}

impl KernelCache {
    pub(super) fn new() -> Self {
        Self { compiled: HashMap::new(), modules: HashMap::new(), aot_hits: 0, misses: Vec::new() }
    }

    /// 核取用:**构建期 cubin 资产唯一路径**(源哈希查表 → fatbin → 符号)。
    ///
    /// 未命中 = `Err`(拒发)。**没有 NVRTC 兜底**(2026-10-13 裁决:零运行期
    /// 编译);源码不在 `include_str!` 闭包内 = 该入口没有构建期资产 =
    /// 编程错误(要么漏建表,要么该入口已废弃应从发射链剔除)。
    pub(super) fn ensure_kernel(
        &mut self,
        ctx: &Arc<CudaContext>,
        name: &str,
        source: &str,
        sink: owl_shared::log::LogSink,
    ) -> Result<CudaFunction, ModelError> {
        if let Some(f) = self.compiled.get(name) {
            return Ok(f.clone());
        }
        let asset = owl_kernels::family::cubins::asset(source).ok_or_else(|| {
            // 机器判据面:整轮 e2e 日志里 `grep -c aot-miss` 应恒 0
            sink.logf(owl_shared::log::LVL_WARN, "aot-miss", format_args!("{name} —— 无构建期 cubin 资产(源 {} B)", source.len()));
            if !self.misses.iter().any(|m| m == name) {
                self.misses.push(name.to_string());
            }
            ModelError::Msg(format!(
                "{name} 无构建期 cubin 资产(源 {} B)—— 零运行期编译硬门:该 .cu 不在 build.rs 的 \
                 include_str! 闭包内(NVRTC 已退役,无兜底路径)",
                source.len()
            ))
        })?;
        let module = match self.modules.get(asset.module) {
            Some(m) => m.clone(),
            None => {
                // fatbin 载入:driver API 与纯 cubin 完全相同(多代际由驱动
                // 按设备挑最匹配那份)
                let m = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
                    .map_err(|e| {
                        ModelError::Msg(format!(
                            "aot load_module({} / {name}): {e:?} —— 资产载入失败(代际不匹配?驱动过旧?)",
                            asset.module
                        ))
                    })?;
                self.modules.insert(asset.module, m.clone());
                m
            }
        };
        let func = module.load_function(name).map_err(|e| {
            ModelError::Msg(format!(
                "aot load_function({name}) in {}: {e:?} —— 符号不在资产里(源哈希命中但符号缺失 = 登记表/源失配)",
                asset.module
            ))
        })?;
        self.aot_hits += 1;
        self.compiled.insert(name.to_string(), func.clone());
        Ok(func)
    }
}

// ============================================================================
// pinned host 码头(非阻塞搬运的 host 侧落点)
// ============================================================================

/// 页锁定 host 缓冲(自管生命周期)。
///
/// ⚠️ 生命周期纪律:pinned 内存在搬运完成前**不得释放**——本类型只允许
/// 被 move 进完成回调,随回调一起 drop。
pub(super) struct Staging {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Staging {}

impl owl_iface::contract::PinnedRegion for Staging {
    fn slice_bytes_mut(&mut self) -> &mut [u8] {
        Staging::slice_mut(self)
    }
    fn as_bytes(&self) -> &[u8] {
        Staging::slice(self)
    }
}

impl Staging {
    /// 分配 n 元素的 pinned 缓冲(未初始化)
    pub(super) fn alloc(n: usize, label: &str) -> Result<Self, ModelError> {
        // 字节口径(f16 基线,2026-09-26):n = 字节数,DMA 按 byte 计数
        let ptr = unsafe { malloc_host(n, 0) }
            .map_err(|e| ModelError::Msg(format!("malloc_host(n={n}B): {e:?}")))? as *mut u8;
        // FIXME(2026-10-13):cudarc 的 `malloc_host` 对失败/零尺寸可能回 Ok(null)
        // ——这里是裸 assert,拿不到任何上下文。先带上尺寸与原因(下同)。
        assert!(
            !ptr.is_null(),
            "Staging::alloc[{label}]: malloc_host(n={n}B) 返回空指针(锁页内存耗尽?n=0 = 零长块写,上游形状错)"
        );
        Ok(Self { ptr, len: n })
    }

    pub(super) fn slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(super) fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // 收尾失败无路可报(Drop 语义);sticky 错误会在后续流操作上暴露
        let _ = unsafe { free_host(self.ptr as *mut _) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 跨账本 id 解析 ⇒ 结构化 CrossLedger 错(含 id + 两个 ledger 号)。
    /// 纯解码/断言,不触 CUDA(可在无卡环境跑)。
    #[test]
    fn cross_ledger_id_is_rejected() {
        let id = pack_block_id(2, 11);
        assert_eq!(ledger_of(id), 2);
        assert_eq!(local_of(id), 11);
        let e = assert_ledger(id, 1).expect_err("跨账本必拒");
        let msg = format!("{e}");
        assert_eq!(
            msg,
            "跨账本块: 块 id=2199023255563 属 ledger 2,被 ledger 1 解析"
        );
        // 同账本放行
        assert!(assert_ledger(id, 2).is_ok());
    }

    /// 位拼接跨账本注入性:同 local 不同 ledger ⇒ id 不同(§27 撞号门根因消除)。
    #[test]
    fn pack_is_injective_across_ledgers() {
        assert_ne!(pack_block_id(1, 11), pack_block_id(2, 11));
        assert_eq!(ledger_of(pack_block_id(3, 99)), 3);
    }

    // ── 全量记名/OOM 按名清单(纯函数;零 CUDA)────────────────────────

    /// **按名清单**:同名字聚合(字节/个数求和)、字节降序、top-N 截断折入"另 N 类"。
    /// 这条就是 OOM 现场那行的口径锁——改渲染必改本测试。
    #[test]
    fn block_report_aggregates_by_name_and_sorts_desc() {
        let es: [(&str, usize); 4] = [
            ("kv.k.0", 1 << 20),
            ("kv.k.0", 1 << 20),
            ("dfk.k.0", 3 << 20),
            ("fi.indices", 512 << 10),
        ];
        let rep = render_block_report(es.iter().copied(), 2);
        assert!(rep.starts_with("在册 4 块 / 合计 5.5MiB(按名 top2):"), "头:{rep}");
        // 降序:dfk(3MiB) 在 kv(2MiB) 之前;同名字合并 ×2
        let dfk = rep.find("dfk.k.0=3.0MiB×1").expect("dfk 行");
        let kv = rep.find("kv.k.0=2.0MiB×2").expect("kv 合并行");
        assert!(dfk < kv, "必须字节降序:{rep}");
        // top2 之外的第三类折入计数(名不出现在正文)
        assert!(rep.ends_with("…另 1 类"), "尾:{rep}");
        assert!(!rep.contains("fi.indices"), "top-N 之外不得列名:{rep}");
    }

    /// **空名必须响亮**(负例):漏记名的块渲染成 `(未命名!)` 且被单独计数——
    /// 否则"全量记名"会退化成"大部分有名",漏网无声无息。
    #[test]
    fn unnamed_blocks_are_loud_not_silent() {
        let es: [(&str, usize); 3] = [("kv.k.0", 1 << 20), ("", 2 << 20), ("tiny.meta", 1536)];
        let rep = render_block_report(es.iter().copied(), 8);
        assert!(rep.contains("(未命名!)=2.0MiB×1"), "空名必须显式子项:{rep}");
        // 单位律:小于 1MiB 的块必须用 KiB 显形(不许渲染成 0.0MiB 而消失量级)
        assert!(rep.contains("tiny.meta=1.5KiB×1"), "小块必须 KiB 口径:{rep}");
        assert_eq!(count_unnamed(es.iter().copied()), 1);
        // 全有名 ⇒ 计数 0(正向对照:门不误报)
        let ok: [(&str, usize); 1] = [("kv.k.0", 1 << 20)];
        assert_eq!(count_unnamed(ok.iter().copied()), 0);
    }

    /// **JSON 结构化清单**:总量/分组/降序/转义。与文本同源 ⇒ 二者总量必须一致。
    #[test]
    fn block_json_shape_and_escapes() {
        let es: [(&str, usize); 3] = [("kv.k.0", 1 << 20), ("kv.k.0", 1 << 20), ("a\"b\\c", 1536)];
        let js = block_report_json(es.iter().copied());
        assert!(js.starts_with("{\"blocks\":[{\"name\":\"kv.k.0\",\"bytes\":2097152,\"blocks\":2}"), "头:{js}");
        assert!(js.contains("{\"name\":\"a\\\"b\\\\c\",\"bytes\":1536,\"blocks\":1}"), "转义:{js}");
        // 尾判据用"片段在 + 总数字面"而不是 ends_with(尾字符判据脆弱,已踩)
        assert!(js.contains("\"total_bytes\":2098688"), "total_bytes:{js}");
        assert!(js.contains("\"total_blocks\":3"), "total_blocks:{js}");
        // 同源:文本口径的总量(2.0MiB + 1.5KiB)与 JSON 字节数互证
        assert!(render_block_report(es.iter().copied(), 8).contains("在册 3 块"));
    }

    /// 峰值快照三重判据:未到门 / 破步进 / 写满 各一条负例 + 两条正例。
    #[test]
    fn peak_snapshot_gate_matrix() {
        let mb = 1u64 << 20;
        // 未到门 ⇒ 不写
        assert_eq!(peak_should_dump(100 * mb, 1000 * mb, 0, 0, 8), None);
        // 到门首次 ⇒ 写,水位 = live
        assert_eq!(peak_should_dump(1000 * mb, 1000 * mb, 0, 0, 8), Some(1000 * mb));
        // 到门但只比上次高 4% ⇒ 不写(步进 5%)
        assert_eq!(peak_should_dump(1040 * mb, 1000 * mb, 1000 * mb, 1, 8), None);
        // 高 6% ⇒ 写
        assert_eq!(peak_should_dump(1060 * mb, 1000 * mb, 1000 * mb, 1, 8), Some(1060 * mb));
        // 写满 ⇒ 不写(即使破步进)
        assert_eq!(peak_should_dump(9999 * mb, 1000 * mb, 1000 * mb, 8, 8), None);
        // 门值 0 = 关
        assert_eq!(peak_should_dump(9999 * mb, 0, 0, 0, 8), None);
    }

    /// 落盘路径解析:门值三态(默认时间戳 / 显式路径 / 空串视作默认)。
    #[test]
    fn vram_dump_path_resolution() {
        assert_eq!(resolve_vram_dump_path("1", 1234), "/tmp/owl-vram-1234.json");
        assert_eq!(resolve_vram_dump_path("", 7), "/tmp/owl-vram-7.json");
        assert_eq!(resolve_vram_dump_path("/tmp/x.json", 9), "/tmp/x.json");
    }

    /// 现场名字 → 毫秒戳:两种自家形态认得出,陌生形态一律 `None`(不误删)。
    #[test]
    fn vram_dump_stamp_forms() {
        assert_eq!(vram_dump_stamp("owl-vram-1792124456019.json"), Some(1792124456019));
        assert_eq!(vram_dump_stamp("owl-vram-peak-1792124456019-64MiB.json"), Some(1792124456019));
        // 陌生/残缺形态 ⇒ None(显式路径恰好长这样也照样不放行)
        assert_eq!(vram_dump_stamp("x.json"), None);
        assert_eq!(vram_dump_stamp("owl-vram-abc.json"), None);
        assert_eq!(vram_dump_stamp("owl-vram-12.json.bak"), None);
        assert_eq!(vram_dump_stamp("owl-vram-peak-x-1MiB.json"), None);
        assert_eq!(vram_dump_stamp("owl-vram-12.json"), Some(12));
    }

    /// 保留裁剪:只删自家现场、保新、陌生文件不动、`keep = 0` 全留。
    #[test]
    fn prune_keeps_newest_and_spares_foreign() {
        let dir = std::env::temp_dir().join(format!("owl-vram-prune-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("临时目录");
        for ms in [40u128, 10, 30, 20] {
            std::fs::write(dir.join(format!("owl-vram-{ms}.json")), "{}").expect("写");
        }
        std::fs::write(dir.join("owl-vram-peak-5-1MiB.json"), "{}").expect("写");
        std::fs::write(dir.join("别人的文件.json"), "{}").expect("写");
        // 5 份自家现场(4 OOM + 1 峰值),keep = 2 ⇒ 删 3 份最旧(戳 5 / 10 / 20)
        assert_eq!(prune_vram_dumps(&dir, 2), 3);
        for ms in [30u128, 40] {
            assert!(dir.join(format!("owl-vram-{ms}.json")).is_file(), "戳 {ms} 应留下");
        }
        for gone in ["owl-vram-10.json", "owl-vram-20.json", "owl-vram-peak-5-1MiB.json"] {
            assert!(!dir.join(gone).is_file(), "{gone} 应被裁掉");
        }
        assert!(dir.join("别人的文件.json").is_file(), "陌生文件不得删");
        // keep = 0 = 关 ⇒ 一份不删
        assert_eq!(prune_vram_dumps(&dir, 0), 0);
        // 目录不存在 ⇒ 0(不炸)
        assert_eq!(prune_vram_dumps(&dir.join("没有这个目录"), 1), 0);
        std::fs::remove_dir_all(&dir).expect("清理");
    }

    /// 空账不炸(捕获窗未开/图外空账都可能读它;报告要能渲染)
    #[test]
    fn empty_ledger_report_is_well_formed() {
        let es: [(&str, usize); 0] = [];
        assert_eq!(render_block_report(es.iter().copied(), 4), "(块账空)");
        assert_eq!(block_report_json(es.iter().copied()), "{\"blocks\":[],\"total_bytes\":0,\"total_blocks\":0}");
        assert_eq!(count_unnamed(es.iter().copied()), 0);
    }

    /// 图引用租约(纯簿记,零 CUDA):窗内解析**不入租约**,收窗才入;
    /// 开新窗不清旧租约(老图还活着);重解析幂等;单图销毁按计数解钉。
    #[test]
    fn graph_refs_window_scope_and_pin() {
        let mut r = GraphRefs::default();
        // 窗内解析:只记窗口集,不入租约(图还没成)⇒ 此刻 free 应放行
        r.begin_window();
        r.record(7);
        r.record(9);
        assert_eq!(r.window_len(), 2);
        assert!(!r.is_pinned(7), "收窗前不得入租约(否则捕获失败的块会被永久钉住)");
        assert_eq!(r.pinned_len(), 0);
        // 收窗:本图引用集 = 2 块,入租约
        let refs = r.adopt();
        assert_eq!(refs.len(), 2);
        assert!(r.is_pinned(7) && r.is_pinned(9));
        assert_eq!(r.window_len(), 0, "收窗须清窗口集");
        // 开第二张图:旧租约不动,新窗独立
        r.begin_window();
        assert!(r.is_pinned(7), "老图还活着 ⇒ 租约不得随新窗清空");
        assert_eq!(r.window_len(), 0);
        // 同一块被第二张图再引用:计数 +1(并集不重复)
        r.record(7);
        r.record(11);
        let refs2 = r.adopt();
        assert_eq!(refs2.len(), 2);
        assert_eq!(r.pinned_len(), 3, "并集:7/9/11");
        // 销毁图 1:9 解钉;7 仍被图 2 引用(计数减到 1)⇒ 不得放行
        let unpinned = r.forget(&refs);
        assert_eq!(unpinned, 1, "只 9 归零");
        assert!(!r.is_pinned(9));
        assert!(r.is_pinned(7), "多图共享块:一张图销毁不得解钉");
        // 销毁图 2:7/11 解钉;9 再 forget 幂等(不得 panic)
        assert_eq!(r.forget(&refs2), 2);
        assert_eq!(r.forget(&refs2), 0, "重复解钉幂等");
        assert!(!r.is_pinned(7) && !r.is_pinned(11));
        assert_eq!(r.pinned_len(), 0);
        // 未解析过的块照常放行(自由块不得被误钉)
        assert!(!r.is_pinned(13));
    }

    /// 捕获窗发射登记表:总数/聚合/Top-N 序稳定(镜像测试用)。
    #[test]
    fn capture_nodes_histogram_and_top() {
        let mut n = CaptureNodes::default();
        assert_eq!(n.total, 0);
        n.add("owl_marlin_gemm_f16_u4b8");
        n.add("owl_marlin_gemm_f16_u4b8");
        n.add("owl_gdn_chunked_delta_rule_fwd");
        n.add("owl_flashinfer_prefill_paged_fp8kv");
        n.add("owl_gdn_chunked_delta_rule_fwd");
        assert_eq!(n.total, 5);
        assert_eq!(n.distinct(), 3);
        let top = n.top(2);
        assert_eq!(
            top,
            vec![
                ("owl_gdn_chunked_delta_rule_fwd".to_string(), 2),
                ("owl_marlin_gemm_f16_u4b8".to_string(), 2)
            ],
            "同数按名序(A13 同款确定性)"
        );
        assert_eq!(n.top(9).len(), 3, "N 超表长只回表内");
    }

    /// 家族面捕获入境判定:不变盒恒放行;走第四条路的动作观察档放行、严格档拒。
    #[test]
    fn fam_window_verdict_matrix() {
        // 不变盒(gdn_scalar 的 soff 范式)= 捕获窗合规
        assert!(fam_window_verdict(FamAct::Upload { invariant: true }, false).is_ok());
        assert!(fam_window_verdict(FamAct::Upload { invariant: true }, true).is_ok());
        // 观察档(缺省):放行(零行为变化)
        assert!(fam_window_verdict(FamAct::ScratchAlloc, false).is_ok());
        assert!(fam_window_verdict(FamAct::Upload { invariant: false }, false).is_ok());
        // 严格档:结构化拒,且拒语必须点出病因(免得后人以为只是"不支持")
        let e = fam_window_verdict(FamAct::ScratchAlloc, true).expect_err("严格档须拒");
        assert!(e.contains("alloc") && e.contains("MEM_ALLOC"), "拒语应可行动:{e}");
        let e = fam_window_verdict(FamAct::Upload { invariant: false }, true).expect_err("严格档须拒");
        assert!(e.contains("upload") && e.contains("不变盒"), "拒语应可行动:{e}");
    }
}

/// **区间是否落在块内**(纯函数;`check_range` 的算术核 —— 可单测)。
///
/// 边界语义:`off + len == n` 合法(写满);`off + len > n` 越界;
/// 溢出(`usize` 相加回绕)按越界处理(不可信输入不许回绕成小值)。
pub(super) fn range_fits(n: usize, off: usize, len: usize) -> bool {
    off.checked_add(len).map(|end| end <= n).unwrap_or(false)
}

#[cfg(test)]
mod range_tests {
    use super::range_fits;

    /// 边界:恰好写满合法 / 超 1 字节越界 / 零长在尾合法 / 起点即越界 / 溢出拒绝
    #[test]
    fn range_fits_boundaries() {
        assert!(range_fits(100, 0, 100), "恰好写满");
        assert!(range_fits(100, 50, 50), "尾部对齐");
        assert!(range_fits(100, 100, 0), "零长在尾(合法:不写)");
        assert!(!range_fits(100, 0, 101), "超 1 字节");
        assert!(!range_fits(100, 100, 1), "起点即越界");
        assert!(!range_fits(100, 101, 0), "起点越界(零长也不放)");
        assert!(!range_fits(usize::MAX, 1, usize::MAX), "相加溢出按越界");
        assert!(!range_fits(0, 0, 1), "空块写入");
    }
}
