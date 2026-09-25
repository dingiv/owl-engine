//! 层协议:计算声明(Module)+ 装载需求(Loadable/Weight/Want)。
//!
//! 两面一个主题(2026-09-26 重组:原 module.rs + loader.rs 合并)——
//! 层与解释器的全部接口面:
//!
//! ```rust,ignore
//! let mut lin = Linear::new("w", 2, 3);   // new = 准备容器(空包指针,零数据)
//! eval_load(&lin, &mut face, &src, &ctx).await?; // 装载执行(驱动 layout)
//! let out = lin.forward(&xs, &ctx);       // 计算声明(total;毒值随链)
//! let y = eval(&lin, &xs, &mut face, &ctx).await?; // 计算执行
//! ```
//!
//! 三律(与 TensorOps 同构):同步(total)· 纯描述;执行都在解释器。

use crate::contract::{Bytes, DeviceClient, Dtype, ModelError, Shape};
use crate::layers::rope::Rope;
use crate::TensorOps;
use std::sync::{Arc, Mutex};

// ============================================================================
// §1 Module:计算声明协议(+ 每步动态依赖词汇)
// ============================================================================

/// 每步动态依赖(forward 的 ctx;C4 定案 2026-09-26:引用注入取代
/// Copy KernelCtx —— rope 表本来就是设备块,ctx 借用 `&Rope` 零双重
/// 所有权;字段 `TensorOps` 值形态让 runner 自选 from_host(调试)/
/// of_block(图捕获),捕获路径不封)。
///
/// runner 每步构造,层只透传不自取(async-runtime §4.3 grab-bag);
/// 无动态依赖的层(mlp/rmsnorm/…)忽略字段。借用结构,无 Default ——
/// 测试用 [`ForwardCtx::minimal`]。
pub struct ForwardCtx<'a> {
    /// 本步 token 数(decode = 1;prefill = 块长)
    pub tokens: usize,
    /// 位置(attention/GDN 用;缺省 = 依赖违约 → 层侧毒值)
    pub pos: Option<&'a TensorOps>,
    /// decode KV 直排引用(attention 用;GDN state 同款随域扩)
    pub kv: Option<&'a KvBuffers>,
    /// 位置编码层(全局一份;表经 eval_load 已是设备块。
    /// 注:类型住在 layers,此处反向引用 —— ctx 是 runner 词汇,
    /// runner 持全局 rope,与旧世界 rotary_emb 注入同一形态)
    pub rope: Option<&'a Rope>,
}

impl<'a> ForwardCtx<'a> {
    /// 最小 ctx(无动态依赖;mlp/rmsnorm/linear/embedding 测试用)
    pub fn minimal(tokens: usize) -> Self {
        Self { tokens, pos: None, kv: None, rope: None }
    }

    /// decode 步 ctx(全量动态依赖)
    pub fn decode(
        tokens: usize,
        pos: &'a TensorOps,
        kv: &'a KvBuffers,
        rope: &'a Rope,
    ) -> Self {
        Self { tokens, pos: Some(pos), kv: Some(kv), rope: Some(rope) }
    }
}

/// decode 步 KV 直排缓冲(引用形态:块归 engine,层零所有权)。
///
/// - `k_cache` / `v_cache`:常驻池块引用(`TensorOps::of_block`),
///   [max_slots, Hkv, HD] f32,块只增不减;
/// - `slots` / `kv_lens`:[bs] f32 数值过线(契约 5;核内 cast);
///   **kv_lens 含本步**(kernel 先写 cache 后打分,caller 传 past+1)。
/// (2026-09-26 C4 重组:自 layers/attention 归位协议层 —— 它是 runner
/// 与层之间的动态依赖词汇,非层私有。)
pub struct KvBuffers {
    pub k_cache: TensorOps,
    pub v_cache: TensorOps,
    pub slots: TensorOps,
    pub kv_lens: TensorOps,
}

/// KV 动态上下文:每步由 runner 构造。(注:decode 直排路径暂走
/// [`KvBuffers`];本结构为 paged 路线预留。)
#[derive(Clone, Debug)]
pub struct KvCtx {
    pub step: u64,
    pub slots: Vec<u32>,
}

/// 模型层统一接口
pub trait Module {
    /// 计算声明:xs → y(同步 · total · 纯描述;ctx = 动态依赖,
    /// 缺依赖 → 层侧毒值声明,eval 边界收割 —— 与未装载槽同构)
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps;
}

// ============================================================================
// §2 数据源:执行器的取数对象(测试 HashMap / 将来 safetensors / 层内表)
// ============================================================================

/// 权重数据源(按槽键取 host f32;层内短键,前缀归上层适配)
pub trait WeightSource {
    fn get(&self, key: &str) -> Option<&[f32]>;
}

impl WeightSource for std::collections::HashMap<String, Vec<f32>> {
    fn get(&self, key: &str) -> Option<&[f32]> {
        self.get(key).map(|v| v.as_slice())
    }
}

/// 容器内表源(rope cos/sin 等纯计算产物;借用层内 Vec,生命周期随层)
pub struct TableSource<'a> {
    entries: Vec<(&'static str, &'a [f32])>,
}

impl<'a> TableSource<'a> {
    pub fn new(entries: Vec<(&'static str, &'a [f32])>) -> Self {
        Self { entries }
    }
}

impl WeightSource for TableSource<'_> {
    fn get(&self, key: &str) -> Option<&[f32]> {
        self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }
}

/// f32 → LE 字节(host 装载辅助;装载域唯一权威)
pub(crate) fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// §3 空包指针:块句柄格(装载产物写入处;decl 读取处)
// ============================================================================

/// 共享块句柄格(空包):装载前 None,执行器物化后 Some(块句柄)
#[derive(Clone, Default)]
pub struct SlotCell(Arc<Mutex<Option<Bytes>>>);

impl SlotCell {
    fn deliver(&self, b: Bytes) {
        *self.0.lock().unwrap() = Some(b);
    }

    fn loaded(&self) -> Option<Bytes> {
        self.0.lock().unwrap().clone()
    }
}

// ============================================================================
// §4 LoaderOps:数据需求清单(纯元数据;零数据;total)
// ============================================================================

/// 数据存在形式(布局)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// 源即装载布局(行主序直给)
    Direct,
    /// 源为 [shape[1], shape[0]] 行主序,装载时 host 转置
    /// (Linear 惯例:权重 safetensors 原生 [out,in],装载 [in,out])
    Transposed,
}

/// 空包回填口(执行器写入;使用者零感知)
#[derive(Clone)]
pub struct Sink(SlotCell);

impl Sink {
    pub(crate) fn deliver(&self, b: Bytes) {
        self.0.deliver(b);
    }
}

/// 单条数据需求(纯元数据 + 回填口)
#[derive(Clone)]
pub struct Want {
    pub key: &'static str,
    pub dtype: Dtype,
    pub shape: Shape,
    pub layout: Layout,
    pub(crate) sink: Sink,
}

/// 需求清单(封闭有序;组合层 = 子层清单 chain 聚合)
#[derive(Clone, Default)]
pub struct LoaderOps {
    wants: Vec<Want>,
}

impl LoaderOps {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn want(w: Want) -> Self {
        Self { wants: vec![w] }
    }

    /// 聚合(组合层;清单顺序 = 装填顺序)
    pub fn chain(mut self, other: LoaderOps) -> Self {
        self.wants.extend(other.wants);
        self
    }

    pub fn wants(&self) -> &[Want] {
        &self.wants
    }
}

// ============================================================================
// §5 LoaderCtx:装载语境(layout 的 ctx;形状切分/dtype 决策的词汇)
// ============================================================================

/// 装载语境(layout 钩子的 ctx;按值传递,Copy)
/// 字段随域扩:并行切分 / dtype 决策 / 容量约束……起步只带两维。
#[derive(Clone, Copy, Debug)]
pub struct LoaderCtx {
    /// 目标 dtype(权重按语境声明需求;F32 起步)
    pub dtype: Dtype,
    /// 并行度(shard 形状切分的词汇;1 = 单卡;TP 随多卡立项)
    pub shard: usize,
}

impl Default for LoaderCtx {
    fn default() -> Self {
        Self { dtype: Dtype::F32, shard: 1 }
    }
}

// ============================================================================
// §6 Loadable:容器装载契约(唯一生命周期钩子 layout)
// ============================================================================

/// 可装载容器:装载生命周期只有一个钩子 ——
/// `layout(&self, ctx) -> LoaderOps`(声明每个空包的排布:同步 · 零 src ·
/// 零 Result)。执行(`eval_load`)与回填(sink)都是解释器/执行器的
/// 内部动作,使用者零感知。
pub trait Loadable {
    /// 装载生命周期:按语境导出数据需求清单(布局声明)
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps;
}

// ============================================================================
// §7 Weight:跨设备权重(空包指针 + 布局标注)
// ============================================================================

/// 跨设备权重:布局标注 + 空包指针。未装载时 decl() → 毒值声明
/// (forward 全程 total,零 panic 零 expect)。
///
/// 与 Tensor<D> 同构(2026-09-25 拍板:句柄落线 A)—— **容器只持句柄,
/// 存放归设备**:CpuFace 解析为 host 值块,GpuClient 解析为显存池块,
/// 层代码零设备感知。
pub struct Weight {
    key: &'static str,
    shape: Shape,
    transposed: bool,
    cell: SlotCell,
}

impl Weight {
    /// 登记权重(new 期;零副作用;Direct 布局)
    pub fn new(key: &'static str, shape: Shape) -> Self {
        Self { key, shape, transposed: false, cell: SlotCell::default() }
    }

    /// 转置权重:源 [rows, cols] 行主序 → 装载声明 [cols, rows]
    /// (Linear 惯例:forward 直 matmul,decode 零转置)
    pub fn new_transposed(key: &'static str, rows: usize, cols: usize) -> Self {
        Self { key, shape: vec![cols, rows], transposed: true, cell: SlotCell::default() }
    }

    /// 装载生命周期:按语境声明需求(元数据直出 + 回填口;零 src 零数据)
    pub fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        LoaderOps::want(Want {
            key: self.key,
            dtype: ctx.dtype,
            shape: self.shape.clone(),
            layout: if self.transposed { Layout::Transposed } else { Layout::Direct },
            sink: Sink(self.cell.clone()),
        })
    }

    /// 权重声明(未装载 → 毒值声明;of_block:eval 时 Block 叶子零操作)
    pub fn decl(&self) -> TensorOps {
        match self.cell.loaded() {
            Some(b) => TensorOps::of_block(b.id, Dtype::F32, self.shape.clone()),
            None => TensorOps::poisoned(
                Dtype::F32,
                self.shape.clone(),
                format!("Weight '{}': 未装载(eval_load 前禁止执行)", self.key),
            ),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.cell.loaded().is_some()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn key(&self) -> &'static str {
        self.key
    }
}

// ============================================================================
// §8 顶层装载函数(最底层基本件:单权重直通,完整自足)
// ============================================================================

/// 单权重装载(最底层基本函数,完整自足):给它权重容器 + 设备 + 数据源,
/// 它把权重物化进设备块并装进空包。取数 → 长度校验 → 转置 → htod →
/// 填空包,五步全在本函数 —— 不经 LoaderOps / eval_load(那是多层
/// 聚合路径)。
pub async fn load_weight<D: DeviceClient, S: WeightSource + ?Sized>(
    weight: &mut Weight,
    face: &mut D,
    src: &S,
) -> Result<(), ModelError> {
    let key = weight.key;

    // 1. 取数:容器内表没有(表走 layout 清单),数据槽直查源
    let data = src
        .get(key)
        .ok_or_else(|| ModelError::Msg(format!("Weight '{key}': 数据源缺键")))?;

    // 2. 长度校验 + 布局变换 → (装载形状, LE 字节)
    let n: usize = weight.shape.iter().product();
    if data.len() != n {
        return Err(ModelError::Msg(format!(
            "Weight '{key}': 元素 {} != shape {:?}({n})",
            data.len(),
            weight.shape
        )));
    }
    let (shape, bytes) = if weight.transposed {
        // 源 [shape[1], shape[0]] → 转置装载 [shape[0], shape[1]]
        let (cols, rows) = (weight.shape[0], weight.shape[1]);
        let mut t = vec![0.0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = data[r * cols + c];
            }
        }
        (weight.shape.clone(), f32b(&t))
    } else {
        (weight.shape.clone(), f32b(data))
    };

    // 3. 物化(解释层原语:htod → 块句柄)
    let b = face
        .htod(Dtype::F32, &shape, &bytes)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{key}': {e}")))?;

    // 4. 填空包
    weight.cell.deliver(b);
    Ok(())
}
