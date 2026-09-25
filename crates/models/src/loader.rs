//! 权重跨设备定义 + LoaderOps(数据需求清单;与 TensorOps 对标)。
//!
//! ## 权重的跨设备统一定义(2026-09-25 拍板:句柄落线 A)
//!
//! 与 Tensor<D> 同构 —— **容器只持句柄,存放归设备**:
//! `Weight = 布局标注 + 空包指针(块句柄格)`;CpuFace 解析为 host 值块,
//! GpuClient 解析为显存池块,层代码零设备感知。
//!
//! ## 装载生命周期(唯一钩子)
//!
//! ```rust,ignore
//! let mut lin = Linear::new("w", 2, 3);  // new = 准备容器(空包指针,零数据)
//! let want = lin.layout();               // layout = 唯一钩子:声明空包的排布
//!                                        //  (字节数/几维形状/dtype/布局;
//!                                        //   同步 · 零 src · 零 Result)
//! eval_load(&want, &mut face, &src).await?; // 执行器:按清单从源取数 →
//!                                        //  物化 → **自动填进空包**
//! let out = lin.forward(&xs);            // forward = 计算声明(空包已满)
//! ```
//!
//! 回填被吸收进执行器(`eval_load` 经 Want.sink 写空包)——使用者只见
//! 一个钩子。缺键/长度不符属执行期数据错误,由 eval_load 结构化收割。

use crate::client::DeviceClient;
use crate::error::ModelError;
use crate::shape::Shape;
use crate::tensor::{Dtype, TensorOps};
use std::sync::{Arc, Mutex};

// ============================================================================
// 数据源:执行器的取数对象(测试 HashMap / 将来 safetensors / 层内表)
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
// 空包指针:块句柄格(装载产物写入处;decl 读取处)
// ============================================================================

/// 共享块句柄格(空包):装载前 None,执行器物化后 Some(块句柄)
#[derive(Clone, Default)]
pub struct SlotCell(Arc<Mutex<Option<crate::client::Bytes>>>);

impl SlotCell {
    fn deliver(&self, b: crate::client::Bytes) {
        *self.0.lock().unwrap() = Some(b);
    }

    fn loaded(&self) -> Option<crate::client::Bytes> {
        self.0.lock().unwrap().clone()
    }
}

// ============================================================================
// LoaderOps:数据需求清单(纯元数据;零数据;total)
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
    pub(crate) fn deliver(&self, b: crate::client::Bytes) {
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
// LoaderCtx:装载语境(layout 的 ctx;形状切分/dtype 决策的词汇)
// ============================================================================

/// 装载语境(layout 钩子的 ctx;按值传递,Copy)
/// 字段随域扩:并行切分 / dtype 决策 / 容量约束……起步只带两维。
#[derive(Clone, Copy, Debug)]
pub struct LoaderCtx {
    /// 目标 dtype(权重按语境声明需求;F32 起步)
    pub dtype: Dtype,
    /// 并行度(shard 形状切分的词汇;1 = 单卡)
    pub shard: usize,
}

impl Default for LoaderCtx {
    fn default() -> Self {
        Self { dtype: Dtype::F32, shard: 1 }
    }
}

// ============================================================================
// Loadable:容器装载契约(唯一生命周期钩子 layout)
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
// Weight:跨设备权重(空包指针 + 布局标注)
// ============================================================================

/// 跨设备权重:布局标注 + 空包指针。未装载时 decl() → 毒值声明
/// (forward 全程 total,零 panic 零 expect)。
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
// 顶层装载函数(最底层基本件:单权重直通,完整自足)
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
    use crate::error::ModelError;
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
