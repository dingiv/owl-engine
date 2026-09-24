//! 运行时 Tensor:**真正的数据张量**(持有设备池块;对照声明链 TensorOps)。
//!
//! 三层区分(2026-09-23/24 定稿):
//!   TensorOps  = 声明链(懒描述;反向多叉树;无数据)
//!   Tensor<D>  = 运行时数据(本模块:设备池块 + dtype/shape 标注)
//!   EvalHandle = eval 的完成路由(采样/等待用)
//!
//! 跨设备统一表达:`Tensor<D: Device>` 对 CPU(`device::Cpu`)与
//! GPU server(owl-cuda 实现 `Device`)同一形状——上层代码零分叉。
//!
//! 与 nn/src/tensor.rs 的对照(那份是 CUDA 特化版):
//!   ptr 缓存/哨兵令牌/池块保活 → 收敛为 `block: D::Bytes` 单一保活
//!   (iface 字节化后池块即唯一载体);offset 支持视图(零拷贝)。

use crate::device::Device;
use crate::dtype::Dtype;
use crate::error::ModelError;

use crate::shape::{numel, Shape};
use crate::tensor::TensorOps;
use std::sync::atomic::{AtomicU64, Ordering};

/// 全局自增 id(与 TensorOps 共用一套进程级身份证;跨线程唯一)
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// 运行时数据张量。
/// - `block`:设备池块(数据所在);`offset`:块内字节偏移(视图);
/// - clone = 共享同一底仓 + 同一偏移(视图克隆,零拷贝);
/// - narrow/reshape 纯元数据;数据写入经设备(装载/算子),不经本类型。
pub struct Tensor<D: Device> {
    dev: D,
    block: D::Bytes,
    /// 块内字节偏移(视图;数据本体从 offset 起读)
    offset: usize,
    dtype: Dtype,
    shape: Shape,
    id: u64,
}

impl<D: Device> Tensor<D> {
    /// 从既有池块装配(纯元数据;容量守卫——cat 案教训)
    pub fn from_bytes(
        dev: D,
        dtype: Dtype,
        shape: Shape,
        block: D::Bytes,
    ) -> Result<Self, ModelError> {
        let n = numel(&shape);
        let need = n * dtype.size_bytes();
        let cap = dev.capacity(&block);
        if need > cap {
            return Err(ModelError::Msg(format!(
                "Tensor::from_bytes: 需 {need}B > 池块 {cap}B"
            )));
        }
        Ok(Self {
            dev,
            block,
            offset: 0,
            dtype,
            shape,
            id: next_id(),
        })
    }

    /// 装载声明 → 执行(host 数据经设备 htod;同步面)
    pub fn from_host(dev: D, dtype: Dtype, shape: Shape, data: &[u8]) -> Result<Self, ModelError> {
        let n = numel(&shape);
        let want = n * dtype.size_bytes();
        if data.len() != want {
            return Err(ModelError::Msg(format!(
                "from_host: 字节数 {} != {want}(dtype×shape)",
                data.len()
            )));
        }
        let block = dev.htod(data)?;
        Ok(Self {
            dev,
            block,
            offset: 0,
            dtype,
            shape,
            id: next_id(),
        })
    }

    /// 清零分配
    pub fn zeros(dev: D, dtype: Dtype, shape: Shape) -> Result<Self, ModelError> {
        let n = numel(&shape);
        let block = dev.alloc(n * dtype.size_bytes())?;
        Ok(Self {
            dev,
            block,
            offset: 0,
            dtype,
            shape,
            id: next_id(),
        })
    }

    // ======================================================================
    // 查询
    // ======================================================================

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn nbytes(&self) -> usize {
        numel(&self.shape) * self.dtype.size_bytes()
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn device(&self) -> D {
        self.dev.clone()
    }

    /// 视图:dim 维收窄(零拷贝;offset 随 start 走)
    /// (仅演示最简形态:dim=0 且行主序时 offset += start × 行宽)
    pub fn narrow_dim0(&self, start: usize, len: usize) -> Result<Self, ModelError> {
        if self.shape.is_empty() || start + len > self.shape[0] {
            return Err(ModelError::Msg(format!(
                "narrow_dim0: [{start}..{}) 越界 shape {:?}",
                start,
                self.shape
            )));
        }
        let row = if self.shape.len() > 1 {
            self.shape[1..].iter().product::<usize>() * self.dtype.size_bytes()
        } else {
            self.dtype.size_bytes()
        };
        let t = Tensor {
            dev: self.dev.clone(),
            block: self.block.clone(),
            offset: self.offset + start * row,
            dtype: self.dtype,
            shape: {
                let mut s = self.shape.clone();
                s[0] = len;
                s
            },
            id: next_id(),
        };
        Ok(t)
    }

    // ======================================================================
    // 数据出入(边界;同步面。GPU 的 async 包装在 client.rs 一层)
    // ======================================================================

    /// device → host 收割(全部数据)
    /// 桥:物化数据 → 声明叶子(Block 节点引用本块的 id)。
    /// 数据本体不动;声明图 eval 时该叶子零操作。
    pub fn as_declaration(&self) -> TensorOps {
        TensorOps::of_block(
            self.id,
            self.dtype,
            self.shape.clone(),
        )
    }

    pub fn to_host(&self) -> Result<Vec<u8>, ModelError> {
        let mut out = vec![0u8; self.nbytes()];
        // 视图收割:块内 offset 起(容量守卫在设备层)
        self.dev.dtoh(&self.block, &mut out)?;
        // offset 版本:设备需支持子区读——一期约束 = narrow 视图暂不支持
        // 跨 offset 收割(需要 server dtoh 带 offset 参数,A3 期扩)。
        if self.offset != 0 {
            return Err(ModelError::Msg(
                format!("to_host: 带偏移视图收割待 server dtoh 扩 offset 参数(offset={})", self.offset),
            ));
        }
        Ok(out)
    }
}
