//! CpuFace:CPU 后端句柄 —— `DeviceClient` 契约的进程内同步实现。
//!
//! 客户端直接持有(`CpuFace::new()` 即后端本体),无管道无 actor:
//! 每个契约方法 = 同步执行 + 直接返回(GPU 版对应物 = GpuClient 提交
//! 进 server 队列,Future pending 至回执)。
//!
//! 与 owl-cuda 的语义对照(契约同源 owl-iface::contract):
//! - `alloc/htod/dtoh/launch/sync`:同一语义,CPU 上"完成"即"返回";
//! - `graph_begin/end/launch`:结构化拒绝 —— 图捕获是异构执行域的
//!   重放机制,CPU 后端重放 = 重算,无捕获价值(与旧 CpuFace 一致)。

use crate::ops;
use crate::value::Value;
use owl_iface::contract::{Arg, Bytes, DeviceClient, Dtype, GraphId, LaunchMsg, ModelError};
use owl_iface::contract::Shape;
use std::collections::HashMap;

/// CPU 后端(与 GPU server 同一契约;测试/对拍/CPU 推理路径)
pub struct CpuFace {
    blocks: HashMap<u64, Value>,
    next: u64,
}

impl Default for CpuFace {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFace {
    pub fn new() -> Self {
        Self { blocks: HashMap::new(), next: 1 }
    }
}

impl DeviceClient for CpuFace {
    async fn graph_begin(&mut self) -> Result<(), ModelError> {
        Err(ModelError::Msg("CpuFace: 图捕获仅 GPU server 支持".to_string()))
    }

    async fn graph_end(&mut self) -> Result<GraphId, ModelError> {
        Err(ModelError::Msg("CpuFace: 图捕获仅 GPU server 支持".to_string()))
    }

    async fn graph_launch(&mut self, _graph: GraphId) -> Result<(), ModelError> {
        Err(ModelError::Msg("CpuFace: 图捕获仅 GPU server 支持".to_string()))
    }

    /// 显存分配(清零;CPU = Vec;无流参数 —— 同步直调即序)
    async fn alloc(&mut self, n_bytes: usize) -> Result<Bytes, ModelError> {
        let v = ops::zeros(Dtype::F32, &vec![n_bytes / 4])?;
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, v);
        Ok(Bytes::new(id, 0))
    }

    /// 装载(host 字节 → f32 值块)
    async fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
    ) -> Result<Bytes, ModelError> {
        let v = ops::htod(dtype, shape, src)?;
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, v);
        Ok(Bytes::new(id, 0))
    }

    /// 收割(值 → LE 字节;长度须与块一致)
    async fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError> {
        let v = self
            .blocks
            .get(&b.id)
            .ok_or_else(|| ModelError::DeadBlock { id: b.id })?;
        let bytes: Vec<u8> = v
            .f32
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        if bytes.len() != out.len() {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {} 字节 {} != 收割 {}",
                b.id,
                bytes.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&bytes);
        Ok(())
    }

    /// 发射:按 kernel 名路由到朴素算子(动作表的 CPU 面;
    /// 同步执行 —— "提交即完成",无 sticky 延迟)
    async fn launch(&mut self, msg: LaunchMsg) -> Result<Bytes, ModelError> {
        // 参数槽解析:Block → 值块;类型化标量
        let mut vals: Vec<Value> = Vec::with_capacity(msg.args.len());
        let mut u64s: Vec<u64> = Vec::new();
        let mut i32s: Vec<i32> = Vec::new();
        let mut f32s: Vec<f32> = Vec::new();
        let mut out_id: Option<u64> = None;
        for a in &msg.args {
            match a {
                Arg::Block { id } => {
                    out_id = Some(*id);
                    vals.push(
                        self.blocks
                            .get(id)
                            .cloned()
                            .ok_or(ModelError::DeadBlock { id: *id })?,
                    )
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
                // 标量与 lower_matmul 对位:m/k/n
                let (m, _k, n) = (i32s[0] as usize, i32s[1] as usize, i32s[2] as usize);
                let shape: Shape = vec![m, n];
                ops::matmul(&vals[0], &vals[1], &shape)?
            }
            "owl_matmul_nt_f32" => {
                // nt:B [n,k] 直读;标量与 lower_matmul_nt 对位:m/k/n
                let (m, _k, n) = (i32s[0] as usize, i32s[1] as usize, i32s[2] as usize);
                let shape: Shape = vec![m, n];
                ops::matmul_nt(&vals[0], &vals[1], &shape)?
            }
            "owl_rmsnorm_f32" => {
                // 标量与 lower_rmsnorm 对位:cols(I32)/eps(F32)/w_off(I32)
                ops::rmsnorm(&vals[0], &vals[1], i32s[0] as usize, f32s[0], i32s[1] != 0)?
            }
            other => {
                return Err(ModelError::Msg(format!(
                    "CpuFace::launch: 未注册的动作 {other}"
                )))
            }
        };
        // 写回 eval 预 alloc 的 out 块(与 GPU 语义一致:发射原位写输出)
        let id = out_id
            .ok_or_else(|| ModelError::Msg("launch: args 中无输出块".to_string()))?;
        self.blocks.insert(id, out);
        Ok(Bytes::new(id, msg.out_elems))
    }

    async fn upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: &Bytes,
        offset_elems: usize,
    ) -> Result<(), ModelError> {
        let v = self
            .blocks
            .get_mut(&dst.id)
            .ok_or(ModelError::DeadBlock { id: dst.id })?;
        let data = buf.as_f32();
        let end = offset_elems + data.len();
        if v.f32.len() < end {
            return Err(ModelError::Msg(format!(
                "upload_pinned: 块 {} 长 {} < 写入终点 {end}",
                dst.id,
                v.f32.len()
            )));
        }
        v.f32[offset_elems..end].copy_from_slice(data);
        Ok(())
    }

    async fn write_block_f32(
        &mut self,
        dst: &Bytes,
        offset_elems: usize,
        data: &[f32],
    ) -> Result<(), ModelError> {
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

    async fn sync(&mut self) -> Result<(), ModelError> {
        Ok(()) // CPU:无在飞操作(同步直调,天然无积压)
    }
}
