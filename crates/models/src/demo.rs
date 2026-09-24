//! 共享 demo 管线:一份 MLP 声明,CPU/GPU 两个执行器跑同一份代码。
//! models/examples/mlp.rs(CpuFace)与 owl-cuda/examples/mlp_server.rs
//! (GpuClient)都调这里——保证两路执行的是**同一份描述**。

use crate::client::{DeviceClient, KvCtx};
use crate::tensor::Dtype;
use crate::error::ModelError;
use crate::TensorOps;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// MLP 层(声明式;静态依赖构造期捕获)
pub struct Mlp {
    w_gate: TensorOps, // [hidden, intermediate]
    w_up: TensorOps,
    w_down: TensorOps, // [intermediate, hidden]
}

impl Mlp {
    /// 构造期:唯一保留 Result 的 layer 位置(装载可败)
    pub fn new(
        hidden: usize,
        intermediate: usize,
        gate_w: &[f32],
        up_w: &[f32],
        down_w: &[f32],
    ) -> Result<Self, ModelError> {
        if gate_w.len() != hidden * intermediate
            || up_w.len() != hidden * intermediate
            || down_w.len() != intermediate * hidden
        {
            return Err(ModelError::ShapeMismatch {
                path: "mlp".into(),
                expected: vec![hidden, intermediate],
                got: vec![gate_w.len()],
            });
        }
        Ok(Self {
            w_gate: TensorOps::from_host(Dtype::F32, vec![hidden, intermediate], &f32b(gate_w)),
            w_up: TensorOps::from_host(Dtype::F32, vec![hidden, intermediate], &f32b(up_w)),
            w_down: TensorOps::from_host(Dtype::F32, vec![intermediate, hidden], &f32b(down_w)),
        })
    }

    /// forward:同步 · total · 纯描述(零 ? 零 await 零 Tx)
    pub fn forward(&self, xs: &TensorOps, _kv: &KvCtx) -> TensorOps {
        let gate = xs.matmul(&self.w_gate);
        let up = xs.matmul(&self.w_up);
        let act = gate.silu().add(&up);
        let down = act.matmul(&self.w_down);
        down.rmsnorm(&alpha_ones(&down), 1e-6, false)
    }
}

fn alpha_ones(t: &TensorOps) -> TensorOps {
    let n = t.shape().iter().product::<usize>();
    TensorOps::from_host(Dtype::F32, t.shape().to_vec(), &f32b(&vec![1.0f32; n]))
}

/// 共享管线:声明 MLP → forward → reduce_gpu → f32 输出。
/// face 注入执行器(CpuFace = CPU;GpuClient 的 face = GPU server)。
/// 同一份描述,换 face 即换后端——这就是"相位/后端无感"的验收形态。
pub async fn mlp_pipeline<D: DeviceClient>(face: &mut D) -> Result<Vec<f32>, ModelError> {
    let (hidden, intermediate) = (4usize, 8usize);

    // 构造期(可败边界:装载)
    let mlp = Mlp::new(
        hidden,
        intermediate,
        &vec![0.1; hidden * intermediate],
        &vec![0.2; hidden * intermediate],
        &vec![0.3; intermediate * hidden],
    )?;

    // 描述期(纯;零 ? 零 await)
    let xvec = vec![0.5f32; hidden];
    let x = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&xvec));
    let kv = KvCtx { step: 7, slots: vec![0] };
    let logits = mlp.forward(&x, &kv);

    // 执行边界(GPU/CPU 解释器经 GpuFace 归约;全程数据留设备)
    let bytes = crate::client::eval(logits.step(), face).await?;
    let n: usize = logits.shape().iter().product();
    let mut out = vec![0u8; n * 4];
    face.dtoh(&bytes, &mut out).await?;
    Ok(out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
