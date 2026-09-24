//! MLP 层样例:声明式 layer 形态的可执行验收(design §三)。
//!
//! 注意 forward 签名:**没有 Tx、没有 Result、没有 async**——
//! `forward(&self, xs, kv) -> TensorOps`,与 xinfer 骨架同构,纯函数。

use owl_models::client::KvCtx;
use owl_models::dtype::Dtype;
use owl_models::error::ModelError;
use owl_models::interpreter::{reduce, CpuInterpreter};

use owl_models::TensorOps;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// MLP(xinfer 同骨架;声明式形态)
// ============================================================================

pub struct Mlp {
    w_gate: TensorOps, // [hidden, intermediate](行向量约定)
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
            w_gate: TensorOps::from_host(
                Dtype::F32,
                vec![hidden, intermediate],
                &f32b(gate_w),
            ),
            w_up: TensorOps::from_host(Dtype::F32, vec![hidden, intermediate], &f32b(up_w)),
            w_down: TensorOps::from_host(
                Dtype::F32,
                vec![intermediate, hidden],
                &f32b(down_w),
            ),
        })
    }

    /// forward:同步 · total · 纯描述。
    /// 静态依赖在 self;动态依赖(kv)走参数;返回惰性声明。
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

// ============================================================================
// 验收
// ============================================================================

fn main() {
    let (hidden, intermediate) = (4usize, 8usize);

    // 构造(可败边界)
    let mlp = Mlp::new(
        hidden,
        intermediate,
        &vec![0.1; hidden * intermediate],
        &vec![0.2; hidden * intermediate],
        &vec![0.3; intermediate * hidden],
    )
    .expect("装载");

    // forward(纯描述:零 ? 零 await 零 Tx)
    let xvec = vec![0.5f32; hidden];
    let x = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&xvec));
    let kv = KvCtx { step: 7, slots: vec![0] };
    let logits = mlp.forward(&x, &kv);

    // 执行边界:CPU 参考解释器归约(生产 = GPU 解释器,同签名)
    let mut itp = CpuInterpreter::new();
    let got = reduce(logits.step(), &mut itp).expect("归约");
    println!("forward 输出 = {:?}", got.f32);

    // 重复消费:值语义 = 重复计算(配置面成本);声明树可重放
    let again = reduce(logits.step(), &mut itp).unwrap();
    assert_eq!(got.f32, again.f32);
    println!("demo: 声明式 MLP 端到端 ✓(无 Tx;无 Result;无 async)");
}
