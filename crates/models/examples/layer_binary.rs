//! 双输入 layer 样例:具名多输入 forward + Block 叶子经 bind 入同步参考归约。
//!
//! 2026-09-26 收束迁影:原 rt::Tensor/Cpu 物化改 `TensorOps::of_block`
//! 直声明(Block 叶子按 id 反查;设备唯一标准 = iface DeviceClient)。

use owl_models::reference::{reduce, CpuInterpreter, Value};
use owl_models::tensor::Dtype;
use owl_models::TensorOps;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// 双输入层:out = (a + b) × w(行向量约定;w = [4,4])
pub struct FuseAdd {
    w: TensorOps,
}

impl FuseAdd {
    /// 静态依赖(权重)构造期捕获
    pub fn new(w: &[f32]) -> Self {
        Self { w: TensorOps::from_host(Dtype::F32, vec![4, 4], &f32b(w)) }
    }

    /// 具名双输入 forward:纯描述(total;零 ? 零 await)
    pub fn forward(&self, a: &TensorOps, b: &TensorOps) -> TensorOps {
        let s = a.add(b);
        s.matmul(&self.w)
    }
}

fn main() {
    // ---- 外部数据 → Block 叶子(id = 同步解释器 bind 的对账键)----
    let a = TensorOps::of_block(7001, Dtype::F32, vec![1, 4]);
    let b = TensorOps::of_block(7002, Dtype::F32, vec![1, 4]);
    println!("a.id = {}, b.id = {}", a.id(), b.id());

    // ---- layer:静态依赖(权重)+ 具名双输入 forward ----
    let layer = FuseAdd::new(&[1.0; 16]);
    let out = layer.forward(&a, &b);

    // ---- 执行:CPU 参考解释器;Block 叶子按 id 解析(bind 登记表)----
    let mut itp = CpuInterpreter::new();
    itp.bind(7001, Value::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]));
    itp.bind(7002, Value::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]));
    let got = reduce(out.step(), &mut itp).expect("归约");

    // host 参考:(a+b) = [11,22,33,44];× 全 1 阵 = 行和
    assert_eq!(got.f32, vec![110.0; 4]); // w 全 1 → out = s 的行和
    println!("out = {:?}\n双输入 layer + Block 叶子端到端 ✓", got.f32);
}
