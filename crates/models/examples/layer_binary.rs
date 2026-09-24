//! 双输入 layer 样例:具名多输入 forward + rt::Tensor 经 Block 叶子入声明图。

use owl_models::device::Cpu;
use owl_models::dtype::Dtype;
use owl_models::interpreter::{reduce, CpuInterpreter};
use owl_models::rt::Tensor;
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
    // ---- 物化数据(rt::Tensor;真数据,池块在"设备"上)----
    let a_data = Tensor::from_host(
        Cpu,
        Dtype::F32,
        vec![1, 4],
        &f32b(&[1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    let b_data = Tensor::from_host(
        Cpu,
        Dtype::F32,
        vec![1, 4],
        &f32b(&[10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();

    // ---- 桥:物化数据 → 声明叶子(Block 节点;id 同一身份证)----
    let a = a_data.as_declaration();
    let b = b_data.as_declaration();
    println!("a.id = {}, b.id = {}", a.id(), b.id());

    // ---- layer:静态依赖(权重)+ 具名双输入 forward ----
    let layer = FuseAdd::new(&[1.0; 16]);
    let out = layer.forward(&a, &b);

    // ---- 执行:CPU 解释器;Block 叶子按 id 解析(登记表)----
    let mut itp = CpuInterpreter::new();
    itp.bind(a_data.id(), owl_models::interpreter::Value::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]));
    itp.bind(b_data.id(), owl_models::interpreter::Value::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]));
    let got = reduce(out.step(), &mut itp).expect("归约");

    // host 参考:(a+b) = [11,22,33,44];× 单位阵 = 不变
    assert_eq!(got.f32, vec![110.0; 4]); // w 全 1 → out = s 的行和
    println!("out = {:?}\n双输入 layer + Block 叶子端到端 ✓", got.f32);
}

