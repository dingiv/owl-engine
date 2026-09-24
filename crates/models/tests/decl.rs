//! 声明式模型的边界验收:毒值、形状违约、数值对拍。

use owl_models::tensor::Dtype;
use owl_models::error::ModelError;
use owl_models::interpreter::{reduce, CpuInterpreter};

use owl_models::TensorOps;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[test]
fn chain_matches_host_reference() {
    let x = TensorOps::from_host(Dtype::F32, vec![1, 4], &f32b(&[1.0, 2.0, -3.0, 4.0]));
    let out = x.silu();
    let got = reduce(out.step(), &mut CpuInterpreter::new()).unwrap();

    let want: Vec<f32> = [1.0f32, 2.0, -3.0, 4.0]
        .iter()
        .map(|v| v / (1.0 + (-v).exp()))
        .collect();
    assert!(got.f32.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-6));
}

#[test]
fn shape_mismatch_becomes_poison_not_panic() {
    let a = TensorOps::from_host(Dtype::F32, vec![4], &f32b(&[1.0; 4]));
    let b = TensorOps::from_host(Dtype::F32, vec![3], &f32b(&[1.0; 3]));

    // 描述期零失败(毒随链流)
    let chain = a.add(&b).silu();
    assert!(chain.is_poisoned());

    // 边界收割:结构化报错 + 案发 depth 回溯
    let err = reduce(chain.step(), &mut CpuInterpreter::new()).unwrap_err();
    assert!(matches!(err, ModelError::Msg(m) if m.contains("毒值落地")));
}

#[test]
fn poison_flows_through_downstream_ops() {
    let a = TensorOps::from_host(Dtype::F32, vec![4], &f32b(&[1.0; 4]));
    let bad = TensorOps::from_host(Dtype::F32, vec![3], &f32b(&[1.0; 3]));

    let mid = a.add(&bad);
    assert!(mid.is_poisoned());
    let out = mid.silu().add(&mid);
    assert!(out.is_poisoned(), "毒必须流过下游,不能中途消失");

    let err = reduce(out.step(), &mut CpuInterpreter::new()).unwrap_err();
    assert!(format!("{err:?}").contains("形状不符"));
}

#[test]
fn matmul_shapes() {
    let a = TensorOps::from_host(Dtype::F32, vec![1, 4], &f32b(&[1.0, 2.0, 3.0, 4.0]));
    let b = TensorOps::from_host(
        Dtype::F32,
        vec![4, 2],
        &f32b(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    );
    let c = a.matmul(&b);
    let got = reduce(c.step(), &mut CpuInterpreter::new()).unwrap();
    assert_eq!(got.f32, vec![4.0, 6.0]);
    assert_eq!(got.shape, vec![1, 2]);
}

// 反向树可重放:flatten 覆盖全部节点且深度单调不降
#[test]
fn flatten_covers_whole_tree() {
    let a = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[1.0, 2.0]));
    let s = a.silu();
    let sum = s.add(&s);
    // 值语义:重复消费 = 子树深拷贝(s 进 add 双亲 → silu 存在两份拷贝)
    // flatten 数到的是"拷贝后"的节点;结构去重留作解释器层可选 memo。
    let flat = sum.flatten();
    assert_eq!(flat.len(), 5);
    assert!(flat.windows(2).all(|w| w[0].depth() <= w[1].depth()));
    assert!(flat.iter().all(|t| !t.is_poisoned()));
}

// Step 孤儿引用(reduce 直接收 Step;不用 Client 也完整可测)
#[test]
fn zeros_leaf_reduces() {
    let z = TensorOps::zeros(Dtype::F32, vec![2, 3]);
    let got = reduce(z.step(), &mut CpuInterpreter::new()).unwrap();
    assert_eq!(got.f32, vec![0.0; 6]);
}
