//! 运行时 Tensor(CPU 设备)端到端:装载/收割/视图/唯一 id。

use owl_models::device::Cpu;
use owl_models::{Dtype, Tensor};

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() {
    let cpu = Cpu;

    // 装载(数据上"设备")
    let t = Tensor::from_host(cpu.clone(), Dtype::F32, vec![1, 4], &f32b(&[1.0, 2.0, -3.0, 4.0]))
        .expect("from_host");
    println!("t: id={} shape={:?} dtype={:?}", t.id(), t.shape(), t.dtype());

    // 收割(数据回 host;字节→f32 由调用方按 dtype 解释)
    let back = t.to_host().unwrap();
    let vals: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(vals, vec![1.0, 2.0, -3.0, 4.0]);
    println!("to_host 一致 ✓ {vals:?}");

    // 视图(零拷贝;dim0 收窄)
    let row = t.narrow_dim0(0, 1).unwrap();
    let row_v = row.to_host().unwrap();
    assert_eq!(row_v.len(), 16);
    println!("narrow 视图 ✓({}B)", row_v.len());

    // 清零分配
    let z = Tensor::zeros(cpu, Dtype::F32, vec![2, 2]).unwrap();
    assert_eq!(z.nbytes(), 16);
    println!("zeros ✓;跨设备统一表达:同一路径,CPU/GPU 换 backend 即可");
}
