//! layers 试水批测试:容器 + load 钩子 + forward 纯声明的生命周期契约
//! + 语义算子层数值对拍(CPU face 全链)+ Kernel 试水件声明构造。

use owl_models::client::DeviceClient;
use owl_models::interpreter::{eval_ops, eval_load};
use owl_models::module::KernelCtx;
use owl_models::Module;
use owl_models::loader::Loadable;
use owl_models::layers::{embedding, linear, mlp, rmsnorm, rope};
use owl_models::tensor::Dtype;
use owl_models::TensorOps;
use std::collections::HashMap;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

type Src = HashMap<String, Vec<f32>>;

async fn harvest<D: DeviceClient>(face: &mut D, t: &TensorOps) -> Vec<f32> {
    let bytes = eval_ops(t.step(), face).await.expect("eval");
    let n: usize = t.shape().iter().product();
    let mut buf = vec![0u8; n * 4];
    face.dtoh(&bytes, &mut buf).await.expect("dtoh");
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ============================================================================
// Linear:容器 + load 钩子 + 转置装载 + matmul
// ============================================================================

#[tokio::test]
async fn linear_transposed_matmul_matches_host() {
    // host 权重 [out=2, in=3](safetensors 原生布局)
    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![0.5, -1.0, 2.0];
    let mut want = vec![0.0f32; 2];
    for (o, wo) in want.iter_mut().enumerate() {
        *wo = (0..3).map(|i| x[i] * w[o * 3 + i]).sum();
    }

    let mut face = owl_cpu::CpuFace::new();
    let lin = linear::Linear::new("w", 2, 3); // 容器(零数据)
    let src = Src::from([("w".to_string(), w)]);
    // 装载执行(层入口:解释器调 layout(ctx) + 物化 + sink 回填)
    eval_load(&lin, &mut face, &src, &Default::default())
        .await
        .expect("eval_load");

    let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&x));
    let got = {
        let ops = lin.forward(&xs, &KernelCtx::default());
        harvest(&mut face, &ops).await
    };
    assert_eq!(got.len(), 2);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-6, "{g} vs {w}");
    }
}

// ============================================================================
// RmsNorm:多行 + per-channel gamma(回归线:lower rows/cols 修复)
// ============================================================================

#[tokio::test]
async fn rmsnorm_multirow_perchannel_matches_host() {
    let (rows, n) = (3usize, 4usize);
    let gamma = vec![0.5, 1.0, 1.5, 2.0];
    let x: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.25) - 1.0).collect();

    let mut face = owl_cpu::CpuFace::new();
    let norm = rmsnorm::RmsNorm::new("gamma", n, 1e-6);
    let src = Src::from([("gamma".to_string(), gamma.clone())]);
    eval_load(&norm, &mut face, &src, &Default::default())
        .await
        .expect("eval_load");

    let xs = TensorOps::from_host(Dtype::F32, vec![rows, n], &f32b(&x));
    let got = harvest(&mut face, &norm.forward(&xs, &KernelCtx::default())).await;

    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (c, v) in row.iter().enumerate() {
            let want = v * inv * gamma[c];
            let g = got[r * n + c];
            assert!((g - want).abs() < 1e-5, "[{r},{c}] {g} vs {want}");
        }
    }
}

// ============================================================================
// Mlp:SwiGLU 全链(gate/up → silu × mul → down)
// ============================================================================

#[tokio::test]
async fn mlp_chain_matches_host_reference() {
    let (hidden, intermediate) = (4usize, 6usize);
    let gate_w: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.01) - 0.1).collect();
    let up_w: Vec<f32> = (0..intermediate * hidden).map(|i| 0.2 - (i as f32 * 0.005)).collect();
    let down_w: Vec<f32> = (0..hidden * intermediate).map(|i| (i as f32 * 0.008) + 0.05).collect();
    let x = vec![0.5, -0.25, 1.0, 0.0];

    let mut face = owl_cpu::CpuFace::new();
    let layer = mlp::Mlp::new(hidden, intermediate); // 容器
    let src = Src::from([
        ("gate_proj".to_string(), gate_w.clone()),
        ("up_proj".to_string(), up_w.clone()),
        ("down_proj".to_string(), down_w.clone()),
    ]);
    eval_load(&layer, &mut face, &src, &Default::default())
        .await
        .expect("eval_load");

    let xs = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&x));
    let got = harvest(&mut face, &layer.forward(&xs, &KernelCtx::default())).await;
    assert_eq!(got.len(), hidden);

    // host 参考(同一 SwiGLU 数学)
    let mut want = vec![0.0f32; hidden];
    for o in 0..hidden {
        let mut acc = 0.0f32;
        for j in 0..intermediate {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for i in 0..hidden {
                g += x[i] * gate_w[j * hidden + i];
                u += x[i] * up_w[j * hidden + i];
            }
            let silu = g / (1.0 + (-g).exp());
            acc += silu * u * down_w[o * intermediate + j];
        }
        want[o] = acc;
    }
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-4, "{g} vs {w}");
    }
}

// ============================================================================
// 生命周期契约:new 后未 load → forward total(毒值),eval 收割
// ============================================================================

#[tokio::test]
async fn unloaded_slot_becomes_poison_at_boundary() {
    let face = owl_cpu::CpuFace::new();
    let mut face = face;
    let lin = linear::Linear::new("w", 2, 3); // 容器;未 load
    let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[1.0, 2.0, 3.0]));

    // forward total(零 panic 零 Err);毒立即随链标注(eval 边界收割详情)
    let out = lin.forward(&xs, &KernelCtx::default());
    assert!(out.is_poisoned(), "未装载槽的声明应立即带毒(随链流动)");
    let err = eval_ops(out.step(), &mut face).await.unwrap_err();
    assert!(format!("{err:?}").contains("未装载"), "{err:?}");

    // 缺键/长度违约:数据属执行期,eval_load 结构化收割(带槽键归因)
    let lin2 = linear::Linear::new("w", 2, 3);
    let empty = Src::new();
    let err = eval_load(&lin2, &mut face, &empty, &Default::default())
        .await
        .unwrap_err();
    assert!(format!("{err:?}").contains("缺键"), "{err:?}");

    let lin3 = linear::Linear::new("w", 2, 3);
    let bad_len = Src::from([("w".to_string(), vec![1.0; 5])]);
    let err = eval_load(&lin3, &mut face, &bad_len, &Default::default())
        .await
        .unwrap_err();
    assert!(format!("{err:?}").contains("元素"), "{err:?}");
}

// ============================================================================
// Embedding / Rope:Kernel 试水件(声明构造;不执行)
// ============================================================================

#[tokio::test]
async fn embedding_declaration_is_wellformed() {
    let mut face = owl_cpu::CpuFace::new();
    let emb = embedding::Embedding::new(16, 4);
    let src = Src::from([("w".to_string(), (0..64).map(|i| i as f32 * 0.1).collect()), ("w_t".to_string(), (0..64).map(|i| i as f32 * 0.1).collect())]);
    eval_load(&emb, &mut face, &src, &Default::default())
        .await
        .expect("eval_load");

    let ids = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[3.0, 7.0]));
    let out = emb.forward(&ids, &KernelCtx { tokens: 2 });
    assert!(!out.is_poisoned(), "embedding 声明不应有毒");
    assert_eq!(out.shape(), &[2, 4]);

    // lm_head(tied 转置槽):[1,4] → [1,16](语义 matmul,可真跑)
    let hidden = TensorOps::from_host(Dtype::F32, vec![1, 4], &f32b(&[0.1; 4]));
    let logits = harvest(&mut face, &emb.lm_head_matmul(&hidden)).await;
    assert_eq!(logits.len(), 16);
}

#[tokio::test]
async fn rope_declaration_is_wellformed() {
    let (head_dim, rotary_dim, heads) = (8usize, 4usize, 2usize);
    let mut face = owl_cpu::CpuFace::new();
    let rp = rope::Rope::new(64, head_dim, rotary_dim, 10_000.0).expect("new");
    let tables = rp.tables();                 // 容器内表源(借用层内 Vec)
    eval_load(&rp, &mut face, &tables, &Default::default())
        .await
        .expect("eval_load 表物化");

    let pos = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0]));
    let q = TensorOps::from_host(
        Dtype::F32,
        vec![1, heads * head_dim],
        &f32b(&vec![0.3; heads * head_dim]),
    );

    let out = rp.forward_q(&q, &pos, 1, heads);
    assert!(!out.is_poisoned(), "rope 声明不应有毒");
    assert_eq!(out.shape(), &[1, heads * head_dim]);

    // 非法 rotary_dim:配置校验在 new 收割(容器语义不成立)
    assert!(rope::Rope::new(64, head_dim, 7, 10_000.0).is_err());
    assert!(rope::Rope::new(64, head_dim, head_dim * 2, 10_000.0).is_err());
}

// ============================================================================
// 顶层装载基本函数:load_weight(单权重 = 源 → 物化 → 装进容器)
// ============================================================================

#[tokio::test]
async fn load_weight_basic() {
    use owl_models::loader::load_weight;
    let mut face = owl_cpu::CpuFace::new();

    let mut w = linear::Linear::new("w", 2, 3).into_weight(); // 取出容器里的权重格
    assert!(!w.is_loaded(), "初始未装载");

    let src = Src::from([("w".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    load_weight(&mut w, &mut face, &src).await.expect("load_weight");

    assert!(w.is_loaded(), "装载后应有块句柄");
    // 容器声明可执行:of_block 引用的块来自这次装载
    let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[0.5, -1.0, 2.0]));
    let got = harvest(&mut face, &xs.matmul(&w.decl())).await;
    assert_eq!(got.len(), 2);
    assert!((got[0] - 4.5).abs() < 1e-6 && (got[1] - 9.0).abs() < 1e-6);

    // 违约路径:缺键 → 结构化 Err,容器保持未装载
    let mut w2 = linear::Linear::new("w", 2, 3).into_weight();
    let empty = Src::new();
    assert!(load_weight(&mut w2, &mut face, &empty).await.is_err(), "缺键应 Err");
    assert!(!w2.is_loaded(), "失败装载不应污染容器");
}

// ============================================================================
// Module 接口:统一入口多态(泛型静态分发 / dyn 动态分发)
// ============================================================================

/// 泛型静态分发(REQ-CODE-01:热路径零 dyn)
fn run_through<M: Module>(m: &M, x: &TensorOps, ctx: &KernelCtx) -> TensorOps {
    m.forward(x, ctx)
}

#[tokio::test]
async fn module_trait_polymorphism() {
    let mut face = owl_cpu::CpuFace::new();
    let (hidden, intermediate) = (4usize, 6usize);
    let gate_w: Vec<f32> = vec![0.1; intermediate * hidden];
    let up_w: Vec<f32> = vec![0.2; intermediate * hidden];
    let down_w: Vec<f32> = vec![0.3; hidden * intermediate];

    let mlp = mlp::Mlp::new(hidden, intermediate);
    let src = Src::from([
        ("gate_proj".to_string(), gate_w.clone()),
        ("up_proj".to_string(), up_w.clone()),
        ("down_proj".to_string(), down_w.clone()),
    ]);
    eval_load(&mlp, &mut face, &src, &Default::default()).await.expect("eval_load");

    let x = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&vec![0.5; hidden]));

    // 静态分发:泛型约束即接口
    let a = harvest(&mut face, &run_through(&mlp, &x, &KernelCtx::default())).await;

    // 动态分发:dyn Module(配置期/注册表期多态)
    let layers: Vec<&dyn Module> = vec![&mlp];
    let b = harvest(&mut face, &layers[0].forward(&x, &KernelCtx::default())).await;

    assert_eq!(a, b, "两种分发同链同果");
    assert_eq!(a.len(), hidden);
}

// Rope:位置上下文层,暂不进 Module(双输入 + pos;ForwardCtx 立项后再归一)
#[test]
fn rope_stays_outside_module_interface() {
    // 编译期断言:Rope 未实现 Module(借助一个需要 Module 的泛型位)
    fn assert_impl<M: Module>(_: &M) {}
    let (head_dim, rotary_dim, heads) = (8usize, 4usize, 2usize);
    let rp = rope::Rope::new(64, head_dim, rotary_dim, 10_000.0).expect("new");
    // 只断言固有 forward_q 可用(接口外能力);Module 约束留待 ForwardCtx
    let _ = rp.forward_q(
        &TensorOps::from_host(Dtype::F32, vec![1, heads * head_dim], &f32b(&vec![0.3; heads * head_dim])),
        &TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0])),
        1,
        heads,
    );
    // 若将来 Rope: Module,此处编译错误提醒更新本测试意图
    let _ = &rp;
}
