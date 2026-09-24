//! ⚠️ 愿景伪代码(2026-09-23):**站在 Tensor 的角度**,想象它需要的 GPU API。
//! 本文件不编译、不实现——是写给 server 侧的"需求提案":这里出现的每一个
//! 调用,都是 asyncrt/protocol/server 要接的活。
//!
//! 视角说明(用户指令):owl-nn 的核心抽象是 Tensor(合并形态:
//! `Tensor<D>` = 一个池块句柄 + dtype/shape 标注)。nn 的算子层(OpsCtx/
//! erased/kernel)是 Tensor 的**最大客户**。这个客户希望 server 长什么样?
//!
//! 四条愿望(下文代码逐条落地):
//!   W1. 工厂一步到位:dtype+shape 进,Tensor 出——server 只见字节,
//!       dtype/shape 是我(client 侧)的标注,你永远别问我;
//!   W2. host 搬运吸收:from_host/to_host 我只给 &[T]/拿 Vec<T>,
//!       pinned 码头、流序、legacy 规避,全是 server 的事(契约 3 归你);
//!   W3. 指针活性你兜底:我发 launch 时直接给 &Tensor——取指针、
//!       保活登记(哨兵①/租约)一条龙,我不手工抱句柄;
//!   W4. 相位无感:同一份 forward 代码,eager 直发、捕获窗内录图,
//!       我的代码里不出现 if capturing(会话面承担,seam 核心原则)。
//!
//! ⚠️ 本文件假设的 `TensorClient` / `TensorRef` / `CaptureTx` 是**提案中的
//! server 门面扩展**(或 nn 侧薄壳——归属待拍板,见文末开放问题),
//! 现有 client.rs 尚无这些方法。

#![allow(unused_imports, dead_code, unused_variables)]

use owl_cuda::examples_note::*; // 伪代码:占位,无实义

fn main() {
    // 伪代码总纲:整个文件在描述一次"理想形状"的 decode 步进。
}

// ============================================================================
// §0 世界观:Tensor 手里有什么
// ============================================================================
//
// 合并后的 Tensor(client 侧,owl-nn 所有):
//
//   struct Tensor<D> {
//       block:  D::Bytes,   // server 发的池块句柄(存储底座;drop = 延迟归还)
//       dtype:  Dtype,      // 运行时标注(S4;server 不感知)
//       shape:  Vec<usize>, // 运行时标注
//   }
//
// server 发的是字节块;dtype/shape/视图元数据全在 client 侧。
// —— 这就是"池去类型裁决"(2026-09-24)落到 API 上的样子。

// ============================================================================
// §1 W1+W2:工厂与搬运(Tensor 的出生与还乡)
// ============================================================================

async fn birth_and_death(c: &TensorClient) {
    let m: usize = 512;
    let k: usize = 512;

    // --- 出生一:zeros(dtype, shape)。server 一次 malloc;我包一层标注。 ---
    // server 视角:Malloc { pool, bytes = m*k*4 }  → Block
    let a = Tensor::zeros(c, Dtype::F32, &[m, k]).await.unwrap();

    // --- 出生二:host 数据直进(权重装载的主路径)---
    // 我只给 &[f32];server 内部:pinned 码头直填 → 池流 htod async+sync。
    // (契约 3 的 pinned 码头由 server 持有复用——我不想知道码头存在。)
    // server 视角:内部 pinned staging + Htod → Block
    let w_src: Vec<f32> = load_weights("layer0.weight"); // 场景:文件/网络来的
    let w = Tensor::from_host(c, &[k, k], &w_src).await.unwrap();

    // --- 出生三:批量装载(M-Ⅰ 真实形状:几十个权重一次进)---
    // 愿望:一次往返装一批,不是几十个 await 排队。
    // server 视角:BatchedHtod { items } → Vec<Block>(顺序一一对应)
    let batch = Tensor::from_host_batch(
        c,
        &[
            (&[k, k], &w_src[..]),
            (&[k, m], &w_src[..]),
            // ...
        ],
    )
    .await
    .unwrap();

    // --- 还乡:to_host。我拿 Vec<T>;内部 D2H + 等待(server 完成 ready)。---
    let back: Vec<f32> = a.to_host().await.unwrap();

    let _ = (a, w, batch, back);
    // 死亡:Tensor drop → BlockRef drop → server 延迟归还(A1.2)。
    // 我不需要 drop 命令;非 Idle 相的归还延迟是 server 的事。
}

// ============================================================================
// §2 W3:发射面(Tensor 直接当参数;活性你兜底)
// ============================================================================

async fn launch_by_tensors(c: &TensorClient, k: &LoadedKernel) {
    let x = Tensor::zeros(c, Dtype::F32, &[1024]).await.unwrap();
    let mut y = Tensor::zeros(c, Dtype::F32, &[1024]).await.unwrap();

    // --- 愿望:参数给 &Tensor / &mut Tensor,不给裸指针 ---
    // server/client 契约面做的事(我一行都不写):
    //   1. 取 device_ptr(带偏移 = 视图安全;P0-5 的收口就在这);
    //   2. 哨兵①登记:读集合(x)/写集合(y)→ 当前作用域;
    //   3. 保活:块的 Arc 克隆进本次发射的记录(发射在飞,块不许死);
    //   4. u64 打包进协议(Launch 参数面不变——协议无类型,正确)。
    // client.launch(k, launch_args![&x, &mut y], grid(8,1,1), block(128,1,1)).await?;

    // --- dtype/shape 断言在 client 侧:错误不过线 ---
    // server 是字节世界;S1/S2 违约(dtype 不匹配/形状不一致)在我这里拦死,
    // 到 server 的永远是合法的字节请求。(server 报的错只可能是:
    // 池耗尽/相位违约/令牌死亡——治理错,不是数据错。)
}

// ============================================================================
// §3 W4:相位无感(一份 forward,eager/捕获两相通用)
// ============================================================================

/// 模型的一层(forward 片段)。注意签名:**没有相位参数**。
/// `tx` 是"发射上下文"——eager 态它直发,捕获态它录图;我不关心。
fn block_forward(tx: &Tx, w: &Tensor, x: &Tensor, out: &mut Tensor) -> Result<(), BackendError> {
    // 算子面(owl-nn OpsCtx 既有)经 tx 发射:同一签名,eager/捕获自动分派
    // PSEUDO: ops.matmul(tx, blas, w, x, out)?;
    // PSEUDO: ops.rmsnorm(tx, x, alpha, out, eps, w_off)?;
    Ok(())
}

async fn phase_agnostic(c: &TensorClient) {
    let w = Tensor::zeros(c, Dtype::F32, &[64, 64]).await.unwrap();
    let mut x = Tensor::zeros(c, Dtype::F32, &[64]).await.unwrap();
    let mut out = Tensor::zeros(c, Dtype::F32, &[64]).await.unwrap();

    // --- eager:直接跑(tx = Live 发射面)---
    // PSEUDO: block_forward(c.live_tx(), &w, &x, &mut out)?;
    //         c.sync().await?;

    // --- 捕获:同一份 block_forward,一行不改,套进事务 ---
    // PSEUDO: let g = c.capture(CapturePlan {
    // PSEUDO:     bindings: ...,        // 预绑定(frontier/positions/slots;seam §三)
    // PSEUDO:     batch: 4,             // 档位
    // PSEUDO: }).await?;              // → CaptureTx(守卫)
    // PSEUDO: block_forward(&g.tx(), &w, &x, &mut out)?;   // ← 同一个函数!
    // PSEUDO: let graph = g.end().await?;                  // 原子:end+audit+instantiate
    // PSEUDO: c.replay(graph).await?;                      // decode 热路径:提交即回

    // 愿望的实质(对 server 的要求清单):
    //   a. capture 的事务守卫暴露的发射面,与 eager 发射面**同型**
    //      (Tx 抽象;web 上 cuTile DeviceOp 的"描述/执行分离"同一思想);
    //   b. 窗内发射的登记(哨兵①/租约)自动完成——契约 2 的落点;
    //   c. 窗口原子性(end+audit+instantiate 失败 = 全回滚)由 server 保证;
    //   d. replay 提交即回,完成通知走事件档(§五档二)。
}

// ============================================================================
// §4 端到端:理想形状的一次 decode 步(总验收用例草案)
// ============================================================================

async fn decode_step_vision(c: &TensorClient, plan_bs4: CapturePlan, tokens: &[u32]) {
    // 1. 装填(Each-step,EagerOnly;bindings 是预绑定缓冲的 narrow 视图)
    // PSEUDO: c.write_binding("frontier", tokens).await?;   // 图主流 H2D,禁止捕获相

    // 2. 回放(提交即回;整链一次发射)
    // PSEUDO: c.replay(plan_bs4.graph).await?;

    // 3. logits 就在预绑定 logits_out(租约钉住);采样直读,零 D2H
    // PSEUDO: let logits = c.binding_tensor("logits_out")?;
    // PSEUDO: let next = sample_radix(logits, ...)?;        // radix 直吃(可同步查,或档二事件)

    // 全程:我(Tensor 视角)没有出现过 device_ptr/pinned/stream/phase 任何一个词。
}

// ============================================================================
// 开放问题(写给评审)
// ============================================================================
//
// Q1. TensorClient/Tx 归属:server 门面的扩展(asyncrt 侧),还是 nn 侧薄壳
//     (owl-nn 包 client)?倾向:**nn 侧薄壳**——server 保持字节世界纯净,
//     dtype/shape 断言、保活登记的本体都在 nn(Tensor 才知道 dtype);
//     server 只新增"块句柄 → 发射参数"的极薄桥。
// Q2. from_host_batch 的批量化程度:A2 先逐个,批命令随 loader 优化立项。
// Q3. to_host 的完成语义:必是"完成 ready"(D2H 数据到手);档二事件对其
//     意义 = 等待前置 kernel 完成,链路 server 内部消化。
