//! **前缀无感删除的设备半**:后缀 K 的池内原地重转(两 rank 对称)。
//!
//! 决策在 `scheduler::plan_image_delete`(纯 host);本模块只做设备动作:
//! 把拼接后的**后缀物理页**里每层 K 乘上 `R(−S)`(位置帧平移),V 不动。
//!
//! 为什么不能复用 prefill 的 rope 核:池 K 是 kNHD 分页
//! `[nb,hkv,hd/x,page,x]` —— token 在**倒数第二维**,拿不到 token-major 视图;
//! 硬搬只会把一次原地乘变成四趟搬运(见 `cu/elems/kv_rope_shift.cu` 头注)。

use owl_iface::contract::{DeviceClient, Dtype, ModelError};
use owl_models::layers::kvshift::{delta_table, k_shift};
use owl_models::interpreters::eval_ops_scoped_env;
use owl_models::TensorOps;

use crate::scheduler::ReropePlan;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// **删除之后的后缀 K 重转**(`R(−shift)`;每层一发,两 rank 对称)。
    ///
    /// 顺序纪律:必须在任何读这些页的 chunk **之前**(调用点 = `execute_begin`,
    /// 紧接 `write_bt`;此时本 turn 还没有 prefill/decode 读过 KV)。
    pub(crate) async fn execute_k_shift(&mut self, plan: &ReropePlan) -> Result<(), ModelError> {
        if plan.pages.is_empty() || plan.shift == 0 {
            return Ok(());
        }
        let half = self.dev.boot.rope.rotary_dim() / 2;
        let geom = self.dev.pool.k_shift_geom(half)?;
        // Δ 表 = m-RoPE 表在 `[Δ,Δ,Δ]` 处的一行(**与上轮 prefill 同源**);
        // 位置前移 `shift` ⇒ 乘 `R(−shift)`。
        let (cos, sin) = {
            let vlm = self.dev.boot.vlm.as_ref().ok_or_else(|| {
                ModelError::EngineInvariant {
                    what: "K 重转:本 rank 无视觉档(image_evict 的 boot 门应已拒)".into(),
                }
            })?;
            delta_table(&vlm.cfg().mrope, -plan.shift)
        };
        let pack = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()
        };
        let cos_t = TensorOps::from_host(Dtype::F16, vec![geom.half], &pack(&cos));
        let sin_t = TensorOps::from_host(Dtype::F16, vec![geom.half], &pack(&sin));
        let blocks: Vec<f32> = plan.pages.iter().map(|p| *p as f32).collect();
        let blocks_bytes: Vec<u8> = blocks.iter().flat_map(|v| v.to_le_bytes()).collect();
        let blocks_t = TensorOps::from_host(Dtype::F32, vec![blocks.len()], &blocks_bytes);
        let t0 = std::time::Instant::now();
        let n_layers = self.dev.pool.kv_caches().len();
        let bytes = plan.pages.len() * self.dev.pool.block_bytes();
        for (k, _v) in self.dev.pool.kv_caches() {
            let node = k_shift(&k, &cos_t, &sin_t, &blocks_t, geom);
            // env 是**方法调用**(整 &self.dev 借用)⇒ 必须在 face_mut 之前取值,
            // 否则与 graphs 的可变借用打架(`prepare_vlm_rank` 同款顺序)。
            let env = self.dev.env();
            let face = self.dev.graphs.session.face_mut();
            let _ = eval_ops_scoped_env(node.step(), face, env, self.dev.boot.tp.as_ref()).await?;
        }
        // TP2:**同一份 Δ 表 / 同一份页列表**(物理槽两 rank 同号;旋转对称)
        if let Some(peer) = self.peer.as_mut() {
            let geom_p = peer.pool.k_shift_geom(half)?;
            for (k, _v) in peer.pool.kv_caches() {
                let node = k_shift(&k, &cos_t, &sin_t, &blocks_t, geom_p);
                let env = peer.env();
                let face = peer.graphs.session.face_mut();
                let _ = eval_ops_scoped_env(node.step(), face, env, peer.boot.tp.as_ref()).await?;
            }
        }
        owl_shared::counter_add!("image_evict.rotate.bytes", bytes as u64);
        owl_shared::log_info!(
            "img-evict",
            "后缀 K 重转 R(−{}) · 层 {} · 页 {} / {:.1} MiB / {:.2} ms",
            plan.shift,
            n_layers,
            plan.pages.len(),
            bytes as f64 / 1048576.0,
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }
}
