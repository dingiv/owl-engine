//! **host-mapped 面(TP host relay;2026-11-04)** —— 无 P2P 驱动的兜底通道。
//!
//! 与 `peer.rs` 并列但机制不同:这里是 `cudaHostAlloc(MAPPED|PORTABLE)` +
//! `cuMemHostGetDevicePointer_v2`,不是 VMM/BAR1 授予。两卡共享同一段 host
//! 内存;`HostAlloc` 的 owner 负责 `cudaFreeHost`,对端用 `HostAttach` 建
//! 第二份账。分配/attach 必须发生在建图之前(捕获期由 `capture_gate` 拒),
//! 释放前必须先卸图(图租约在 `GpuCtx::host_release` 里拒收)。
//!
//! 设计参考:
//! - `llama.cpp/ggml/src/ggml-cuda/allreduce.cu` 的 pinned host-mapped
//!   staging + 设备内 arrival 轮询(证明消费卡无 P2P 也可行);
//! - `ninfer_duo_3080/src/ops/common/allreduce.cu` 的 UVA `cudaMemcpyAsync`
//!   D2D 在无 peer access 时由 CUDA 透明 host-staged 的行为。

use super::*;

impl GpuServer {
    /// `Command::HostAlloc`:pinned host 段 + 设备别名 + owner 块。
    pub(super) fn handle_host_alloc(
        &mut self,
        bytes: usize,
        name: String,
        ack: Ack<Result<(Bytes, u64), ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        let result = (|| {
            let (id, host) = self.ctx_mut().host_alloc(bytes, &name)?;
            self.trace_alloc_done(id);
            self.sink.logf(
                owl_shared::log::LVL_INFO,
                "host",
                format_args!(
                    "HostAlloc id={id} {}MiB host={host:#x}(name={name};MAPPED|PORTABLE)",
                    bytes >> 20
                ),
            );
            Ok((Bytes::new(id, bytes, Dtype::U8, 1), host))
        })();
        self.sink.timer("srv.seg.host_alloc", t0.elapsed());
        ack.send(result);
    }

    /// `Command::HostAttach`:在另一 actor 的地址空间登记同一 host 段。
    pub(super) fn handle_host_attach(
        &mut self,
        host: u64,
        bytes: usize,
        name: String,
        ack: Ack<Result<Bytes, ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        let result = (|| {
            let id = self.ctx_mut().host_attach(host, bytes, &name)?;
            self.trace_alloc_done(id);
            self.sink.logf(
                owl_shared::log::LVL_INFO,
                "host",
                format_args!(
                    "HostAttach id={id} {}MiB host={host:#x}(name={name})",
                    bytes >> 20
                ),
            );
            Ok(Bytes::new(id, bytes, Dtype::U8, 1))
        })();
        self.sink.timer("srv.seg.host_attach", t0.elapsed());
        ack.send(result);
    }

    /// `Command::HostRelease`:owner `cudaFreeHost`;attach 只销账。
    pub(super) fn handle_host_release(&mut self, ids: Vec<u64>, ack: Ack<Result<(), ModelError>>) {
        let t0 = std::time::Instant::now();
        let result = match self.ctx_mut().host_release(&ids) {
            Ok(n) => {
                self.sink.count("srv.count.host_released", n as u64);
                Ok(())
            }
            Err(e) => Err(e),
        };
        self.sink.timer("srv.seg.host_release", t0.elapsed());
        ack.send(result);
    }
}
