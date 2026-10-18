//! **peer 面(TP2 S1)**——跨卡 PeerShared 分配/映射/释放的执行臂。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §2(判据 = 单笔上限;对称握手序列)。
//! 决策出处:D1 = **直扩 `Command`**(唯一规范;iface 里那套 peer 声明非规范)。
//!
//! # 实现现状(2026-10-16 C1)
//!
//! 三个 handler 都是**真实现**(壳子期桩已替换):
//! - `PeerAlloc`:VMM `reserve→create→map→setAccess(RW)→export` + 单笔硬门;
//! - `PeerImport`:`import→reserve→map→setAccess(R)`(别名只读);
//! - `PeerRelease`:显式释放 + 图引用租约拒收(`Drop` 兜底在 `GpuCtx`)。
//!
//! 捕获期由 server 路由的 catch-all(`Self::reject`)天然拒掉这三个命令
//! (peer 授予必须发生在建图之前 —— 施工设计 §2.2 纪律 1)。
//!
//! # 凭据线格式(显式 28 B 头;同进程 fd 直传)
//!
//! ```text
//! 偏移    宽   内容
//! 0       u32  magic = 0x4F57_4C50("OWLP" LE)
//! 4       u32  version = 1
//! 8       u64  mapped_bytes(分配侧 VMM 映射总字节;粒度对齐后)
//! 16      u64  reserved(0;将来扩用)
//! 24      i32  fd(POSIX_FILE_DESCRIPTOR;同进程 fd 表直传)
//! ```
//!
//! ⚠️ **fd 所有权 = 分配侧**(单进程双 actor ⇒ fd 是同一张表项):
//! 导入侧**绝不 close**(否则分配侧释放时双 close 事故);分配侧在
//! `peer_release` 里 close。跨进程形态才需要 SCM_RIGHTS + 各自副本 ——
//! 本仓 TP2 = 单进程双 actor,不进那条路。

use super::*;

// ============================================================================
// 凭据编解码(纯逻辑;显式头 —— 陌生字节响亮拒,不猜)
// ============================================================================

/// 凭据 magic("OWLP")
const CRED_MAGIC: u32 = 0x4F57_4C50;
/// 凭据线格式版本
const CRED_VERSION: u32 = 1;
/// 凭据总长(magic + version + mapped_bytes + reserved + fd)
const CRED_LEN: usize = 28;

/// 编码(分配侧;`fd` 所有权不移交 —— 见模块头)。
fn cred_encode(mapped_bytes: u64, fd: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(CRED_LEN);
    out.extend_from_slice(&CRED_MAGIC.to_le_bytes());
    out.extend_from_slice(&CRED_VERSION.to_le_bytes());
    out.extend_from_slice(&mapped_bytes.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // reserved
    out.extend_from_slice(&fd.to_le_bytes());
    debug_assert_eq!(out.len(), CRED_LEN);
    out
}

/// 解码(导入侧)。**长度/magic/版本任一不符 = 响亮拒**(不"尽力而为")。
fn cred_decode(cred: &[u8]) -> Result<(u64, i32), ModelError> {
    if cred.len() != CRED_LEN {
        return Err(ModelError::Msg(format!(
            "peer 凭据长度非法:{} B(期望 {CRED_LEN};拒绝 —— 不猜格式)",
            cred.len()
        )));
    }
    let rd_u32 = |o: usize| u32::from_le_bytes(cred[o..o + 4].try_into().expect("越界已保证"));
    let magic = rd_u32(0);
    if magic != CRED_MAGIC {
        return Err(ModelError::Msg(format!(
            "peer 凭据 magic 不符:0x{magic:08X}(期望 0x{CRED_MAGIC:08X};拒绝)"
        )));
    }
    let ver = rd_u32(4);
    if ver != CRED_VERSION {
        return Err(ModelError::Msg(format!(
            "peer 凭据版本不支持:v{ver}(本端支持 v{CRED_VERSION};拒绝)"
        )));
    }
    let mapped_bytes = u64::from_le_bytes(cred[8..16].try_into().expect("越界已保证"));
    let fd = i32::from_le_bytes(cred[24..28].try_into().expect("越界已保证"));
    if mapped_bytes == 0 || fd < 0 {
        return Err(ModelError::Msg(format!(
            "peer 凭据字段非法:mapped_bytes={mapped_bytes} fd={fd}(拒绝)"
        )));
    }
    Ok((mapped_bytes, fd))
}

impl GpuServer {
    /// `Command::PeerAlloc`(TP2 S1):VMM 拼块 + 本地 RW 授予 + 导出凭据。
    ///
    /// 额度硬门在 `GpuCtx::peer_alloc`(server 权威);此处只做编解码与
    /// 探针记账(与 `handle_alloc` 同款薄壳)。
    pub(super) fn handle_peer_alloc(
        &mut self,
        bytes: usize,
        name: String,
        ack: Ack<Result<(Bytes, Vec<u8>), ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        let result = (|| {
            let (id, fd, mapped) = self.ctx_mut().peer_alloc(bytes, &name)?;
            self.trace_alloc_done(id);
            self.sink.logf(
                owl_shared::log::LVL_INFO,
                "peer",
                format_args!(
                    "PeerAlloc id={id} 请求 {}MiB → 映射 {}MiB fd={fd}(name={name})",
                    bytes >> 20,
                    mapped >> 20
                ),
            );
            Ok((Bytes::new(id, bytes, Dtype::U8, 1), cred_encode(mapped, fd)))
        })();
        self.sink.timer("srv.seg.peer_alloc", t0.elapsed());
        ack.send(result);
    }

    /// `Command::PeerImport`(TP2 S1):消费对端凭据 → 本卡 VA 别名(AR 核读对端 staging)。
    pub(super) fn handle_peer_import(
        &mut self,
        credential: Vec<u8>,
        bytes: usize,
        peer_uuid: String,
        name: String,
        ack: Ack<Result<Bytes, ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        let result = (|| {
            let (mapped, fd) = cred_decode(&credential)?;
            let id = self.ctx_mut().peer_import(fd, mapped, bytes, &peer_uuid, &name)?;
            self.trace_alloc_done(id);
            self.sink.logf(
                owl_shared::log::LVL_INFO,
                "peer",
                format_args!(
                    "PeerImport id={id} 请求 {}MiB / 对端映射 {}MiB(peer={peer_uuid} name={name})",
                    bytes >> 20,
                    mapped >> 20
                ),
            );
            Ok(Bytes::new(id, bytes, Dtype::U8, 1))
        })();
        self.sink.timer("srv.seg.peer_import", t0.elapsed());
        ack.send(result);
    }

    /// `Command::PeerRelease`(TP2 S1;D2 = 显式释放 + Drop 兜底)。
    pub(super) fn handle_peer_release(
        &mut self,
        ids: Vec<u64>,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        let result = match self.ctx_mut().peer_release(&ids) {
            Ok(n) => {
                self.sink.count("srv.count.peer_released", n as u64);
                Ok(())
            }
            Err(e) => Err(e),
        };
        self.sink.timer("srv.seg.peer_release", t0.elapsed());
        ack.send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::{cred_decode, cred_encode, CRED_LEN};

    #[test]
    fn cred_roundtrip() {
        let c = cred_encode(64 << 20, 7);
        assert_eq!(c.len(), CRED_LEN);
        let (mb, fd) = cred_decode(&c).expect("自产凭据必须可解");
        assert_eq!(mb, 64 << 20);
        assert_eq!(fd, 7);
    }

    #[test]
    fn cred_rejects_foreign_bytes() {
        // 长度非法
        assert!(cred_decode(&[]).is_err());
        assert!(cred_decode(&[0u8; 8]).is_err());
        // magic 不符
        let mut bad = cred_encode(1 << 20, 3);
        bad[0] ^= 0xFF;
        assert!(cred_decode(&bad).is_err());
        // 版本不符
        let mut bad_v = cred_encode(1 << 20, 3);
        bad_v[4] = 99;
        assert!(cred_decode(&bad_v).is_err());
        // 字段非法(fd 负数)
        let mut bad_fd = cred_encode(1 << 20, 3);
        bad_fd[24..28].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(cred_decode(&bad_fd).is_err());
        // mapped_bytes = 0
        let bad_zero = cred_encode(0, 3);
        assert!(cred_decode(&bad_zero).is_err());
    }
}
