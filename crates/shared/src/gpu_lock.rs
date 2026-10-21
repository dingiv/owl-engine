//! **GPU 占用锁**(polite start / graceful close;2026-10-16)。
//!
//! 一个卡一个锁文件:**文件名 = 卡的 UUID**(唯一身份证 —— 躲开 CUDA ordinal /
//! PCI 枚举序漂移,见 `docs/dev/debug-tools.md` §7),内容 = TOML(零新增依赖;人读友好):
//! `{uuid, pid, port, cfg, started_at}`(不记 flavor —— 它只是"预设配置文件"的名字,
//! 真正决定行为的是最终生效的配置,故只记 `cfg` 路径)。
//!
//! - **polite start**:启动前 `live_locks()` 读全量 + 剔陈旧(进程已死 ⇒ 删文件),
//!   调用方据此挑空闲卡;对已被占的卡 = 响亮拒绝(带持有者信息),不静默抢占。
//! - **graceful close**:`LockGuard` 的 `Drop` 删自己的锁 —— 正常退出路径全覆盖;
//!   信号处理只负责"把控制权交回正常退出"(见 apps/server 接线)。
//! - 锁只是**协调**,不是安全边界:它不能阻止别的进程(人手 `CUDA_VISIBLE_DEVICES`
//!   或非 owl 程序)占卡,故 `acquire` 之后仍要如实上报显存/启动失败。
//!
//! 目录(**锁是运行期状态 ⇒ 优先 `/run`**,重启即清):`OWL_GPU_LOCK_DIR`(显式覆盖)→
//! `$XDG_RUNTIME_DIR/owl/gpu-locks`(典型 `/run/user/<uid>/owl/gpu-locks`)→
//! `/run/user/<uid>/owl/gpu-locks` → `$HOME/.owl/gpu-locks`(兜底)。

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// 锁目录覆盖 env(缺省 `$HOME/.owl/gpu-locks`)
pub const LOCK_DIR_ENV: &str = "OWL_GPU_LOCK_DIR";

/// 锁文件内容(文件名另载 UUID;此处再存一份便于人读与校验)
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LockInfo {
    pub uuid: String,
    pub pid: u32,
    pub port: u16,
    pub cfg: Option<String>,
    pub started_at: u64,
}

impl LockInfo {
    /// 当前进程 + 给定卡/端口/档位的锁(时间戳取 now)
    pub fn new(uuid: &str, port: u16, cfg: Option<String>) -> Self {
        Self {
            uuid: uuid.to_string(),
            pid: std::process::id(),
            port,
            cfg,
            started_at: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        }
    }

    /// 持锁进程还活着吗(Linux:`/proc/<pid>` 是否存在)
    pub fn pid_alive(&self) -> bool {
        pid_alive(self.pid)
    }
}

/// 进程存活判据(Linux;`/proc` 不存在 = 死)
pub fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// 缺省锁目录(`OWL_GPU_LOCK_DIR` 覆盖)
pub fn default_lock_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let uid = home
        .as_deref()
        .and_then(|h| std::fs::metadata(h).ok())
        .map(|m| {
            use std::os::unix::fs::MetadataExt;
            m.uid()
        });
    let cands = lock_dir_candidates(
        std::env::var_os(LOCK_DIR_ENV),
        std::env::var_os("XDG_RUNTIME_DIR"),
        home,
        uid,
    );
    // 首个"已存在或能建出来"的目录胜(顺带 ensure);全失败 ⇒ 返回末位(acquire 时响亮报错)
    for c in &cands {
        if c.is_dir() || std::fs::create_dir_all(c).is_ok() {
            return c.clone();
        }
    }
    cands.last().cloned().unwrap_or_else(|| PathBuf::from("/tmp/owl-gpu-locks"))
}

/// 锁目录候选(纯函数,按优先级;**锁是运行期状态 ⇒ 优先 `/run`**,重启即清):
/// ① `OWL_GPU_LOCK_DIR`(显式覆盖/测试)
/// ② `$XDG_RUNTIME_DIR/owl/gpu-locks`(典型 = `/run/user/<uid>/owl/gpu-locks`)
/// ③ `/run/user/<HOME 属主 uid>/owl/gpu-locks`(XDG 未设但用户运行期目录在)
/// ④ `$HOME/.owl/gpu-locks`(兜底;非 tmpfs,陈旧锁靠 pid 复活检测清)
pub fn lock_dir_candidates(
    env_override: Option<std::ffi::OsString>,
    xdg_runtime: Option<std::ffi::OsString>,
    home: Option<PathBuf>,
    home_owner_uid: Option<u32>,
) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Some(d) = env_override {
        v.push(PathBuf::from(d));
    }
    if let Some(x) = xdg_runtime {
        v.push(PathBuf::from(x).join("owl").join("gpu-locks"));
    }
    if let Some(uid) = home_owner_uid {
        v.push(
            PathBuf::from(format!("/run/user/{uid}"))
                .join("owl")
                .join("gpu-locks"),
        );
    }
    if let Some(h) = home {
        v.push(h.join(".owl").join("gpu-locks"));
    }
    v
}

fn lock_path(dir: &Path, uuid: &str) -> PathBuf {
    dir.join(format!("{uuid}.lock"))
}

/// 读某目录下的锁:**陈旧者就地删除**(进程已死;删失败则响亮 warn),返回活锁
pub fn live_locks_in(dir: &Path) -> Vec<LockInfo> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("lock") {
            continue;
        }
        let raw = match std::fs::read_to_string(&p) {
            Ok(r) => r,
            Err(e) => {
                crate::log_warn!("gpu-lock", "锁文件读取失败(跳过):{}: {e}", p.display());
                continue;
            }
        };
        match toml::from_str::<LockInfo>(&raw) {
            Ok(info) if info.pid_alive() => out.push(info),
            Ok(info) => {
                // 陈旧锁:持有者进程已死 ⇒ 清掉(不静默)
                match std::fs::remove_file(&p) {
                    Ok(()) => crate::log_info!("gpu-lock", "清理陈旧锁(pid {} 已死):{}", info.pid, p.display()),
                    Err(e) => crate::log_warn!("gpu-lock", "陈旧锁删除失败:{}: {e}", p.display()),
                }
            }
            Err(e) => crate::log_warn!("gpu-lock", "锁文件解析失败(跳过):{}: {e}", p.display()),
        }
    }
    out.sort_by(|a, b| a.uuid.cmp(&b.uuid));
    out
}

/// 缺省目录版的 [`live_locks_in`]
pub fn live_locks() -> Vec<LockInfo> {
    live_locks_in(&default_lock_dir())
}

/// 某卡的活锁持有者(None = 空闲)
pub fn holder_of(uuid: &str) -> Option<LockInfo> {
    live_locks().into_iter().find(|l| l.uuid == uuid)
}

/// 持锁守卫:`Drop` = 删自己的锁(graceful close 的正常退出面)
#[derive(Debug)]
pub struct LockGuard {
    path: PathBuf,
}

impl LockGuard {
    /// 取锁(指定目录;同 uuid 已有**活锁** ⇒ 响亮 Err 带持有者信息)
    pub fn acquire_in(dir: &Path, info: &LockInfo) -> Result<Self, String> {
        if let Some(h) = live_locks_in(dir).into_iter().find(|l| l.uuid == info.uuid) {
            return Err(format!(
                "GPU {} 已被占用:pid={} port={} cfg={:?} 起于 {}",
                info.uuid, h.pid, h.port, h.cfg, h.started_at
            ));
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("锁目录 {} 建失败: {e}", dir.display()))?;
        let path = lock_path(dir, &info.uuid);
        let body = toml::to_string(info).map_err(|e| format!("锁序列化失败: {e}"))?;
        std::fs::write(&path, body).map_err(|e| format!("锁写入失败 {}: {e}", path.display()))?;
        crate::log_info!("gpu-lock", "持锁 {:?}(pid {} port {})", info.uuid, info.pid, info.port);
        Ok(Self { path })
    }

    /// 缺省目录版的 [`LockGuard::acquire_in`]
    pub fn acquire(info: &LockInfo) -> Result<Self, String> {
        Self::acquire_in(&default_lock_dir(), info)
    }

    /// 锁文件路径(诊断/测试用)
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// **显式释放**(信号路径用 —— `process::exit` 不跑 `Drop`;幂等,已不存在视作成功)
pub fn release(path: &std::path::Path) {
    match std::fs::remove_file(path) {
        Ok(()) => crate::log_info!("gpu-lock", "释放锁 {}", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => crate::log_warn!("gpu-lock", "释放锁失败 {}: {e}", path.display()),
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.path) {
            Ok(()) => crate::log_info!("gpu-lock", "释放锁 {}", self.path.display()),
            // 已不存在 = 视作已释放(不报错);其他失败响亮 warn(律 #1)
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => crate::log_warn!("gpu-lock", "释放锁失败 {}: {e}", self.path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("owl_gpu_lock_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// 取锁 → 同卡二次取锁被拒(带持有者)→ Drop 后可再取(优雅释放)
    #[test]
    fn acquire_refuse_then_release() {
        let d = tmp("acq");
        let g = LockGuard::acquire_in(&d, &LockInfo::new("GPU-test-1", 8137, Some("agent-27b.toml".into())))
            .expect("首取");
        let err = LockGuard::acquire_in(&d, &LockInfo::new("GPU-test-1", 8138, None)).unwrap_err();
        assert!(err.contains("已被占用") && err.contains("8137"), "拒绝信息应带持有者: {err}");
        drop(g);
        LockGuard::acquire_in(&d, &LockInfo::new("GPU-test-1", 8138, None)).expect("释放后可再取");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 陈旧锁(死 pid)自动清理;活锁保留
    #[test]
    fn stale_lock_is_reaped() {
        let d = tmp("stale");
        std::fs::create_dir_all(&d).expect("mkdir");
        let mut dead = LockInfo::new("GPU-dead", 1, None);
        dead.pid = 0x7fff_ffff; // 不存在的 pid
        std::fs::write(lock_path(&d, "GPU-dead"), toml::to_string(&dead).unwrap()).expect("w");
        let mut live = LockInfo::new("GPU-live", 1, None);
        live.pid = std::process::id();
        std::fs::write(lock_path(&d, "GPU-live"), toml::to_string(&live).unwrap()).expect("w");
        let got = live_locks_in(&d);
        assert_eq!(got.len(), 1, "只应剩活锁:{got:?}");
        assert_eq!(got[0].uuid, "GPU-live");
        assert!(!lock_path(&d, "GPU-dead").exists(), "陈旧锁应被删");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 坏文件不炸:解析失败 ⇒ 跳过(响亮 warn),其余仍可用
    #[test]
    fn garbage_lock_skipped() {
        let d = tmp("junk");
        std::fs::create_dir_all(&d).expect("mkdir");
        std::fs::write(lock_path(&d, "GPU-junk"), "{not json").expect("w");
        assert!(live_locks_in(&d).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 候选链顺序:env 覆盖 → XDG(/run/user/<uid>)→ /run/user/<uid> → $HOME 兜底
    #[test]
    fn lock_dir_priority() {
        use std::ffi::OsString;
        let c = lock_dir_candidates(
            Some(OsString::from("/tmp/x")),
            Some(OsString::from("/run/user/1000")),
            Some(PathBuf::from("/home/user")),
            Some(1000),
        );
        assert_eq!(c[0], PathBuf::from("/tmp/x"), "显式覆盖最优先");
        assert_eq!(c[1], PathBuf::from("/run/user/1000/owl/gpu-locks"), "XDG 运行期目录次之");
        assert_eq!(c[2], PathBuf::from("/run/user/1000/owl/gpu-locks"));
        assert_eq!(c[3], PathBuf::from("/home/user/.owl/gpu-locks"), "HOME 兜底");
        // env 与 XDG 都缺 ⇒ 仍要有 /run 与 HOME 两级
        let c2 = lock_dir_candidates(None, None, Some(PathBuf::from("/home/user")), Some(7));
        assert_eq!(c2[0], PathBuf::from("/run/user/7/owl/gpu-locks"));
        assert_eq!(c2[1], PathBuf::from("/home/user/.owl/gpu-locks"));
    }
}
