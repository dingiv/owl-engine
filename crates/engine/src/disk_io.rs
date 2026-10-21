//! 磁盘层的**执行面**(S6):store 出计划,这里做真 I/O。
//!
//! # 三条纪律(设计卷 §15.4 的落地)
//!
//! 1. **I/O 只在阻塞池**:`std::fs` 一律跑在 [`tokio::task::spawn_blocking`] ——
//!    27B 一条快照 159 MiB,直接在 actor 里写会钉死整条命令管道;
//! 2. **原子写**:先写 `*.bin.tmp` 再 `rename` 到 `*.bin` ⇒ 崩在半路
//!    不会留下被下一次启动当成好块的半截文件(启动扫描还会用大小兜底);
//! 3. **目录自建**:boot 只算路径,第一次写盘时按需 `create_dir_all`。
//!
//! store 侧**零 I/O**(§21):它只给 [`owl_store::SpillPlanOf`] /
//! [`owl_store::FillPlanOf`] / 驱逐计划;本模块是唯一的执行者。

use std::path::{Path, PathBuf};

use owl_store::{DiskKey, DiskTier};

/// 写盘失败(**内容原样带回** —— 调用方据此回滚到宿主,内容不丢)。
pub(crate) struct WriteFail {
    /// 未写成的内容(所有权回到调用方)。
    pub bytes: Vec<u8>,
    /// 失败原因(响亮;含路径)。
    pub err: String,
}

/// 原子的把一块内容写到磁盘(写 `.tmp` → `rename`)。
///
/// 失败时把 `bytes` **还回来**:溢写是"内容移动",失败必须能放回宿主
/// (用户裁决:不急着丢)。
pub(crate) async fn write_block(
    path: PathBuf,
    tmp: PathBuf,
    bytes: Vec<u8>,
) -> Result<usize, WriteFail> {
    let n = bytes.len();
    let label = path.display().to_string();
    let joined = tokio::task::spawn_blocking(move || -> Result<(), (Vec<u8>, String)> {
        let fail = |e: std::io::Error, b: Vec<u8>| (b, format!("{e}"));
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                return Err(fail(e, bytes));
            }
        }
        if let Err(e) = std::fs::write(&tmp, &bytes) {
            return Err(fail(e, bytes));
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            return Err(fail(e, bytes));
        }
        Ok(())
    })
    .await;
    match joined {
        Ok(Ok(())) => Ok(n),
        Ok(Err((b, e))) => Err(WriteFail { bytes: b, err: format!("磁盘写失败({label}):{e}") }),
        Err(e) => Err(WriteFail {
            bytes: Vec::new(),
            err: format!("磁盘写任务 join 失败({label}):{e}(内容已丢)"),
        }),
    }
}

/// 从磁盘读回一块内容。
pub(crate) async fn read_block(path: PathBuf, bytes: usize) -> Result<Vec<u8>, String> {
    let label = path.display().to_string();
    let v = tokio::task::spawn_blocking(move || std::fs::read(&path))
        .await
        .map_err(|e| format!("磁盘读任务 join 失败:{e}"))?
        .map_err(|e| format!("磁盘读失败({label}):{e}"))?;
    if v.len() == bytes {
        Ok(v)
    } else {
        Err(format!(
            "磁盘块大小不符:{label} 实得 {} ≠ 账上 {}(文件被外部改动?)",
            v.len(),
            bytes
        ))
    }
}

/// 删除一批文件(磁盘驱逐;返回真删掉的文件数)。**不报错上抛**:
/// 文件不在(已被外部清掉)也算腾出成功 —— 账由调用方按计划出。
pub(crate) async fn remove_files(paths: Vec<PathBuf>) -> usize {
    tokio::task::spawn_blocking(move || {
        let mut n = 0usize;
        for p in paths {
            match std::fs::remove_file(&p) {
                Ok(()) => n += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => n += 1,
                Err(e) => {
                    eprintln!("[disk] 删除失败({}):{e}(账已出,文件残留)", p.display());
                }
            }
        }
        n
    })
    .await
    .unwrap_or(0)
}

/// 扫一个目录(启动回填):`(键, 路径, 大小)`;不认识的形状**跳过**
/// (`.tmp` 残件、外来文件 —— 由调用方决定报不报)。
///
/// **同步版**:装配期(`into_running` 是同步函数)专用 —— boot 一次,阻塞可接受;
/// 运行期一律走 async 版(它调用同一个实现体)。
pub(crate) fn scan_dir_sync<K: DiskKey>(dir: &Path) -> Vec<(K, PathBuf, usize)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out; // 目录不存在 = 空(第一次跑)
    };
    for ent in rd.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Some(key) = DiskTier::<K>::key_of_file(name) else {
            if name.ends_with(".tmp") {
                // 崩在 rename 前的残件:删掉(它不是账的一部分)
                let _ = std::fs::remove_file(&path);
            }
            continue;
        };
        let Ok(md) = ent.metadata() else { continue };
        out.push((key, path, md.len() as usize));
    }
    out
}

/// 扫一个目录里的 **`.idx` 索引文件**(S6.1;`(键, 路径)`)。名字形状 === `.bin` 的键规则。
pub(crate) fn scan_index_sync<K: DiskKey>(dir: &Path) -> Vec<(K, PathBuf)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for ent in rd.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Some(stem) = name.strip_suffix(".idx") else {
            continue;
        };
        if stem.ends_with(".tmp") {
            let _ = std::fs::remove_file(&path); // 崩在 rename 前的残件
            continue;
        }
        if let Some(k) = K::parse_stem(stem) {
            out.push((k, path));
        }
    }
    out
}

/// 文件是否存在(单次 `stat`;只用于"索引是否已写"的幂等判定 —— 不是数据 I/O)。
pub(crate) fn exists(p: &Path) -> bool {
    std::fs::metadata(p).is_ok()
}
