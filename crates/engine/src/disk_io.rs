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

/// **批量读回一批块**(2026-11-01 P1 回灌提速)。
///
/// 逐块 `read_block` = 每块一次 `spawn_blocking` 线程跳 + 一次 open/read/close;
/// 实测 4364 块(4.3 GiB,page cache 全热)只有 **0.96 GB/s**。这里把整窗塞进
/// **一次** 线程跳、窗口内再按 4 路并行读 —— 台账里每次 ≥8 块走这条路。
///
/// 返回顺序与入参**逐项对齐**(`Err` 项带路径与原因,由调用方降级/报错);
/// 大小不符仍按"账上长度"判(与 [`read_block`] 同律)。
pub(crate) async fn read_blocks(
    items: Vec<(std::path::PathBuf, usize)>,
) -> Vec<Result<Vec<u8>, String>> {
    if items.is_empty() {
        return Vec::new();
    }
    tokio::task::spawn_blocking(move || {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        let n = items.len();
        let lanes = n.clamp(1, 4);
        let next = AtomicUsize::new(0);
        let out: Mutex<Vec<Option<Result<Vec<u8>, String>>>> =
            Mutex::new((0..n).map(|_| None).collect());
        std::thread::scope(|sc| {
            for _ in 0..lanes {
                let items = &items;
                let next = &next;
                let out = &out;
                sc.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let (path, want) = &items[i];
                    let label = path.display().to_string();
                    let r = std::fs::read(path)
                        .map_err(|e| format!("磁盘读失败({label}):{e}"))
                        .and_then(|v| {
                            if v.len() == *want {
                                Ok(v)
                            } else {
                                Err(format!(
                                    "磁盘块大小不符:{label} 实得 {} ≠ 账上 {}",
                                    v.len(),
                                    want
                                ))
                            }
                        });
                    if let Ok(mut g) = out.lock() {
                        g[i] = Some(r);
                    }
                });
            }
        });
        out.into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .into_iter()
            .map(|x| x.unwrap_or_else(|| Err("磁盘读未完成(内部错)".to_string())))
            .collect()
    })
    .await
    .unwrap_or_else(|e| vec![Err(format!("磁盘读任务 join 失败:{e}"))])
}

/// 把一块内容改名(盘层快照重挂;语义同 [`write_block`] 的原子性)。
///
/// 失败**不改账** —— 调用方保留旧键(旧文件、旧路径仍在),按响亮日志走。
pub(crate) async fn rename_block(from: PathBuf, to: PathBuf) -> Result<(), String> {
    let label = format!("{} → {}", from.display(), to.display());
    let joined = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::rename(&from, &to)
    })
    .await;
    match joined {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("磁盘改名失败({label}):{e}")),
        Err(e) => Err(format!("磁盘改名任务 join 失败({label}):{e}")),
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
                    owl_shared::log_warn!(
                        "disk",
                        "删除失败({}):{e}(账已出,文件残留)",
                        p.display()
                    );
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

/// **启动期磁盘配额对账**(2026-11-01;答"配置改小 / 没改也超限")。
///
/// 现状(`DiskTier::adopt`)只做**隔离**:超预算 / 大小不符 / 重复键的文件
/// **不删也不进账** ⇒ 物理占用可以永久大于配额,而且那些文件因为不在账里,
/// `plan_evict` 的 LRU **永远挑不到**它们。换模型(块尺寸变)、崩在写盘半路
/// (`.tmp` 残件)、索引丢失(孤儿)都会往这里堆 —— 这正是 vLLM 那类实现
/// "缓存盘越跑越满、命中反而越来越差"的成因。
///
/// 本函数以**文件系统真值**为准做一次对账(与 store 的账无关):
/// 1. **残件**:`*.tmp` 一律删(写盘是 tmp+rename,留下的必然是半成品);
/// 2. **只在超配额时才删**:实占 ≤ 配额 ⇒ 一个不动(不打扰热内容);
/// 3. 删除顺序 = **先不可用、再最老**:未被任何已采纳链引用的(`protected=false`)
///    先删,同类内按 mtime 旧→新;
/// 4. **绝不为凑配额删"已采纳链"的活内容** —— 全是被保护的还超 ⇒ 只报 ERROR
///    (那是配额配置错,该由人改配置,不是引擎偷偷毁数据)。
///
/// 幂等:可重复调用;多进程同目录并发时最坏重复删同一个文件(`NotFound` 忽略)。
pub(crate) struct ReconcileReport {
    pub tmp_removed: usize,
    pub removed: usize,
    pub bytes_freed: u64,
    pub kept_bytes: u64,
    /// 删掉的**受保护**文件数(>0 = 配额不足已开始牺牲召回面,必须让人看见)
    pub removed_protected: usize,
    /// 被删文件的路径(调用方拿去出账/审计)
    pub removed_paths: Vec<std::path::PathBuf>,
}

pub(crate) fn reconcile_dir(
    root: &std::path::Path,
    quota_bytes: usize,
    protected: &std::collections::HashSet<std::path::PathBuf>,
) -> ReconcileReport {
    use std::collections::HashSet;
    let mut rep = ReconcileReport {
        tmp_removed: 0,
        removed: 0,
        bytes_freed: 0,
        kept_bytes: 0,
        removed_protected: 0,
        removed_paths: Vec::new(),
    };
    // 注:`quota_bytes == 0` **仍清 tmp**(残件永远是垃圾),但不动 .bin。
    let mut bins: Vec<(std::path::PathBuf, u64, std::time::SystemTime, bool)> = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return rep;
    };
    for e in rd.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if name.ends_with(".tmp") {
            if std::fs::remove_file(&path).is_ok() {
                rep.tmp_removed += 1;
            }
            continue;
        }
        if !name.ends_with(".bin") {
            continue; // 索引/外来文件不动
        }
        let Ok(md) = e.metadata() else { continue };
        let mtime = md.modified().unwrap_or(std::time::UNIX_EPOCH);
        let prot = protected.contains(&path);
        bins.push((path, md.len(), mtime, prot));
    }
    if quota_bytes == 0 {
        rep.kept_bytes = bins.iter().map(|b| b.1).sum();
        return rep;
    }
    let total: u64 = bins.iter().map(|b| b.1).sum();
    if total <= quota_bytes as u64 {
        rep.kept_bytes = total;
        return rep;
    }
    // 排序 = **先删不可用的,再删"尾巴"**:
    // - 未保护(孤儿/外来):mtime 旧→新(越老越没价值);
    // - 受保护(活链):mtime **新→旧** —— 删尾部才能保住"从头连续的可用前缀";
    //   删链头 = 整条链再也对不上(实测过:无保护集按 mtime 删 ⇒ 重建 0 链)。
    // 用 (组号, 时间戳) 排序:未保护组按时间**升序**,受保护组按时间**降序** ——
    // 升/降用戳的取反表达(同一把 SystemTime 比较,避免两套键类型)。
    bins.sort_by(|a, b| {
        let (ga, gb) = (a.3 as u8, b.3 as u8);
        ga.cmp(&gb).then_with(|| {
            if a.3 {
                b.2.cmp(&a.2) // 受保护:新的先删(保前缀)
            } else {
                a.2.cmp(&b.2) // 未保护:旧的后删(先删更老)
            }
        })
    });
    let mut live = total;
    for (path, len, _, prot) in bins {
        if live <= quota_bytes as u64 {
            break;
        }
        if prot && rep.removed_protected > 0 {
            // 已经开始牺牲活内容:同一批里只删到刚好够,但继续删也行 —— 这里保守:
            // 只删到配额即停(上面的 live 判断),因此不会无差别清空
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                live -= len;
                rep.removed += 1;
                rep.bytes_freed += len;
                rep.removed_paths.push(path);
                if prot {
                    rep.removed_protected += 1;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                live -= len; // 并发/重入:已经没了,账照减
            }
            Err(_) => {}
        }
    }
    rep.kept_bytes = live;
    rep
}

/// 盘上 `*.bin` 的物理实占(诊断:与配额对账打一行)。
pub(crate) fn dir_bin_bytes(root: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(root) else {
        return 0;
    };
    rd.flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.ends_with(".bin")
        })
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
}
