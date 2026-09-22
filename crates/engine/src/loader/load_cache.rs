//! 烘焙缓存(T2-三 搬运;origin: xinfer utils/load_cache.rs,S3-P3)。
//!
//! 行为契约(xinfer 定版,文件格式语义保持):
//! - 开关(OWL_LOAD_CACHE,兼容 XINFER_LOAD_CACHE)关闭 = 零读写;
//! - 镜像目录 <model_dir>/owl-load-cache/(xinfer 为 xinfer-load-cache,
//!   owl 目录独立,两引擎缓存互不污染);
//! - checksum.txt 两行摘要:源文件(名/大小/mtime)×量化签名×架构×dtype×
//!   world + 各 rank bin 尺寸;任一不符 = 未命中,整仓重烘焙(自愈);
//! - 摘要哈希:XxHash64 → FNV-1a 64(手写,零依赖;目录独立所以
//!   与 xinfer 缓存不互通是预期行为)。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

const FORMAT_VERSION: u32 = 1;
const MODE_OFF: u8 = 0;
const MODE_LOAD: u8 = 1;
const MODE_STORE: u8 = 2;

static ENABLED: OnceLock<bool> = OnceLock::new();
static MODE: AtomicU8 = AtomicU8::new(MODE_OFF);
static LOGGED: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        let owl = std::env::var("OWL_LOAD_CACHE").ok();
        let xinfer = std::env::var("XINFER_LOAD_CACHE").ok();
        owl.as_deref() == Some("1") || xinfer.as_deref() == Some("1")
    })
}

#[derive(Serialize, Deserialize, Clone)]
struct Entry {
    name: String, // "{tag}::{tensor}"
    shape: Vec<usize>,
    offset: u64,
    len: usize,
}

struct Store {
    entries: Vec<Entry>,
    file: Option<fs::File>,
    offset: u64,
}

struct Session {
    dir: PathBuf,
    rank: usize,
    /// world 规模随会话登记(Load 模式下仅存档;Store 路径经参数传递)
    #[allow(dead_code)]
    world: usize,
    idx: HashMap<String, Entry>,
    bin_path: PathBuf,
    store: Mutex<Store>,
}

static SESSION: OnceLock<Session> = OnceLock::new();

fn session() -> Option<&'static Session> {
    SESSION.get()
}

fn mode() -> u8 {
    MODE.load(Ordering::Relaxed)
}

fn source_metas(source_files: &[PathBuf]) -> Vec<(String, u64, i64)> {
    source_files
        .iter()
        .filter_map(|p| {
            let md = fs::metadata(p).ok()?;
            let mtime = md
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs() as i64;
            Some((p.file_name()?.to_string_lossy().to_string(), md.len(), mtime))
        })
        .collect()
}

/// FNV-1a 64(摘要专用;与 xinfer XxHash64 不互通,见模块头)
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn canonical_sources_digest(
    sources: &[(String, u64, i64)],
    quant_sig: &str,
    arch: &str,
    dtype: &str,
    world: usize,
    w4a8: bool,
) -> String {
    let mut desc = format!(
        "v{FORMAT_VERSION}|arch={arch}|dtype={dtype}|world={world}|quant={quant_sig}|w4a8={w4a8}"
    );
    let mut sorted = sources.to_vec();
    sorted.sort();
    for (name, size, mtime) in sorted {
        desc.push_str(&format!("\nsrc:{name}:{size}:{mtime}"));
    }
    format!("{:016x}", fnv1a64(desc.as_bytes()))
}

fn bin_sizes_line(dir: &Path, world: usize) -> Option<String> {
    let mut line = String::new();
    for r in 0..world {
        let md = fs::metadata(dir.join(format!("rank{r}.bin"))).ok()?;
        line.push_str(&format!("rank{r}.bin={} ", md.len()));
    }
    Some(line.trim_end().to_string())
}

fn env_flag(owl: &str, compat: &str) -> bool {
    std::env::var(owl).as_deref() == Ok("1")
        || std::env::var(compat).as_deref() == Ok("1")
}

/// 会话初始化:开关关闭 → no-op;否则探测镜像并决定 Load/Store 模式。
#[allow(clippy::too_many_arguments)]
pub fn session_init(
    model_dir: &Path,
    world: usize,
    rank: usize,
    source_files: &[PathBuf],
    quant_sig: &str,
    arch: &str,
    dtype: &str,
) {
    if !enabled() || world == 0 {
        return;
    }
    let dir = model_dir.join("owl-load-cache");
    let sources = source_metas(source_files);
    let w4a8 = env_flag("OWL_W4A8", "XINFER_W4A8");
    let digest = canonical_sources_digest(&sources, quant_sig, arch, dtype, world, w4a8);

    let bin_path = dir.join(format!("rank{rank}.bin"));
    let mut hit = false;
    if let Ok(text) = fs::read_to_string(dir.join("checksum.txt")) {
        let lines: Vec<&str> = text.lines().collect();
        let src_ok = lines.first() == Some(&digest.as_str());
        let sizes_ok = lines
            .get(1)
            .map(|want| bin_sizes_line(&dir, world).as_deref() == Some(*want))
            .unwrap_or(false);
        let all_present = (0..world).all(|r| {
            dir.join(format!("rank{r}.bin")).is_file()
                && dir.join(format!("rank{r}.idx.json")).is_file()
        });
        hit = src_ok && sizes_ok && all_present;
    }

    if hit {
        let idx: HashMap<String, Entry> = fs::read(dir.join(format!("rank{rank}.idx.json")))
            .ok()
            .and_then(|b| serde_json::from_slice::<Vec<Entry>>(&b).ok())
            .map(|v| v.into_iter().map(|e| (e.name.clone(), e)).collect())
            .unwrap_or_default();
        if SESSION
            .set(Session {
                dir: dir.clone(),
                rank,
                world,
                idx,
                bin_path: bin_path.clone(),
                store: Mutex::new(Store {
                    entries: Vec::new(),
                    file: None,
                    offset: 0,
                }),
            })
            .is_err()
        {
            return;
        }
        MODE.store(MODE_LOAD, Ordering::Relaxed);
    }

    if mode() == MODE_OFF {
        let _ = fs::create_dir_all(&dir);
        let part = dir.join(format!("rank{rank}.bin.part"));
        let file = fs::File::create(&part).ok();
        let _ = fs::remove_file(dir.join(format!("rank{rank}.idx.json")));
        if SESSION
            .set(Session {
                dir: dir.clone(),
                rank,
                world,
                idx: HashMap::new(),
                bin_path: bin_path.clone(),
                store: Mutex::new(Store {
                    entries: Vec::new(),
                    file,
                    offset: 0,
                }),
            })
            .is_err()
        {
            return;
        }
        MODE.store(MODE_STORE, Ordering::Relaxed);
    }
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "[load-cache] rank{rank}: mode={} dir={}",
            if mode() == MODE_LOAD { "LOAD" } else { "STORE" },
            SESSION
                .get()
                .map(|s| s.dir.display().to_string())
                .unwrap_or_default()
        );
    }
}

/// 张量直读(Load 模式)。返回 shape + 原始小端字节。
pub fn get_raw(tag: &str, tensor: &str) -> Option<(Vec<usize>, Vec<u8>)> {
    if mode() != MODE_LOAD {
        return None;
    }
    let s = session()?;
    let e = s.idx.get(&format!("{tag}::{tensor}"))?;
    let mut f = fs::File::open(&s.bin_path).ok()?;
    f.seek(SeekFrom::Start(e.offset)).ok()?;
    let mut buf = vec![0u8; e.len];
    f.read_exact(&mut buf).ok()?;
    Some((e.shape.clone(), buf))
}

/// 烘焙写入(Store 模式;其余 no-op)。
pub fn put_raw(tag: &str, tensor: &str, shape: Vec<usize>, bytes: &[u8]) {
    if mode() != MODE_STORE {
        return;
    }
    let Some(s) = session() else { return };
    let Ok(mut store) = s.store.lock() else { return };
    let Some(file) = store.file.as_mut() else { return };
    if file.write_all(bytes).is_err() {
        return;
    }
    let (offset, len) = (store.offset, bytes.len());
    store.offset += len as u64;
    store.entries.push(Entry {
        name: format!("{tag}::{tensor}"),
        shape,
        offset,
        len,
    });
}

pub fn put_u32(tag: &str, tensor: &str, shape: Vec<usize>, v: &[u32]) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    put_raw(tag, tensor, shape, &bytes);
}

pub fn put_f32(tag: &str, tensor: &str, shape: Vec<usize>, v: &[f32]) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    put_raw(tag, tensor, shape, &bytes);
}

fn bytes_to_u32(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn bytes_to_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

pub fn get_u32(tag: &str, tensor: &str) -> Option<Vec<u32>> {
    let (_shape, bytes) = get_raw(tag, tensor)?;
    bytes_to_u32(&bytes)
}

pub fn get_f32(tag: &str, tensor: &str) -> Option<Vec<f32>> {
    let (_shape, bytes) = get_raw(tag, tensor)?;
    bytes_to_f32(&bytes)
}

/// 会话收尾(Store:flush bin + idx/meta/checksum;Load:no-op)。
#[allow(clippy::too_many_arguments)]
pub fn session_commit(
    model_dir: &Path,
    world: usize,
    source_files: &[PathBuf],
    quant_sig: &str,
    arch: &str,
    dtype: &str,
) {
    let _ = model_dir;
    if !enabled() || mode() != MODE_STORE {
        return;
    }
    let Some(s) = session() else { return };
    let (entries, file, bin_path, dir, rank) = {
        let Ok(mut store) = s.store.lock() else { return };
        (
            store.entries.clone(),
            store.file.take(),
            s.bin_path.clone(),
            s.dir.clone(),
            s.rank,
        )
    };
    let idx_bytes = serde_json::to_vec(&entries).unwrap_or_default();
    let idx_path = dir.join(format!("rank{rank}.idx.json"));
    let _ = fs::write(&idx_path, idx_bytes);
    drop(file);
    let part = dir.join(format!("rank{rank}.bin.part"));
    if fs::rename(&part, &bin_path).is_err() {
        tracing::warn!("[load-cache] rank{rank}: bin rename failed, cache incomplete");
        return;
    }

    if rank == 0 {
        let sources = source_metas(source_files);
        let meta = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "arch": arch,
            "quant_sig": quant_sig,
            "world": world,
            "sources": sources,
        });
        let tmp = dir.join("meta.json.tmp");
        if fs::write(&tmp, serde_json::to_vec_pretty(&meta).unwrap_or_default()).is_ok() {
            let _ = fs::rename(&tmp, dir.join("meta.json"));
        }
    }

    let mut waited = 0u32;
    while waited < 120_000 {
        if (0..world).all(|r| dir.join(format!("rank{r}.bin")).is_file()) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        waited += 100;
    }
    let sizes = match bin_sizes_line(&dir, world) {
        Some(l) => l,
        None => {
            tracing::warn!("[load-cache] rank{rank}: bins incomplete, checksum not written");
            return;
        }
    };
    if rank == 0 {
        let sources = source_metas(source_files);
        let w4a8 = env_flag("OWL_W4A8", "XINFER_W4A8");
        let digest = canonical_sources_digest(&sources, quant_sig, arch, dtype, world, w4a8);
        let content = format!("{digest}\n{sizes}\n");
        let tmp = dir.join("checksum.txt.tmp");
        if fs::write(&tmp, content).is_ok() {
            let _ = fs::rename(&tmp, dir.join("checksum.txt"));
        }
        tracing::info!("[load-cache] rank0: bake committed at {}", dir.display());
    }
}
