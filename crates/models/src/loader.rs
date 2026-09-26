//! safetensors 权重源(通用;模型无关)。
//!
//! **流式形态(2026-09-26 M-e 性能,用户裁决:读一点装一点)**:
//! - 文件 = libc::mmap 只读映射(打开 27µs;页缓存支撑零堆拷贝);
//! - 张量只登记**条目索引**(字节区间 + dtype,堆上 KB 级);
//! - `take(key)` = 查到才从映射区转换出 f32 **并取走所有权** ——
//!   上传后随调用方消亡,主机驻留只剩"在途"份(0.8B 全量常驻 3.4GB
//!   的旧形态作废;visual/mtp 153 张量永不转换,0.5GB 根本不发生)。
//! 量化格式另立 source,勿在此堆积。
//!
//! 形状事实:conv1d 等 3D 权重按扁平字节直读(row-major 连续,
//! [6144,1,4] ≡ [6144,4]),shape 元数据由层侧 Want 声明,此处不搬。

use crate::contract::ModelError;
use crate::module::WeightSource;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// mmap 只读映射(Linux,MAP_PRIVATE;Drop = munmap)。
/// 页缓存支撑、零堆拷贝 —— 替代 std::fs::read 的整文件堆读;
/// 页为 file-backed clean page,内存压力下内核可直接回收。
struct Mmap {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

impl Mmap {
    fn open(path: &Path) -> Result<Self, ModelError> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path)
            .map_err(|e| ModelError::Msg(format!("mmap: open {path:?}: {e}")))?;
        let len = file
            .metadata()
            .map_err(|e| ModelError::Msg(format!("mmap: metadata {path:?}: {e}")))?
            .len() as usize;
        if len == 0 {
            return Err(ModelError::Msg(format!("mmap: {path:?} 空文件")));
        }
        // SAFETY:fd 合法、len 来自 metadata;只读私有映射,进程存活期稳定
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error();
            return Err(ModelError::Msg(format!("mmap: {path:?}: {errno}")));
        }
        // 顺序读提示:内核加大预读窗口(转换流式顺序扫描整仓)
        unsafe {
            libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        }
        // SAFETY:mmap 成功返回值非空(MAP_FAILED 已排除)
        Ok(Self {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr.cast::<u8>()) },
            len,
        })
    }
}

impl Mmap {
    /// 归还已消费区间映射页(页对齐向下/向上取整;只读私有映射,
    /// DONTNEED 后再访问会从文件重新缺页,数据无恙)
    pub fn dontneed(&self, start: usize, len: usize) {
        let page = 4096usize;
        let s = start / page * page;
        let e = ((start + len) + page - 1) / page * page;
        let (a, b) = (s.min(self.len), e.min(self.len));
        if a < b {
            unsafe {
                libc::madvise(self.ptr.as_ptr().add(a) as *mut _, b - a, libc::MADV_DONTNEED);
            }
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY:ptr/len 来自成功的 mmap,未被改动
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

impl std::ops::Deref for Mmap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY:映射区 [ptr, ptr+len) 在 munmap 前始终有效且只读
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

// 只读文件映射:内容稳定、无内部可变性
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

/// 流式权重源:mmap + 条目索引(堆上 KB 级),数据按需转换、取走即弃。
pub struct SafeTensorsSource {
    maps: Vec<Mmap>,
    /// 待取条目(take 即移除;锁只护索引,转换在锁外读映射区)
    index: Mutex<HashMap<String, Entry>>,
}

/// 张量条目(映射区内字节区间 + dtype)
#[derive(Clone, Copy)]
struct Entry {
    map_ix: usize,
    start: usize,
    nbytes: usize,
    dtype: safetensors::Dtype,
}

impl Entry {
    /// 元素宽(字节):F32=4,BF16=2
    fn esz(&self) -> usize {
        match self.dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::BF16 => 2,
            _ => unreachable!("open_dir 只登记 F32/BF16"),
        }
    }
}

impl SafeTensorsSource {
    /// 打开目录下全部 `*.safetensors`(文件名字典序;分片键天然互斥,
    /// 无需 index.json;单文件仓同样适用)。只建映射 + 索引,零数据拷贝。
    pub fn open_dir(dir: &Path) -> Result<Self, ModelError> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| ModelError::Msg(format!("safetensors: 读目录 {dir:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(ModelError::Msg(format!("safetensors: {dir:?} 无 .safetensors 文件")));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for p in paths {
            let t0 = std::time::Instant::now();
            let mmap = Mmap::open(&p)?;
            eprintln!("[st] {p:?} mmap {}MB ({:.1?})", mmap.len() / 1_000_000, t0.elapsed());
            let map_ix = maps.len();
            let base = mmap.as_ptr() as usize;
            let st = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| ModelError::Msg(format!("safetensors: 解析 {p:?}: {e}")))?;
            for (name, t) in st.iter() {
                let dtype = t.dtype();
                if !matches!(dtype, safetensors::Dtype::F32 | safetensors::Dtype::BF16) {
                    return Err(ModelError::Msg(format!(
                        "safetensors: {p:?}:{name} dtype {dtype:?} 不支持(量化另立 source)"
                    )));
                }
                let start = t.data().as_ptr() as usize - base;
                index.insert(
                    name.to_string(),
                    Entry { map_ix, start, nbytes: t.data().len(), dtype },
                );
            }
            maps.push(mmap);
        }
        Ok(Self { maps, index: Mutex::new(index) })
    }

    pub fn len(&self) -> usize {
        self.index.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SafeTensorsSource {
    /// 条目区间 → f32(不持锁;F32 直读 / BF16 升位)。
    /// 消费完立即 `madvise(MADV_DONTNEED)` 归还映射页 —— 进程 RSS
    /// 恒定在"在途块"量级,不随仓大小增长(读一点装一点)。
    fn convert_range(&self, e: &Entry, offset_elems: usize, len: usize) -> Vec<f32> {
        let esz = e.esz();
        let s = e.start + offset_elems * esz;
        let nbytes = len * esz;
        let bytes = &self.maps[e.map_ix][s..s + nbytes];
        let converted = match e.dtype {
            safetensors::Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            _ => unreachable!("open_dir 只登记 F32/BF16"),
        };
        self.maps[e.map_ix].dontneed(s, nbytes);
        converted
    }
}

impl WeightSource for SafeTensorsSource {
    fn elem_len(&self, key: &str) -> Option<usize> {
        let e = self.index.lock().unwrap().get(key).cloned()?;
        Some(e.nbytes / e.esz())
    }

    /// 整取(F32 直读 / BF16 升位),条目即从索引移除。
    /// 流式装载主路径走 `take_range`(分块,条目保留)。
    fn take(&self, key: &str) -> Option<Vec<f32>> {
        let e = self.index.lock().unwrap().remove(key)?;
        // 元素数按条目真实位宽(F5 修:原 /4 硬编码对 bf16 条目
        // 返回一半元素 —— 直转路径首次踩中;分块路径按 len 显式未暴露)
        Some(self.convert_range(&e, 0, e.nbytes / e.esz()))
    }

    /// 区间转换(mmap 直读,**不移除条目** —— 分块上传同键多块重复取)
    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Option<Vec<f32>> {
        let e = self.index.lock().unwrap().get(key).cloned()?;
        if (offset_elems + len) * e.esz() > e.nbytes {
            return None;
        }
        Some(self.convert_range(&e, offset_elems, len))
    }

    /// 分块转换**直写** dst(mmap → 解码 → 目标缓冲,零中间 Vec),
    /// 消费完归还映射页(DONTNEED,读一点装一点)
    fn convert_chunk_into(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [f32],
    ) -> Option<()> {
        let e = match self.index.lock().unwrap().get(key).cloned() {
            Some(e) => e,
            None => {
                eprintln!("[st][DIAG] 索引未命中: {key}");
                return None;
            }
        };
        let esz = e.esz();
        if (offset_elems + len) * esz > e.nbytes {
            eprintln!(
                "[st][DIAG] OOB: {key} off{offset_elems} len{len} esz{esz} nbytes{}",
                e.nbytes
            );
            return None;
        }
        let s = e.start + offset_elems * esz;
        let bytes = &self.maps[e.map_ix][s..s + len * esz];
        match e.dtype {
            safetensors::Dtype::F32 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(4)) {
                    *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            safetensors::Dtype::BF16 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
                    *d = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                }
            }
            _ => return None,
        }
        self.maps[e.map_ix].dontneed(s, len * esz);
        Some(())
    }
}
