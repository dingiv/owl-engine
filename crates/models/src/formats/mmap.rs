//! 只读映射底座(两源共享:SafeTensorsSource / W4A16Source)。
//!
//! libc::mmap 只读私有映射(file-backed clean page)+ **全 dtype** 条目
//! 索引(字节区间 + dtype + shape,堆上 KB 级)—— 打开即返回,零数据
//! 拷贝;消费侧按需转换,DONTNEED 还页(RSS 恒定 = 在途份)。

use crate::contract::ModelError;
use std::collections::HashMap;
use std::path::Path;

/// mmap 只读映射(Linux,MAP_PRIVATE;Drop = munmap)。
/// 页缓存支撑、零堆拷贝 —— 替代整文件堆读(file_loader::read)的 mmap 形态;
/// 页为 file-backed clean page,内存压力下内核可直接回收。
pub(crate) struct Mmap {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    /// 保留 fd 供**后台预取**用 `pread` 填充页缓存（mmap 本身不持 fd）。
    file: std::fs::File,
}

impl Mmap {
    pub(crate) fn open(path: &Path) -> Result<Self, ModelError> {
        use std::os::unix::io::AsRawFd;
        let file = owl_shared::file_loader::open(path)
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
            file,
        })
    }
}

impl Mmap {
    /// **后台预取一段区间**（`pread` 填充页缓存；与 `madvise(WILLNEED)`
    /// 不同，`pread` 的可控队列深度更稳定，冷盘实测 ~2.1GB/s/流）。
    /// 只读、可重入；错误由调用方记账，不改变映射内容。
    pub fn readahead(&self, start: usize, len: usize) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        const CHUNK: usize = 4 << 20;
        let mut buf = vec![0u8; CHUNK.min(len.max(1))];
        let mut off = start;
        let end = start.saturating_add(len).min(self.len);
        while off < end {
            let n = (end - off).min(buf.len());
            let got = self.file.read_at(&mut buf[..n], off as u64)?;
            if got == 0 {
                break;
            }
            off += got;
        }
        Ok(())
    }

    /// 归还已消费区间映射页(页对齐向下/向上取整;只读私有映射,
    /// DONTNEED 后再访问会从文件重新缺页,数据无恙)
    pub fn dontneed(&self, start: usize, len: usize) {
        let page = 4096usize;
        let s = start / page * page;
        let e = (start + len).div_ceil(page) * page;
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

/// 全 dtype 原始条目(量化源共享底座用;含 shape 供尺寸推导)
#[derive(Clone)]
pub(crate) struct RawEntry {
    pub map_ix: usize,
    pub start: usize,
    pub nbytes: usize,
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
}

impl RawEntry {
    /// 天然元素宽度(字节)
    pub(crate) fn esz(&self) -> usize {
        use safetensors::Dtype as S;
        match self.dtype {
            S::F64 | S::I64 | S::U64 => 8,
            S::F32 | S::I32 | S::U32 => 4,
            S::F16 | S::BF16 | S::I16 | S::U16 => 2,
            S::U8 | S::I8 | S::BOOL | S::F8_E5M2 | S::F8_E4M3 | S::F8_E8M0 | S::F8_E4M3FNUZ
            | S::F8_E5M2FNUZ | S::F4 | S::F6_E2M3 | S::F6_E3M2 => 1,
            // safetensors::Dtype 是 #[non_exhaustive]:未列出的窄格式按 1B 兜底。
            // 两条入口(量化源 native_dtype / convert 臂)先经 dtype 白名单筛过,
            // 未支持 dtype 到不了本函数(dtype 白名单失配 = 上游响亮 None)。
            _ => 1,
        }
    }
}

/// **公共字节拷贝口**(页缓存直读 → pinned 租约/堆缓冲):大块分线程 memcpy。
///
/// 早测账(本机):单线程 memcpy ≈19GB/s、四线程 ≈55GB/s,而 27B 整装
/// CPU 侧曾是全载最大单相 ⇒ 装载域所有"原样字节"搬运统一走本口
/// (与 `safetensors::copy_native_into` 同款,2026-10-16 提速同源)。
pub(crate) fn copy_par(src: &[u8], dst: &mut [u8]) {
    assert!(dst.len() >= src.len(), "copy_par: 目标缓冲 {} < 源 {}", dst.len(), src.len());
    let dst = &mut dst[..src.len()];
    const PAR_THRESHOLD: usize = 8 << 20;
    if src.len() >= PAR_THRESHOLD {
        let chunk = src.len().div_ceil(4);
        let ds: Vec<&mut [u8]> = dst.chunks_mut(chunk).collect();
        let ss: Vec<&[u8]> = src.chunks(chunk).collect();
        std::thread::scope(|sc| {
            for (d, s) in ds.into_iter().zip(ss) {
                sc.spawn(move || d.copy_from_slice(s));
            }
        });
    } else {
        dst.copy_from_slice(src);
    }
}




/// mmap 源共享底座:打开目录下全部 `*.safetensors`(文件名字典序;
/// 分片键天然互斥,无需 index.json;单文件仓同样适用),登记**全 dtype**
/// 条目索引(含 shape),零数据拷贝。调用方自行做 dtype/业务校验
/// (f16 基线只收 F32/BF16;量化源 permissive)。
pub(crate) fn open_raw_index(
    dir: &Path,
) -> Result<(Vec<Mmap>, HashMap<String, RawEntry>), ModelError> {
    let mut paths: Vec<_> = owl_shared::file_loader::read_dir(dir)
        .map_err(|e| ModelError::Msg(format!("safetensors: 读目录 {dir:?}: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(ModelError::Msg(format!(
            "safetensors: {dir:?} 无 .safetensors 文件"
        )));
    }
    let mut maps = Vec::new();
    let mut index = HashMap::new();
    for p in paths {
        let t0 = std::time::Instant::now();
        let mmap = Mmap::open(&p)?;
        owl_shared::log_info!("st", "{p:?} mmap {}MB ({:.1?})", mmap.len() / 1_000_000, t0.elapsed());
        let map_ix = maps.len();
        let base = mmap.as_ptr() as usize;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| ModelError::Msg(format!("safetensors: 解析 {p:?}: {e}")))?;
        for (name, t) in st.iter() {
            let start = t.data().as_ptr() as usize - base;
            index.insert(
                name.to_string(),
                RawEntry {
                    map_ix,
                    start,
                    nbytes: t.data().len(),
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                },
            );
        }
        maps.push(mmap);
    }
    Ok((maps, index))
}

