//! 只读映射底座(两源共享:SafeTensorsSource / W4A16Source)。
//!
//! libc::mmap 只读私有映射(file-backed clean page)+ **全 dtype** 条目
//! 索引(字节区间 + dtype + shape,堆上 KB 级)—— 打开即返回,零数据
//! 拷贝;消费侧按需转换,DONTNEED 还页(RSS 恒定 = 在途份)。

use crate::contract::ModelError;
use std::collections::HashMap;
use std::path::Path;

/// mmap 只读映射(Linux,MAP_PRIVATE;Drop = munmap)。
/// 页缓存支撑、零堆拷贝 —— 替代 std::fs::read 的整文件堆读;
/// 页为 file-backed clean page,内存压力下内核可直接回收。
pub(crate) struct Mmap {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

impl Mmap {
    pub(crate) fn open(path: &Path) -> Result<Self, ModelError> {
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

/// 全 dtype 原始条目(量化源共享底座用;含 shape 供尺寸推导)
#[derive(Clone)]
pub(crate) struct RawEntry {
    pub map_ix: usize,
    pub start: usize,
    pub nbytes: usize,
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
}




/// mmap 源共享底座:打开目录下全部 `*.safetensors`(文件名字典序;
/// 分片键天然互斥,无需 index.json;单文件仓同样适用),登记**全 dtype**
/// 条目索引(含 shape),零数据拷贝。调用方自行做 dtype/业务校验
/// (f16 基线只收 F32/BF16;量化源 permissive)。
pub(crate) fn open_raw_index(
    dir: &Path,
) -> Result<(Vec<Mmap>, HashMap<String, RawEntry>), ModelError> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
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
        eprintln!("[st] {p:?} mmap {}MB ({:.1?})", mmap.len() / 1_000_000, t0.elapsed());
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

