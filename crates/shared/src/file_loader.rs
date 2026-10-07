//! 文件读写统一入口(file loader;2026-10-10 clean-code 轮立)。
//!
//! **定位**:应用运行时的所有文件 I/O 走本模块 —— 现为 `std::fs` 薄包装
//! (零行为差异),收口为未来统一控制点(访问记账 / 路径沙箱 / 只读纪律)
//! 留出的唯一入口。新增文件 I/O 一律经此,禁止散点直呼 `std::fs::*`。
//!
//! **边界**:
//! - 构建脚本(`build.rs`)例外 —— 构建期基础设施,不经运行时读取面;
//! - mmap 装载经 [`open`] 取句柄后自管映射(零拷贝路径不做重复包装);
//! - 网络 IO(TcpStream reader)不属本模块。

use std::fs::{File, Metadata, ReadDir};
use std::io::Result;
use std::path::{Path, PathBuf};

/// 整文件读入字节(权重小文件/配置/golden)
pub fn read(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    std::fs::read(path)
}

/// 整文件读入 UTF-8(tokenizer.json 等)
pub fn read_to_string(path: impl AsRef<Path>) -> Result<String> {
    std::fs::read_to_string(path)
}

/// 写文件(截断式;调试转储/golden 生成)
pub fn write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> Result<()> {
    std::fs::write(path, contents)
}

/// 递归建目录
pub fn create_dir_all(path: impl AsRef<Path>) -> Result<()> {
    std::fs::create_dir_all(path)
}

/// 删文件(测试清场;不存在时报错,调用方按需 `let _ =` 显式吞)
pub fn remove_file(path: impl AsRef<Path>) -> Result<()> {
    std::fs::remove_file(path)
}

/// 元数据(存在性 + 尺寸)
pub fn metadata(path: impl AsRef<Path>) -> Result<Metadata> {
    std::fs::metadata(path)
}

/// 存在性(不区分文件/目录;IO 错误按不存在处置,与 std 语义一致)
pub fn exists(path: impl AsRef<Path>) -> bool {
    path.as_ref().exists()
}

/// 列目录(调用方自滤扩展名/排序)
pub fn read_dir(path: impl AsRef<Path>) -> Result<ReadDir> {
    std::fs::read_dir(path)
}

/// 开句柄(mmap 装载等需要 File 的路径)
pub fn open(path: impl AsRef<Path>) -> Result<File> {
    File::open(path)
}

/// 路径拼接便利(调用点免 `PathBuf::from(x).join(y)` 样板)
pub fn join(base: impl AsRef<Path>, rest: &str) -> PathBuf {
    base.as_ref().join(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pid + 计数唯一临时路径(测试自清场)
    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "owl_file_loader_{}_{}_{tag}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    #[test]
    fn write_read_roundtrip_and_remove() {
        let p = tmp_path("roundtrip");
        write(&p, b"hello owl").expect("write");
        assert!(exists(&p));
        assert_eq!(read(&p).expect("read"), b"hello owl");
        assert_eq!(read_to_string(&p).expect("read_to_string"), "hello owl");
        assert_eq!(metadata(&p).expect("meta").len(), 9);
        remove_file(&p).expect("remove");
        assert!(!exists(&p));
    }

    #[test]
    fn create_dir_all_and_open() {
        let dir = tmp_path("nested_dir");
        let nested = join(&dir, "sub");
        create_dir_all(&nested).expect("mkdir");
        let f = join(&nested, "f.txt");
        write(&f, b"x").expect("write");
        assert!(open(&f).is_ok());
        assert_eq!(read_dir(&dir).expect("read_dir").count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
