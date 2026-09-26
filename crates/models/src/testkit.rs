//! 测试套件(仅测试构建;`#[cfg(test)]`)。
//!
//! 就近测试的公共件:层文件内 `#[cfg(test)] mod tests` 直接
//! `use crate::testkit::*`。四组能力:
//!
//! - **声明收割**:`f32b` / `f32_of` / `harvest`(任一 face 求值 + dtoh);
//! - **HF 进程协议**:`st_write` / `st_read` / `hf_python` /
//!   `parity_enabled` / `skip_note`(python 与层同目录就近:`src/layers/<layer>.py`);
//! - **GPU 门控**:`gpu_enabled` / `gpu_client`(OWL_TEST_DEVICE 约定);
//! - **数值对拍**:`assert_close`(allclose 带坐标报错)。

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::contract::DeviceClient;
use crate::TensorOps;

pub type Src = std::collections::HashMap<String, Vec<f32>>;

pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("owl_testkit_{name}_{}.safetensors", std::process::id()))
}

pub fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn f32_of(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 声明求值 + dtoh 收割(测试/对拍公共路径)
pub async fn harvest<D: DeviceClient>(face: &mut D, t: &TensorOps) -> Vec<f32> {
    let n: usize = t.shape().iter().product();
    let bytes = crate::interpreters::eval_ops(t.step(), face)
        .await
        .expect("eval");
    let mut buf = vec![0u8; n * 4];
    face.dtoh(&bytes, &mut buf).await.expect("dtoh");
    f32_of(&buf)
}

/// f32 张量集 → safetensors 文件(HF baseline 交换)
pub fn st_write(path: &Path, tensors: &[(&str, &[f32], Vec<usize>)]) {
    use safetensors::tensor::TensorView;
    // 两段式:字节先驻留,再建借用视图(View 借数据,不能指向临时)
    let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, v, shape)| {
            (
                name.to_string(),
                shape.clone(),
                v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            )
        })
        .collect();
    let data: Vec<(String, TensorView)> = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView 构造"),
            )
        })
        .collect();
    safetensors::serialize_to_file(data, None, path).expect("safetensors 写入");
}

/// safetensors 文件 → 具名 f32 张量
pub fn st_read(path: &Path, name: &str) -> Vec<f32> {
    let buf = std::fs::read(path).expect("safetensors 读入");
    let st = safetensors::SafeTensors::deserialize(&buf).expect("safetensors 解析");
    let t = st.tensor(name).expect("张量名");
    assert_eq!(t.dtype(), safetensors::Dtype::F32, "{name} 应为 F32");
    f32_of(t.data())
}

/// fork 层旁的 python 金标准脚本(`src/layers/<layer>.py`,与 .rs 就近;
/// uv 项目根 = crate 根);返回 stdout(manifest json)
pub fn hf_python(script: &str, args: &[&str]) -> String {
    let script_path = manifest_dir().join("src/layers").join(script);
    let out = Command::new("uv")
        .arg("run")
        .arg("python")
        .arg(script_path)
        .args(args)
        .current_dir(manifest_dir())
        .output()
        .expect("spawn uv(python env 未就绪?先在 crates/models 下 `uv sync`)");
    if !out.status.success() {
        panic!(
            "python 脚本失败({script}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// HF parity 门控(OWL_HF_PARITY=1;平时 cargo test 零 python 依赖)
pub fn parity_enabled() -> bool {
    std::env::var("OWL_HF_PARITY")
        .map(|v| v == "1")
        .unwrap_or(false)
}

pub fn skip_note() {
    eprintln!("skip: OWL_HF_PARITY=1 未开(首次会同步 torch CPU 轮)");
}

/// GPU 后端门控(OWL_TEST_DEVICE 已设;与 owl-cuda 测试同一约定)
pub fn gpu_enabled() -> bool {
    std::env::var_os("OWL_TEST_DEVICE").is_some()
}

/// GPU face(owl-cuda actor;门控已过才调用)
pub async fn gpu_client() -> owl_cuda::GpuClient {
    owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(
        owl_cuda::test_device_ordinal(),
    ))
    .expect("gpu server boot(OWL_TEST_DEVICE 指向空闲卡)")
}

/// allclose 对拍(带坐标报错)
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: 长度不符");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "[{tag}][{i}] {g} vs {w}(atol {atol})"
        );
    }
}
