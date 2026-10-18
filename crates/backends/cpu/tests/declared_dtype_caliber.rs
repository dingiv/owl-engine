//! **声明 dtype 口径门**(第三轮审查 §A1 的守门用例;2026-10-17)。
//!
//! 病史(实测三连,修前):
//! 1. `memset_zero` / `copy_block_at` **硬除 4** ⇒ F16/BF16 声明块**静默半做**
//!    (8 元素清零只落 4 槽;拷贝 4 元素只落 2);
//! 2. `swap` 默认实现(逐片 `dtoh_at` + `upload_pinned`)跨两套口径 ⇒
//!    F16 块字节级往返**静默腐蚀**([1..8] → [0,1.875,0,2,…]);
//! 3. `dtoh` / `dtoh_at` 只认 f32 编码 ⇒ `testkit::harvest_f16` 那类
//!    **真卡口径**(n×2 字节)在 CPU 面被响亮拒 —— F16 链路无法在无卡机上测。
//!
//! 口径(修后,唯一一条):**设备字节 = 声明 dtype 字节**(`len × dsz`),
//! 值平面恒 f32(一槽一元素),所有入参字节数经 `slots_at` 折算,
//! 非整除/越界 = 响亮拒(与真卡 `GpuServer::check_range` 同义)。
//!
//! ⚠️ 这四个用例若变红,意味着 mock 又开始"说谎" —— CPU 面跑绿不再等于
//! 真卡语义成立(测试假绿/假红)。

use owl_cpu::{CpuFace, MockLimits};
use owl_iface::contract::{
    Bytes, ConnSlot, Connector, DeviceClient, Dtype, ModelError, SwapDir, SwapPiece,
};
use std::collections::HashMap;

/// 最小宿主连接器(内存槽表)。
#[derive(Default)]
struct MemConn {
    slots: HashMap<u64, Vec<u8>>,
}

impl Connector for MemConn {
    fn put(&mut self, slot: ConnSlot, off: usize, bytes: &[u8]) -> Result<(), ModelError> {
        let buf = self.slots.entry(slot.0).or_default();
        if buf.len() < off + bytes.len() {
            buf.resize(off + bytes.len(), 0);
        }
        buf[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn get(&self, slot: ConnSlot, off: usize, out: &mut [u8]) -> Result<(), ModelError> {
        let buf = self
            .slots
            .get(&slot.0)
            .ok_or_else(|| ModelError::Msg(format!("槽 {} 不存在", slot.0)))?;
        out.copy_from_slice(&buf[off..off + out.len()]);
        Ok(())
    }
    fn tier_name(&self) -> &'static str {
        "mem"
    }
    fn unbox(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// f16 编码(真卡 `dtoh` 口径:每元素 2 字节)。
fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
}

fn f16_of(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

/// 以**声明 dtype 字节**写入一整个 F16 块(生产调用点同口径)。
async fn put_f16(f: &mut CpuFace, b: &Bytes, vals: &[f32]) {
    let src = f16_bytes(vals);
    let mut lease = f.alloc_pinned(src.len(), "t").await.expect("租约");
    lease.slice_bytes_mut()[..src.len()].copy_from_slice(&src);
    f.upload_pinned(lease, b, 0, src.len()).await.expect("上传");
}

/// 以**声明 dtype 字节**收割整块(F16)。
async fn get_f16(f: &mut CpuFace, b: &Bytes) -> Vec<f32> {
    let mut buf = vec![0u8; b.len * Dtype::F16.size_bytes()];
    f.dtoh(b, &mut buf).await.expect("dtoh");
    f16_of(&buf)
}

/// 门 1(F16 声明块):`memset_zero` 按声明字节全量清零(**修前半做**)。
#[tokio::test]
async fn memset_zero_uses_declared_dtype_bytes() {
    let mut f = CpuFace::new();
    let b = f.alloc(Dtype::F16, 8, "t").await.expect("alloc");
    put_f16(&mut f, &b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).await;
    assert_eq!(get_f16(&mut f, &b).await, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "上传");
    // 调用点口径(models/interpreters/eval.rs:731)= n_elems × dtype.size_bytes()
    f.memset_zero(&b, 8 * Dtype::F16.size_bytes()).await.expect("清零");
    assert_eq!(get_f16(&mut f, &b).await, vec![0.0; 8], "整块清零");
}

/// 门 2(F16 声明块):`copy_block_at` = **字节级**拷贝(偏移/长度按声明 dtype)。
#[tokio::test]
async fn copy_block_at_is_byte_level_on_f16() {
    let mut f = CpuFace::new();
    let src = f.alloc(Dtype::F16, 8, "src").await.expect("src");
    let dst = f.alloc(Dtype::F16, 8, "dst").await.expect("dst");
    put_f16(&mut f, &src, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).await;
    let esz = Dtype::F16.size_bytes();
    // src[0..4] → dst[2..6](长度 4 元素 = 8 字节;与 load.rs:922 同口径)
    f.copy_block_at(&src, 0, &dst, 2 * esz, 4 * esz).await.expect("拷贝");
    assert_eq!(
        get_f16(&mut f, &dst).await,
        vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
        "4 元素按 8 字节落位"
    );
}

/// 门 3(F16 声明块):字节级往返(默认 `swap` 实现)必须**逐字节等价**
/// —— store 的整条纪律就是 byte-exact,而生产 KV 块正是 f16(**修前静默腐蚀**)。
#[tokio::test]
async fn swap_roundtrip_f16_is_byte_exact() {
    let mut f = CpuFace::new();
    let n = 8usize;
    let a = f.alloc(Dtype::F16, n, "a").await.expect("alloc a");
    let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    put_f16(&mut f, &a, &vals).await;

    let piece = SwapPiece {
        block: a.id,
        dev_off: 0,
        conn_off: 0,
        len: n * Dtype::F16.size_bytes(),
        slot: ConnSlot(0),
    };
    let conn = Box::new(MemConn::default());
    let conn = f.swap(SwapDir::Out, std::slice::from_ref(&piece), conn).await.expect("换出");
    let b = f.alloc(Dtype::F16, n, "b").await.expect("alloc b");
    let piece_b = SwapPiece { block: b.id, ..piece };
    let _ = f.swap(SwapDir::In, std::slice::from_ref(&piece_b), conn).await.expect("换回");
    assert_eq!(get_f16(&mut f, &b).await, vals.to_vec(), "字节级往返等价");
}

/// 门 4:收割按**声明 dtype 编码**;长度不符 = 响亮拒(不静默截断)。
#[tokio::test]
async fn dtoh_encodes_declared_dtype_and_refuses_wrong_len() {
    let mut f = CpuFace::new();
    let b = f.alloc(Dtype::F16, 4, "t").await.expect("alloc");
    put_f16(&mut f, &b, &[1.0, 2.0, 3.0, 4.0]).await;
    let mut good = vec![0u8; 4 * Dtype::F16.size_bytes()];
    f.dtoh(&b, &mut good).await.expect("8B 收割");
    assert_eq!(f16_of(&good), vec![1.0, 2.0, 3.0, 4.0], "f16 位型收割");
    // 旧口径(4B/元素)现在响亮拒 —— `testkit::harvest_f16` 正好走 n×2
    let mut wrong = vec![0u8; 4 * 4];
    let err = f.dtoh(&b, &mut wrong).await.expect_err("长度须按声明 dtype");
    assert!(format!("{err}").contains("声明"), "{err}");
    // 非整除的 memset = 响亮拒(修前静默少做)
    assert!(f.memset_zero(&b, 3).await.is_err(), "3B 非 f16 整除");
}

/// 门 5:`launch` 的 `Arg::BlockSlice` 折算 = 声明 dtype;**越界 = 响亮拒**
/// (修前:硬除 4 + 越界静默保留整块,还 `shape = vec![n]` 撒谎)。
#[tokio::test]
async fn block_slice_arg_is_bounds_checked() {
    use owl_iface::contract::{Arg, KernelSource, LaunchMsg};

    fn msg(src: u64, out: u64, off: usize, elems: usize) -> LaunchMsg {
        LaunchMsg {
            kernel: KernelSource { name: "owl_silu_f32".into(), source: String::new() },
            args: vec![
                Arg::BlockSlice { id: src, byte_offset: off as u64, elems: elems as u64, dtype: Dtype::F32, rank: 1 },
                Arg::Block { id: out, dtype: Dtype::F32, rank: 1 },
            ],
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            out_elems: elems,
        }
    }

    let mut f = CpuFace::new();
    let src = f.alloc(Dtype::F32, 8, "src").await.expect("src");
    let out = f.alloc(Dtype::F32, 4, "out").await.expect("out");
    f.write_block_f32(&src, 0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).await.expect("写");
    // 合法:src[4..8] → out(silu 单调,这里只验落位与长度)
    f.launch(msg(src.id, out.id, 16, 4)).await.expect("合法切片");
    let mut buf = vec![0u8; 16];
    f.dtoh(&out, &mut buf).await.expect("收割");
    let got: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let silu = |x: f32| x / (1.0 + (-x).exp());
    for (i, v) in [5.0f32, 6.0, 7.0, 8.0].iter().enumerate() {
        assert!((got[i] - silu(*v)).abs() < 1e-6, "第 {i} 槽 = {} (want silu({v}))", got[i]);
    }
    // 越界:src[6..12] 超出 8 槽 ⇒ 响亮拒(修前静默返回整块并谎报 shape)
    let err = f.launch(msg(src.id, out.id, 24, 6)).await.expect_err("越界必拒");
    assert!(format!("{err}").contains("越界"), "{err}");
}

/// **门 6:未初始化块的诚实度**(2026-10-17 审查 §三轮-C1)。
///
/// 默认档 = 与 `alloc` 同(清零,存量行为不变);`poison_uninit` 置位 ⇒
/// 新块填 NaN ⇒ "池复用忘 memset / 依赖 alloc 恒零"这类缺陷**可见**。
#[tokio::test]
async fn uninit_poison_is_optin_and_visible() {
    // 默认档:未初始化 = 清零(现状行为,逐条不变)
    let mut f = CpuFace::new();
    let b = f.alloc_uninit(Dtype::F32, 4, "t").await.expect("alloc_uninit");
    let mut buf = vec![0u8; 16];
    f.dtoh(&b, &mut buf).await.expect("dtoh");
    let vals: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(vals, vec![0.0; 4], "默认档 == alloc(清零)");

    // opt-in:NaN 位型(与自身不相等)
    let mut g = CpuFace::mock(MockLimits { poison_uninit: true, ..Default::default() });
    let c = g.alloc_uninit(Dtype::F32, 4, "t").await.expect("alloc_uninit(mock)");
    let mut buf2 = vec![0u8; 16];
    g.dtoh(&c, &mut buf2).await.expect("dtoh");
    let vals2: Vec<f32> = buf2.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
    for v in &vals2 {
        assert!(v.is_nan(), "毒值应为 NaN,得 {v}");
    }
    // 而 alloc(Zeros 语义)不受影响
    let z = g.alloc(Dtype::F32, 4, "t").await.expect("alloc");
    let mut buf3 = vec![0u8; 16];
    g.dtoh(&z, &mut buf3).await.expect("dtoh");
    assert!(buf3.iter().all(|x| *x == 0), "alloc 仍恒零");
}
