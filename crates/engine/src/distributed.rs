//! 分布式标识(从 xinfer `models/layers/distributed.rs` 搬运,
//! 只取非 nccl 分支的 `Id`;nccl 路径随 A2.5 自研通信另行设计)。

#[derive(Debug, Clone, Copy)]
pub struct Id {
    pub internal: [::core::ffi::c_char; 128usize],
}

impl Id {
    pub fn as_bytes(&self) -> &[u8] {
        // Safe reinterpretation of `c_char` as `u8`
        unsafe { std::slice::from_raw_parts(self.internal.as_ptr() as *const u8, 128) }
    }
}
