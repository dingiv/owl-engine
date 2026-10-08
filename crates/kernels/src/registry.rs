//! 算子注册表 —— **名字字面量全库唯一住址**(P0 名字无关的名字源)。
//!
//! [`KernelSpec`] = 胖算子契约(原子单位;每个在用算子一个 struct 实现)。
//! [`OpRegistry`] = boot 装配 + server 唯一入口(名字无关)。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};

/// 胖算子契约(原子单位)。
///
/// init = boot 装配(装载/校验/预钉,失败 = Err 拒启);
/// run = 唯一执行入口(Result 全链);
/// validate = 参数校验(interpreter 发射前强制)。
pub trait KernelSpec: Send {
    fn id(&self) -> OpId;
    fn names(&self) -> Vec<&'static str> {
        vec![self.id().0]
    }
    fn linkage(&self) -> crate::contract::Linkage;
    fn init(&mut self, res: &mut dyn crate::device::DeviceRes, exec: &mut crate::device::Exec) -> Result<(), OpError>;
    fn validate(&self) -> Result<(), OpError>;
    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, exec: &mut crate::device::Exec) -> Result<Bytes, OpError>;
}

/// 注册表(编译期穷举;分派在注册表内部,server 主循环零算子知识)。
pub struct OpRegistry {
    families: Vec<Box<dyn KernelSpec>>,
}

impl OpRegistry {
    /// boot 一步:全家族装配 + init(P4 门;失败 = Err 拒启)。
    pub fn boot(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Self, OpError> {
        Self::init_all(crate::server::all_families(), res, exec)
    }

    pub fn init_all(
        mut families: Vec<Box<dyn KernelSpec>>,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Self, OpError> {
        for f in families.iter_mut() {
            f.init(res, exec)?;
        }
        Ok(Self { families })
    }

    /// server 唯一算子入口:名字 → 家族 runtime 的分派在注册表内部。
    pub fn execute(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        let f = self
            .families
            .iter_mut()
            .find(|f| f.names().iter().any(|n| *n == msg.kernel.name))
            .ok_or_else(|| OpError::Contract {
                op: msg.kernel.name.clone(),
                field: "name",
                expect: "已注册家族".into(),
                got: format!("{}(未注册)", msg.kernel.name),
            })?;
        f.run(msg, res, exec)
    }
}

pub fn is_foreign_name(name: &str) -> bool {
    let mut foreign = name == crate::client::gdn_chunked::GDN_CHUNKED
        || name == crate::client::gdn_scalar::GDN_SCALAR;
    #[cfg(feature = "cublas")]
    {
        foreign = foreign
            || name == crate::client::cublas::GEMM_F16
            || name == crate::client::cublas::GEMM_BF16;
    }
    #[cfg(feature = "marlin")]
    {
        foreign = foreign
            || name == crate::client::marlin::GEMM_W4A16
            || name == crate::client::marlin::GEMM_W4A16_AWQ
            || name == crate::client::marlin::GEMM_W4A16_BF16;
    }
    #[cfg(feature = "flashinfer")]
    {
        foreign = foreign
            || name == crate::client::flashinfer::PREFILL_FI
            || name == crate::client::flashinfer::PREFILL_FI_FP8KV;
    }
    foreign
}
