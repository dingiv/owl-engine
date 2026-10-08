//! 算子注册表 —— **名字字面量全库唯一住址**(P0 名字无关的名字源)。
//!
//! [`KernelSpec`] = 胖算子契约(原子单位;每个在用算子一个 struct 实现)。
//! [`OpRegistry`] = boot 装配 + server 唯一入口(名字无关)。
//!
//! 名字路由(2026-10-12 review D 案收口):foreign/native 双路由判定
//! = [`OpRegistry::knows`](boot 表自洽,零第二份清单)—— 旧
//! `is_foreign_name` 硬编码枚举(与 all_families 双真源)已删除。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};
use std::collections::HashSet;

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
    /// 名字速查(boot 期自 families 收口;knows()/execute 同源)
    names: HashSet<&'static str>,
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
        let mut names = HashSet::new();
        for f in families.iter_mut() {
            f.init(res, exec)?;
            for n in f.names() {
                if !names.insert(n) {
                    return Err(OpError::Contract {
                        op: n.to_string(),
                        field: "registry",
                        expect: "家族名全库唯一".into(),
                        got: "重复登记(两家族声明同名)".into(),
                    });
                }
            }
        }
        Ok(Self { families, names })
    }

    /// 名字路由判定(foreign/native 双路唯一真源;boot 表自洽)。
    /// 线格式名 ∈ 注册表 = foreign 家族臂,否则走 native nvrtc 路径。
    pub fn knows(&self, name: &str) -> bool {
        self.names.contains(name)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// knows() = execute() 同源(knows 为真 → execute 找得到家族;
    /// knows 为假 → execute 结构化拒)。家族表零实例化即可判定的前提
    /// 是两处同表 —— 本测试锁死这条纪律。
    #[test]
    fn knows_and_execute_share_the_table() {
        // 零 DeviceRes 环境无法 boot 真家族(init 需 GPU);
        // 这里用空家族表验证空集行为 + 名字集构造逻辑。
        let mut reg = OpRegistry::init_all(vec![], &mut NopRes, &mut crate::device::Exec::new()).expect("空表");
        assert!(!reg.knows("anything"));
        let msg = LaunchMsg {
            kernel: crate::contract::KernelSource { name: "anything".into(), source: String::new() },
            args: vec![],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: 0,
        };
        let err = reg
            .execute(&msg, &mut NopRes, &mut crate::device::Exec::new())
            .expect_err("空表必拒");
        assert!(matches!(err, OpError::Contract { field: "name", .. }), "{err}");
    }

    struct NopRes;
    impl crate::device::DeviceRes for NopRes {
        fn context(&self) -> Result<std::sync::Arc<cudarc::driver::CudaContext>, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "ctx", expect: "-".into(), got: "-".into() })
        }
        fn stream(&self) -> Result<std::sync::Arc<cudarc::driver::CudaStream>, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "stream", expect: "-".into(), got: "-".into() })
        }
        fn resolve(&self, _b: &Bytes) -> Result<u64, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "resolve", expect: "-".into(), got: "-".into() })
        }
        fn alloc(&mut self, _bytes: usize, _tag: &'static str) -> Result<crate::device::ScratchBuf, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "alloc", expect: "-".into(), got: "-".into() })
        }
        fn capturing(&self) -> Result<bool, OpError> {
            Ok(false)
        }
        fn record_capture(&mut self, _note: crate::device::LaunchNote) -> Result<(), OpError> {
            Ok(())
        }
        fn upload(&mut self, _dst: u64, _src: &[u8]) -> Result<(), OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "upload", expect: "-".into(), got: "-".into() })
        }
        fn device_ordinal(&self) -> Result<i32, OpError> {
            Ok(0)
        }
    }
}
