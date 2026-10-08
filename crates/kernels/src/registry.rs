//! 算子注册表 —— **名字字面量全库唯一住址**(P0 名字无关的名字源)。
//!
//! 结构(operator-contract 设计 v0.4 §4.4):
//! - [`OpId`](crate::contract::OpId):models 只拿这个,不拼名字字符串;
//! - [`registry!`]:编译期穷举的注册表(单二进制,无插件加载),
//!   名字→Linkage→Runtime 构造的映射在此**声明一次**;
//! - [`OpRegistry`]:boot 期 init(符号/指纹双门,§4.5 装配门)后交
//!   server 持有;`execute` = server 唯一算子入口(名字无关)。
//!
//! M2 骨架:宏 + 类型 + 注册表壳;runtime 家族随 M3/M4 接入
//! (接入点 = `registry!` 的 arms 与 [`FamilyRuntime`] 实现者)。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};

/// 家族运行时统一面(服务端面;每家族一个实现者)。
///
/// 生命周期:init = boot 装配(装载/校验/预钉,缺一 = Err 拒启);
/// run = 唯一执行入口(Result 全链,零 ack 侧信道)。
pub trait FamilyRuntime: Send {
    fn id(&self) -> OpId;
    /// 本家族响应的全部核名(别名;GEMM_BF16/AWQ/fp8kv 等同族多名)
    fn names(&self) -> Vec<&'static str> {
        vec![self.id().0]
    }
    fn linkage(&self) -> crate::contract::Linkage;
    /// boot 装配(P4:符号逐一校验 × manifest 指纹 × workspace 预钉;
    /// 装载原语经 env.exec,上下文经 env.res)
    fn init(&mut self, env: &mut RunEnv) -> Result<(), OpError>;
    /// 唯一执行入口;msg 解析(契约 parse)在此家族内完成
    fn run(&mut self, msg: &LaunchMsg, env: &mut RunEnv) -> Result<Bytes, OpError>;
}

/// 执行环境:资源面 + 执行引擎的打包视图(家族 runtime 的全部所需)。
pub struct RunEnv<'a> {
    pub res: &'a mut dyn crate::device::DeviceRes,
    pub exec: &'a mut crate::device::Exec,
}

impl<'a> RunEnv<'a> {
    pub fn new(res: &'a mut dyn crate::device::DeviceRes, exec: &'a mut crate::device::Exec) -> Self {
        Self { res, exec }
    }
}

/// 注册表(编译期穷举;分派在注册表内部,server 主循环零算子知识)。
///
/// [`is_foreign_name`] = server 的 foreign/native 路由谓词(名字清单
/// 唯一住址;server 只问"这是不是你的",不问"这是谁")。
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
pub struct OpRegistry {
    families: Vec<Box<dyn FamilyRuntime>>,
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self { families: Vec::new() }
    }
}

impl OpRegistry {
    /// boot 装配:逐家族 init(P4 门在各自 init 内);缺一 = Err 拒启。
    pub fn init_all(
        mut families: Vec<Box<dyn FamilyRuntime>>,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Self, OpError> {
        for f in families.iter_mut() {
            let mut env = RunEnv::new(res, exec);
            f.init(&mut env)?;
        }
        Ok(Self { families })
    }

    /// boot 一步:全家族装配([`crate::server::all_families`])+ init
    /// (P4 门;失败 = Err 拒启)。
    pub fn boot(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Self, OpError> {
        Self::init_all(crate::server::all_families(), res, exec)
    }

    /// server 唯一算子入口:名字 → 家族 runtime 的分派在注册表内部
    /// (线性查表 = 家族数个比较,ns 级 vs kernel ms 级)。
    pub fn execute(
        &mut self,
        msg: &LaunchMsg,
        env: &mut RunEnv,
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
        f.run(msg, env)
    }
}

// 注册表声明宏(M4 接入时落地;名字字面量唯一住址的语法位)。
// 形态:`registry! { ops { GdnChunked => ("名字", AotCubin, GdnChunkedRuntime), ... } }`
// —— 展开为 families 向量构造 + 静态 OpId 常量表;M2 暂以 Vec<Box<dyn>>
// 手工装配,宏在首个家族接入时引入(避免无消费者的死宏)。