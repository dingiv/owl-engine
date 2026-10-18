//! 算子注册表 —— **名字字面量全库唯一住址**(P0 名字无关的名字源)。
//!
//! [`KernelFamily`] = **族/运行时**契约(一族一个实例:构造/析构 + 发射 +
//! 资源作用域;原 `KernelSpec` 更名 —— 名字错位修正,2026-10-13 P0)。
//! [`OpRegistry`] = boot 装配 + server 唯一入口(名字无关)。
//!
//! 名字路由(2026-10-12 review D 案收口):foreign/native 双路由判定
//! = [`OpRegistry::knows`](boot 表自洽,零第二份清单)—— 旧
//! `is_foreign_name` 硬编码枚举(与 all_families 双真源)已删除。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// 族工厂:构造即装配门(§4.1)。boot 逐个调用,任一 `Err` ⇒ 拒启。
///
/// 关联函数指针(非方法调用)—— `assemble` 带 `where Self: Sized`,
/// 不进 vtable,故 `Box<dyn KernelFamily>` 的 dyn 兼容不受影响(实测)。
pub type FamilyFactory =
    fn(&mut dyn crate::device::DeviceRes, &mut crate::device::Exec) -> Result<Box<dyn KernelFamily>, OpError>;

/// 族契约(原子单位 = 一族 runtime;每族一个 struct 实现)。
///
/// - `assemble` = **构造函数**(工厂语义):族内一次完成符号/资产校验 +
///   资源预钉;返回 `Ok` ⟺ 本族已准入(漏调 = 拿不到实例,物理不可发射);
/// - `close` = **显式析构**(可失败 ⇒ 调用方 `?`;Rust 纪律:不吞可失败返回值);
/// - `run` = 唯一发射入口(Result 全链)。
///
/// `Drop` 只做 best-effort 兜底(见 [`FamilyGuard`]):未经 `close()` 即 drop
/// ⇒ warning + 计数,把"忘记收尾"变成可见事件。
pub trait KernelFamily: Send {
    fn id(&self) -> OpId;

    /// **本族算子**(注册语义;设计 §2.2):返回本族的准入声明表。
    ///
    /// **派生**(2026-10-15 声明驱动):缺省 = `ops::of_family(self.id())` ——
    /// 单一真源 = 每个算子的 [`KernelSpec::FAMILY`](crate::spec::KernelSpec::FAMILY);
    /// 族侧**不再手写** `const OPS`(旧形态在 6 个族里各抄一份,与声明面
    /// 可能漂移)。
    fn ops(&self) -> Vec<&'static crate::spec::KernelMeta> {
        crate::ops::of_family(self.id()).collect()
    }

    /// 本族声明的名字 = **族 id(主名)∪ 已建表算子名**。
    ///
    /// 这样每族都不必手列名字(消灭"名字两处写"这个漂移源):
    /// - 全量建表的族(marlin/cublas/xqa/gdn_*)→ 全部由 `ops()` 派生;
    /// - 部分建表的族(主名不是任何算子 ENTRY 的族)→ 主名(id)+ 已建表项。
    ///
    /// ⚠️ 不写成"`ops()` 非空则纯派生":那样**部分建表**的族会丢掉主名
    /// (2026-10-13 实测:FI 的 f16 臂主名 + gdn_scalar 会静默从路由表消失)。
    fn names(&self) -> Vec<&'static str> {
        let mut v = vec![self.id().0];
        for m in self.ops() {
            if !v.contains(&m.entry) {
                v.push(m.entry);
            }
        }
        v
    }


    /// **构造函数**(工厂语义)。带 `where Self: Sized` 保住 trait 的 dyn 兼容。
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError>
    where
        Self: Sized;

    /// **显式析构**。默认无收尾(纯 FFI/无持有资源族);有资源族覆写。
    fn close(self: Box<Self>) -> Result<(), OpError> {
        Ok(())
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError>;

    /// **FI plan 表同步**(W3;仅 FI 族实现)。缺省 = **结构化拒**(不静默):
    /// 非 FI 族被要求刷 plan 表即调用方违约。
    fn plan_sync(
        &mut self,
        _t: usize,
        _ctx_total: usize,
        _hq: usize,
        _hkv: usize,
        _hd: usize,
        _page: usize,
        _res: &mut dyn crate::device::DeviceRes,
    ) -> Result<(), OpError> {
        Err(OpError::Asset {
            op: self.id().0.into(),
            detail: "本族无 FI plan 同步面(仅 flashinfer 族实现)".into(),
        })
    }
}

// ============================================================================
// 族生命周期守卫(§4.1 析构纪律)
// ============================================================================

/// 未显式 `close()` 即 drop 的族计数(全局,`Relaxed` = 只做度量不打点)。
static UNCLOSED_DROPS: AtomicU64 = AtomicU64::new(0);

/// best-effort 兜底:记一次"未经 close 即 drop"并打 warning(不静默)。
pub fn note_unclosed_drop(id: &str) {
    let n = UNCLOSED_DROPS.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!("[kernels] warning: 族 {id} 未经 close() 即被 drop(累计未收尾 {n})");
}

/// 未收尾 drop 计数快照(测试/运维用)。
pub fn unclosed_drop_count() -> u64 {
    UNCLOSED_DROPS.load(Ordering::Relaxed)
}

/// 族生命周期守卫:族实例的**唯一合法持有形态**(工厂产物)。
///
/// - 显式 `close()` ⇒ 静默;`close` 会把内层族的 `close()` 走一遍;
/// - 未经 `close()` 即 drop ⇒ [`note_unclosed_drop`](warning + 计数)。
///
/// 内部 `Option` 仅为在 `close` 中把内层族移出(`F::close` 需 `Box<F>`);
/// 运行/查询路径经 `expect` 断言"未 close 才可用"(close 后再用 = 编程错误)。
pub struct FamilyGuard<F: KernelFamily + 'static> {
    inner: Option<Box<F>>,
    id: OpId,
    closed: bool,
}

impl<F: KernelFamily + 'static> FamilyGuard<F> {
    /// 工厂收尾糖:把具体族擦进 dyn 守卫。
    pub fn boxed(inner: F) -> Box<dyn KernelFamily> {
        let id = inner.id();
        Box::new(Self { inner: Some(Box::new(inner)), id, closed: false })
    }

    fn family(&self) -> &F {
        self.inner.as_deref().expect("族已 close,不可再用(生命周期违例)")
    }

    fn family_mut(&mut self) -> &mut F {
        self.inner.as_deref_mut().expect("族已 close,不可再用(生命周期违例)")
    }
}

impl<F: KernelFamily + 'static> KernelFamily for FamilyGuard<F> {
    fn id(&self) -> OpId {
        self.id
    }
    fn ops(&self) -> Vec<&'static crate::spec::KernelMeta> {
        self.family().ops()
    }
    fn names(&self) -> Vec<&'static str> {
        self.family().names()
    }
    /// 守卫由族工厂创建,不可作为工厂(类型错误拦在编译期之外)。
    fn assemble(
        _res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        Err(OpError::Contract {
            op: "FamilyGuard".into(),
            field: "assemble",
            expect: "由具体族工厂创建".into(),
            got: "直接调用守卫工厂".into(),
        })
    }
    fn close(mut self: Box<Self>) -> Result<(), OpError> {
        self.closed = true;
        match self.inner.take() {
            Some(inner) => inner.close(),
            None => Ok(()),
        }
    }
    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        self.family_mut().run(msg, res, exec)
    }

    /// **转发**(2026-10-18 W3):守卫层必须显式转发每个新方法 —— 否则
    /// `Box<dyn KernelFamily>` 走的是 guard 的**缺省实现**(实测症状:
    /// `本族无 FI plan 同步面` 而内层族明明实现了它)。新增 trait 方法时
    /// 请连带检查本 impl(与 `run`/`close` 同律)。
    fn plan_sync(
        &mut self,
        t: usize,
        ctx_total: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        page: usize,
        res: &mut dyn crate::device::DeviceRes,
    ) -> Result<(), OpError> {
        self.family_mut().plan_sync(t, ctx_total, hq, hkv, hd, page, res)
    }
}

impl<F: KernelFamily + 'static> Drop for FamilyGuard<F> {
    fn drop(&mut self) {
        if !self.closed {
            note_unclosed_drop(self.id.0);
        }
    }
}

/// 拒启路径收尾:逐族 `close()`(错误已在上游记录;此处只保证不留噪声)
fn close_all(families: Vec<Box<dyn KernelFamily>>) {
    for f in families {
        let _ = f.close();
    }
}

/// 注册表(编译期穷举;分派在注册表内部,server 主循环零算子知识)。
pub struct OpRegistry {
    families: Vec<Box<dyn KernelFamily>>,
    /// 名字速查(boot 期自 families 收口;knows()/execute 同源)
    names: HashSet<&'static str>,
    /// **名字 → 族下标**(boot 期一次建;`execute` 从"线性扫族×名"变 O(1),
    /// 2026-10-15 声明驱动路由)
    route: std::collections::HashMap<&'static str, usize>,
    /// **名字 → 准入声明**(boot 期自 [`crate::ops::METAS`] 建一次;
    /// 发射期 check 走它 —— O(1),热路径零扫描)。
    ///
    /// ⚠️ 与 `names` **必须分离**:原生核有元数据但**不得**进 `names`
    /// (否则会被误分派到家族 `run`,原生 NVRTC 路径被截断)。
    metas: std::collections::HashMap<&'static str, &'static crate::spec::KernelMeta>,
}

impl OpRegistry {
    /// boot 一步:全家族装配(构造函数即装配门;失败 = Err 拒启)。
    pub fn boot(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Self, OpError> {
        Self::init_all(crate::server::family_factories(), res, exec)
    }

    /// **生产档 boot**(白名单在场):族装配 → 白名单校验 → 池写集审计 →
    /// 装配期自检。任一失败 = Err ⇒ 引擎拒启(设计 §八 P4/P5 的装配门全链)。
    pub fn boot_with(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
        allowed: Option<&[&str]>,
    ) -> Result<Self, OpError> {
        // 装配(族构造即门 + 白名单 + 池审计)→ **然后**跑自检:
        // foreign 算子的 oracle 要经注册表发射(dispatch),所以自检必须在
        // 注册表建成之后(2026-10-13 P5c 的调用点搬迁)。
        let mut reg = Self::init_all_with(crate::server::family_factories(), res, exec, allowed)?;
        reg.run_selfchecks(res, exec, allowed)?;
        Ok(reg)
    }

    /// **装配期自检**(设计 §5.5:独立 oracle,失败 = Err ⇒ 拒启)。
    ///
    /// **随白名单生效**:白名单在场 = 生产档 ⇒ 只对生效算子跑 oracle
    /// (装配期成本按档位裁,"该跑谁"也是确定的);门关(dev/测试档)不跑 ——
    /// 那类环境常无设备资源,自检由 GPU 测试面覆盖
    /// (`tests/selfcheck_oracles.rs`:正例 + 判据敏感性负例)。
    fn run_selfchecks(
        &mut self,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
        allowed: Option<&[&str]>,
    ) -> Result<(), OpError> {
        let Some(a) = allowed else { return Ok(()) };
        let active: Vec<&'static crate::spec::KernelMeta> = self
            .metas
            .values()
            .copied()
            .filter(|m| a.contains(&m.entry))
            .collect();
        let n = crate::server::selfcheck::run_selfchecks(active.into_iter(), res, exec, self)?;
        if n > 0 {
            eprintln!("[kernels] 装配期自检:{n} 个算子通过(oracle 独立参考)");
        }
        Ok(())
    }

    /// 带白名单的装配(生产档入口:白名单同时喂给池审计与装配期自检)
    pub fn init_all_with(
        factories: Vec<FamilyFactory>,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
        allowed: Option<&[&str]>,
    ) -> Result<Self, OpError> {
        Self::init_all_inner(factories, res, exec, allowed)
    }

    pub fn init_all(
        factories: Vec<FamilyFactory>,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Self, OpError> {
        Self::init_all_inner(factories, res, exec, None)
    }

    fn init_all_inner(
        factories: Vec<FamilyFactory>,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
        allowed: Option<&[&str]>,
    ) -> Result<Self, OpError> {
        let mut names = HashSet::new();
        let mut route: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
        let mut families: Vec<Box<dyn KernelFamily>> = Vec::with_capacity(factories.len());
        for factory in factories {
            // 装配失败路径也要收尾已建成的族(§4.4 同律:错误路径不留
            // "未经 close 即 drop" 的噪声 —— 否则拒启日志被兜底告警淹没)
            let f = match factory(res, exec) {
                Ok(f) => f,
                Err(e) => {
                    close_all(families);
                    return Err(e);
                }
            };
            // 装配期一致性门(设计 §三 铁律 4:门由 boot 执行,不是文档纪律):
            // **族声明了元数据的算子必须也在 names() 里** —— 否则那份元数据
            // 永远发不出去(名字路由收不到它),是纯粹的摆设。
            // 注意方向:允许 `names() ⊋ ops()`(族内尚有未建表算子,如 FI 的
            // f16 臂),不允许 `ops() ⊄ names()`。
            let ops = f.ops();
            if !ops.is_empty() {
                let names = f.names();
                let missing: Vec<&str> = ops
                    .iter()
                    .map(|m| m.entry)
                    .filter(|e| !names.contains(e))
                    .collect();
                if !missing.is_empty() {
                    let e = OpError::Contract {
                        op: f.id().0.to_string(),
                        field: "ops/names",
                        expect: "已建表算子 ∈ 族声明名字(否则元数据发不出去)".into(),
                        got: format!("元数据在册但名字未声明:{missing:?}"),
                    };
                    let _ = f.close();
                    close_all(families);
                    return Err(e);
                }
            }
            let idx = families.len();
            for n in f.names() {
                if !names.insert(n) {
                    let e = OpError::Contract {
                        op: n.to_string(),
                        field: "registry",
                        expect: "家族名全库唯一".into(),
                        got: "重复登记(两家族声明同名)".into(),
                    };
                    close_all(families);
                    return Err(e);
                }
                // 名字 → 族下标(boot 一次;`execute` O(1);旧实现每次发射
                // 线性扫 `families × names`)
                route.insert(n, idx);
            }
            families.push(f);
        }
        // 准入索引(纯元数据,与发射路由无关)
        let metas: std::collections::HashMap<_, _> =
            crate::ops::all().map(|m| (m.entry, m)).collect();

        // ── 声明驱动路由的自洽门(2026-10-15)───────────────────────────
        // A) **声明的家族必须真实存在**:非 Native 的算子,其 `FAMILY` 必须在
        //    已装配的族 id 里 —— 否则那份元数据永远发不出去(路由收不到它),
        //    是纯粹的摆设。这是"声明驱动"的正面判据(旧设计靠 ops⊆names 事后校)。
        //
        //    **射程 = 激活集**(2026-10-15 用户裁决):`allowed = Some(白名单)`
        //    时只审白名单里的声明 —— 生产 boot 用 flavor 白名单,语义 = "**会被
        //    发射的**声明必须有族";白名单外的声明本就不参与路由。`allowed = None`
        //    (= 全量激活,测试/工具面)保持原严格度。
        //    副作用(正收益):空激活集 ⇒ 零孤儿 ⇒ **裸注册表可构造**,
        //    `knows/execute 同源` 这类不依赖真家族的纪律可测(旧行为下空表必撞
        //    本门 ⇒ 该测试自 A16 起挂红)。
        let ids: HashSet<&'static str> = families.iter().map(|f| f.id().0).collect();
        let orphan: Vec<&str> = crate::ops::all()
            .filter(|m| m.is_family() && !ids.contains(m.family.0))
            .filter(|m| match allowed {
                Some(a) => a.contains(&m.entry),
                None => true,
            })
            .map(|m| m.entry)
            .collect();
        if !orphan.is_empty() {
            close_all(families);
            return Err(OpError::Contract {
                op: "registry".into(),
                field: "KernelSpec::FAMILY",
                expect: "激活集里的非 Native 算子,其家族必须已装配(白名单外不审)".into(),
                got: format!("孤儿声明(家族不存在):{orphan:?}"),
            });
        }
        // B) **路由表里的名字若已建表,其 FAMILY 必须指向本族**:反向判据
        //    (拦住"名字挂在甲族、声明却写乙族"——那会让元数据与执行者分家)。
        for f in families.iter() {
            let id = f.id();
            for n in f.names() {
                if let Some(m) = metas.get(n) {
                    if m.family != id {
                        let e = OpError::Contract {
                            op: n.to_string(),
                            field: "KernelSpec::FAMILY",
                            expect: format!("{} (本族 id)", id.0),
                            got: format!("{}(声明与路由分家)", m.family.0),
                        };
                        close_all(families);
                        return Err(e);
                    }
                }
            }
            // C) 空 ops() = 只能靠主名路由(未建表);不报错但必须可见
            if f.ops().is_empty() {
                eprintln!(
                    "[kernels] 族 {} 无已建表算子(只能靠主名路由;声明面零约束)",
                    id.0
                );
            }
        }
        // F 轮(2026-10-13):登记表退役 ⇒ 唯一真源 = 声明面。boot 报告改为
        // 三件事:**声明表张数 / 家族臂数 / 编出但未声明的符号数**(后者是
        // 新的 R1 判据:构建期符号集 ⊆ 声明集)。
        let n_declared = crate::ops::declared_count();
        let n_family = names.len();
        let undeclared = crate::ops::undeclared_symbols();
        eprintln!(
            "[kernels] 算子准入:已声明 {} 张(原生表 + 家族臂 {})/ 编出符号 {} 条(未声明 {})",
            n_declared,
            n_family,
            crate::cu_sig::ASSET_SYMBOLS.iter().map(|(_, s)| s.len()).sum::<usize>(),
            undeclared.len()
        );
        if !undeclared.is_empty() {
            // R1(F 轮换源:编出的符号 ⊆ 声明集)—— 未声明清单要可数、可见
            eprintln!("[kernels] R1 编出但未声明的核符号 {} 条:{undeclared:?}", undeclared.len());
        }
        // 装配期自检(设计 §5.5:独立 oracle,失败 = 拒启)。
        // 白名单在场 ⇒ 只跑生效算子(装配期成本按档位裁);否则跑全部已建表。
        // ── 白名单硬门(装配期三件事)──────────────────────────────
        if let Some(a) = allowed {
            for n in a {
                if !metas.contains_key(n) {
                    return Err(OpError::Contract {
                        op: (*n).to_string(),
                        field: "allowed_ops",
                        expect: "白名单算子必须已建表(有 KernelSpec 声明)".into(),
                        got: "未建表(仅引入)—— 不许进生产档".into(),
                    });
                }
            }
            crate::ops::audit_active(Some(a))?;
            eprintln!("[kernels] 生产白名单:{} 项生效(池写集审计通过)", a.len());
        }

        Ok(Self { families, names, route, metas })
    }

    /// **准入声明查询**(发射期 check 的唯一入口;未建表 = `None` = 仅引入)。
    pub fn meta(&self, name: &str) -> Option<&'static crate::spec::KernelMeta> {
        self.metas.get(name).copied()
    }

    /// **整体收尾**(析构的唯一官方出口):逐族 `close()`,返回首个错误。
    ///
    /// 为什么必须有:各族的 `assemble` 把自己包在 `FamilyGuard` 里,而
    /// `Drop` 只做"未经 close 即 drop"的兜底告警(design §四)。若收尾路径
    /// 直接裸 `drop(registry)`,则**每次正常退出都会刷 N 条告警** ⇒ 告警疲劳,
    /// 真漏反被淹没(2026-10-13 复核实测:一次正常 e2e 启动刷 5 条)。
    /// teardown 路径调用方已无法上抛 ⇒ 错误由调用方记录+计数。
    pub fn close(mut self) -> Result<(), OpError> {
        let families = std::mem::take(&mut self.families);
        let mut first: Option<OpError> = None;
        for f in families {
            if let Err(e) = f.close() {
                eprintln!("[kernels] 族 close() 失败: {e}");
                if first.is_none() {
                    first = Some(e);
                }
            }
        }
        self.names.clear();
        self.route.clear();
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// 名字路由判定(foreign/native 双路唯一真源;boot 表自洽)。
    /// 线格式名 ∈ 注册表 = foreign 家族臂,否则走 native 路径。
    pub fn knows(&self, name: &str) -> bool {
        self.route.contains_key(name)
    }

    /// server 唯一算子入口:名字 → 家族 runtime 的分派在注册表内部。
    pub fn execute(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        let idx = *self.route.get(msg.kernel.name.as_str()).ok_or_else(|| {
            OpError::Contract {
                op: msg.kernel.name.clone(),
                field: "name",
                expect: "已注册家族".into(),
                got: format!("{}(未注册)", msg.kernel.name),
            }
        })?;
        self.families[idx].run(msg, res, exec)
    }

    /// **FI plan 表同步**(W3):按名字路由到族(与 [`Self::execute`] 同表同行纪律;
    /// 名字不在表 ⇒ 结构化拒)。调用方 = 后端 server 命令层。
    #[allow(clippy::too_many_arguments)]
    pub fn plan_sync(
        &mut self,
        name: &str,
        t: usize,
        ctx_total: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        page: usize,
        res: &mut dyn crate::device::DeviceRes,
    ) -> Result<(), OpError> {
        let idx = *self.route.get(name).ok_or_else(|| OpError::Contract {
            op: name.to_string(),
            field: "name",
            expect: "已注册家族".into(),
            got: format!("{name}(未注册)"),
        })?;
        self.families[idx].plan_sync(t, ctx_total, hq, hkv, hd, page, res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// knows() = execute() 同源(knows 为真 → execute 找得到家族;
    /// knows 为假 → execute 结构化拒)。家族表零实例化即可判定的前提
    /// 是两处同表 —— 本测试锁死这条纪律。
    ///
    /// **为何走白名单构造**(2026-10-15)：A16 的声明驱动门(A)要求"激活集里的
    /// 非 Native 声明必须有族" —— 空工厂表 + 全量激活 = 11 条孤儿 ⇒ 直接拒
    /// (本测试自 A16 起挂红)。用**空激活集** `Some(&[])` ⇒ 零激活声明 ⇒
    /// 裸注册表可构造，纪律照测。
    #[test]
    fn knows_and_execute_share_the_table() {
        // 零 DeviceRes 环境无法 boot 真家族(assemble 需 GPU);这里用空工厂表
        // + 空激活集验证空集行为 + 名字集构造逻辑。
        let mut reg = OpRegistry::init_all_with(
            vec![],
            &mut NopRes,
            &mut crate::device::Exec::new(),
            Some(&[]),
        )
        .expect("空激活集");
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

    /// **`names()` = 族主名(id)∪ 已建表算子名**(2026-10-13 回归锁)。
    ///
    /// 实测过的坑:写成"`ops()` 非空则纯派生"会让**部分建表**的族丢掉主名
    /// (FI 的 f16 臂主名 + gdn_scalar 会静默从路由表消失 ⇒ 该臂回落原生
    /// NVRTC 路径后必然失败)。
    #[test]
    fn names_is_id_union_ops() {
        struct Fake(&'static [&'static crate::spec::KernelMeta]);
        impl KernelFamily for Fake {
            fn id(&self) -> OpId {
                OpId("fake_family")
            }
            fn ops(&self) -> Vec<&'static crate::spec::KernelMeta> {
                self.0.to_vec()
            }
            fn assemble(
                _res: &mut dyn crate::device::DeviceRes,
                _exec: &mut crate::device::Exec,
            ) -> Result<Box<dyn KernelFamily>, OpError> {
                Err(OpError::Contract {
                    op: "fake".into(),
                    field: "-",
                    expect: "-".into(),
                    got: "-".into(),
                })
            }
            fn run(
                &mut self,
                _msg: &LaunchMsg,
                _res: &mut dyn crate::device::DeviceRes,
                _exec: &mut crate::device::Exec,
            ) -> Result<Bytes, OpError> {
                Err(OpError::Contract {
                    op: "fake".into(),
                    field: "-",
                    expect: "-".into(),
                    got: "-".into(),
                })
            }
        }
        // 未建表:只剩主名
        assert_eq!(Fake(&[]).names(), vec!["fake_family"]);
        // 部分/全量建表:主名 + 算子名(去重)
        const WITH_OP: &[&crate::spec::KernelMeta] = &[&crate::ops::xqa::XQA_VERIFY];
        assert_eq!(
            Fake(WITH_OP).names(),
            vec!["fake_family", "owl_xqa_verify"]
        );
    }

    struct NopRes;
    impl crate::device::DeviceRes for NopRes {
        fn context(&self) -> Result<std::sync::Arc<cudarc::driver::CudaContext>, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "ctx", expect: "-".into(), got: "-".into() })
        }
        fn stream(&self) -> Result<std::sync::Arc<cudarc::driver::CudaStream>, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "stream", expect: "-".into(), got: "-".into() })
        }
        fn resolve(&self, _id: u64) -> Result<u64, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "resolve", expect: "-".into(), got: "-".into() })
        }
        fn resolve_len(&self, _id: u64) -> Result<usize, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "resolve_len", expect: "-".into(), got: "-".into() })
        }
        fn alloc(&mut self, _bytes: usize, _tag: &str) -> Result<crate::device::ScratchBuf, OpError> {
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
        fn upload_block(
            &mut self,
            _b: &crate::contract::Bytes,
            _src: &[u8],
        ) -> Result<(), OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "upload_block", expect: "-".into(), got: "-".into() })
        }
        fn alloc_block(
            &mut self,
            _bytes: usize,
            _tag: &str,
        ) -> Result<crate::contract::Bytes, OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "alloc_block", expect: "-".into(), got: "-".into() })
        }
        fn download(&mut self, _src: crate::device::DevPtr, _dst: &mut [u8]) -> Result<(), OpError> {
            Err(OpError::Contract { op: "nop".into(), field: "download", expect: "-".into(), got: "-".into() })
        }
    }
}
