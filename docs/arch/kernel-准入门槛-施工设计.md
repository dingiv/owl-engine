# 算子准入门槛与显存块治理 —— 施工设计

> 版本 v1.1(2026-10-13;v1.0 评审后按用户裁决修订)。
> 承接:用户指令「引入 KernelSpec / KernelFamily 升级版」+ 前一轮失败分类学讨论。
> 前置:`operator-contract-施工设计.md`(P0-P5 六原则 / Error 模型 / 双面基础设施)、
> `kernel-error-ledger.md`(E1-E15 + 缺口闭环)。
> **v1.1 相对 v1.0 的四处修订**(全部来自用户裁决):
> ① 算子 struct **住各 family 目录内**;② `list.rs` **只留注册语义**,元数据由算子自托管;
> ③ 新增 **构造/析构(工厂 + 作用域)纪律**一节;④ 生产 flavor **白名单硬门**立项。
> 状态:**设计待评审(不动代码)**。

---

## 一、病灶(为什么现在必须动)

### 1.1 四条现状事实(当日核实)

| 事实 | 数字/证据 | 含义 |
|---|---|---|
| .cu 里"有入口"的算子 | **30 个 .cu / 103 个 `extern "C" __global__`** | = 已**引入** |
| 元数据现状 | **110 条集中条目**(`list.rs` Entry:name+source+sig+dtype) | 元数据**住在注册表里**,而不是算子身上 |
| 实现契约的 struct | **只有 7 个**,且全是**族级 runtime** | 族 = 有一个;算子 = 没有 |
| `KernelSpec::validate()` 调用者 | **0** | **"运行时检查"这一层现在根本不存在**(占位符) |

⇒ 现状 = **"能被发射"就等于"能被使用"**。没有任何声明说"这个算子需要什么几何、
哪些参数合法、它写哪块内存、它数值该满足什么"。**准入 = 无**。

### 1.2 由此漏进来的真实案

| 类 | 案 | 本该拦住的地方 |
|---|---|---|
| 几何 | `xqa_sem` 零长槽(2026-10-13)→ 锁页分配空指针崩 | "任何块长度 ≥1" + 形状族 |
| 几何 | `M_TILESIZE` 实例化选错 → q=1 走 32 行 tile,白烧 5.3× | 形状族(核声明的行数上界) |
| 数值语义 | raw g 案(E7)→ e²⁴ 爆炸,染毒 60 层 | 出口/中间不变量 + 独立 oracle |
| 数值语义 | 同源对拍双错(E8)→ 层 0-2 全绿假象 | **独立** oracle(boot 期强制对拍) |
| 池访问 | 融合插池 fp8 臂缺失(E10)→ 槽区全零 | 池读写集声明 + 毒值自检 |
| 池寻址 | 草稿池双寻址(E14)、slot 契约腐烂(E15) | 布局标签类型化 |
| 时序 | 装载期 static / 跨账本 id(坑册 §26/§27) | 账本身份 + 装配期静态检查 |

**最痛的是"数值语义"类**:不崩、输出合法、但模型胡言乱语,只能一层一层挖。
本文档的核心目标:给这一类配上**机器判据**,而不是靠人肉逐层二分。

---

## 二、术语迁移与注册表瘦身(本设计的第一刀)

### 2.1 更名

```rust
// 旧(名字错位:trait 描述的是"族/运行时",却叫 Spec)
pub trait KernelSpec { fn id(); fn names(); fn linkage(); fn init(); fn validate(); fn run(); }
                              ↑ 这 7 个实现全是"族"

// 新
pub trait KernelFamily { id / ops / linkage / assemble / close / run }  // 族 = 作用域所有者
pub trait KernelSpec   { /* 单算子声明式契约,见 §四 */ }                // 算子 = 一个 struct
```

| | KernelFamily | KernelSpec |
|---|---|---|
| 单位 | 一族(marlin / xqa / gdn_chunked …) | **一个算子**(如 `owl_naive_attn_nc_bf16_v2`) |
| 数量 | 7 | 目标 = 生产在用的每一个(起步 ~20,上限 103) |
| 职责 | **构造/析构 + 装配门 + 发射 + 资源作用域** | **声明 + 校验**(参数/几何/内存/数值律) |
| 与 .cu | 一组 .cu 的链接面 | **一对一**某个 `extern "C"` 入口 |
| 住哪 | `kernels/src/server/<family>.rs` | **同目录** `kernels/src/server/<family>/<op>.rs`(用户裁决①) |

### 2.2 `list.rs` 退化为**纯注册表**(用户裁决②)

**现状问题**:110 条 `Entry{name, source, sig, dtype}` 集中在一张表里 —— 算子的身份、
源码、槽序、dtype **住在远离算子自己的地方**,于是必然出现"两处对不上"(E3/E15 的根因)。

**新形态**:元数据**全部由算子 struct 自托管**;`list.rs` 只剩"谁被登记了"这一件事。

```rust
// ── list.rs(瘦身后全文约 30 行):只有注册语义 ─────────────────────
pub fn family_factories() -> &'static [FamilyFactory] {
    &[ MarlinFamily::assemble, CublasFamily::assemble, GdnChunkedFamily::assemble,
       FiFamily::assemble, XqaFamily::assemble, GdnScalarFamily::assemble ]
}

// ── 族自己登记它的算子(注册语义的第二半) ────────────────────────
impl KernelFamily for MarlinFamily {
    fn ops(&self) -> &'static [&'static dyn KernelSpec] {
        &[ &marlin::W4A16, &marlin::W4A16Awq, &marlin::W4A16Bf16 ]
    }
}

// ── 算子自带全部元数据(自托管;全部是**关联常量**,编译期定值) ────
//    用户裁决(2026-10-13):能写成 const 的一律写 const,
//    不要写成 `fn entry(&self) -> &'static str { "..." }`。
pub struct NcAttnBf16V2;                       // 标记类型(零大小,只承载常量)
impl KernelSpec for NcAttnBf16V2 {
    const ENTRY:  &'static str = "owl_naive_attn_nc_bf16_v2";
    const SOURCE: &'static str = include_str!("../../cu/owl/dflash2.cu");
    const PARAMS: &'static [Param] = &[ /* 见 §5.1 */ ];
    const SHAPE:  ShapeFamily = ShapeFamily { /* 见 §5.3 */ };
    const POOL:   &'static [PoolAccess] = &[ /* 见 §5.4 */ ];
    const LAWS:   &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SELFCHECK: Option<&'static dyn BootCheck> = Some(&NC_ORACLE);
    // check() 不写 —— 默认实现由 PARAMS 自动生成(§5.2)
}
```

**为什么这个简化是对的(不只是省字)**:
- 常量可以在 **const 上下文**里用(`static`/`const` 表、数组长度、编译期断言),
  方法不能;于是整张算子表能在 **编译期** 建好,零运行期初始化;
- **不可能有"假实现"**:以前 `fn entry(&self)` 允许写任意函数体(可以返回任何东西),
  `const ENTRY` 只能是一个字符串字面量;
- 静态派发:没有虚表,没有 `&self` 间接,`check` 可取函数指针直接调。

**收益**:
- 名字/源码/槽序/dtype **物理上只有一个住址**(在算子身上)⇒ 第二份手抄不可能存在;
- 新增算子 = **在族目录里加一个文件** + 在 `ops()` 里加一行(注册语义);
- `boot` 仍从族走 `ops()` 收口 **名字唯一性** 校验(现有机器门保留);
- "**引入 vs 启用**"变成显式的:有 `impl KernelSpec` = 启用 ⇒ 可发射;否则 ∈ `INTRODUCED_ONLY`
  (由 `.cu` 入口清单 − 已登记 `entry()` 差集算出,boot 打印)。

### 2.3 属于注册表/族的、与属于算子的(边界)

| 信息 | 归属 | 理由 |
|---|---|---|
| entry 名 / .cu 源码 / 参数表 / 形状族 / 池访问 / 数值律 | **算子** | 它的契约 |
| 链接形态(StaticLib/AotCubin/Nvrtc)/ 资产 manifest / 符号解析 | **族** | 一族共享 |
| 装配期资源(workspace / plan / 句柄 / pinned 缓冲) | **族持有**,算子借用 | 见 §四(作用域) |
| 名字唯一性 / 白名单 / 未启用清单 | **注册表(boot 收口)** | 全局性质 |

---

## 三、三层准入(原则)

| 层 | 时机 | 拦什么 | 成本 | 失败表现 |
|---|---|---|---|---|
| **L1 编译期** | 写代码时 | 参数类型/数量/顺序(已有 `KernelCall`);**无 struct = 不可发射** | 零 | 不编译 |
| **L2 装配期** | boot(**构造函数**内) | 符号/资产/**形状族/长度律/池写集/oracle 对拍/不变量指纹** | 一次性 | **结构化拒启 + 一行日志** |
| **L3 运行期** | 每次发射 | 零成本哨兵:长度>0、账本标签、指针非空、范围断言 | ~0 | 指名道姓的 `OpError` |

**铁律**:
1. **能放 L2 的不放 L3**(boot 失败便宜;推理期失败 = 胡言乱语,昂贵且不可定位);
2. **L3 只留便宜且能指名的检查**;重检查(不变量、oracle)全部放 L2(warmup 窗口合法);
3. **错误信息 = 算子名 + 参数名 + 期望 + 实得 + 块 id/账本**(今日已验证:`[htod_chunks]`、`槽 xqa_sem 声明为 0 元素` 都是一句话定位);
4. **门由 boot 自动执行,不是文档纪律**(E3/E15 的教训:写下来的纪律会腐烂)。

---

## 四、构造与析构纪律(用户裁决③;本设计的骨架)

> 用户原话:"用好 struct 的构造函数,家族也有构造函数,还有析构函数。"
> 这条把"装配即校验(P4)"从**一个可选钩子**升级为**类型系统级的不变量**:
> **构造成功 ⟺ 本作用域内全部准入门已通过**。

### 4.1 族:构造函数 = 装配门,析构函数 = 资源释放

```rust
pub type FamilyFactory =
    fn(&mut dyn DeviceRes, &mut dyn Exec) -> Result<Box<dyn KernelFamily>, OpError>;

pub trait KernelFamily: Send + Sync {
    fn id(&self) -> FamilyId;
    fn ops(&self) -> &'static [&'static dyn KernelSpec];
    fn linkage(&self) -> Linkage;

    /// **构造函数**(工厂语义):族内一次完成
    ///   ① 符号/资产校验(Linkage 相关;manifest sha256 对账)
    ///   ② 每算子的形状族/长度律/池写集 静态门
    ///   ③ 每算子的 boot_selfcheck(独立 oracle + 毒值自检)
    ///   ④ 资源预钉(workspace/plan/句柄)
    /// **返回 Ok ⇒ 本族全部算子"已准入";Err ⇒ 引擎拒启**(不需要再有人记得调 init)
    fn assemble(res: &mut dyn DeviceRes, exec: &mut dyn Exec)
        -> Result<Box<dyn KernelFamily>, OpError> where Self: Sized;

    /// **析构(显式)**:可失败 ⇒ 必须被调用方 `?`(Rust 纪律:不吞可失败返回值)。
    /// shutdown/换档/测试收尾路径调用。
    fn close(self: Box<Self>) -> Result<(), OpError> { Ok(()) }

    /// 发射(名字无关;注册表分派到族)
    fn run(&self, msg: &LaunchMsg, res: &mut dyn DeviceRes, exec: &mut dyn Exec)
        -> Result<Bytes, OpError>;
}
```

**`Drop` 的定位(必须写清楚,否则会被滥用)**:
- `Drop` **只做最后一手兜底**(best-effort 释放 + 记 metrics),**不允许承载"必须成功"的语义** ——
  因为 `Drop` 无法返回 `Result`,吞错就违反了「不允许静默失败」;
- 因此官方收尾路径 = **显式 `close()`**;`Drop` 里若发现"没被显式 close 就 drop 了",
  **打 warning + 计数**(把"忘记收尾"变成可见事件,而不是静默泄漏)。

### 4.2 算子:默认 ZST 单例;有资源者由族持有、算子借用

```rust
/// 无状态算子(= 绝大多数):**标记类型**(零大小),元数据全是关联常量。
/// 构造 = 常量,不可能失败 ⇒ 不需要工厂;**也不需实例** ——
/// 族的 `OPS` 表用 `meta_of::<T>()` 在编译期把它擦成数据(见 §5.2)。
pub struct NcAttnBf16V2;                 // ZST 标记类型,没有字段、没有 static 实例

/// 需要资源的算子(workspace/句柄/plan):资源**不属于算子**,由族在 assemble 期
/// 构造并持有,算子只拿 `&'a Res`(显式生命周期;rust-style「显式生命周期」律)。
pub struct PagedDecodeV2<'a> { ws: &'a PinnedWorkspace }   // 借用,不拥有 ⇒ Drop 无责
```

**三条纪律**:
1. **默认无状态**;有状态必须显式声明生命周期,**禁止把设备资源塞进算子自己的字段**(那是族的作用域);
2. 算子的"构造"只有两种形态:① **标记类型 + 关联常量**(无资源,零构造)② 族工厂内构造
   (有资源,生命周期挂在族上);**没有第三条路径** ⇒ "先构造一个半成品再 init" 这种危险形态被物理排除;
3. 元数据必须是 `&'static`(编译期常量)**或**由族构造时注入的**不可变**引用 ——
   禁止运行期改元数据(否则"准入"就成了一句空话)。

### 4.3 与现有 `init()` 的关系(迁移)

| 现在 | 新 |
|---|---|
| `KernelSpec::init(&mut self, res, exec)`(族实现,可被漏调) | `KernelFamily::assemble(res, exec) -> Result<Box<dyn Self>>`(**构造即门**;漏调 = 拿不到实例,物理不可能发射) |
| `Registry::boot()` 逐个 `init` | 逐个调 `FamilyFactory`,任一 Err ⇒ 拒启(语义同,但"门在构造函数里") |
| 无收尾 | `close()` 显式 + `Drop` 兜底告警 |
| `validate()`(0 调用者) | **算子**的 `check()`,由**上层统一调用**(见 §5.2) |

---

## 五、新 `KernelSpec`:单算子声明式契约

### 5.1 参数声明表(单一真源 → 派生五处视图)

```rust
pub enum ParamKind  { Tensor, Scalar }
pub enum ParamRole  { In, Out, InOut, Scratch, Const }      // 输出/临时/常量必须显式
pub enum ScalarTy   { F32, I32, U32, F32Bits, I32Bits }     // ★位型走私显式化(§5.6)
pub enum Dtype      { F16, BF16, F32, U32, U8 }

pub struct Param {
    pub name:  &'static str,
    pub kind:  ParamKind,
    pub role:  ParamRole,
    pub ty:    ParamType,           // Tensor{dtype, rank} | Scalar(ScalarTy)
    pub shape: ShapeRule,           // 符号维 + 维间关系
    pub range: RangeRule,           // 数值/长度范围
    pub align: Option<usize>,       // 向量化访存前提(8B/16B)
}

pub struct ShapeRule {
    pub dims: &'static [&'static str],   // N/B/Hq/Hkv/D/PAGE… 同名维全局一致(自动互证)
    pub rels: &'static [Rel],            // D == Hkv*HD、out.numel == in.numel、mask=ceil(Q/32)…
}
pub struct RangeRule { pub min: Option<f64>, pub max: Option<f64>, pub set: Option<&'static [f64]> }
```

**由这一张表派生(五处视图,零第二份手抄)**:
1. **线槽序**(取代此刻集中在 `list.rs` 的 `sig` 字符串);
2. **发射期 `check()` 主体**(表能表达的自动生成,无需逐算子手写);
3. **客户端 builder**(models 侧 `kernel_call` 消费同表);
4. **`.cu` 签名机器互证**(C2 升级:解析算子自己那份 `source()` 的 `extern "C"` 签名 ⇔ 表);
5. **文档**(`--print-op-table` 直接打印,取代手写矩阵)。

### 5.2 trait 草案:关联常量 + 编译期擦除(**零虚表**)

> ⚠️ **一个必须记住的连带后果(已用 rustc 实证 E0038)**:**带关联常量的 trait 不能做
> trait 对象**(`dyn KernelSpec` 不成立,因为常量没有 vtable 槽)。
> 这不是代价,而是**逼出更好的形态**:算子侧彻底静态化、元数据在编译期擦成纯数据表。

```rust
// ── 算子 trait:只有关联常量 + 一个可默认的静态方法(无 &self) ──────
pub trait KernelSpec: 'static {
    const ENTRY:  &'static str;                       // 与某 .cu 入口一对一
    const SOURCE: &'static str;                       // include_str!(同目录 .cu)
    const PARAMS: &'static [Param];                   // ① 参数表(单一真源)
    const SHAPE:  ShapeFamily;                        // ② 几何
    const POOL:   &'static [PoolAccess] = &[];        // ③ 内存
    const LAWS:   &'static [Law] = &[];               // ④ 数值律
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;      // ⑦
    const DETERMINISM: Determinism = Determinism::Deterministic;       // ⑦
    const SELFCHECK: Option<&'static dyn BootCheck> = None;            // ⑥

    /// ⑤ 运行时检查(上层统一调用)。默认由 PARAMS 自动生成;
    /// 只有跨参耦合的算子才覆写(如 XQA 的 `q_seq_len ↔ mask` 耦合)
    fn check(args: &[Arg], geom: &Geom) -> Result<(), OpError> {
        default_from_params::<Self>(args, geom)
    }
}

// ── 擦除形态:纯数据(可 const 构造 ⇒ 零运行期初始化;无 dyn 特征对象) ──
pub struct KernelMeta {
    pub entry:  &'static str,
    pub source: &'static str,
    pub params: &'static [Param],
    pub shape:  ShapeFamily,
    pub pool:   &'static [PoolAccess],
    pub laws:   &'static [Law],
    pub capture: CaptureEvidence,
    pub determinism: Determinism,
    pub check:  fn(&[Arg], &Geom) -> Result<(), OpError>,  // = <T as KernelSpec>::check
    pub selfcheck: Option<&'static dyn BootCheck>,
}

/// 编译期擦除(实测可行:关联常量 + trait 方法取 fn 指针,均在 const 上下文内)
pub const fn meta_of<T: KernelSpec>() -> KernelMeta {
    KernelMeta { entry: T::ENTRY, source: T::SOURCE, params: T::PARAMS, shape: T::SHAPE,
                 pool: T::POOL, laws: T::LAWS, capture: T::CAPTURE,
                 determinism: T::DETERMINISM, check: T::check, selfcheck: T::SELFCHECK }
}
```

**族侧保持 dyn 兼容**(族**有状态**:workspace/plan/句柄 ⇒ 需要实例 ⇒ `Box<dyn _>`):
实测 `fn assemble() -> Result<Box<dyn KernelFamily>, OpError> where Self: Sized;`
**不进 vtable、不破坏 dyn 兼容** ⇒ 工厂函数指针表照旧可用。

```rust
pub trait KernelFamily: Send + Sync {
    fn id(&self) -> FamilyId;
    fn ops(&self) -> &'static [KernelMeta];   // 必填方法(实现里一行 `Self::OPS`)
    fn linkage(&self) -> Linkage;
    fn run(&self, msg, res, exec) -> Result<Bytes, OpError>;
    fn close(self: Box<Self>) -> Result<(), OpError> { Ok(()) }
    fn assemble(res, exec) -> Result<Box<dyn KernelFamily>, OpError> where Self: Sized;
}

impl XqaFamily {
    /// 算子表是 **const**:编译期建好,零运行期初始化
    const OPS: &'static [KernelMeta] = &[meta_of::<XqaVerify>(), meta_of::<XqaDecode>()];
}
impl KernelFamily for XqaFamily {
    fn ops(&self) -> &'static [KernelMeta] { Self::OPS }   // 一行样板
    // …
}
```

**唯一的调用点(新增;这是"由上层统一调用"的落地)**:

```
上层(models/interpreter)发射前:
    let meta = registry.meta(msg.kernel.name)?;      // 未启用 = 结构化拒(deny-by-default)
    (meta.check)(&msg.args, &geom)?;                 // ← 统一入口,所有算子同一条代码路径
    family.run(msg, res, exec)?;                     // 实际发射(注册表分派)
```

> **连带好处**:`EXTENT/几何/池访问/数值律/捕获安全` 全是常量 ⇒ 族构造函数里的装配期门
> 与 `--print-op-table` 文档生成 **共用同一份数据**(不存在"文档与代码漂移"这个失败模式)。

### 5.3 形状族 `ShapeFamily`

```rust
pub struct ShapeFamily {
    pub pool_layout: LayoutTag,     // Knhd | Classic | DraftPool(类型化,取代隐式协定 E14)
    pub kv_dtype:    Dtype,         // F16 | BF16 | Fp8E4M3
    pub head_dim:    OneOf<[u32]>,  // {128} | {256} | {128,256}
    pub page:        OneOf<[u32]>,  // {16,32,64,128}
    pub grp:         Option<u32>,   // hq/hkv(GQA)
    pub extent:      ExtentLaw,     // 见下
    pub tile:        Option<Tile>,  // 如 XQA 的 M_TILESIZE(实例化行数上界)
}
```

**`ExtentLaw`**:`numel > 0`(**全局硬律**,直接拦 `xqa_sem` 零长槽)/ `numel(shape) == len` /
对齐前提(256B 基底,MISALIGNED_ADDRESS 案机器化)/ page ∈ 声明集 / 维间关系。

### 5.4 `PoolAccess`(拦"池写不全/写错布局")

```rust
pub struct PoolAccess {
    pub pool: &'static str,   // "kv.main" / "kv.draft" / "gdn.rec" …
    pub mode: AccessMode,     // Read | Write | ReadWrite
    pub layout: LayoutTag,    // 必须与池的权威布局一致
    pub dtype: Dtype,
    pub region: Option<RegionSpec>,   // 负责写的区域(写集穷举审计用)
}
```

**装配期两道审计(在族的构造函数里跑)**:
1. **写集穷举**:每个池的每段区域**恰好一个**权威写者;缺(漏写者 = E10 全零案)/双 ⇒ **拒启**;
2. **毒值自检**:池灌 NaN → 跑一次 warmup forward → 断言 (a) 出口无 NaN (b)
   **实际被改写的字节集 == 声明写集**(哨兵逐字节比对)。
   ⇒ E10 的"99.6% 输出未写"在 boot 期 1 秒暴露,而非线上全零。

### 5.5 独立 oracle(拦"静默错值")

```rust
pub trait BootCheck { fn run(&self, res: &mut dyn DeviceRes) -> Result<CheckReport, OpError>; }
```
三条纪律:①**参考必须是独立实现**(E8 的教训:同源副本对不出双错);②**装配期默认执行**,失败 = 拒启;③报告含逐元素 `max|err|` + 首个越界元素。

### 5.6 位型走私显式化

现状 `f32::from_bits(kv_len)` / i32 页表位型走私 —— 全靠注释。
**新规**:`ScalarTy::{U32Bits,I32Bits}` 是**显式 token**;表里声明 `U32Bits` 的槽,
`check()` 自动断言发射端确实用 `as_u32_bits()` 构造;内核侧 parse 必须同型。
⇒ `xqa_seq`(整请求总 KV 长)/`page_table`(i32)/`q_scale_bits` 从"注释约定"变"类型契约"。

### 5.7 数值指纹

固定微小输入 → 每算子输出 checksum → 入 `assets/ops-fingerprint.toml`(进 git)。
变化 = **警告**(合法优化也会变),要求改动者显式更新并在提交信息说明。
⇒ 本轮 `M_TILESIZE` 这种"几何变了没人知道"会被自动点出。

---

## 六、显存块结构体扩展(`Bytes` → `VramBlock`)

### 6.1 现状与代价

```rust
pub struct Bytes { pub id: u64, pub len: usize }   // len 是"元素数(f32)"
```
- dtype/形状/布局/角色/生命周期/**账本**全丢;
- cuda 后端 `next_block` **每个 GpuServer 实例各自从 1 计数** ⇒ id 撞号 ⇒ 坑册 §27 的
  `DeadBlock{id:11}` 六步定位(元凶 = 跨账本解析);
- `len` 单位是 f32 元素 ⇒ 2B/1B 池靠心算,错一次 = OOB 或半截写。

### 6.2 新结构(字段 × 谁填 × 拦哪类案)

```rust
pub struct VramBlock {
    // 身份(消除多账本)
    pub ledger: u32,          // 分配它的 GpuServer/actor 序号(分配时签)
    pub local_id: u64,        // 该账本内单调计数
    // 布局与类型(消除单位/布局歧义)
    pub dtype: Dtype,
    pub shape: Shape,
    pub layout: Option<LayoutTag>,
    // 生命周期与角色
    pub place: Place,         // Device | PinnedHost | StackTemp(E12 机器面)
    pub origin: Origin,       // Owned(cudaMalloc) | Carved(捕获 slab 切块)
    pub role: BlockRole,      // Persist | Scratch | Activation | Output
    pub generation: u32,      // 扩容换代(E11 坟场案)
    // 溯源
    pub owner_op: Option<OpId>,   // 哪次发射创建/最近写入(仅 debug 填)
}
```

| 字段 | 谁填 | 拦什么 | 成本 |
|---|---|---|---|
| `ledger` | 分配点 | **跨账本 id 撞号**(§27) | 0 |
| `dtype`+`shape` | 分配点 | OOB / 半截写 / 单位心算错 | 0 |
| `layout` | 池分配点 | E14 双寻址、E15 契约腐烂 | 0 |
| `place` | 分配点 | E12 staging 生命周期 | 0 |
| `origin` | 分配点 | 捕获期误 `cudaMalloc` | 0 |
| `generation` | 扩容点 | **E11 坟场案** | 0 |
| `owner_op` | 发射点 | 定位到算子 | debug only |

### 6.3 两道门

- **L2(族构造函数内)**:所有池块的 `(dtype, shape, layout)` 与 `shape_family()` 逐一对表 ⇒ 不匹配 = 拒启;
- **L3(零成本)**:
  - 解析时断言 `ledger == 本账本` ⇒ 失败 = `块 id=11 属 ledger 2,被 ledger 1 解析`(**6 步 → 1 行**);
  - `numel(shape) == len`、`numel > 0`、`align` 满足;
  - `generation` 与调用方期望代不符 ⇒ 拒(E11 当场炸在源头)。

---

## 七、与既有资产的关系(不重复造)

| 既有 | 新设计里的位置 |
|---|---|
| `KernelCall` 声明期状态机(models) | L1 保留;消费点从"手抄 sig"改成"`params()` 派生" |
| `list.rs`(110 条 Entry) | **瘦身为纯注册表**;元数据迁到算子自托管(§2.2) |
| registry `KernelSpec`(7 个族) | **更名 `KernelFamily`**;`init` → `assemble`(构造函数) |
| driver 谓词(`paged_decode_ok` …) | **收编进 `ShapeFamily`** |
| `kv_manifest_gate` | 并入 `PoolAccess` 写集穷举 |
| golden / 恒等门 | 保留为人工深门;`selfcheck` 是它的**自动化前置层** |
| `Law`/`FieldStats`/`OpError::Invariant` | 复用;不变量门从 debug-only 升为 **warmup 强制** |
| `InvariantBox` | 保留;`place=StackTemp` 建在其上 |

---

## 八、增量施工路径(每步独立可验)

| 步 | 内容 | 验收 | 量级 |
|---|---|---|---|
| **P0** | **更名 + 注册表瘦身**:`KernelSpec`→`KernelFamily`(7 处 impl);`init`→`assemble`(构造函数)+ `close`(析构);`list.rs` 只剩 `family_factories()`;`ops()` 上移到族 | 编译过 + kernels lib 17/17 + 全量 e2e 17/17(行为零变化) | 3h |
| **P1** | **`VramBlock` 扩字段**(ledger/dtype/shape/align)+ 分配点填充 + 解析点断言 | 单测:跨账本拒 / 单位错拒;e2e 绿 | 1d |
| **P2** | **L3 哨兵**:`numel>0` / `numel==len` / `ledger` 匹配 —— **先把今天两个 bug 的类堵死** | 故意造零长槽/跨账本 = 结构化报错(带名字) | 0.5d |
| **P3** | **算子自托管元数据 + 参数表 + `check()` 自动生成**:先给 5 个生产算子(`nc_bf16_v2` / K0 / `qknorm_rope_kv_insert` / `paged_attn_v2_*` / marlin)建表;`.cu` 签名互证升级 | 表 ⇔ 各自 `.cu` 逐参一致;`check()` **真有调用者**(今天 0) | 2d |
| **P4** | **`ShapeFamily` + `ExtentLaw` + `PoolAccess`**(收编 driver 谓词与 manifest 门)+ **白名单硬门**(生产 flavor 列出允许算子;未准入出现 = 拒启) | boot 拒非法几何;写集穷举 + 毒值自检 | 2d |
| **P5** | **独立 oracle**(`selfcheck`)给"曾出过事"的算子(gdn_chunked / marlin / NC / K0) | 逐元素误差报告;故意注错 = boot 拒启 | 2d |
| **P6** | **算子级 checksum + 自动二分**(用户裁决④:**立项**,排 P5 之后):SSA 拓扑序 diff,一次运行定位首个不一致算子 | 注入单点错误 → 工具一次指认 | 2-3d |

**范围裁剪**:P3-P5 **不要求 103 个入口全建表**,只要求生产在用的(起步 ~20)。
未建表的入口保持"仅引入",物理上发不出去 —— 这正是 deny-by-default 的价值。

---

## 九、验收门(定量)

1. `check()` 调用者 **≥1 且唯一**(上层统一);未启用算子被发射 ⇒ 结构化拒;
2. `.cu` 签名 ⇔ 算子参数表 **逐参机器互证**;
3. **零长 / 跨账本 / 单位错** 三类各有一个"故意注入"的负例测试,必须结构化报错(带算子名+参数名);
4. `selfcheck` 覆盖"曾出过事"的算子清单,失败 = 拒启;
5. **构造函数即门**:任一装配期门失败 ⇒ `assemble` 返回 Err ⇒ 引擎拒启(不允许"起来了但没检查");
6. **析构可见**:未显式 `close()` 就 `drop` ⇒ 打 warning + 计数;
7. 全量 e2e **17/17** 与恒等门(AL 逐位不变)不回归;`cargo clippy` 保持 0。

---

## 十、风险与取舍

| 风险 | 缓解 |
|---|---|
| 字段膨胀伤热路径 | 全字段 32-64B 量级;`owner_op` 仅 debug 填 |
| **门变成形式主义** | 门在**构造函数**里跑 + 失败拒启 + 验收门第 3 条(注入负例)强制它真跑 |
| 建表 vs 快速试验的摩擦 | 提供"**仅引入**"档:新算子可无表跑测试,**但不能进生产 flavor**(§八 P4 硬门) |
| 独立 oracle 维护成本 | 只对"曾出过事"的算子强制;其余只留不变量指纹 |
| 更名/瘦身带来大面积 diff | P0 单独一刀、纯机械、零行为变化;先合入再加能力 |
| 数值指纹假阳性 | 只警告不改行为,要求提交信息说明 |
| `Drop` 被当成收尾路径 | 文档 + 验收门第 6 条:未 close 即警告计数 |

---

## 十一、待决点(仍需裁决)

1. **族目录布局**:`server/<family>.rs` + `server/<family>/<op>.rs` 是否可接受?
   (Rust 允许同名 `foo.rs` 与 `foo/` 并存;若不喜欢,可改 `server/families/<family>.rs` + `.../<family>/<op>.rs`)
2. **`VramBlock` 的 id 编码**:`(ledger<<40)|local` 单 u64(改动最小)vs 双字段(可读性更好)?
3. ~~算子暴露方式:动态派发 vs 静态派发~~ ⇒ **已由用户裁决(关联常量)解决**:
   `const` ⇒ 无 `dyn KernelSpec`(E0038 实测);算子侧全部静态化 + 编译期擦成 `KernelMeta` 表;
   族侧因有状态仍用 `Box<dyn KernelFamily>`(`assemble` 带 `where Self: Sized` 不破 dyn 兼容)。
   **新待决**:`KernelMeta` 里 `check` 用 `fn` 指针(已选)vs 泛型 `<T>::check` 单态化(零间接但表要泛型化)。
4. **P0 是否一并把 `server/` 下的族文件重命名**为 `<family>_family.rs`(可读性)还是保持 `<family>.rs`(diff 最小)?
5. **白名单硬门的粒度**:flavor 里列**算子名**白名单,还是列**族**白名单 + 族内自动全启用?

---

## 十二、裁决记录(2026-10-13 用户授权"怎么好写怎么来")

| # | 议题 | 裁决 | 理由 |
|---|---|---|---|
| 1 | 族/算子目录布局 | `server/<family>.rs` 与 `server/<family>/<op>.rs` **并存**(Rust 允许) | 改动最小;算子与它的 `.cu` 对望 |
| 2 | `VramBlock` id 编码 | **单 u64 位拼接** `(ledger<<40)|local` + 访问器 | 线格式 `Bytes{id,len}` 不变,churn 最小;报错时可同时打印两段 |
| 3 | 族文件名 | **不改**(`xqa.rs` 保持) | diff 最小 |
| 4 | 生产白名单粒度 | **算子名白名单**(缺省空 ⇒ 生产档必须显式列);`--print-op-table` 输出可粘贴清单 | 与 deny-by-default 的本意一致 |
| 5 | `KernelMeta.check` 形态 | **fn 指针**(表是纯数据) | 关联常量已定;fn 指针让表可 const 构造,写起来最省 |
| 6 | 算子 trait 元数据 | **全部关联常量**(用户提案);算子侧零虚表 | E0038 实测 ⇒ 不可 `dyn KernelSpec`;编译期擦成 `KernelMeta` |

## 十三、首批施工范围(第一棒)

本棒 = **P0 + P2(+P1 最小面)**,按序、每步可验、任一 checkpoint 可停:

| 序 | 内容 | 完成判据 |
|---|---|---|
| **D1** | **P0**:`KernelSpec`→`KernelFamily`(7 实现 + 全库引用);`init`→`assemble`(构造函数语义)+ `close()`(析构,可失败)+ `Drop` 未 close 告警计数。**行为零变化**;`list.rs`/元数据位置**本棒不动**(自托管是 P3) | 编译过 + kernels lib 17/17 + clippy 0 + 恒等门(单场景) |
| **D2** | **P2 哨兵**:全局 `numel > 0` 律(零长块/槽 ⇒ 结构化报错带名字);`numel(shape) == len`;解析点错误带块 id | 注入负例测试(零长槽/零长块)必须带名字报错 |
| **D3** | **P1 最小面**:块账加 `ledger` + `dtype` + `shape`(分配点填;id 按裁决 2);解析点断言 `ledger == 本账本` | 负例:跨账本 id ⇒ 报"块 id=N 属 ledger A,被 ledger B 解析" |

### 4.4 补充:兜底告警必须"正常路径静默"(2026-10-13 复核实测)

`Drop` 里那条"未经 `close()` 即 drop ⇒ warning + 计数"的兜底,**只有在收尾路径真的
走 `close()` 时才有信号价值**。本棒首版漏了这一步:`GpuServer::shutdown_fence` 里仍是
`drop(self.registry.take())`(裸 drop)⇒ 一次正常 e2e 启动就刷 **5 条**告警
(每族一条)⇒ **告警疲劳,真漏反被淹没**。

修:新增 `OpRegistry::close(self) -> Result<(), OpError>`(逐族 `close()`,返回首个错误;
teardown 无法上抛 ⇒ 调用方记录),并在 `shutdown_fence` 里走它(次序不变:仍在 ctx 拆掉之前)。
复验:正常退出**零告警** + 恒等门 `AL=1.29` 逐位一致 + clippy 0。

**律**:**兜底路径的告警,必须证明它在正常路径上是静默的**;否则"加了监控"其实是"加了噪声"。

---

## 十四、生产算子审查(实测,2026-10-13;P3 建表范围的唯一依据)

> 用户裁决(2026-10-13):「**审查当前的生产中的算子,并且仅对生产中的算子进行建表**;
> 重新设计 `KernelPick` 抽象,我们需要根据新的 `KernelSpec` 来挑选算子」。
> 本节 = 那次审查的**机器证据**(不靠代码阅读推断)。

### 14.1 方法

真跑生产档(不是测试场景):`target/release/server` + `[cuda] launch_time = true`
(每次发射打印 `[launch-time] <耗时> <核名>`),3090 Ti(空闲卡),发两条请求
(短 prompt 64 tok + 2540 tok 长 prefill),再对日志做 `uniq -c`。

| 档 | 配置 | 发射总数 |
|---|---|---:|
| **spec-dflash**(27B AWQ + DFlash2 d7 + fp8 主池 + gdn_chunked + XQA verify/decode + int4 头) | `--flavor spec-dflash` + 检查点目录 | 13108 |
| **awq27b**(裸 27B AWQ + fp8 主池;无草稿、无 XQA) | `--flavor awq27b` + 检查点目录 | 5239 |

原始日志:`/tmp/owl-audit.log` / `/tmp/owl-audit-bare.log`(临时;下表即其提炼)。

### 14.2 生产算子清单(并集 46 个;spec 列 = 发射次数,bare 列同)

| # | 算子 | 形态 | spec | bare | 备注 |
|---:|---|---|---:|---:|---|
| 1 | `marlin_gemm_w4a16_awq` | foreign/marlin | 2822 | 2464 | 权重 GEMM 主力 |
| 2 | `marlin_gemm_w4a16_bf16` | foreign/marlin | 175 | 0 | 草稿 BF16 权重 |
| 3 | `owl_gdn_conv_fwd_f16` | native | 2016 | 864 | chunked prefill 批核 |
| 4 | `owl_narrow_strided_f16` | native | 1666 | 1362 | 逐 token 视图 |
| 5 | `owl_gdn_l2norm_f16` | native | 672 | 576 | |
| 6 | `owl_rmsnorm_f16` | native | 520 | 455 | |
| 7 | `owl_silu_and_mul_f16` | native | 512 | 448 | |
| 8 | `owl_fused_add_rmsnorm_f16` | native | 512 | 448 | |
| 9 | `owl_add_f16` | native | 512 | 448 | fc 部分和累加 |
| 10 | `cublas_gemm_f16` | foreign/cublas | 480 | 483 | lm_head 等 |
| 11 | `owl_gdn_recurrence_varlen_gqa_f16` | native | 432 | 288 | |
| 12 | `owl_ct_repack_u32` | native | 393 | 352 | **装载域**(非轮内) |
| 13 | `owl_sigmoid_f16` | native | 336 | 288 | |
| 14 | `owl_gdn_norm_act_f16` | native | 336 | 288 | |
| 15 | `owl_gdn_gating_g_f16` | native | 336 | 288 | |
| 16 | `gdn_chunked_delta_rule_fwd` | foreign/gdn_chunked | 240 | 0 | fork-bf16 六核编排 |
| 17 | `owl_norm_rope_f16` | native | 224 | 192 | |
| 18 | `owl_gemv_dual_f16` | native | 144 | 96 | b/a 投影 |
| 19 | `owl_sigmoid_gate_mul_f16` | native | 128 | 112 | attention 门融合 |
| 20 | `owl_reshape_and_cache_fp8kv_knhd` | native | 112 | 96 | K0 写池(fp8/kNHD) |
| 21 | `flashinfer_prefill_paged_fp8kv` | foreign/flashinfer | 80 | 80 | **FI 恒启用**(≥128 tok) |
| 22 | `owl_norm_rope_bf16` | native | 55 | 0 | |
| 23 | `owl_gdn_decode_step_v2_f16` | native | 48 | 48 | D1-v2 融合 decode |
| 24 | `owl_gdn_conv_upd_dual_f16` | native | 48 | 48 | |
| 25 | `owl_dflash_conv_bf16` | native | 40 | 0 | DFlash2 2-tap conv |
| 26 | `owl_cast_f16_bf16` | native | 37 | 0 | 铸边界 |
| 27 | `vllm_reshape_and_cache_bf16` | native | 35 | 0 | 草稿池 K0 |
| 28 | `owl_add_bf16` | native | 28 | 0 | |
| 29 | `cublas_gemm_bf16` | foreign/cublas | 22 | 0 | |
| 30 | `owl_fused_add_rmsnorm_bf16` | native | 20 | 0 | |
| 31 | `vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256` | native | 16 | 16 | 短 prefill 尾段 |
| 32 | `owl_xqa_verify` | foreign/xqa | 16 | 0 | verify 臂 |
| 33 | `owl_xqa_decode` | foreign/xqa | 16 | 0 | 裸 decode 臂(② 转正) |
| 34 | `owl_qknorm_rope_kv_insert_f16_fp8kv_knhd` | native | 16 | 16 | 融合插池 |
| 35 | `owl_argmax_f32idx_f16` | native | 11 | 3 | 采样 |
| 36 | `owl_silu_and_mul_bf16` | native | 10 | 0 | |
| 37 | `owl_naive_attn_nc_bf16_v2` | native | 10 | 0 | 草稿 NC 注意力(③ 转正) |
| 38 | `owl_embed_f16` | native | 10 | 7 | |
| 39 | `owl_rmsnorm_bf16` | native | 9 | 0 | |
| 40 | `owl_concat_rows_f16` | native | 6 | 0 | |
| 41 | `owl_topk16_f16` | native | 2 | 0 | 草稿 top-16 |
| 42 | `owl_dflash_select_bf16` | native | 2 | 0 | selector 打分 |
| 43 | `owl_cast_bf16_f16` | native | 2 | 0 | |
| 44 | `owl_concat_rows_f32` | native | 1 | 0 | 首轮 f32 拼接 |
| 45 | `vllm_paged_attention_v2_fp8_knhd_hd256bs32` | native | 0 | 16 | 裸档 decode 主核 |
| 46 | `vllm_paged_attention_v2_reduce_f16_hd256` | native | 0 | 16 | 裸档 decode 归并 |

**形态分布**:native NVRTC **37** / foreign FFI **9**(marlin×2、cublas×2、
flashinfer×1、gdn_chunked×1、xqa×2 + `gdn_scalar` 未用)。

### 14.3 审查的三个结构性发现

1. **`prefill_split` 家族在生产档里彻底不在场** —— `dispatch.prefill_split` 缺省
   `false`,且 FI 在 `tokens ≥ 128` 时优先接管(`attention.rs` 分派序:FI → split
   → chunked)。所以 `owl_prefill_split*` / `prefill_split_reduce*` 六个入口 + 全部
   kNHD 变体 = **仅引入、未启用**。这直接推翻"长 ctx 走 split"的旧印象(那是 FI
   影子池时代的事)。
2. **生产实际只用 4 个池布局/量化组合**:主池 fp8 + kNHD(`*_fp8kv_knhd`)、
   草稿池 bf16 + classic、主池 fp8 + classic(chunked 尾段)、f16 归并。**classic
   布局的 f16 主池全族、fp8 非 kNHD 全族都不在生产**。
3. **dev/f32 演示核与逃生舱核全部不在生产**(`owl_*_f32` 约 30 条、`probe_ts_f16`、
   `owl_naive_decode_attn_*`、`owl_gdn_*_f32` …)—— 它们是"引入"而非"启用",
   与 §八 P3「不要求 103 个入口全建表」一致。

### 14.4 P3 建表范围(定稿)

| 范围 | 条目 | 处理 |
|---|---|---|
| **生产集** | 上表 46 条 | **建表**(PARAMS/形状族/池访问/数值律) |
| **生产相邻(条件臂)** | `owl_reshape_and_cache_fp8kv`(classic 池档)、`owl_qknorm_rope_kv_insert_f16_fp8kv`(classic 池档)、`owl_naive_attn_nc_fp8kv_bf16`(fp8 草稿池档) | 一并建表(同一族的兄弟;缺表 = 换池档时无门) |
| **勘误** | ~~`vllm_paged_attention_v2_fp8_seq_knhd_hd256`~~ | **不存在这个核**:`*_seq_knhd` 是 **driver 的 op-id**(`attn.paged_decode_v2_fp8_seq_knhd`),落到同一个核名 `..._hd256bs32`(E16 修后核内读 `blockIdx.y` 承担多序)。已在产集内,无需另建 |
| **其余(~55 条)** | split 家族、paged v1、naive decode、f32 演示核 | 保持"仅引入";P4 白名单硬门保证生产档不可能发射它们 |

### 14.5 `KernelPick` 重设计(用户指令)

现状:`KernelPick { name: &'static str, shape: Shape }` —— 名字是**裸字符串**,
发射端再按名字去 `list.rs` 查源/查槽序(第二份真源),准入检查也只能靠名字。

**实际采用的形态**(2026-10-13 落地;与本节初稿的"加 `meta` 字段"有一处偏离,
理由如下):

```rust
pub struct KernelPick { pub name: &'static str, pub shape: Shape }   // 字段不变

impl KernelPick {
    /// **按 KernelSpec 拾取**(生产臂用):名字 = 算子关联常量(编译期耦合)
    pub const fn of<T: KernelSpec>(shape: Shape) -> Self { Self { name: T::ENTRY, shape } }
    /// 准入声明查询(未建表 = None = 仅引入)
    pub fn meta(&self) -> Option<&'static KernelMeta> { crate::ops::meta(self.name) }
}
```

- **偏离理由**:driver 里有 **131 处** `KernelPick { .. }` 字面量,加字段 =
  131 处纯机械改写(设计稿 §十 自己列的风险项"大面积 diff")。改成"按名查表"
  后:生产臂用 `of::<T>()` 拿到**类型耦合**的名字(核名字面量在 driver 里消失),
  未建表臂零改动 —— 语义等价,diff 最小。
- `list::with_pick(pick)`:meta 在场 ⇒ `Kernel{ name/source/launch/meta }` 全取元数据
  (foreign 的 `SOURCE` 为空 ⇒ 源码空串,server 走家族 FFI);否则回落登记表。
- `Kernel::slot_tokens()` = **槽序唯一权威**(优先序:元数据 → 登记表 → 逃生舱签名);
  `models::lower_kernel` 不再自己 `split(',')` 手拼权威。
- server 侧 `check()` 走 `OpRegistry::meta(name)`(boot 期建一次的名字→元数据索引,
  与 `knows()` 路由表**分离** —— 原生核有元数据但不得进路由表)。
- 名字字面量在 driver 里绝迹(改从算子关联常量取),`gname/leak_name` 类拼名彻底无必要。


### 14.6 一处结构性偏离:算子声明住 `kernels/src/ops/`,不是 `server/native/`

用户裁决(2026-10-13)是「原生算子 struct 住 `server/native/`」。落地时发现
**物理上做不到**,原因是一条 feature 边界:

| 事实 | 证据 |
|---|---|
| `server` 模块 = `#[cfg(feature = "device")]` | `kernels/src/lib.rs` |
| `server::KernelFamily` 的签名引用 `device::Exec` | `registry.rs` |
| **models 以 `features = ["marlin"]` 依赖本 crate**(无 device) | `crates/models/Cargo.toml` |
| `driver.rs` 是**零门**的,models 经它拾取 | `driver::resolve` |

⇒ 若算子声明住 `server/`,models 就看不到 `KernelPick` 携带的元数据,
"按 KernelSpec 挑选算子"在编译期就断了。
**最终形态**:声明(参数表/形状族/池/律)= 零门模块 `kernels/src/ops/`
(按 driver 的域分模块:`ops::attn` ↔ `driver::attn`,以此类推);
`server/native.rs` 保留为**原生算子的服务端面**(注册/装配门,device 门内),
两处引用**同一份** `KernelMeta`。设计稿 §2.1 的"算子与 .cu 对望"由
`ops::<域>` 与 `cu/<域>` 的目录对称性承担。

### 14.7 P3 进度(2026-10-13;**B1+B2 完工 —— 生产集 100% 建表**)

| 项 | 状态 | 证据 |
|---|---|---|
| `spec.rs` 词汇(Param/ShapeFamily/PoolAccess/KernelMeta/meta_of/check) | ✅ | `cargo test -p owl-kernels --lib spec::` 6/6 |
| 槽序**派生**(const fn;`PARAMS` 唯一真源) | ✅ | `spec::tests::args_derivable_in_const_context` |
| 生产算子建表(**50 张**) | ✅ | `ops/` 十一域;**生产集 46 个算子 100% 覆盖**(+4 张生产相邻/家族补齐);含生产里每一个池写者/读者(E10/E14/E15 的机器面) |
| 表 ⇔ `.cu` 签名机器互证 | ✅ | `tests/op_tables.rs` 6/6:**41 张原生表逐参一致**;9 张 foreign 表与 face 常量对拍 |
| `KernelPick::of::<T>()` + `meta()` | ✅ | `driver.rs`;**45 个生产臂全部携带声明**(门:`production_picks_carry_admission_meta`) |
| `Kernel::slot_tokens()` 唯一槽序权威 | ✅ | `list.rs`;`models::lower_kernel` 改消费 |
| `OpRegistry::meta()` 索引 + boot 报告 | ✅ | boot 打印 `已建表 50 / 原生登记 110(未建表 69)/ 家族臂 11(未建表 2)` |
| **族注册语义** `KernelFamily::ops()` | ✅ | 六族接入(cublas/marlin/xqa×2/gdn_chunked/FI);`names()` = **族主名 ∪ ops 名**(手列名字全删) |
| 族装配期一致性门 + 拒启路径收尾 | ✅ | `init_all`:`ops ⊄ names` ⇒ 拒启;拒启前逐族 `close()`(不留兜底告警噪声) |
| 命名 static 单一提升点 | ✅ | 门:`ops::tests::domain_tables_share_one_promotion_point`(地址相等) |
| `names()` 并集律 | ✅ | 门:`registry::tests::names_is_id_union_ops`(实测坑:纯派生会静默丢掉部分建表族的主名) |
| **L3 哨兵唯一调用点**(foreign+native 汇合) | ✅ | `backends/cuda/server/mod.rs::admission_check` |
| 生产实跑零违规 | ✅ | 真 server(`--flavor spec-dflash`)短+长 prompt:**44 个核、全谱发射、准入错误 0 条**(FI / chunked / 草稿池 K0 / XQA / marlin / GDN 全族) |
| `Dtype::U8`(池块字节承载如实标注) | ✅ | `contract.rs` + 3 处消费点 |
| 恒等门零行为差异 | ✅ | **同命令 A/B**:改动前后 `AL=1.17 / rounds=18 / sum_m=21` 逐位一致,生成文本逐字相同 |
| clippy | ✅ | 0(仅 4 行 build.rs 预编资产提示) |

### 14.8 P4 进度(2026-10-13;**形状族 + 池写集 + 白名单硬门三件完工**)

| 项 | 状态 | 证据 |
|---|---|---|
| **几何门**(`ShapeFamily::accepts`,拾取即验) | ✅ | 声明驱动 + **未知容忍**(0/None = 跳过);`driver::resolve` 唯一调用点;门:`spec::shape_family_accepts_declaration_driven` |
| 几何事实入 `OpEnv` | ✅ | `head_dim`/`grp` 由引擎 boot 自 `ModelDims` 合入(被动律);`kv.head_dim/grp` 单源 |
| 形状族填充 | ✅ | 池/注意力 11 张表(head_dim/page/grp/kv_dtype/tile);**其余显式 `NONE`**("无几何约束"是显式声明,不是漏写) |
| **池写集穷举审计**(`ops::audit_pools`) | ✅ | 三条规则:R1 读必有人写(E10)/ R2 声明自洽 / R3 活跃写者不歧义(E14);`POOL` 声明 15 张表 |
| 审计输入 = 生效集合 | ✅ | `audit_active(Some(白名单))`;门:生产集 ✓ / classic 池档 ✓ / fp8 草稿池档 ✓ / 混布局 ⇒ 拒 |
| **白名单硬门**(`dispatch.allowed_ops`) | ✅ | 装配期(已建表校验 + 池审计)⇒ 拒启;发射期 ⇒ deny-by-default;`spec-dflash` 44 项 / `awq27b` 27 项 |
| 白名单两处一致锁 | ✅ | `apps/server/tests/ops_whitelist.rs`:flavor 清单 ⇔ `ops::PROD_*_2026_10_13` 逐项一致 + 全建表 + 池审计 |
| 谓词收编 | ✅ | `paged_decode_ok` ⇔ `ShapeFamily` 声明互证(门:`predicate_lock`) |
| 实测拒启 | ✅ | 从白名单去掉 `owl_sigmoid_gate_mul_f16` ⇒ boot capture 当场拒启(指名道姓) |
| 全量 e2e | ✅ | **17/17 场景 365.6s**,几何门零误伤(0.8B/27B × f16/fp8 × spec/裸 × 全池) |
| clippy | ✅ | 0(默认 + 全 feature 两档) |

**未完成(P5/P6)**:
1. **毒值自检(NaN 哨兵)**:设计 §5.4 的"池灌 NaN → warmup → 断言出口无 NaN
   + 实际改写字节集 == 声明写集" —— 需要引擎级 warmup 与逐字节比对管道,
   本棒只落了**声明间互证**那一半(R1-R3);
2. **`selfcheck` 独立 oracle**(P5):`SELFCHECK` 字段已立,尚无实现;
3. **`ShapeRule.dims/rels`**:符号维与维间关系目前全部 `NONE` ——
   真正的"形状族逐参约束"(如 `D == Hkv*HD`)未填;
4. **数值指纹**(设计 §5.7):`assets/ops-fingerprint.toml` 未立;
5. 顺带:`owl_gdn_decode_step_f16`(非 v2)与 `gdn_scalar` 臂仍未建表(不在生产集);
6. **新发现(非本专项)**:`apps/server/tests/e2e.rs` 的"算术 2+2"用例在**改动前
   的树上同样失败**(A/B 实证),返回的是**上一个 session 的答案**(Paris)——
   即 server 路径存在会话串扰类缺陷,已挂账(与算子准入无关)。
