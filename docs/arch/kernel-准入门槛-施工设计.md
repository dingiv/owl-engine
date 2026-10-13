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

### 14.9 P5 第一棒(2026-10-13;独立 oracle + 毒值自检落地)

| 项 | 状态 | 证据 |
|---|---|---|
| 自检运行时(ungated `CheckRunner` ↔ device 侧 `Runner` 适配) | ✅ | `spec::CheckRunner` / `server::selfcheck::Runner`;`LaunchVal` 下沉 contract(自检面零门) |
| `DeviceRes::download`(设备→host 回读唯一出口) | ✅ | 装配期 oracle 的必备原语(与 `upload` 对称) |
| **毒值自检** | ✅ | K0 fp8 池写:池灌 e4m3-NaN(`0x7F`)→ 发射 → **声明写集外字节仍为毒值**(实测未写 64000 字节)+ 越界写当场红 |
| **独立 oracle**(K0 fp8 knHD) | ✅ | e4m3 RNE 参考**逐字节相等**(0 处不符);发现并修正参考侧 ±0.0 符号语义 |
| **独立 oracle**(NC v2) | ✅ | host 全量 softmax 参考(滑窗+自块)`max|err|=3.904e-3 ≤ 2e-2` |
| **独立 oracle**(融合插池 fp8 knHD) | ✅ | host norm+rope 参考 + e4m3 逐字节 + 毒值律:`max|err|=1.953e-3(相对 1e-2)`,未写字节 7936 —— **E10 案 ② + rope 配对案(2026-10-02)两处现场同时上锁** |
| **判据敏感性证明**(负例 ×3) | ✅ | K0 换 classic 寻址 ⇒ **1512 处**不符;NC 关滑窗 ⇒ `1.119e-1` vs 噪声 `3.9e-3`(**28×**);融合插池参考换历史 **rope 配对错** ⇒ `4.478` vs 噪声 `1.95e-3`(**2300×**)+ classic 池寻址 ⇒ 248 处 |
| boot 期执行 | ✅ | 生产 server 三行自检:融合插池 ✓(未写 7936)/ K0 fp8 ✓(未写 64000)/ NC v2 ✓ |
| 门:自检随白名单生效 | ✅ | 白名单在场 ⇒ 只跑生效算子(dev/测试档不跑 —— 那类环境无设备资源) |
| clippy / 回归 | ✅ | clippy 0;kernels 72(默认)/55(全 feature)、models 119、engine 22、shared 27、server 2;e2e 恒等门 `AL=1.17` 逐位一致 |

### 14.10 P5c(2026-10-13;**foreign 自检运行面**完工)

| 项 | 状态 | 证据 |
|---|---|---|
| `DeviceRes::alloc_block`(真账本块) | ✅ | `GpuRes` 的 `ctx` 改 `&mut`(foreign 发射路径字段级解构);自检块**不计** slab/vram 音账(否则池预算随自检项数漂移) |
| `DeviceRes::upload_block` / `download` | ✅ | 块句柄读写一对;与 `upload`(scratch-only 白名单)并列 —— 语义不同不合并(坑册 §38) |
| `CheckRunner::{alloc_block, upload_block, download_block, dispatch}` | ✅ | `dispatch` = **经注册表**(与生产同一路径 `OpRegistry::execute`) |
| 自检调用点搬迁 | ✅ | `init_all` 内部 → `boot_with` 的**注册表建成之后**(foreign oracle 要 dispatch) |
| **第一个 foreign oracle**(cublas GEMM) | ✅ | `alloc_block + dispatch + upload_block + download_block` 四件全走;`max\|err\|=7.507e-3`(相对 1e-2) |
| 判据敏感性(负例 ×1) | ✅ | 参考按 `a·b`(布局错)⇒ `41.04` vs 噪声 `7.5e-3`(**5500×**) |
| 拒启可见性 | ✅ | boot 失败加 `eprintln!` + 自检**先打名再跑**(实测:原先只剩"族被 drop"告警,查不到原因) |
| 生产 boot | ✅ | 4 条自检全绿(融合插池/K0 fp8/NC v2/**cublas**),引擎照常就绪+捕获+服务 |
| 回归 | ✅ | clippy 0;kernels 72/55、models 119、engine 22、shared 27、oracle 6/6;e2e `AL=1.17` 逐位一致 |

### 14.11 P5d(2026-10-13;**marlin oracle** 落地 —— 生产发射量第一)

| 项 | 状态 | 证据 |
|---|---|---|
| **marlin W4A16 AWQ oracle** | ✅ | 自家 packer 造输入(`pack_marlin_b/s/z`)+ 经注册表 dispatch + **独立 host 反量化参考** `(q−zp)·s`:`max\|err\|=3.065e-4`,噪声/信号 RMS `1.7e-4`(parity 门阈值 2e-2 的 **1/100**) |
| oracle 与门的接触点 | ✅ | oracle 住 `#[cfg(feature="marlin")]`(要用 packer);`SELFCHECK` 用**一对互斥 cfg 常量**挂 —— 表零门、oracle 门内,接触面只有这一点 |
| 判据敏感性(负例 ×1 双形态) | ✅ | 忽略零偏(zp≡8)⇒ RMS **22.79**(134000×);按 `[k,n]` 读权重 ⇒ **1.85**(11000×) |
| 生产 boot | ✅ | **5 条自检全绿**:marlin AWQ / cublas / 融合插池 / K0 fp8 / NC v2 |
| 实测约束 | ✅ | marlin 打包要求 `k%128==0 && n%256==0`(小形状被 packer 直接 assert)—— 自检几何取 m=4/n=256/k=128/g=32 |
| 回归 | ✅ | clippy 0;kernels 72/55、models 119、engine 22、shared 27、oracle **8/8**;e2e `AL=1.17` 逐位一致 |

**未完成(P5 剩余 / P6)**:
1. **更多 oracle**:native 侧还差 `vllm_reshape_and_cache_bf16`(草稿池写者,逐位参考)
   与 `owl_reshape_and_cache_fp8kv`(classic 池写者)—— K0 oracle 已是参数化结构,
   加变体 = 加寻址/编码开关;
   **foreign 侧**:运行面 + marlin oracle 已就绪(P5c/P5d);还差
   gdn_chunked(金标参考)、xqa、FI 三家的 oracle;
2. **毒值自检的"出口 NaN"半边**:本棒验的是"池被写过且写集穷举";"出口无 NaN"
   需要引擎级 warmup 管道(设计与实现另案);
3. **`ShapeRule.dims/rels`**:符号维与维间关系目前全部 `NONE` —— 需要服务端
   账本携带**真实 shape**(现在 `BlockEntry.shape` 是占位 `vec![bytes]`),
   即设计 §6.2 的 `VramBlock` 完整面;
4. **数值指纹**(设计 §5.7):`assets/ops-fingerprint.toml` 未立(oracle 已覆盖
   两个高危算子,指纹的边际价值在无参考测试的算子上);
5. 顺带:`owl_gdn_decode_step_f16`(非 v2)与 `gdn_scalar` 臂仍未建表(不在生产集);
6. **新发现(非本专项)**:`apps/server/tests/e2e.rs` 的"算术 2+2"用例在**改动前
   的树上同样失败**(A/B 实证),返回的是**上一个 session 的答案**(Paris)——
   即 server 路径存在会话串扰类缺陷,已挂账(与算子准入无关)。


---

## 十五、资产形态与源码归属(2026-10-13 用户裁决 —— 两条硬规则)

### 15.1 R4:资产只允许 `.a`;**`.cubin` 形态淘汰**

| 允许 | 禁止 |
|---|---|
| `.a`(静态库)+ 一个 `extern "C"` **host 入口**(marlin/xqa/FI 先例:`rustc-link-lib=static`) | `.cubin` + `include_bytes!` + 自研 loader + `json` 元数据 + `manifest` 指纹 |
| 系统库动态调用(cublas,唯一例外) | 运行期 NVRTC 之外的任何"自造加载机制" |

⇒ **`Linkage::AotCubin` 变体退役**,`Linkage` 只剩 `StaticLib` / `Nvrtc` 两态。
**机制数 6 → 3**:

| 机制 | 定位 | 成员 |
|---|---|---|
| **M1 源码在树 + NVRTC 直发** | 无状态单核 | native 全部(110) |
| **M2 源码在树 + build.rs 预编 `.a`** | 有状态/多核/需 host 入口 | marlin、xqa、flashinfer(头 vendor 后)、gdn_scalar、gdn_chunked(源 vendor 后) |
| **M3 外部二进制库** | 无源可编 | cublas(系统 `.so`);gdn_chunked 若不 vendor 源也归此(入库 `.a`) |

**R4 的前提与出口(2026-10-13 补注;必须与裁决同读)**:

| 载体 | 跨 OS | 跨 CPU 架构 | 跨 GPU 架构 | 本项目 |
|---|---|---|---|---|
| 源码(`.cu`/Triton `.py`)→ 本机编译 | ✅ | ✅ | ✅ | **M1**(native 110) |
| PTX(驱动 JIT) | ✅ | ✅ | ✅ | 未用(备选出口) |
| `.cubin` | ✅ | ✅ | ❌(SASS 绑 sm) | **淘汰**(R4) |
| `.fatbin`(多 `-gencode`) | ✅ | ✅ | ⚠️ 部分 | `.a` 内嵌形态 |
| `.a` / `.so` | ❌ | ❌ | ❌ | **M2/M3** |

- **前提**:本工作区锁 Linux + x86_64 + sm86(驱动 615/595 + P2P 补丁 + 全 Ampere),
  因此"`.a` 不跨 OS/不跨 CPU 架构"**在本项目内不构成损失**;
- **裁决理由 = 消除自造机制 + 形态统一**(`.cubin` 逼出自研 loader + 符号校验 +
  `json` 元数据 + `manifest` 指纹 + Rust 侧编排;`.a` 只需 host 入口 + FFI);
- **出口(将来跨平台/跨架构时)**:见 §15.1.1 —— **出口 = 源在树 + 能在目标平台重编**,
  资产格式不是决定因素(`.a` 是**产物**:源在树就能在 Windows 上再产一份 `.lib`);
- **两个隐性事实**:① `.a` 路线的设备码靠 **cudart** 注册(实测 `libmarlin.a` 引用
  `__cudaRegisterFatBinary*`,故必须 `-lcudart`),而 `.cubin` 路线走 **driver API**
  (`libcuda`);② gdn_chunked 转 `.a` 后**仍绑 sm86**(内嵌 fatbin),
  换 40/50 系同样要重编/重采 —— 这个维度两条路都没解决,只有源码/PTX 解决。

#### 15.1.1 跨平台出口(2026-10-13 追问 marlin/Windows 后补)

**分层结论**:

| 目标 | 需要什么 | 本仓库现状 |
|---|---|---|
| 跨 **OS / CPU 架构** | **能在目标平台重编源码** | marlin/xqa/FI/gdn_scalar 源在树 ⇒ ✅(build.rs 加 Windows 分支:`nvcc -c` + MSVC `lib.exe` → `.lib`);**gdn_chunked 源不在树 ⇒ ❌ 会卡** |
| 跨 **GPU 架构** | **源码** 或 **PTX**(驱动 JIT) | 只有源码路线(M1 NVRTC 天然跨 sm);`.a`/`.cubin` 都绑 sm |
| 资产格式(`.a` vs `.cubin`) | **不是**决定因素 | `.a` 是**产物**,不是源 |

**三条 Windows 路线**(若将来真要):
1. **A(推荐)**:build.rs 平台分支 + `lib.exe` 重编 `.lib` —— 源在树即可,工作量小;
2. B:只复用设备码(`.nv_fatbin`/cubin 跨 OS)+ 主机侧走 driver API 自己编排 —— **与 R4 冲突**(正是要删的自造 loader/编排),不取;
3. C(A 增强):设备码用 **PTX**(`-gencode …code=compute_86`)⇒ 设备码可 JIT 到未来架构;但 **PTX 不解决跨 OS**。

**但 Windows 对 owl 整体不是"归档格式"问题**(第一道门槛在别处):
- 引擎依赖**打过 PCIe P2P 补丁的 `open-gpu-kernel-modules`**(Linux 内核模块);
  Windows 驱动闭源无法打补丁 ⇒ 双卡 P2P 无等价物;
- **NCCL 官方只发 Linux** ⇒ 两引擎的 TP/通信路径需另找方案;
- 其余:`/dev/shm` offload、`libnccl` 收编、进程组纪律、BMC/iKVM 运维面……
⇒ **owl = "Linux + Ampere + 打过补丁的驱动"三位一体**;现实的跨平台场景是
"**单个算子/组件搬到别的平台复用**",那种场景正解 = 路线 A。

**因此"保住出口"三条(不猜将来,只保出口)**:
1. **所有族的源必须在树**(⇒ gdn_chunked 必须解决 —— 这比 R4 的"形态统一"更硬);
2. build.rs 重编路径**平台无关化**(`nvcc`/`ar` 探测可覆盖 + 平台分支 + `lib.exe`);
3. 文档写明本律(本节)。

**迁移动作**:
- `gdn_scalar`:`build.sh` 的 `nvcc -cubin` → `nvcc -c` + `ar` → `libowl_gdn_scalar.a` + 一个 C host 入口包装 launch;build.rs 加 `rustc-link-lib`;删 `include_bytes!`/`load_cubin`/Rust 侧编排。
- `gdn_chunked`:七核 cubin **作为数据内嵌进 `.a`**(`ld -r -b binary` 或 `xxd -i`),`.a` 导出 `owl_gdn_chunked_fwd(...)`(内部 `cuModuleLoadData` + 六核流水);删 `json`/`manifest`/自研 loader/Rust 侧编排。
- 可选(另案):`GDN_FORCE_BUILD=1` 走 triton 重编(源从 `packages/vllm/vllm/third_party/flash_linear_attention` vendor 到 owl `cu/gdn/fla/`,采集脚本 `testdata/gdn_fla/harvest_fork.py` 一并搬)。

### 15.2 R5:源码归**算子自托管**;`sources.rs` 集中表淘汰

- **规则**:每个算子的 `SOURCE` **就地** `include_str!("../../cu/<域>/<file>.cu")` —— 与它的算子表同文件、同目录;
- **理由**:`sources.rs` 是"名字 → 源"的第二张表(设计 §2.2 立意 = 元数据/源码物理上只有一个住址);实测 `cu/owl/fused.cu` 被 **5 个常量**指向 ⇒ 一文件多算子 + 集中表 = 又一次"两处对不上"的温床;
- **进度(2026-10-13)**:`ops/` 下 **64 张表 + 3 个 oracle 已就地化**(`grep sources::` 在 `ops/` 内为空);
- **✅ 2026-10-13 C 轮结案**:见 §15.12 —— 17 个消费者已清零,**`sources.rs` 模块已删除**
  (常量就地并入 `list.rs` §0),**`.cu` 已按「一算子族一文件」拆开**。
- **✅ 待裁项已决**:“一文件多算子是否拆”→ **按「算子族」拆、helper 随族走、不抽公共头**
  (用户裁决;C 轮执行,7 个多算子文件 → 30 个族文件,`pagedattention_f16.cu` 的 vllm
  模板机体**不拆**——拆它等于抽公共头,正好被同一裁决排除)。


### 15.3 "JIT 红利"的实测审计(2026-10-13;用户裁决"不要运行期编译"后必读)

用户裁决:**设备码一律 cubin(构建期产出) + 主机侧一律 Rust + 零运行期编译**。
要问"丢了什么",先查**我们实际吃到哪几条**——实测结论:四条里只吃到 1.5 条。

| 设计稿宣称的 JIT 红利 | 我们的**实际**用法(实测) | 改 cubin 后 |
|---|---|---|
| ① 同源多 dtype 免预编("dtype 参数化近乎免费") | **没吃**:变体在源码里用宏**逐个写出**(`ops_pair.cu` 11 处 `OWL_*_KERNEL(name, T, ...)` 实例)⇒ 一个文件编一次即含全部变体 | **零损失**(构建期同样"一文件一 cubin,含全部变体") |
| ② 按当前设备编("换卡即跑") | **吃了**:server 路径 `CompileOptions{arch: None}` ⇒ nvrtc 出**通用 PTX** → **驱动载入时 JIT** 到实际设备(即"运行期编译"×2) | ❌ **真丢**:cubin 是 SASS,绑 sm;可用 fatbin 多目标**部分补**(sm86+sm89+sm120),**未来新架构补不了** |
| ③ 源码即资产(免构建期 nvcc / 免二进制入库+指纹) | **吃了**:`.cu` 进 git 即全部,二进制是运行期产物 | ⚠️ 变"构建期产物":要么每次构建编(需 nvcc),要么入库 + 指纹(`*_FORCE_BUILD` 先例) |
| ④ 开发期即时迭代(改一行 `.cu` 不用等构建) | 吃了(开发便利) | ⚠️ 变构建期几秒(只编改动文件;且一次出全文件所有核,比 NVRTC 逐核快) |

**换来的收益(对称)**:
- **启动快**:现在 boot 有一部分是 NVRTC 首用编译 + 驱动 JIT(~20 个 `.cu` 文件、110 个核);
  改 cubin 后**运行期零编译**;
- **零运行期编译风险**:编译错误从"运行期首次发射"提前到**构建期**(fail fast);
- **跨 OS 更简单**(cubin 跨 OS;NVRTC 需要目标平台的 CUDA toolkit);
- **形态统一**:1 种机制(cubin + Rust 编排)+ 1 例外(库调用)。

**结论**:在本项目(**锁 Linux + x86_64 + sm86**)里,真丢的只有 ②"换卡即跑",
而它**被"锁 sm86"这条前提屏蔽**;①③④ 是"成本从运行期搬到构建期"。
⇒ 裁决的实际代价小,收益实在;**未来换架构的出口 = 源在树 ⇒ 重编**(R5 的硬价值)。

> 勘误:`docs/arch/operator-matrix.md` 的"推论 2(JIT 红利)"把①说成"运行期按宏多编译",
> 与实测不符(变体本就写死在源码里) —— 以本节为准。


### 15.4 cubin 预编的形态与架构维度(2026-10-13 实测)

**用户定调**:"编译的时候就要很明确地感知显卡架构;编译选项里肯定会有这种东西;
针对不同的架构,编出来的东西不一样;**可以在编译时选择支持多少种架构**"。
⇒ **架构 = 构建期一等输入**(不是运行期 JIT 决定)。

| 事实(实测) | 结论 |
|---|---|
| `nvcc -cubin` **不允许**多 code instance:`Option '--cubin' is not allowed when compiling for multiple GPU code instances` | 多代际容器 = **`.fatbin`**(内含每代际一份 cubin;`cuModuleLoadData` 载入 API 完全相同,运行期由**驱动**挑最匹配) ⇒ 统一用 `-fatbin` |
| `OWL_CUBIN_ARCHS=sm_86,sm_89,sm_120` → 一个 **185 KB** fatbin,`cuobjdump --list-elf` 显示内嵌 `ops_pair.{1,2,3}.sm_{86,89,120}.cubin` | "编译时选择支持多少种架构"**已落地**:一个资产含多代际 |
| `ops_pair.cu` 里 f16/bf16/f32 变体是**源码里用宏逐个实例化**(11 处) | **build.rs 唯一需要参数化的维度 = arch**;dtype/量化变体随源码走(与 §15.3 审计一致) |

**新增的门(待接)**:`装配期架构门` —— 设备 sm ∉ 资产 arch 集 ⇒ **拒启**,
把 `no kernel image is available for execution on the device` 从**发射时**提前到**装配期**
(资产 manifest 已记 `archs` 集,设备 sm 由 `cuDeviceGetAttribute(COMPUTE_CAPABILITY_*)` 查)。

**构建期旋钮**:`OWL_CUBIN_ARCHS`(CSV,缺省 `sm_86`)/ `OWL_NVCC` / `OWL_CUBIN_ONLY`(排障)。

---

### 15.5 C2 施工实录(2026-10-13;**native 全量迁 cubin,运行期零编译**,已验收)

用户裁决("零运行期编译,设备码一律 cubin")的落地。**终态 = 构建期 `nvcc -fatbin`
产资产 + 运行期按源哈希查表 + NVRTC 仅作诊断出口**。

#### 15.5.1 源清单 = `include_str!` 闭包(零第二份清单)

> ⚠️ **已被 §15.14(E 轮)取代**:扫描目标从 `include_str!` 闭包改为
> **资产 id 字面量**(`const SRC: &'static str = "cu/…/x.cu";`)+ 逐条
> **路径存在性校验**;"零第二份清单"这一立意不变,本条按历史存档。

`build.rs` 不另立"要编哪些 `.cu`"的清单 —— 它**扫描** `src/**/*.rs` 里所有
`include_str!("…/*.cu")`,解析成真实路径去重。命中的正是

- `KernelSpec::SOURCE`(R5 就地 `include_str!`,64 张表),以及
- `list.rs::REGISTRY` 经 `sources.rs` 引的源。

⇒ **新增算子 = 加表(自带 `include_str!`)⇒ 构建期自动纳入**。物理上不可能出现
"加了表但忘了编"(E3/E15 类"两处对不上"在构建面也堵死)。实测闭包 = **19 个文件**:

| 域 | 文件 |
|---|---|
| `cu/attention/` | `pagedattention_f16` / `prefill_paged_attn_f16` / `prefill_split_f16` / `reshape_and_cache` / `reshape_and_cache_fp8kv` |
| `cu/owl/` | `argmax_f16` / `attention` / `cast` / `concat` / `dflash2` / `embed_f32` / `fused` / `gdn` / `gemv_dual_f16` / `ops_pair` / `rope_f32` / `sigmoid_gate_mul_f16` / `ts_probe` |
| `cu/marlin/` | `marlin_repack_ct`(native 通道;**不是** `marlin_host`/`vllm_marlin/*` 那些 `.a` 设备码) |

家族自管源(`cu/xqa/**`、`cu/flashinfer/**`、`cu/gdn/**`、`cu/marlin/vllm_marlin/**`)
不在闭包内 ⇒ 天然排除,无需排除表。

#### 15.5.2 新发现:`__CUDACC_RTC__` 方言 —— cubin 预编**不是**免费的

19 个源里 **16 个 nvcc 直编即过**;3 个不行,暴露出 NVRTC 与 nvcc 的**语义差**:

| 文件 | nvcc 报错 | 根因 | 修法(双载体可编) |
|---|---|---|---|
| `attention/pagedattention_f16.cu`、`attention/prefill_paged_attn_f16.cu` | `a __global__ function call must be configured`(11 / 9 处) | 文件按 NVRTC 习惯写:**`extern "C" __global__` 入口直接当函数调用内层 `__global__` 模板**(全文件零 `<<<>>>`)。NVRTC 无宿主编译 ⇒ 一切皆 device 语义,合法;nvcc 走真宿主/设备双通道 ⇒ 未配置的 `__global__` 调用是硬错 | 内层核限定符分臂:`__CUDACC_RTC__` → `__global__`(与旧文本逐字相同),否则 → `__device__`。**语义等价**:内层从不被 host 发射 |
| 同上 + `owl/argmax_f16.cu` | `invalid redeclaration of type name "int64_t"` / `identifier "uint16_t" is undefined` | 文件自带 **mini-stdint `typedef`**(nvrtc 默认头集极简,`stdint.h` 不可开);nvcc 下真 `stdint.h` 已由 CUDA 头引入 ⇒ 撞重声明(或反向缺失) | `#ifdef __CUDACC_RTC__` 圈住 `typedef` 块;`#else` 加 `#include <stdint.h>` |

**两条纪律(本次固化)**:

1. **新建 `.cu` 必须双载体可编**(`nvcc -fatbin` 是构建期硬门之一);
   NVRTC 方言(`__global__` 当函数调、mini-stdint 序言)一律进
   `#ifdef __CUDACC_RTC__` 分臂 —— 否则该文件进不了资产表 = `native_aot=require` 下拒启。
2. 三处改动在 `__CUDACC_RTC__` 分支下产生的**预处理文本与改动前逐字相同**
   ⇒ NVRTC 路径行为零变化(A/B 已由 e2e 对照复核)。

#### 15.5.3 资产登记表(生成的 Rust 文件,不手改)

`build.rs` → `$OUT_DIR/cubin_assets.rs` → `kernels::family::cubins`(`include!` 进来):

```text
CUBIN_ARCHS : &[&str]                       // 构建期 OWL_CUBIN_ARCHS 解析结果
CUBIN_ASSETS: &[CubinAsset { asset, src, src_hash, src_len, bytes }]
```

- **查表键 = 源文本哈希**(FNV-1a 64)+ **源长度**(双判据)⇒ 查的是**内容**不是名字,
  与 `KernelSpec::SOURCE` 天然一一对应(不存在第二份名字→资产映射);
- 每个资产另带 `src: include_str!(绝对路径)` —— 反查/自洽锁/诊断用;
- **哈希算法两处一份**(build.rs 与 Rust 侧)**不自证**:由
  `cubins::tests::generated_hashes_match_rust_hash_fn` 对全部资产回算锁死(漂移一处即红);
- 该模块门 = `feature = "device"`(models 侧零成本);
- `OWL_CUBIN_SKIP=1` ⇒ 写空表(纯 NVRTC 基线),同一文件形态免分支。

#### 15.5.4 装载面:kernels 侧零改动,执行面一处收口

| 层 | 变化 |
|---|---|
| `family/cubins.rs`(**新**) | 资产表 + `asset_of_source` / `has_aot` / `coverage()` + 3 个静态门测试 |
| `owl-cuda::state::KernelCache` | `ensure_kernel(name, source)`:**先查资产**(命中 ⇒ 载 fatbin 模块 + `load_function`),未命中按载体档处理。附带收益:**模块按 `.cu` 共享**(旧路径是**每核一个模块**,逐核 nvrtc + 重复装载) |
| `owl-cuda::state::GpuCtx::new` | **装配期架构门**(设备 sm ∉ `CUBIN_ARCHS` ⇒ 拒启)+ **AOT 覆盖率报告/硬门** |
| `shared::config` | 新键 `cuda.native_aot` = `require`(**缺省**)/ `prefer` / `off` |

**缺省 = `require`**(不是 `prefer`):"零运行期编译"是律不是优化;覆盖率已实测
110/110 且有测试锁(`registry_coverage_is_total`),故缺省在门内。`prefer`/`off`
是刻意留的诊断出口(A/B 对照)。

#### 15.5.5 验收(全部实测)

| 判据 | 结果 |
|---|---|
| 构建期编出资产 | **19 个 `.cu` → 19 个 fatbin**(sm_86) |
| 登记表覆盖 | **110/110**(boot 报告 `[aot] … 登记表覆盖 110/110`) |
| 已建表算子源全命中 | 64 张表全绿(`every_declared_op_source_has_asset`) |
| 哈希自洽 | 19/19 逐资产回算一致(`generated_hashes_match_rust_hash_fn`) |
| **零运行期编译** | 整轮 e2e 真机跑完 **`grep -c aot-miss` = 0** |
| **数值等价**(逐位) | `cubin_launch_probe` 6/6 绿:smem 归约 `owl_rmsnorm_f16`(8192 元素 × 两 w_off 臂)/ 平局语义 `owl_argmax_f32idx_f16` / **分区注意力链** `v2_f16_hd256bs32 + reduce`(2048 元素)—— nvrtc vs fatbin **逐位一致** |
| 回归 | kernels lib 58(全 feature)/ 55(device)/ 50(默认)· models 117 · engine 22 · shared 27 · op_tables 7 · oracles 8 · 白名单 2 —— 全绿;clippy 两档 0 警 |
| 既有缺陷(与本役无关) | `apps/server` e2e 的「算术 2+2」用例仍返回上一 session 答案(**A/B 实证:`native_aot=off` 旧路径同样复现** ⇒ 非本役引入,挂账未修) |

#### 15.5.6 收益(冷 `~/.nv/ComputeCache` 三次采样)

| 载体档 | boot→health | 首次对话(首用核编译) | **到首答** |
|---|---|---|---|
| `off`(NVRTC + 驱动 JIT) | 32.2 / 33.4 / 34.4 s | 38.6 / 38.7 / 39.1 s | **≈ 71–74 s** |
| `require`(构建期 cubin) | 27.2 / 28.1 / 28.4 s | **0.4 / 0.4 / 0.4 s** | **≈ 28 s** |

⇒ **到首答省 ≈ 43 s(−60%)**;NVRTC 那 ~39 s 是"首用某核才编译"的尾巴
(实测:同一进程第二次请求即 0.4 s,故它只在**每个新核首次发射**时付)。
附带一条:该路径此前**依赖驱动 ComputeCache**(本机 `~/.nv/ComputeCache` 已 770 MB)
才在重启后免于重付 —— cubin 路线**与该缓存彻底解耦**。

#### 15.5.7 剩余(下一步)

- **C3**:marlin/xqa/FI 三个 `.a` 家族 —— 设备码改 cubin、C host 编排搬 Rust(删 `.a`);
- **C4**:gdn 两家 —— `cu/gdn/gdn_chunk_scalar.cu` 已可纳入同一闭包(替换
  `assets/gdn_scalar/*.cubin`);gdn_chunked(Triton 采集)源 vendor 进树后同办;
- **C5**:删 `Linkage` 三态 / `sources.rs` / `list::REGISTRY`(C2 后它们只剩兜底职责);
- `R1` 余 55 条未建表入口 **不需要**单独安排:只要它的 `.cu` 在 `include_str!` 闭包内,
  资产已自动就位(建表是**准入声明**问题,与编译载体解耦)。

---

### 15.6 C2e —— **NVRTC 退役**(2026-10-13 收官;零运行期编译从"策略"变"结构")

§15.5 把**生产 native 路径**搬到了 cubin,但有**三条运行期编译调用点被绕过**
(它们走 `kernels::device::Exec::load_nvrtc`,不经过 server 的 `KernelCache`,
所以 `native_aot` 门管不到)。本节把它们一并收口,并把"零运行期编译"变成
**结构性判据**而不是自觉。

#### 15.6.1 三条漏网调用点

| 调用点 | 用途 | 处置 |
|---|---|---|
| `server/gdn_scalar.rs::assemble` | `owl_cast_f32_f16`(源 `cu/owl/cast.cu`) | → `Exec::load_aot`(源哈希查资产) |
| `server/gdn_chunked.rs::assemble` | `owl_cast_{f16_bf16,bf16_f16,f16_f32}` + `owl_state_{kv_to_vk,vk_to_kv}` | → `Exec::load_aot` |
| `server/selfcheck.rs::Runner::launch_nvrtc` | 装配期 oracle 的 native 发射 | 改名 `launch_aot`,`CheckRunner` trait 同步改名(3 个 oracle 调用点) |

#### 15.6.2 删掉的东西(不留双路径)

| 删除 | 位置 |
|---|---|
| `Exec::load_nvrtc` + `nvrtc: HashMap<…CudaModule>` + `nvrtc_include` 字段 | `kernels/src/device.rs` |
| `KernelCache::ensure_kernel_nvrtc` + `compile_ptx_with_opts/CompileOptions` 导入 | `owl-cuda/src/state.rs` |
| `pub mod nvrtc { compile_ptx_with_opts, CompileOptions }` re-export | `owl-cuda/src/ffi.rs` |
| 配置键 `cuda.nvrtc_include` / `cuda.native_aot`(+ `NativeAot` 枚举) | `shared/src/config.rs`、`assets/owl.example.toml` |

**为什么连 `native_aot` 档一起删**:`Require/Prefer/Off` 三档的存在本身就是
"还留着 NVRTC"的意思。裁定"零运行期编译"之后,留一个 `off` 档 = 留一条
生产可走的旧路径(设计纪律:定谳后固化其一,不留双路径)。A/B 数据已录
§15.5.6,历史可回查,路径不再留。

> `cudarc` 的 `nvrtc` feature **保留**:它提供 `Ptx::from_binary`(纯数据
> 类型,`load_module` 的入参);保留 ≠ 允许编译,由下节的门机器执行。

#### 15.6.3 新结构性门:`tests/no_runtime_compile.rs`

运行期断言只能证明"本轮没编译";证明不了"代码里没有编译入口"。本门扫描
源码:

- **范围** = `crates/**/src/**` + `apps/**/src/**`(跳过 `tests/`、`examples/`);
- **禁词** = `compile_ptx*` / `nvrtcCompileProgram` / `CompileOptions` /
  `nvrtcCreateProgram` / `nvrtcGetCUBIN` / `nvrtcGetPTX`;
- **白名单** = 含 `Ptx::from_binary` 的行(纯数据包装);
- **自证** = `banned_matcher_is_sensitive` 用正/负样例锁判据本身(防"扫描器坏了 ⇒ 永远绿")。

门的**首次运行就抓到了 §15.6.2 表里最后两行**(`state.rs` 与 `ffi.rs`),说明它是活的。

#### 15.6.4 顺带修掉的真实缺陷:同 tag 多 cubin 的模块撞车

`Exec` 原按 `load_cubin(asset="gdn_chunked", bytes, symbol)` 的 **tag** 做模块缓存
—— 而 `gdn_chunked` 一族**六个 Triton cubin 共用一个 tag** ⇒ 后五个 cubin 全拿到
第一个的模块。实测症状(真机 boot):

```text
[kernels] ❌ 算子注册表装配失败(拒启): asset gdn_chunked_delta_rule_fwd:
          符号 chunk_scaled_dot_kkt_fwd_kernel 不在 gdn_chunked
          (DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found"))
```

修法:**块身份 = (tag, 字节首址, 长度)**(`BlobKey`)—— AOT 资产(`include_bytes!`)
与自管 cubin(static 表项)各自唯一,同名不同块物理上不可能撞。函数缓存键
一并改用 `(BlobKey, symbol)`(否则同名符号会静默取到别的块里的函数)。

**这条缺陷的价值在"它被哪一层拦住"**:不是错数值,而是**装配期拒启 +
指名道姓**(`Linkage::AotCubin` 的符号逐一校验门)。这正是 P4/P5 那套门要拦的
东西 —— 本役新增的"模块共享"优化若要出错,也只会往这个方向出。

#### 15.6.5 门的负例验证(真机)

| 构造 | 期望 | 实测 |
|---|---|---|
| `OWL_CUBIN_SKIP=1` 构建(空资产表) | 拒启、报缺哪些 | ✅ `server_exit=1`,`零运行期编译硬门:110 个登记表核无构建期 cubin 资产 …前几个: ["owl_add_f32", …]` |
| `OWL_CUBIN_ARCHS=sm_89` 构建,跑在 sm_86 卡 | 架构门拒启 | ✅ `装配期架构门:设备 sm_86(ordinal 1)不在构建期 cubin 代际集 ["sm_89"] 内 —— 重建时请加 OWL_CUBIN_ARCHS=...,sm_86` |
| 顺带修:`OWL_CUBIN_SKIP` 的 `rerun-if-env-changed` 必须在**两条分支之外**发 | 改 env 必触发重跑 | ✅(踩坑记录:指令只在 build.rs 真跑过那次才被 cargo 记住) |

#### 15.6.6 验收(全部实测)

| 判据 | 结果 |
|---|---|
| 生产路径零编译调用 | `no_runtime_compile` 2/2 绿(含判据自证) |
| 真机 boot + 首答 | `[aot] 覆盖 110/110`、`aot-miss` = **0**、boot 29–31 s、首答 0.4 s |
| 装配期自检 | 5 个 oracle 全绿(走新 `launch_aot` 面):`1.953e-3` / `64000` / `3.065e-4` / `3.904e-3` / `7.507e-3` |
| GDN 两家(改道后的 CAST + 六 cubin) | `gdn_scalar_golden` 1/1、`gdn_chunked_golden` 1/1 绿 |
| 其余回归 | kernels lib 58/58 · op_tables 7 · oracles 8 · cublas 隔离 1 · cubin 探针 6 · models 117 · engine 22 · shared 27 · 白名单 2 —— 全绿 |
| clippy | `--workspace --all-targets` 与全 feature 两档 **0 警** |

#### 15.6.7 剩余(下一步)

- **C3**:marlin / xqa / flashinfer 三个 `.a` 家族 —— 设备码改 cubin、C host 编排搬 Rust
  (删 `prebuilt/*.a`)。**注意 FI 还带一个树外依赖**(`owl_fi_attention.cu` 头
  vendor 自 `~/.cudaforge/git/checkouts/flashinfer-*`),不满足 §15.1.1 的
  "源必须在树",要先 vendor 头;
- **C4**:`cu/gdn/gdn_chunk_scalar.cu` 纳进同一 `include_str!` 闭包(替换
  `assets/gdn_scalar/*.cubin` 与 `Linkage::AotCubin` 的最后残余);
  gdn_chunked 的 Triton 源 vendor 进树后同办;
- **C5**:删 `Linkage` 三态 / `sources.rs` / `list::REGISTRY`。

---

### 15.7 C4a —— `gdn_scalar` 归入统一资产面(2026-10-13;入库 cubin blob 清零)

`gdn_scalar` 是 §15.1 R4 时代"**预编 cubin 入库 + `include_bytes!` + 自管清单**"
的最后一个活样本(`assets/gdn_scalar/{chunk_scalar_f32.cubin, build.sh, manifest.toml}`)。
它的源本来就在树(`cu/gdn/gdn_chunk_scalar.cu`),只是没被 `include_str!` 引用 ⇒
不在 §15.5.1 的闭包内 ⇒ 也就没有构建期资产。

| 动作 | 内容 |
|---|---|
| **源自托管** | `family/gdn_scalar.rs`:`cubin` 模块 → **`asset`** 模块,`SOURCE = include_str!("../../cu/gdn/gdn_chunk_scalar.cu")`(就地,§R5 律) |
| **闭包自动纳入** | 源被 `include_str!` 引用 ⇒ `build.rs` 扫到 ⇒ 编 fatbin ⇒ 进资产登记表(资产数 **19 → 20**) |
| **装载改道** | `server/gdn_scalar.rs`:`exec.load_cubin(…bytes…)` → `exec.load_aot(res, asset::SOURCE, asset::KERNEL_F32, OP)` |
| **删资产** | `crates/kernels/assets/gdn_scalar/` 整目录删除(cubin blob + build.sh + manifest)—— 一处机制,不留自管资产 |
| 测试面 | `gdn_scalar_golden` / `gdn_big_t_arms_parity` 改取 `asset::bytes()`(资产表反查,不再读 blob) |

**验收**:`gdn_scalar_golden` 1/1 绿;真机把控制臂打开
(`dispatch.gdn_scalar=true / gdn_chunked=false`)boot + 首答成功(aot 覆盖 110/110、
资产 20、零 `aot-miss`)⇒ `assemble` 里 cast 与主核都走了资产表。

> **顺带印证白名单硬门**:同一次尝试(未放开 `allowed_ops`)被拦在 boot:
> `[算子准入] contract owl_cast_f16_f32: allowed_ops 期望 ∈ 本档白名单(44 项) 实得 未列入`。
> 控制臂要用的算子在**生产白名单**里没有 ⇒ 拒启 —— 这正是 P4 那道门的设计意图
> (生产档只允许它声明过的算子集)。

> **顺带暴露的一条资产纪律问题**:`crates/kernels/assets/` 整个目录是
> **gitignore**(`crates/kernels/.gitignore:1`)—— 也就是说旧的"cubin 入库"其实
> **没进 git**:fresh clone 里这些二进制不存在,装载会在运行期失败,得有人记得手工
> 跑一次 `build.sh`。新的方案把资产变成**构建产物**(源在树 ⇒ `cargo build` 即产),
> clone 即可复现 —— 这比"跑得快"更要紧的一层:R4 时代的资产面**不可复现**。

**剩余(C4b)**:`gdn_chunked` 的六个 Triton cubin(`assets/gdn_chunked*/`)仍走
`load_cubin` 老路 —— 要归入统一面必须先把它**源 vendor 进树**(Triton `.py` +
采集脚本,§15.1.1 律 1"所有族的源必须在树")并给出构建期编译路径(需 triton/torch
开发期依赖)。另:`Linkage` 三态已确认**零消费者**(只有 `FamilyGuard` 直通),
C5 可整块删。

---

### 15.8 C3a —— XQA:`.a` → 构建期 cubin + **Rust host 发射器**(2026-10-13)

三个 `.a` 家族里最小的一支(host 发射器 ~40 行)。**1.5 MB 二进制 + 223 行 C 出局**,
换成 2 个构建期 fatbin(551 KB / 518 KB)+ 一个 Rust 发射器。

#### 15.8.1 上游自带的出口:`-DGENERATE_CUBIN`

`mha.h` 里本来就有:
```cpp
#ifdef GENERATE_CUBIN
#define CUBIN_EXPORT extern "C"      // 上游的"只出设备码"模式
#else
#define CUBIN_EXPORT static
#endif
```
⇒ 用它编译,`kernel_mha` 落成 **STB_GLOBAL STO_ENTRY**(可直接 `cuModuleGetFunction`)。
**不定义它则一切符号被内部化**(实测 `STB_LOCAL`),driver 按名取不到。

**owl 本地 port 两处**(都写进 `mha.cu` 的 `#ifndef GENERATE_CUBIN` 注记):
1. owl 加的 host 发射器段(`configureKernel` + `launchMHAFlashInfer`)原本**没**被
   `GENERATE_CUBIN` 排除 ⇒ 该模式下 `utils.h` 的 `checkCuda` / `hostUtils.h` 的
   `makeLaunchConfig` 已被条件排除 ⇒ 未定义标识符。**补上 `#ifndef GENERATE_CUBIN`**;
2. 见 15.8.2。

#### 15.8.2 关键设计:**资产自描述**(编译期常量不复制到 Rust)

单 TU `-cubin` 会把**设备变量内部化**:实测 `smemSize`(以及 `__device__` /
`__constant__` 全局,加 `__attribute__((visibility("default")))` 也一样)全部落
`STB_LOCAL` ⇒ `cuModuleGetGlobal` 取不到。而 host 发射器需要这些**编译期常量**:
dynamic smem 尺寸、`ctaTile`、`headGrpSize`、`rowsPerBlock`、`tokensPerPage`、
`validElemsPerHead`、`allowMultiBlock`、cta 形状、`sizeof(KVCacheList<true>)`。

⇒ 内核侧发一个 1 线程探针核 `owl_xqa_launch_consts(uint32_t* out)` 把常量写进缓冲;
Rust 侧**装配期发一次 + 44 B D2H**(~50 µs,一次性)。

**为什么值得**:替代方案是"Rust 复刻 C++ 的 constexpr 推导"—— 那正是本项目反复
打击的"两处对不上"(改一处编译选项,Rust 常量静默失配 ⇒ 错几何 ⇒ 错值/越界)。
自描述让**单一真源**留在 `mha.cu`,且带一道**布局硬门**:第 12 槽 =
`sizeof(KVCacheList<true>)`,与 Rust 镜像 `size_of` 比对,不等即**装配期拒启**。

#### 15.8.3 契约面新增:`LaunchVal::Blob`(by-value 结构体参数)

`kernel_mha` 有一个 **40 字节 by-value 结构体**参数(`KVCacheList<true>`)。

**踩坑(实测 `CUDA_ERROR_MISALIGNED_ADDRESS`)**:先按"CUDA 参数区紧密排布"把结构体
拆成 **5 个 `U64` 槽** —— **错的**。`cuLaunchKernel` 的 `kernelParams` 是
"**每参数一个指针**",驱动按内核元数据里的 `paramSize`(实测 `EIATTR_KPARAM_INFO`
= ordinal 10 / offset 0x48 / size **0x28**)从该指针拷 40 字节 ⇒ 从第一个 u64 的地址
拷 40 字节 = 读到邻居栈。

**正解**:`LaunchVal::Blob { len, bytes: [u8; 64] }`(内联镜像,**值随 `LaunchVal`
owned**,压槽时取地址天然安全,无悬垂),`launch_vals` 对结构体只压**一个指针**。
镜像由调用方 `#[repr(C)]` + `const` size 断言保证,并可经 15.8.2 的自描述**机器锁死**。

顺带:原 `Exec::launch` 的压槽循环提为 `device::launch_vals`(**唯一映射点**)——
生产与探针共用,"测试绿 ⇒ 生产绿"。

#### 15.8.4 另一条实测坑:探针槽数

`CONSTS_SLOTS` 从 11 涨到 12 时,内核侧探针核**忘了同步** ⇒ 尾三槽读到全零
(`tokens_per_page=0` ⇒ `maxNbPagesPerSeq` 爆表、`valid_elems_per_head=0` ⇒ stride
不换算)⇒ 同样是 `MISALIGNED_ADDRESS`。**教训**:探针核与 host 的槽数是**一份契约的
两半**,`XqaConsts::from_slots` 的长度门只拦"太短",拦不了"内核少写"(零值合法)。
⇒ 缓解:数值探针(`xqa_verify/decode_probe`)是最终判据,而它确实抓到了。

#### 15.8.5 构建期增量闸(顺手修的开发回路问题)

`build.rs` 的 rerun 面含 `src/**/*.rs`(源清单 = `include_str!` 闭包)⇒ **改任何一个
Rust 文件都会重跑 build.rs**。没有闸的话每次 Rust 编辑要把 20 个 `.cu` + `mha.cu`×2
重编一遍。

⇒ `compile_fatbin` 带**指纹增量闸**(源 mtime+长度 + 架构集 + 编译选项 → `.stamp`);
指纹一致即跳过 nvcc。实测:**touch 一个 .rs 后整轮构建 0.5 s**(全增量命中,
零 nvcc;日志显式打"(全增量命中,零 nvcc)");**改 `mha.cu` 后 = 19.6 s**
(只重编两个 xqa fatbin,20 个 native 源仍增量命中)。

#### 15.8.6 验收(全部实测)

| 判据 | 结果 |
|---|---|
| `xqa_verify_probe`(数值 + 速度) | **PASS**:ctx8192 XQA **0.505 ms** vs v2 2.673 ms = **5.29×**;`max\|XQA−host\|` **4.51e-4**(历史 0.495ms / 5.40× / 4.5e-4 —— 一致) |
| `xqa_decode_probe`(两实例化 A/B) | **PASS**:ctx8192 decode 实例化 **0.286 ms** vs v2 0.421 = **1.47×**(历史 1.46×) |
| 资产自描述 | `verify: smem=83584 ctaTile=(256,32) rowsPerBlock=32` / `decode: smem=67200 ctaTile=(256,16) rowsPerBlock=16`;**`sizeof(KVCacheList)=40` = Rust 镜像** ✓ |
| **真机生产档**(`spec-dflash`:`verify_xqa=true` + `decode_xqa=true`,白名单含两臂) | boot 30.4 s + 首答 **0.4 s** 成功;`[aot] 110/110`、`aot-miss`=**0** ⇒ XQA 两臂在**真实图里**走的就是新发射器 |
| 回归 | kernels lib 58 · no_runtime_compile 2 · op_tables 7 · oracles 8 · cubin 探针 6 · models 117 · engine 22 · shared 27 · 白名单 2 —— 全绿;clippy `--workspace --all-targets` 与全 feature **0 警** |
| 出局资产 | `prebuilt/libowl_xqa{,_decode}.a`(1.5 MB,git 内)+ `owl_xqa_launch.cu` + `xqa_wrapper.cu`(死文件:引用不存在的 `../tvm_ffi_utils.h`) |

#### 15.8.7 剩余

- **C3b marlin**(19 MB `.a`):4 个 sm80 设备码 → cubin;`marlin_host.cu`(514 行
  exec-config/occupancy 启发式)搬 Rust —— 需 `cuOccupancyMaxActiveBlocksPerMultiprocessor`
  等 driver 能力,**三者中最重**;
- **C3c flashinfer**(8.2 MB `.a`):还差**树外头 vendor**(`owl_fi_attention.cu` 的头
  来自 `~/.cudaforge/git/checkouts/flashinfer-*`,不满足 §15.1.1"源必须在树");
- **C4b gdn_chunked**:13 个 Triton cubin(仍 `load_cubin` 老路)→ 需 vendor Triton
  源 + 构建期 triton 编译路径;
- **C5**:`Linkage` 三态(已确认零消费者)/ `sources.rs` / `list::REGISTRY`。

---

### 15.9 C3-α —— marlin / flashinfer:预编 `.a` → **构建期从源编**(2026-10-13)

**27.2 MB 二进制出局**(marlin 19.1 MB + FI 8.2 MB,原先都在 git 里)。两支的
**host 侧仍留在 C**(见 §15.9.5 的成本账),本节做的是"**资产面**"那一半:
入库二进制 → 构建产物 + 源/头全在树 + 增量闸 + 两道新门。

#### 15.9.1 三个实测发现(都指向同一句话:**预编件不是本仓产物**)

| 发现 | 证据 | 后果 |
|---|---|---|
| **①`libmarlin.a` 是外来二进制** | 它的成员含 `marlin_quant.o` / `sampler_core.o` / `silu_int8_quant.o` / `bf16inst.o` / `sm80_kernel_s8_u4b8_bfloat16.o` —— **本仓没有这些源**(那些是 xinfer marlin-ffi 的对象) | "预编入库"实为**不可复现的外来 blob**;而且 `build.rs` 的 `FORCE_BUILD` 路径(3 源)**从未链接通过** |
| **②selector 引用未实例化变体** | `kernel_selector.h` 引用 **306** 个 `Marlin<...>` 变体,树内 4 个 `sm80_kernel_*.cu` 只实例化 **258** ⇒ 缺 **48** 条 `kS8×kU4B8×kBFloat16`(其源文件 `sm80_kernel_s8_u4b8_bfloat16.cu` **不在树**) | 从源编 = **链接期 `undefined hidden symbol`**(实测)。修:源清单补上在树的第 4 支(`s8→f16`)+ 从 selector 删那 48 条 + **加构建期机器门**(见 15.9.3) |
| **③FI 从源编缺编译门** | `owl_fi_prefill_run_fp8kv` 被 `#ifdef FLASHINFER_ENABLE_FP8_E4M3` 圈住,而 build.rs 的从源命令**没带该宏** ⇒ 符号缺失(实测链接失败);旧预编件带该宏 | 补 `-DFLASHINFER_ENABLE_FP8_E4M3`(owl 生产档 `quant=Fp8E4M3 + fi=true` **必需**) |

#### 15.9.2 FI 头闭包 vendor 进树(§15.1.1 律 1)

原缺省 `-I ~/.cudaforge/git/checkouts/flashinfer-0f06c2305a276bcb/include` ——
**树外**检出 ⇒ clone 重建不出 `.a`。实测 `nvcc -M` 头闭包 = **22 文件 / 696 KB
(不含 cutlass)** ⇒ 整体 vendor 到 `cu/flashinfer/include/`(保留相对布局;
复现法写进其 README)。⇒ **两个 `.a` 现在都能从 clone 重建**。

#### 15.9.3 新门:`check_marlin_selector`(selector ⊆ 实例化)

`get_marlin_kernel` 靠生成的 `kernel_selector.h` 选核,该头与实例化文件一旦漂移,
后果是**链接期**报错(实测踩过;E5-DF3 "新旧分支混链" 同款)。新门在 **build.rs**
里把两边的 `Marlin<...>` 变体集抽出来比对,引用而未实例化 ⇒ **直接拒构建**
(带首个缺失变体)。实测输出:

```
marlin selector ⊆ 实例化 ✓(258 变体引用 / 258 个已实例化)
```

#### 15.9.4 增量闸(两支共用)

`stamp_hit` / `stamp_write`(源+头 `len:mtime` 指纹 + 编译旗子 → `<lib>.stamp`)。

| 场景 | 实测 |
|---|---|
| touch 一个 `.rs`(build.rs 重跑) | **0.6 s**(native cubin 20 + marlin + FI **全增量命中,零 nvcc**) |
| 改 `mha.cu` | 19.6 s(只重编 xqa 两个 fatbin) |
| 冷编 marlin(4 支) | ≈ 3.5 min |
| 冷编 FI | ≈ 1.5 min |

> 旗子纪律:`FORCE_BUILD` **不进指纹**(一次性旁路,不是构建输入;进了会导致
> "用 FORCE_BUILD 跑过一次之后,普通构建反而重编"—— 实测踩过)。

#### 15.9.5 验收

| 判据 | 结果 |
|---|---|
| **marlin 数值** | 装配期 oracle `marlin_gemm_w4a16_awq`: `max\|err\| = 3.065e-4` —— 与预编件时代**逐位同值**(§15.6.6 记录值) |
| **FI 数值/通路** | 生产档 boot:`[boot] FlashInfer prefill 面启用(FI 直读主池 kNHD 单池;quant=Fp8E4M3)`;首答 0.5 s |
| 链接 | `cargo build --workspace` 通过(此前从源路径**链接失败**) |
| 真机 boot | 27.7 s + 首答 0.5 s;`[aot] 110/110`、`aot-miss` = **0** |
| 回归 | kernels lib 58 · no_runtime_compile 2 · op_tables 7 · oracles 8 · cubin 探针 6 · models 117 · engine 22 · shared 27;clippy `--workspace --all-targets` **0 警** |
| 出局 | `cu/marlin/prebuilt/libmarlin.a`(19.1 MB)+ `cu/flashinfer/prebuilt/libowl_flashinfer.a`(8.2 MB) |

#### 15.9.6 剩余(C3-β:host 侧搬 Rust 的成本账)

| 家族 | host 侧是什么 | 搬 Rust 的成本 |
|---|---|---|
| **marlin** | `marlin_host.cu` 514 行:exec-config/occupancy 启发式(`cudaOccupancyMaxActiveBlocksPerMultiprocessor`)+ `MARLIN_KERNEL_PARAMS` 20 参装配 + 从 mangled 符号选核 | 中(需 driver occupancy 绑定 + 符号名表;可由 build.rs 从 `cuobjdump -symbols` 抽 mangled 名生成) |
| **flashinfer** | `owl_fi_attention.cu` 309 行,但 `plan` 面 = **flashinfer 自己的 `PrefillPlan` 调度器**(C++ 模板,数百行算法) | **高**(等于 fork 上游调度算法;Rust 侧现只消费不透明的 `plan15`) |

⇒ 两者是"形态统一 + 跨 OS"的收益,**不**影响"零运行期编译 / clone 可重建 / 源在树"
(这四条本节已全部达成)。建议按上表排序另立工单,不在本设计内顺手做。

#### 15.9.7 全资产面现状(本役后)

| 家族 | 载体 | git 内二进制 |
|---|---|---|
| native 110 核 + gdn_scalar + xqa(2 变体) | 构建期 fatbin(源哈希/变体名查表) | **0** |
| marlin | 构建期 `.a`(4 支 sm80 设备码 + C host) | **0** |
| flashinfer | 构建期 `.a`(设备码 + C host;头闭包在树) | **0** |
| gdn_chunked(13 cubin) | `assets/`(**gitignore** —— 本就是运行期必需但未入库!) | 0(但**不可复现**,C4b 待办) |
| cublas | 系统 `.so`(唯一例外) | 0 |

---

### 15.10 收尾(2026-10-13):死代码 + 资产入库 + 终态

#### 15.10.1 `Linkage` / `FunctionKey` 删除(C5-α)

`Linkage` 三态(`StaticLib`/`AotCubin`/`Nvrtc`)自 §15.5 起就只剩**声明**
作用:全仓唯一引用是 `KernelFamily::linkage()` 的**直通**(`FamilyGuard` 转发),
**零消费者**。装载形态现在由**实现事实**决定(资产表 / FFI 直调),声明式字段
只会成为"声明与实现漂移"的温床 ⇒ **整块删除**(trait 方法 + 7 个族实现 +
测试桩 + `contract.rs` 枚举)。同批删 `device::FunctionKey`(死枚举:三链路
合一后无人构造)。

#### 15.10.2 `assets/` 改回入库(C4b-α;clone 可编)

**实测缺陷**:`crates/kernels/.gitignore` 忽略整个 `assets/`,而
`family/gdn_chunked.rs` 用 `include_bytes!("../../assets/gdn_chunked/*.cubin")`
⇒ **clean clone 根本编不过**(`include_bytes!` 找不到文件),生产档
`gdn_chunked` prefill 臂也无从运行。这与 §15.7 发现 gdn_scalar blob 的问题
**同类同因**("gitignore 掉了运行期必需的资产")。

处置:**去掉 `.gitignore` 条目 + 资产入库**(13 cubin + provenance json +
manifest,1.7 MB);新增 `assets/README.md` 记**来源/复现配方/根治工单**
(源是 Triton `.py`,构建期编译需 triton+torch ≈1.8 GB 开发期依赖;
源当前在 `packages/vllm` 检出里 = owl 树外,故先入库止血)。

#### 15.10.3 终态:资产面一览

| 家族 | 载体 | 源在树 | git 内二进制 | clone 可重建 |
|---|---|---|---|---|
| native 110 核 + gdn_scalar + xqa(2 变体) | 构建期 fatbin | ✅ | 0 | ✅ |
| marlin | 构建期 `.a`(4 支设备码 + C host) | ✅ | 0 | ✅ |
| flashinfer | 构建期 `.a`(设备码 + C host) | ✅(头闭包已 vendor) | 0 | ✅ |
| gdn_chunked(13 cubin) | 采集产物入库 + 配方在树 | ⚠️(源在 vllm 检出) | 1.7 MB | ⚠️(需 triton;工单在案) |
| cublas | 系统 `.so`(唯一例外) | — | 0 | ✅ |

**四条律全部达成**:① 零运行期编译(结构性门 + `aot-miss`=0)
② 源在树(唯一例外 = gdn_chunked 的 Triton 源,已立工单)
③ 形态统一(一处机制:构建期资产;一处例外:系统库)
④ 架构 = 构建期一等输入(`OWL_CUBIN_ARCHS` + 装配期架构门)。

**终态判据**(真机):boot 26.6 s + 首答 0.5 s;`[aot] 110/110`、`aot-miss` = 0;
marlin oracle `max|err| = 3.065e-4`(与预编件时代同值);xqa 5.29×(0.505 ms vs
2.673 ms)/ decode 1.47×,与历史记录一致;回归全绿;clippy 0 警。

> 过程留档(值得记的一条,含一次**误诊更正**):收尾时出现过
> `[xqa] 布局失配:内核 0 B vs Rust 镜像 40 B`。当时先判"旧 OUT_DIR 陈旧 fatbin",
> **错了** —— 真因是 **回读竞态**:`DeviceRes::download` 走 `cuMemcpyDtoH_v2`
> (阻塞在 **NULL/legacy 默认流**),与 compute 流不同流 ⇒ 不构成同步点 ⇒ 间歇
> 读到全零(同一份资产 verify 读到 40 / decode 读到 0)。修法 = 回读前显式
> `res.stream()?.synchronize()`(oracle 面本来就有 `r.sync()` 同款纪律,见 §15.11.3)。
> **两次都是布局硬门拦下的**(fail-closed 而非静默错值)—— 这正是 §15.8.2
> "资产自描述"要买的东西。顺带修掉一条 cargo 跟踪隐患:资产经 `rustc-env`
> 发**指纹**(`OWL_ASSET_FP`),指纹变 ⇒ 强制重编 crate(见 §15.11.3 末行)。

---

### 15.11 C4b-β —— gdn_chunked 六核**我们自编**(2026-10-13;opt-in,产物逐字节复现)

用户追问:"这些预编译文件不能由我们自己来编吗?" ⇒ 能,而且**编出来与入库件
逐字节相同**。

#### 15.11.1 结论先给

| 核 | AOT 自编 | 入库采集件 | 对比 |
|---|---|---|---|
| cumsum | 9952 B `c1c4285f76d5` | 9952 B `c1c4285f76d5` | **字节一致 ✓** |
| kkt | 39264 B `c6c8b7e1b79e` | 39264 B `c6c8b7e1b79e` | **字节一致 ✓** |
| merge | 134880 B `9ff7c2ab80b4` | 134880 B `9ff7c2ab80b4` | **字节一致 ✓** |
| wu | 81120 B `0a16fe8bbf83` | 81120 B `0a16fe8bbf83` | **字节一致 ✓** |
| h | 124000 B `e80afeef2002` | 124000 B `e80afeef2002` | **字节一致 ✓** |
| o | 86624 B `f4803d0235b1` | 86624 B `f4803d0235b1` | **字节一致 ✓** |

⇒ 采集件不再是"不可复现的外来 blob":**源在树 + 配置烘焙 ⇒ 可重建**。

#### 15.11.2 三件事让它成为可能

1. **源 vendor 进树**(§15.1.1 律 1):`triton/fla/` = 六核 import 闭包(9 文件,
   236 KB)。原文件从 vllm 取 `triton_utils`/`platforms` ⇒ 加
   `fla/triton_shim.py` 等价垫片 + 改 import(语义不变);
   `ops/__init__.py` 清空(未 vendor 的模块会 ModuleNotFoundError)。
2. **配置烘焙**:`assets/gdn_chunked/*.json` 记的是当年 autotune 选中值的
   **结果**(`num_warps/num_stages/shared`),而源码 `@triton.autotune(configs=…)`
   的 config 还带 constexpr kwargs(`BK`/`BV`)。kkt/h/o 的 `(w,s)` 各有
   3/2/4 个候选 ⇒ **逐个候选编译、用编译产物回读的 `shared` 与记录值比对**,
   唯一命中者即选中配置(全部不中 = 报错退出,不猜)。实测命中:
   `kkt BK=64` / `h BV=64` / `o BK=32,BV=64`。
3. **AOT 而非跑流水线**:内核对象被换成 `_WarmupProxy` —— wrapper 照常算
   args/constexpr/grid,但 `kernel[grid](…)` 变成 `jit_fn.warmup(…)`
   (**只编译不发射**)。⇒ 复用 wrapper 的**逐字配方**(零手抄 args),
   不跑流水线、不重跑 autotune(后者会让产物**不确定**)。

#### 15.11.3 踩坑留档(每一条都实测复现过)

| 坑 | 症状 | 正解 |
|---|---|---|
| 换掉模块符号 | `dynamic_func() missing IS_VARLEN` | **必须保留外层 `Heuristics`**(它注入 IS_VARLEN),只顶替它内层的 autotuner/jit |
| 候选轮询取"当前值" | `'_WarmupProxy' object has no attribute 'warmup'` | `Heuristics` 是**就地被改**的 ⇒ 每轮从**原始对象**重建,且提前 `_jit_of` 剥到底 |
| grid 回调吃 meta | `KeyError: 'BV'` | `def grid(meta)` 要用**并入候选 kwargs 之后**的 meta 求值 |
| dtype 配方 | `Both operands must be same dtype. Got fp16 and fp32`(o 核)/ `Expected fp32 but got fp16`(kkt) | 引擎先 `owl_cast_f16_bf16` 把输入转 **bf16** ⇒ 激活域 = bf16(`h_buf` 亦 bf16),`g/g_cum/A/state` = f32。**按资产 `origin`("bf16+g_f32")与引擎实参走**,别信 harvest 脚本里遗留的 `DT=f16`(那会让 cumsum 从 9952 变 10080) |
| 默认 CUDA 设备 | autotuner bench 内 `Pointer argument cannot be accessed from Triton (cpu tensor?)` | `torch.cuda.set_device(dev)` 必须与张量设备一致(否则 bench 在设备 0 分配、张量在设备 1) |
| build.rs env 指令位置 | 设 `GDN_TRITON_BUILD=1` 不触发重编 | `rerun-if-env-changed` 必须**无条件**发(同 §15.6.5 的 `OWL_CUBIN_SKIP` 坑) |
| **装配期回读竞态**(§15.8 同款,误诊过一次) | `download` 读到全零(间歇) | `download` = `cuMemcpyDtoH_v2` 阻塞在 **NULL 流**,与 compute 流不同流 ⇒ **回读前显式 `stream().synchronize()`** |
| `include_bytes!` 指向 `$OUT_DIR` 产物 | 资产已更新、rlib 仍嵌旧字节 | 资产经 `rustc-env` 发**指纹**(`OWL_ASSET_FP`,boot 报告可见)⇒ 指纹变即强制重编 crate |

> 另:原 `testdata/gdn_fla/harvest_fork.py`(跑流水线的旧配方)**并非"跑不通"**,
> 而是要求默认 CUDA 设备 = 张量设备(见上表);它的问题在别处 ——
> autotune 现场选配置 ⇒ 产物不确定,且要 GPU 跑完整流水线。故自建走 AOT。

#### 15.11.4 接线与验收

| 层 | 变化 |
|---|---|
| `triton/fla/`(新) | FLA 六核闭包 vendor + `triton_shim.py` + README(来源/改动/升级法) |
| `triton/gdn_aot_compile.py`(新) | AOT 编译脚本(配置自校验;`--out` / `--device`) |
| `build.rs` | `GDN_TRITON_BUILD=1`(opt-in)⇒ 跑脚本 → `$OUT_DIR/gdn/*.cubin` → 生成 `GDN_ASSETS` 表;**增量闸** + 解释器探测(`GDN_TRITON_PY`,缺省探工作区 `vllm-test/.venv`)+ `GDN_TRITON_DEVICE` |
| `family/gdn_chunked.rs` | `cubins::*` 由 `const` 改 **`fn` + `resolve()`**:自编资产优先,入库采集件兜底(`h_noh0` 明确标注"未注册,保留备用") |
| `server/gdn_chunked.rs` / `tests/gdn_chunked_golden.rs` | 取资产处改调 `cubins::x()` |

**验收**:① 自编 6 核与入库件**逐字节一致**(表见上);② `GDN_TRITON_BUILD=1`
构建 + `gdn_chunked_golden` **绿**,且 **c5(同源同核紧容差)max|Δ| = 0.000e0**
(g_cum/A/w/u/h/v_new/o/ht 全项);③ 缺省构建(不设 env)**不需要** Python/triton,
走入库件,golden 同样绿。

#### 15.11.5 翻默认 + `assets/` 整体退役(2026-10-13 用户裁决)

用户追问两点,直接改变了设计:
> "既然你都引入这个 triton 了……那为什么又要留着它呢?还有这个 `.json`,它到底是
> 个什么东西?为什么源码里面还需要有一个 `.json`?每个 cubin 都必须要有吗?"

**① 翻默认**:`GDN_TRITON_BUILD` 由 opt-in 改为**默认开** ⇒ **入库采集件整体删除**
(`crates/kernels/assets/` 退役,~1.8 MB:6 生产 cubin + `h_noh0` + pipfla_f32
退役备份 + manifest)。六核与 native cubin / marlin / flashinfer **同形态**:
**资产 = 构建产物**。

**② 每个 cubin 都要一份 json 吗?—— 不要**。区分两类:

| 类 | 编译输入 | 需要配置表吗 |
|---|---|---|
| native(20 个 fatbin) | `.cu` 源 + 架构(`OWL_CUBIN_ARCHS`) | **不需要** —— 输入从源码唯一确定 |
| Triton(gdn_chunked 六核) | 源 + **autotune 选中那组选项** | **需要** —— 选项空间是**搜出来的**,不从源码唯一导出 |

⇒ 六份 per-kernel json **合并成一张表** `triton/gdn_chunked_aot.json`(角色写在文件名与
`_note` 里:"本族 AOT 编译配置表";含 `num_warps`/`num_stages`/`kwargs(BK,BV)` +
`shared` 期望值)。脚本**按表直接编译**(不再搜候选),并把产物回读的 `shared` 与
表值比对 —— 不等即拒用(**源/配置漂移的机器判据**)。它更像 lockfile,不是
per-binary 元数据。

**③ 逃生舱**:`GDN_TRITON_BUILD=0` = 显式跳过自编 ⇒ 资产表为空 ⇒ 该族
**装配期拒启**(实测:`asset gdn_chunked_delta_rule_fwd: gdn_chunked 资产缺失
["cumsum","kkt","merge","wu","h","o"] … 缺省应自动自编(需 Python+triton)`),
但 crate 本身**仍能编过** —— 给"没有 triton 的环境"留一条只编不跑的路。

**④ 未注册资产一并删除**:`h_noh0`(无初态变体)与 `pipfla_f32`(退役变体备份)
是设备码但**没有任何装载路径**;将来要用 = 配置表加一行 + `initial_state=None`
重编即可(一行改动)。

**验收**:缺省构建自动自编(日志 `GDN chunked 六核 **AOT 自编**完成`);
`gdn_chunked_golden` 绿(c5 紧容差);真机 boot 33.6 s + 首答 0.5 s、`[aot] 110/110`、
`aot-miss`=0;`GDN_TRITON_BUILD=0` 路径按设计拒启;全回归 + clippy 0 警。

#### 15.11.6 构建环境:**uv 项目声明依赖 + 零 GPU**(2026-10-13 用户指令)

用户指出:既然构建期要 Python,就该**用 uv 把依赖声明清楚**,而不是靠"工作区里
恰好有个装了 triton 的 venv"。

⇒ `crates/kernels/pyproject.toml`(uv 项目;**含源目录声明** ——
`[tool.uv.build-backend] module-root = "triton"` / `module-name = "fla"`,即
"源根 = `crates/kernels/triton/`,顶层模块 = vendor 的 `fla`";`uv sync` 把它
**editable 装进 venv** ⇒ 脚本 `import fla.ops` 直接可用,不再手插 `sys.path`):

| 依赖 | 理由 |
|---|---|
| `triton==3.7.1`(**钉死**) | 编译器版本变 ⇒ SASS 变 ⇒ 资产不再逐字节可复现 |
| `torch`(**CPU 轮子**,`[tool.uv.sources]` 指 `download.pytorch.org/whl/cpu`) | FLA wrapper 的真张量簿记(`empty_like`/`len`/`.to`);AOT 不发射 ⇒ 不需要 CUDA 版(省 ~2.5 GB) |

```bash
uv sync --project crates/kernels      # → crates/kernels/.venv
cargo build                           # build.rs 自动用该 venv
```

**零 GPU**(`--device none`,缺省):AOT 只编译不发射 ⇒ `CompiledKernel._init_handles()`
(需 `driver.utils.load_binary` + 设备)**从不被调用**;`run()` 起手只要
`get_current_device/get_current_stream`(缓存键)+ `get_current_target()` ⇒
注入**桩驱动**给这三个,其余方法一被调用即报错(设备触达哨兵)。

**实测**:`uv` 环境 + 零 GPU 的产物与"带 GPU 真实设备"产物**逐字节一致**
(6/6,sha16 全同);`GDN_TRITON_DEVICE=cuda:N` 保留为排障/对照路径。

**多族可扩**(用户追问"后面要加其他 Triton 算子怎么办"):`triton/` 是**多族共用
的源根**,`module-name` 用**列表**形式(uv_build 支持 `["foo","bar"]`,官方文档在案)
⇒ 加族 = ①vendor 到 `triton/<name>/`(保持上游包名)②`module-name` 加一个名字
③该族 `triton/<name>_aot.{py,json}` ④`build.rs` 加一段(照 `build_gdn_assets`)
⑤family 侧 `resolve()` + 数值判据。**实测**(dummy 族)建目录 + 加名 + `uv sync`
⇒ 两个模块都 import 通、都指回树内(editable)。
不用 `namespace = true`(官方注明**关掉安全检查**);也不把 `crates/kernels` 塞进
`sys.path`(会遮蔽真正的 `triton` 包)。配方写进 `triton/README.md`。

**顺带两处 vendor 补丁**(都写进 `triton/fla/ops/utils.py` 注记):
① `import torch` 改可选(用到 torch 的三处:AOT 不需要 `get_all_max_shared_mem`;
`is_intel_alchemist` 恒短路;`is_nvidia_hopper` 见②);
② `is_nvidia_hopper` 原式 `torch.cuda.get_device_name(0)` +
`get_device_capability()` 需 **torch + 真实 GPU** ⇒ 改为从**编译目标**
(`GPUTarget.arch >= 90`)判 —— 语义等价(Ampere ⇒ False),且跟随 target 而非宿主 GPU。

#### 15.11.7 剩余(未做)

- marlin / flashinfer 的 **host 侧搬 Rust**(§15.9.6 成本账)仍为独立工单;
- 构建期 Python+triton 是**硬依赖**(本机由工作区 `vllm-test/.venv` 提供;
  `GDN_TRITON_PY` 可指向其它解释器)。

---

### 15.12 C 轮:算子目录重构 —— `sources.rs` 退役 + 一算子族一 `.cu`(2026-10-13)

用户指令:「继续收尾算子目录重构」→ 依次完成 A → B → C。三棒一句话:**"名字 → 源"从两张表变一张,再让 .cu 文件与算子族对望。**

#### 15.12.1 A:17 个消费者迁移到唯一取源口

新增唯一读取口(解析序 = 算子自托管优先,登记表兜底):

```rust
// kernels/src/ops/mod.rs
pub fn source(name: &str) -> &'static str {
    match meta(name) { Some(m) => m.source, None => crate::list::registry_source(name) }
}
```

| 消费者 | 处数 | 迁移 |
|---|---|---|
| `server/gdn_chunked.rs` | 1(5 符号) | `sources::attention::CAST` → `ops::source("owl_cast_f16_bf16")` |
| `server/gdn_scalar.rs` | 1 | → `ops::source("owl_cast_f32_f16")` |
| 13 个探针 + `backends/cuda/tests/marlin_parity.rs` | 22 | `sources::X` → `ops::source("<本探针实际编译的入口名>")` |

**为什么用「入口名」而不是「文件常量名」**:探针编译的是具名入口,按名取源与它要做的事同构;
且未建表入口走登记表兜底 —— 两条路返回**同一份文本**(已逐条核对 15 个映射的源文件一致)。

#### 15.12.2 B:`sources.rs` 模块删除 + 两道新门

- `sources.rs` 的 `include_str!` 常量**就地并入 `list.rs` §0**(`mod cu`,`pub(crate)` 可见 ——
  家族私有核如 GDN 状态转置没有 `ops::source` 出口,需要它);
- 解析序收口:`ops::source` 是**唯一实现**,`list::source` 退化为转发;登记表侧原名
  `registry_source`(只给兜底分支用);
- **新门 1** `registry_source_agrees_with_op_meta`:登记表与算子声明同名的源必须**逐字节一致**
  (实测 55/110 一致,余 55 = 仅引入)—— 把"两处同时有"这份冗余**机器锁住**,不许漂移;
- **新门 2** `source_resolves_every_registered_name`:110 条登记名逐个解析,必须等于登记表所记。

**为什么保留一份冗余而不立刻消灭**:已建表算子的源已只有算子一个住址;登记表那份是
**未建表条目的唯一住址**,两者字段暂时同形。待 R1 建表补齐(§15.13),`REGISTRY` 由
`ops::all()` 派生 ⇒ 冗余与 `registry_source` 一起蒸发。

#### 15.12.3 C:一算子族一 `.cu`(用户裁决:按族拆、helper 随族走、**不抽公共头**)

**拆点定义**:一个 `.cu` 只装**一个算子族**的全部变体(dtype / 布局 / knhd / v1-v2 都算变体);
族之间不共享任何宏或 helper(共享 = 同族)。拆出的文件与母文件**同目录**(目录级对望
`cu/<域>/ ↔ ops/<域>.rs` 是纯 `git mv`,留作独立小步,不与本轮混)。

| 母文件 | 行数/入口 | 拆成 | 械 |
|---|---|---|---|
| `cu/owl/ops_pair.cu` | 133 / 18 变体 | `add.cu` `mul.cu` `silu.cu` `sigmoid.cu` `matmul.cu` `rmsnorm.cu` | 6 |
| `cu/owl/fused.cu` | 686 / 10 | `norm_rope.cu` `silu_and_mul.cu` `fused_add_rmsnorm.cu` `qknorm_insert.cu` | 4 |
| `cu/owl/gdn.cu` | 1068 / 15 | `gdn_{conv_upd,gating,delta_dec,norm_act,l2norm,conv_fwd,recurrence,decode_step}.cu` | 8 |
| `cu/owl/dflash2.cu` | 1226 / 15 | `dflash_conv.cu` `dflash_select.cu` `dflash_topk.cu` `naive_attn_nc.cu` | 4 |
| `cu/owl/cast.cu` | 52 / 6 | `cast.cu` + `state_transpose.cu` | 2 |
| `cu/owl/attention.cu` | 136 / 4 | `narrow_strided.cu` + `naive_decode_attn.cu` | 2 |
| `cu/attention/pagedattention_f16.cu` | 2132 / 24 | **不拆**(v1/v2/reduce 同族;拆它 = 抽 1500 行 vllm 模板机) | 0 |

**helper 随族走**:`gdn16_{to,from}_float` / `gdn16_warp_reduce_sum` 等在各 GDN 族文件里各一份;
`OWL_GDN*_MAX_KD` / `WARP_PER_BLOCK` / `OWL_NC_DPL_MAX` / `DFLASH_TOPK_*` / `OWL_MAX_KV`
等宏随使用它的族文件走。每文件**自足**(单 TU 编一个 fatbin,无跨文件符号解析问题)。

**顺带清掉的两条历史包袱**:
- `cu::attention::CAST` → 移入 `cu::owl::CAST`(owl 自研位,不是 attention port);
- `cu::text::GDN_F32` / `ATTENTION_F32`(f16 臂寄居在 `text` 模块)—— 随拆分回到各自族常量。

#### 15.12.4 机器门升级:**文件边界 = 算子族边界**

`cubin_launch_probe` 的判据 2 原为「一个文件里全变体一次出」,拆开后**反过来**了,故升级:

1. 每个已建表入口在**它自己 `SOURCE`** 的资产里**必须可见**;
2. 同一入口在**别的**资产的模块里**必须不可见**(`load_function` 必须报 NOT_FOUND)
   —— 这条同时证明"拆干净了"与"没有同一核两处各留一份"。

清单由 `ops::all()` 按源分组**自动生成**(零第二份清单);加的族/文件自动纳入。

#### 15.12.5 验收(全部实测)

| 项 | 数字 |
|---|---|
| 构建期 cubin 源 | 20 → **40**(拆出 30 个族文件,删 7 个母文件 + `sources.rs`) |
| `no_runtime_compile` | 2/2(零运行期编译仍是结构) |
| `op_tables`(表 ⇔ `.cu` 签名逐参) | **7/7**(覆盖全部 55 张原生表 —— 拆分后签名零漂移) |
| `selfcheck_oracles` / `cubin_launch_probe` | 8/8 · **6/6**(新边界门全绿) |
| kernels lib / models / shared | 60/60 · 117/117 · 27/27 |
| clippy / workspace check | **0 警** · 0 错 |
| **GPU 验收套件**(3090 Ti,`OWL_TEST_DEVICE=1`)| **11 套件 / 18 用例全绿**:cubin_launch(6) dflash2_nc(2) k0_fp8(1) kv_knhd_read(2) kv_knhd_write(2) gdn_chunked_golden(1) gdn_scalar_golden(1) gdn_big_t_arms(1) attention_split(1) chunked_prefill_fp8(1) gdn_chunked_chain(1) |

#### 15.12.6 踩坑留档(两条都实测复现过)

1. **`CUDA_VISIBLE_DEVICES` 会改探针语义**:`dflash2_nc_probe` 硬编码
   `CudaContext::new(1)`,而运行纪律是 **不设 CVD** 的原用法(`OWL_TEST_DEVICE=1`)。
   我加的 `CUDA_VISIBLE_DEVICES=1,2` 一路上报 `CUDA_ERROR_OUT_OF_MEMORY`(在
   `memcpy_htod` 的暂存面),换成原用法 **2/2 全绿** —— 假回归,别再踩。
   (同款教训已有先例:CUDA FASTEST_FIRST 序 ≠ nvidia-smi 序。)
2. **脚本化拆分必须校验路径拼接**:`ops/` 用 `../../cu/...`、`oracle` 用 `../../../cu/...`、
   `list.rs` 用 `../cu/...` —— 一次性 sed 很容易拼出 `cu/cu/...`。本次踩了两次,
   均被 `build.rs` 的 `include_str!` 解析门当场拦下(好门)。收尾统一
   `grep -r "cu/cu/"` 复查。

#### 15.12.7 剩余(未做)

- 目录级对望(`cu/<域>/` ↔ `ops/<域>.rs`)的纯搬移 —— 与拆分解耦,可在 R1 后一次做;
- `list.rs` / `REGISTRY` 整体退役 —— 依赖 R1(见 §15.13)。

---

### 15.13 D 轮:R1 归零 + 目录对望 + 配置漂移修复(2026-10-13)

用户指令:「把用得到的放进来先实现就行」+「继续,都做」。

#### 15.13.1 病灶回顾(R1 = 55 条"仅引入")

| 后果 | 实测 |
|---|---|
| **白名单硬门放不进它们** | `--flavor awq27b testdata/server/bench-27b-std.toml` **整档起不来**:流式请求全被准入门拒,`out=0`、TTFT 假性 14ms(测速工具把空响应算成 39912 t/s 的伪影) |
| **池写集穷举审计不覆盖** | E10(漏写者⇒槽区全零)/ E14(双寻址)两类"不崩但胡说"的 bug 无机器判据 |
| **`REGISTRY` 退不了役** | 名字→源/槽序/dtype 的第二份手抄继续存在(§15.12.2 留的冗余) |

#### 15.13.2 建表方法:**从登记表 + `.cu` 派生,不手抄**

生成器两步(见 §15.13.6 的复现要点):
1. 取登记表该条目的 `args` 串(它已被 `list::registry_sig_matches_cu_signatures` 门锁住 ⇒ 可信);
2. 从目标 `.cu` 解析**参数名与 C 类型**(字面核 + 单层宏实例化两种形态),按 token 逐位映射成 `Param`:
   `T→Param::t_in/t_out(名, dtype, 2)`、`sz→Param::sz`、`i32→Param::i32`、`f32→Param::f32`。

⇒ 生成物**一次过 `op_tables` 门**(110 张表逐参互证),零人眼对账。

#### 15.13.3 三批进度

| 批 | 域 | 张数 |
|---|---|---:|
| 1 | `elems` 16 + `ln` 1 + `dflash` 2 | 19 |
| 2 | `gdn` 8 + `fused` 2 | 10 |
| 3 | `attn` 26(**KV 池重口径**) | 26 |
| 3' | 家族臂 FFI 虚核:`flashinfer_prefill_paged_f16` + `gdn_scalar_delta_rule_fwd` | 2 |

```
起点 [R1] 未声明 55 / 登记 110 / 已建表 64
终点 [R1] 未声明  0 / 登记 110 / 已建表 121(110 原生 + 11 家族臂)
[op-tables] 110 张表与 .cu 签名逐参一致 ✓
```
**R1 门阈值 55 → 0**(`r1_undeclared_inventory`):此后新增任何核(含 FFI 家族臂)不带声明立刻红。

#### 15.13.4 准入门自己抓到的两个真错(本轮的"门有用"实证)

1. **输出槽角色错**:生成器忠实照抄登记表,而登记表对 GDN 族**没给输出槽打 `O`**(写的是 `T`)⇒ 17 处 `Param::t_in("out", …)`(角色=输入)。已全部改回 `t_out`,顺带**把门变强**:`op_tables` 现在会断言 `O` 槽在 `.cu` 里必须是**指针位**。
2. **`ShapeFamily.page` 猜窄**:26 张 attn 表我按名字后缀(`bs32`→32,否则 16)推 `page`,而 cache 写核与片大小无关 —— `models::specs::qwen35::tests::gpu_real_weights_prefill` 当场红:
   ```
   driver::resolve("attn.k0_write_knhd") -> vllm_reshape_and_cache_f16_knhd:
   几何不接受 —— page 要求 ∈ [16],实得 32
   ```
   修正律:**池几何 = 配置决定 ⇒ 声明"受支持集合"**(`page: &[16,32,64,128]`,head_dim 无后缀者放宽 `[128,256]`),细窄留给真正的特化核(`*_hd128`)。

#### 15.13.5 目录级对望(用户裁决「都做」)

规则:`cu/<域>/` ↔ `ops/<域>.rs`;私有/无声明归位见下。

```
cu/attn/   ← ops/attn.rs     (8)    cu/gdn/   ← ops/gdn.rs      (10, 含 family 私有 state_transpose)
cu/elems/  ← ops/elems.rs    (14)   cu/load/  ← ops/load.rs     (1)
cu/ln/     ← ops/ln.rs       (3)    cu/probe/ (ts_probe)
cu/fused/  ← ops/fused.rs    (1)    cu/marlin/ cu/xqa/ cu/flashinfer/  (家族自管,不动)
cu/dflash/ ← ops/dflash.rs   (2)
```
- **搬 39 个文件 + 改写 id 字面量**;`cu/owl/`、`cu/attention/` 两个"杂项目录"消失;
- 安全性由 §15.14 引入的 **build.rs 路径存在性校验**兜住(搬完仍 40 源,只重编搬动那批);
- 踩坑:首次脚本用仓库根相对路径(`cu/…`)跑,`git mv/mv` 双双失败 → 脚本第一项就 abort ⇒ **零文件被移动**(幸而如此);加 `crates/kernels/` 前缀后一次成功 —— 批量搬移**必须前置断言源文件存在**(与批量替换必须断言命中同理)。

#### 15.13.6 in-tree 配置漂移定谳 + 修复

`testdata/server/bench-27b-std.toml` 与 flavor 白名单**两面同时对不上**(该 config 早于 P4 白名单硬门):

| 面 | config | `awq27b` flavor 台账 |
|---|---|---|
| 池 dtype | **f16**(`kv_fp8` 缺省 false) | **fp8**(`PROD_AWQ27B_2026_10_13`) |
| GDN 臂 | `gdn_chunked = true` | 只有 gdn_scalar 臂 |

修法:**不动 flavor 台账**(它是"实测发射集"声明,不许无凭据改),改为在该 config 的文件层给 `allowed_ops`(四层叠加里文件层 > flavor 层,config 头注已列明"第 4 处有意覆盖")。R1 归零后 f16 那一撮全有声明,清单才写得出来 —— **这正是 D 的可见收益**。

**实测(修复后首次真跑)**:`引擎就绪(捕获态 Captured)` + `生产白名单:28 项生效(池写集审计通过)`,`std` 五点三轮 **15/15**:prefill avg **1309.60** · decode avg **39.24** · TTFT@512 385-392ms —— 对拍同 f16 池口径历史基线(1318.11 / 39.14)与 C 轮后复测(1319.10 / 39.05):**±0.7% 内**。

#### 15.13.7 验收(全部实测)

| 项 | 数字 |
|---|---|
| `[R1]` | 未声明 **0** / 登记 110 / 已建表 **121** |
| `op_tables` | **110 张表**逐参互证 |
| kernels lib / models / shared | 62/62(含新增家族臂孪生门) · 117/117 · 27/27 |
| clippy / workspace check | **0 警** / 0 错 |
| GPU 套件 | cubin_launch 6/6 + **13 套探针全绿** |
| 真机 boot | fp8 档(spec-dflash)与 **f16 档(awq27b)** 双双 Captured;std 均 15/15 |

#### 15.13.8 剩余(未做)

- **`REGISTRY` 退役**:全部 121 项都有声明后,登记表可由 `ops::all()` 派生(含 `Entry.src` 那份冗余),`registry_src` 一并删除 —— 独立小步,判据 = `registry_source_agrees_with_op_meta` 门失去对象时。
- `ops/attn.rs` 的 26 张新表可进一步用 `macro_rules!` 收敛(现为生成器一次写出的展开形态)。

---

### 15.14 E 轮:**运行期源文本退役**(2026-10-13)

用户指令(指着 `const SOURCE: &'static str = include_str!("../../cu/owl/gdn_delta_dec.cu");`):
> 「我们走的预编译了,不再使用内联 JIT」

#### 15.14.1 病灶:同一份 `.cu` 文本在二进制里躺了两遍

预编译后,源文本**只参与构建期**(决定编什么),运行期只需要**资产身份**。但当时:

```
KernelSpec::SOURCE = include_str!("…")              ← 第一份文本(躺在 kernels rlib)
CubinAsset { asset, src: include_str!("…"),         ← 第二份文本(build.rs 生成进资产表)
             src_hash, src_len, bytes }
运行期真正用到:src_hash / src_len / bytes —— src 只给测试/诊断
查表: asset_of_source(源文本) ⇒ 每次装载真算一遍 FNV-1a64
```

「内容当键」的自证性(设计 §15.5 的立意:名字是"第三份表")在**构建期**依然成立;运行期还随身带文本纯属历史包袱。

#### 15.14.2 改动面

| 面 | 之前 | 之后 |
|---|---|---|
| 声明 | `const SOURCE: &'static str = include_str!(…)` | `const SRC: &'static str = "cu/…/x.cu"`(**资产 id**;改名以免"一名两义") |
| 资产表 | `{ asset, **src**, src_hash, **src_len**, bytes }` | `{ **id**, **module**, src_hash, bytes }` |
| 查表 | `asset_of_source(文本)` | `asset(id)`(按键查;`has_aot(id)`) |
| 发现规则(build.rs) | 扫 `include_str!` 闭包 | 扫 **`const … = "cu/…/x.cu"` 字面量** + **逐条存在性校验**(路径写错 = 构建期失败);仍零第二份清单,`rerun-if-changed` 覆盖 `.cu` 本体 |
| "源里有没有这个核" | `source_defines(文本, 名)` 正则扫文本 | 构建期生成的 **`ASSET_SYMBOLS`**(每资产符号清单;字面核 + 宏实例化)→ 判据从"文本像不像"升级为"编译期真有没有" |
| 测试/工具面取源 | 直接吃内嵌文本 | `list::cu_text(id)` **按需读盘**(`CARGO_MANIFEST_DIR`) |

`ops::source(name)` 保留为**唯一读取口**(现在返回资产 id),`list::source` 转发;`KernelSource.source`(线格式字段)语义同为 id,名字未改(挂账)。

#### 15.14.3 验证:二进制里真的没有源文本了

```
target/release/server (39M)
--- 源文本特征串(应未命中)---
  extern "C" __global__ void                    未命中 ✓
  OWL_ADD_KERNEL / OWL_RMSNORM_KERNEL           未命中 ✓
  __nv_fp8_e4m3(__half2float                    未命中 ✓
  blockIdx.x * blockDim.x + threadIdx.x         未命中 ✓
  DFLASH_TOPK_BLOCK / OWL_NC_SLIDING_WINDOW     未命中 ✓
--- 只应有的资产 id 串 ---
  cu/elems/add.cu / cu/attn/pagedattention_f16.cu / cu/gdn/gdn_chunk_scalar.cu   命中 ✓
```

#### 15.14.4 踩坑留档(四条,全部实测复现)

1. **两个函数共用同一行代码**:`source_defines` 与 `parse_cu_sig` 都有 `let clean: String = source`,批量替换把**解析器**一起改坏 —— 靠编译错误定位。**替换前先 `grep` 数一遍目标行数**。
2. **扫描规则切换会静默丢文件**:`include_str!` 闭包 → 字面量扫描后源数 **40 → 39**,漏的是 `family/gdn_scalar.rs` 里那份"就地 include_str"(既不在 `ops/` 也不在 `list.rs`)。补法:拿 `find cu/**/*.cu`(减家族自管目录)对账 —— **数量核对是这类规则切换的唯一保险**。
3. **宏形参漏进符号清单**:宏定义体里的 `NAME` 被当成核名(如 `CHUNKED_KNHD_ENTRY` 族)。修法:核名一律含小写 ⇒ 只收"含小写字母"的标识符。另有一处 id 少了 `cu/` 前缀(符号与 id 错位),一并修。
4. **补丁静默不命中**:我给的 oldText 写的是 `SRC` 而文件里还是 `SOURCE`,`str.replace` 无声无息 ⇒ `op_tables` 仍在拿 **id 当文本**比对(报"SOURCE 不是核源码")。**纪律**:所有批量替换必须 `assert` 命中(§15.12.6 已立过同款规矩,本轮再犯一次)。

#### 15.14.5 验收(全部实测)

| 项 | 结果 |
|---|---|
| 构建期 cubin | **40 源**(与重构前一致) |
| `op_tables` | 110 张表逐参互证(文本从磁盘取) |
| lib / oracles / no_runtime_compile | 62/62 · 8/8 · 2/2 |
| **GPU** | cubin_launch 6/6(含 NVRTC↔fatbin **逐位等价**,NVRTC 对照臂源从磁盘取)+ 13 套探针全绿 |
| **真机 boot** | spec-dflash / awq27b 双双 `Captured`,问答输出连贯 |

#### 15.14.6 剩余(未做)

- `KernelSource.source`(线格式字段)名字未改,语义已是 id;
- 家族自管资产(marlin/xqa/flashinfer/gdn_chunked cubin 表)仍按**名字**查表(它们不经 build.rs 的 `.cu` 编译面),与本轮统一化是两条链,可另立项;
- 目录对望已做(§15.13.5),但 `ops/attn.rs` 内 26 张表尚未宏收敛。
