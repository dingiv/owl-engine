//! **GGUF 容器解析**(纯 host;header / KV / tensor table / 对齐 / mmap)。
//!
//! 设计见 `docs/arch/gguf-格式支持-设计与施工.md` §四 L1。三条纪律:
//!
//! 1. **纯索引,零数据拷贝**:整文件只读 mmap;解析只产出 KB 级索引
//!    (张量表 + KV 表),权重字节按需切片取用;
//! 2. **命中即定,残缺/越界/未知 = 响亮拒**:一切数字都做边界校验
//!    (防御性上限借 `mistralrs-quant/src/gguf/archive.rs` 的取值);
//!    解析**不因型未知而失败**(可跑性由 `owl_kernels::family::gguf::entry_for` 单独回答;
//!    执行/装载侧才拒);
//! 3. **字节布局是事实,不是推导**:张量字节数由 `ne[0]` 的块几何算出,
//!    并逐条与 mmap 长度对账(截断文件当场拒,不留半装载)。
//!
//! 权威源:本机 `llama.cpp` 树(`ggml/include/gguf.h` 的类型编号与版本、
//! `gguf-py/gguf/constants.py` 的块几何 `GGML_QUANT_SIZES`)。
//! 对拍门 = `tests::matches_gguf_py`(独立第三源,逐字段)。

use crate::contract::ModelError;
use crate::formats::mmap::Mmap;
use owl_kernels::family::gguf::GgmlTy;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// GGUF magic(`"GGUF"` 小端读出的 u32)
pub const MAGIC: u32 = 0x4655_4747;
/// 本仓支持的容器版本(2 / 3)。v1 的张量表没有 `n_dims` 字段 ⇒ 不解析。
pub const VERSIONS: &[u32] = &[2, 3];
/// 默认对齐(未声明 `general.alignment` 时;`gguf.h: GGUF_DEFAULT_ALIGNMENT`)
pub const DEFAULT_ALIGNMENT: u64 = 32;

// ---- 防御性上限(借 mistralrs-quant archive.rs;防越界/无限膨胀)----
const MAX_STRING_LEN: u64 = 1 << 30;
const MAX_ARRAY_ELEMS: u64 = 1 << 30;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_VALUE_DEPTH: usize = 64;
/// KV 条数与张量数的上限(防"计数撒谎"型畸形文件把解析拖进天量循环)
const MAX_ENTRIES: u64 = 1 << 26;

// ---- gguf.h: enum gguf_type(判别值权威)----
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// GGUF 元数据值(KV 面;`ARRAY` 可嵌套)
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// 类型名(GGUF 官方拼法;诊断用)
    pub fn type_name(&self) -> &'static str {
        match self {
            GgufValue::U8(_) => "UINT8",
            GgufValue::I8(_) => "INT8",
            GgufValue::U16(_) => "UINT16",
            GgufValue::I16(_) => "INT16",
            GgufValue::U32(_) => "UINT32",
            GgufValue::I32(_) => "INT32",
            GgufValue::F32(_) => "FLOAT32",
            GgufValue::Bool(_) => "BOOL",
            GgufValue::String(_) => "STRING",
            GgufValue::Array(_) => "ARRAY",
            GgufValue::U64(_) => "UINT64",
            GgufValue::I64(_) => "INT64",
            GgufValue::F64(_) => "FLOAT64",
        }
    }

    /// 渲染(诊断/日志;字符串加引号,数组 CSV)
    pub fn render(&self, max: usize) -> String {
        let s = match self {
            GgufValue::String(s) => format!("{s:?}"),
            GgufValue::Array(v) => {
                let mut parts: Vec<String> = v.iter().take(8).map(|x| x.render(24)).collect();
                if v.len() > 8 {
                    parts.push(format!("…(+{})", v.len() - 8));
                }
                format!("[{}]", parts.join(", "))
            }
            GgufValue::Bool(b) => b.to_string(),
            GgufValue::F32(x) => x.to_string(),
            GgufValue::F64(x) => x.to_string(),
            GgufValue::U8(x) => x.to_string(),
            GgufValue::I8(x) => x.to_string(),
            GgufValue::U16(x) => x.to_string(),
            GgufValue::I16(x) => x.to_string(),
            GgufValue::U32(x) => x.to_string(),
            GgufValue::I32(x) => x.to_string(),
            GgufValue::U64(x) => x.to_string(),
            GgufValue::I64(x) => x.to_string(),
        };
        s.chars().take(max).collect()
    }
}

/// KV 表(保序 = 文件序;GGUF 规范要求键有序,但本表不依赖该性质)
#[derive(Debug, Clone, Default)]
pub struct KvTable {
    entries: Vec<(String, GgufValue)>,
}

impl KvTable {
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &GgufValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    // ---- 带键名的类型校验取值(缺/型错 = 响亮拒;`opt_*` 缺 = None)----

    fn want<T>(&self, key: &str, f: impl FnOnce(&GgufValue) -> Option<T>) -> Result<T, String> {
        match self.get(key) {
            None => Err(format!("缺键 `{key}`")),
            Some(v) => match f(v) {
                Some(x) => Ok(x),
                None => Err(format!("键 `{key}` 类型不符:实得 {}", v.type_name())),
            },
        }
    }

    fn opt<T>(&self, key: &str, f: impl FnOnce(&GgufValue) -> Option<T>) -> Result<Option<T>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => match f(v) {
                Some(x) => Ok(Some(x)),
                None => Err(format!("键 `{key}` 类型不符:实得 {}", v.type_name())),
            },
        }
    }

    /// 无符号整数(接受 U8/U16/U32/U64;GGUF 写者惯例不一)
    fn as_u64(v: &GgufValue) -> Option<u64> {
        match v {
            GgufValue::U8(x) => Some(*x as u64),
            GgufValue::U16(x) => Some(*x as u64),
            GgufValue::U32(x) => Some(*x as u64),
            GgufValue::U64(x) => Some(*x),
            GgufValue::I8(x) if *x >= 0 => Some(*x as u64),
            GgufValue::I16(x) if *x >= 0 => Some(*x as u64),
            GgufValue::I32(x) if *x >= 0 => Some(*x as u64),
            GgufValue::I64(x) if *x >= 0 => Some(*x as u64),
            _ => None,
        }
    }

    fn as_f64(v: &GgufValue) -> Option<f64> {
        match v {
            GgufValue::F32(x) => Some(*x as f64),
            GgufValue::F64(x) => Some(*x),
            GgufValue::U32(x) => Some(*x as f64),
            GgufValue::I32(x) => Some(*x as f64),
            GgufValue::U64(x) => Some(*x as f64),
            GgufValue::I64(x) => Some(*x as f64),
            _ => None,
        }
    }

    pub fn u64(&self, key: &str) -> Result<u64, String> {
        self.want(key, Self::as_u64)
    }
    pub fn opt_u64(&self, key: &str) -> Result<Option<u64>, String> {
        self.opt(key, Self::as_u64)
    }
    pub fn usize(&self, key: &str) -> Result<usize, String> {
        Ok(self.u64(key)? as usize)
    }
    pub fn opt_usize(&self, key: &str) -> Result<Option<usize>, String> {
        Ok(self.opt_u64(key)?.map(|x| x as usize))
    }
    pub fn u32(&self, key: &str) -> Result<u32, String> {
        let v = self.u64(key)?;
        u32::try_from(v).map_err(|_| format!("键 `{key}` 超 u32 域({v})"))
    }
    pub fn opt_u32(&self, key: &str) -> Result<Option<u32>, String> {
        match self.opt_u64(key)? {
            None => Ok(None),
            Some(v) => u32::try_from(v)
                .map(Some)
                .map_err(|_| format!("键 `{key}` 超 u32 域({v})")),
        }
    }
    pub fn f32(&self, key: &str) -> Result<f32, String> {
        self.want(key, Self::as_f64).map(|x| x as f32)
    }
    pub fn opt_f32(&self, key: &str) -> Result<Option<f32>, String> {
        Ok(self.opt(key, Self::as_f64)?.map(|x| x as f32))
    }
    pub fn bool(&self, key: &str) -> Result<bool, String> {
        self.want(key, |v| match v {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        })
    }
    pub fn opt_bool(&self, key: &str) -> Result<Option<bool>, String> {
        self.opt(key, |v| match v {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        })
    }
    pub fn str(&self, key: &str) -> Result<&str, String> {
        match self.get(key) {
            None => Err(format!("缺键 `{key}`")),
            Some(GgufValue::String(s)) => Ok(s.as_str()),
            Some(v) => Err(format!("键 `{key}` 类型不符(应 STRING):实得 {}", v.type_name())),
        }
    }
    pub fn opt_str(&self, key: &str) -> Result<Option<&str>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(GgufValue::String(s)) => Ok(Some(s.as_str())),
            Some(v) => Err(format!("键 `{key}` 类型不符(应 STRING):实得 {}", v.type_name())),
        }
    }
    /// 数组 → u32 序列(如 `attention.sliding_window_pattern` / `rope.dimension_sections`)
    pub fn opt_u32s(&self, key: &str) -> Result<Option<Vec<u32>>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(GgufValue::Array(v)) => {
                let mut out = Vec::with_capacity(v.len());
                for x in v {
                    out.push(Self::as_u64(x).ok_or_else(|| {
                        format!("键 `{key}` 数组元素类型不符:实得 {}", x.type_name())
                    })? as u32);
                }
                Ok(Some(out))
            }
            Some(v) => Err(format!("键 `{key}` 类型不符(应 ARRAY):实得 {}", v.type_name())),
        }
    }
    /// 数组 → bool 序列(`attention.sliding_window_pattern` 的布尔写法)
    pub fn opt_bools(&self, key: &str) -> Result<Option<Vec<bool>>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(GgufValue::Array(v)) => {
                let mut out = Vec::with_capacity(v.len());
                for x in v {
                    match x {
                        GgufValue::Bool(b) => out.push(*b),
                        other => {
                            return Err(format!(
                                "键 `{key}` 数组元素类型不符(应 BOOL):实得 {}",
                                other.type_name()
                            ))
                        }
                    }
                }
                Ok(Some(out))
            }
            Some(v) => Err(format!("键 `{key}` 类型不符(应 ARRAY):实得 {}", v.type_name())),
        }
    }
    /// 数组 → 字符串序列(tokenizer 词表用)
    pub fn opt_strs(&self, key: &str) -> Result<Option<Vec<String>>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(GgufValue::Array(v)) => {
                let mut out = Vec::with_capacity(v.len());
                for x in v {
                    match x {
                        GgufValue::String(s) => out.push(s.clone()),
                        other => {
                            return Err(format!(
                                "键 `{key}` 数组元素类型不符(应 STRING):实得 {}",
                                other.type_name()
                            ))
                        }
                    }
                }
                Ok(Some(out))
            }
            Some(v) => Err(format!("键 `{key}` 类型不符(应 ARRAY):实得 {}", v.type_name())),
        }
    }
    /// 数组 → f32 序列
    pub fn opt_f32s(&self, key: &str) -> Result<Option<Vec<f32>>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(GgufValue::Array(v)) => {
                let mut out = Vec::with_capacity(v.len());
                for x in v {
                    out.push(Self::as_f64(x).ok_or_else(|| {
                        format!("键 `{key}` 数组元素非数值:{}", x.type_name())
                    })? as f32);
                }
                Ok(Some(out))
            }
            Some(v) => Err(format!("键 `{key}` 类型不符(应 ARRAY):实得 {}", v.type_name())),
        }
    }
}

/// 张量条目(索引面;字节仍在 mmap 里按需取)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ty: GgmlTy,
    /// **GGML 序**:`ne[0]` = 最内维(= Linear 的 `in_dim`)。
    /// ⚠️ 转 owl `[out, in]` 的翻转在**源侧**一次完成(设计 I-1)。
    pub ne: Vec<u64>,
    /// 相对**数据区起点**的字节偏移(规范要求对齐)
    pub offset: u64,
    /// 字节数(= `ne[0] / block.0 × block.1 × ne[1..] 连乘`)
    pub nbytes: u64,
}

impl TensorInfo {
    /// 元素总数
    pub fn elems(&self) -> u64 {
        self.ne.iter().product()
    }
    /// 行数(= 除最内维外连乘;GGML 序下的"外维")
    pub fn rows(&self) -> u64 {
        self.ne[1..].iter().product()
    }
    /// 最内维(= 块轴)
    pub fn inner(&self) -> u64 {
        self.ne[0]
    }
    /// 一行字节数(块轴 × 块字节)
    pub fn row_bytes(&self) -> u64 {
        let (be, bb) = self.ty.block();
        self.ne[0] / be as u64 * bb as u64
    }
}

/// 打开好的 GGUF 文件(索引 + mmap 句柄)
pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    pub kv: KvTable,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// 数据区起点(文件绝对偏移)
    pub data_offset: u64,
    map: Mmap,
    by_name: HashMap<String, usize>,
}

impl Gguf {
    /// 打开并解析(`path` 须为 `*.gguf` 单文件;目录 ⇒ 拒)
    pub fn open(path: &Path) -> Result<Gguf, ModelError> {
        let ctx = |m: String| ModelError::Msg(format!("gguf: {}: {m}", path.display()));
        if path.is_dir() {
            return Err(ctx("是目录;GGUF 是单文件格式(设计 §四 L4:model.dir 需支持文件路径)".into()));
        }
        let map = Mmap::open(path)?;
        let buf: &[u8] = &map;
        let mut c = Cur { buf, pos: 0, ctx: &ctx };

        let magic = c.u32()?;
        if magic != MAGIC {
            return Err(c.err(format!("magic {magic:#010x} ≠ {MAGIC:#010x}(\"GGUF\")")));
        }
        let version = c.u32()?;
        if !VERSIONS.contains(&version) {
            return Err(c.err(format!("容器版本 {version} 不受支持(本仓 {:?})", VERSIONS)));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if n_tensors > MAX_ENTRIES || n_kv > MAX_ENTRIES {
            return Err(c.err(format!("计数越界(tensors {n_tensors} / kv {n_kv})")));
        }

        // ---- KV ----
        let mut entries = Vec::with_capacity(n_kv.min(1 << 16) as usize);
        for _ in 0..n_kv {
            let key = c.string()?;
            let v = c.value(0)?;
            if entries.iter().any(|(k, _): &(String, GgufValue)| *k == key) {
                return Err(c.err(format!("KV 键重复 `{key}`")));
            }
            entries.push((key, v));
        }
        let kv = KvTable { entries };

        // ---- 张量表 ----
        let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16) as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims == 0 || n_dims > MAX_TENSOR_DIMS {
                return Err(c.err(format!("张量 `{name}` 维数 {n_dims} 越界(1..={MAX_TENSOR_DIMS})")));
            }
            let mut ne = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = c.u64()?;
                if d == 0 {
                    return Err(c.err(format!("张量 `{name}` 有零维")));
                }
                ne.push(d);
            }
            let tcode = c.u32()?;
            let ty = GgmlTy::from_code(tcode).ok_or_else(|| {
                c.err(format!("张量 `{name}` 型编码 {tcode} 不在 ggml_type 表中"))
            })?;
            let offset = c.u64()?;

            // 块轴整除(规范要求;不满足 = 字节布局无定义)
            let (be, bb) = ty.block();
            if ne[0] % be as u64 != 0 {
                return Err(c.err(format!(
                    "张量 `{name}` 最内维 {} 不是块大小 {be} 的整数倍({} 型)",
                    ne[0],
                    ty.name()
                )));
            }
            let rows: u64 = ne[1..].iter().product();
            let nbytes = (ne[0] / be as u64) * bb as u64 * rows;
            tensors.push(TensorInfo { name, ty, ne, offset, nbytes });
        }

        // ---- 对齐 + 数据区起点 ----
        let alignment = kv.opt_u64("general.alignment").map_err(&ctx)?.unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ctx(format!("general.alignment = {alignment} 非 2 的幂")));
        }
        let data_offset = align_up(c.pos, alignment);
        if data_offset > buf.len() as u64 {
            return Err(ctx(format!(
                "数据区起点 {data_offset} 超出文件长度 {}",
                buf.len()
            )));
        }

        // ---- 逐条对账:偏移对齐 + 区间在文件内 + 不重叠 ----
        let mut by_name = HashMap::with_capacity(tensors.len());
        let mut spans: Vec<(u64, u64, &str)> = Vec::with_capacity(tensors.len());
        for t in &tensors {
            if t.offset % alignment != 0 {
                return Err(ctx(format!(
                    "张量 `{}` 偏移 {} 未按 alignment {alignment} 对齐",
                    t.name, t.offset
                )));
            }
            let start = data_offset + t.offset;
            let end = start + t.nbytes;
            if end > buf.len() as u64 {
                return Err(ctx(format!(
                    "张量 `{}` 区间 [{start}, {end}) 超出文件长度 {}",
                    t.name,
                    buf.len()
                )));
            }
            spans.push((start, end, t.name.as_str()));
            if by_name.insert(t.name.clone(), by_name.len()).is_some() {
                return Err(ctx(format!("张量名重复 `{}`", t.name)));
            }
        }
        spans.sort();
        for w in spans.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(ctx(format!("张量区间重叠:`{}` 与 `{}`", w[0].2, w[1].2)));
            }
        }

        Ok(Gguf { path: path.to_path_buf(), version, kv, tensors, alignment, data_offset, map, by_name })
    }

    /// 按名取条目索引
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// 按名取条目(索引面;缺 = None —— 调用方决定拒还是兜底)
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index_of(name).map(|i| &self.tensors[i])
    }

    /// 条目字节(零拷贝切片)
    pub fn bytes(&self, ix: usize) -> &[u8] {
        let t = &self.tensors[ix];
        let s = (self.data_offset + t.offset) as usize;
        &self.map[s..s + t.nbytes as usize]
    }

    /// 架构名(`general.architecture`;**必须在场** —— 无它无法选键面/超参)
    pub fn arch(&self) -> Result<&str, ModelError> {
        self.kv
            .str("general.architecture")
            .map_err(|e| ModelError::Msg(format!("gguf: {}: {e}", self.path.display())))
    }

    /// 数据区归还(mmadvise DONTNEED;**只对已消费区间**调用 —— 再访问会重新缺页)
    pub fn release(&self, ix: usize) {
        let t = &self.tensors[ix];
        self.map.dontneed((self.data_offset + t.offset) as usize, t.nbytes as usize);
    }

    /// 摘要(诊断;条目数 + 型直方图 + 架构)
    pub fn summary(&self) -> String {
        let mut hist: Vec<(GgmlTy, usize)> = Vec::new();
        let mut bytes = 0u64;
        for t in &self.tensors {
            match hist.iter_mut().find(|(k, _)| *k == t.ty) {
                Some((_, n)) => *n += 1,
                None => hist.push((t.ty, 1)),
            }
            bytes += t.nbytes;
        }
        hist.sort_by_key(|(t, _)| t.code());
        let h: Vec<String> = hist.iter().map(|(t, n)| format!("{}×{n}", t.name())).collect();
        format!(
            "v{} · arch {} · {} 张 / {:.2}G · {}",
            self.version,
            self.arch().unwrap_or("<无 general.architecture>"),
            self.tensors.len(),
            bytes as f64 / (1u64 << 30) as f64,
            h.join(" ")
        )
    }
}

/// 对齐向上取整
fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

/// 字节游标(一切读取都过边界校验;错误带文件路径)
struct Cur<'a, 'b> {
    buf: &'a [u8],
    pos: u64,
    ctx: &'b dyn Fn(String) -> ModelError,
}

impl Cur<'_, '_> {
    fn err(&self, m: String) -> ModelError {
        (self.ctx)(format!("@{}({m})", self.pos))
    }

    fn take(&mut self, n: u64) -> Result<&[u8], ModelError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| self.err(format!("长度溢出({n} 字节)")))?;
        if end > self.buf.len() as u64 {
            return Err(self.err(format!(
                "截断:需 {n} 字节,仅余 {}",
                self.buf.len() as u64 - self.pos
            )));
        }
        let s = &self.buf[self.pos as usize..end as usize];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8, ModelError> {
        Ok(self.take(1)?[0] as i8)
    }
    fn u16(&mut self) -> Result<u16, ModelError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16, ModelError> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, ModelError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, ModelError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, ModelError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, ModelError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// GGUF 字符串:u64 长度 + 裸字节(**非** NUL 结尾);必须 UTF-8
    fn string(&mut self) -> Result<String, ModelError> {
        let n = self.u64()?;
        if n > MAX_STRING_LEN {
            return Err(self.err(format!("字符串长度 {n} 越界")));
        }
        let b = self.take(n)?;
        std::str::from_utf8(b)
            .map(|s| s.to_string())
            .map_err(|e| self.err(format!("字符串非 UTF-8: {e}")))
    }

    /// 值(按 `gguf_type` 判别;`ARRAY` 递归,带深度上限)
    fn value(&mut self, depth: usize) -> Result<GgufValue, ModelError> {
        if depth > MAX_VALUE_DEPTH {
            return Err(self.err("KV 嵌套深度越界".into()));
        }
        let t = self.u32()?;
        Ok(match t {
            T_U8 => GgufValue::U8(self.u8()?),
            T_I8 => GgufValue::I8(self.i8()?),
            T_U16 => GgufValue::U16(self.u16()?),
            T_I16 => GgufValue::I16(self.i16()?),
            T_U32 => GgufValue::U32(self.u32()?),
            T_I32 => GgufValue::I32(self.i32()?),
            T_F32 => GgufValue::F32(self.f32()?),
            T_BOOL => GgufValue::Bool(self.u8()? != 0),
            T_STRING => GgufValue::String(self.string()?),
            T_U64 => GgufValue::U64(self.u64()?),
            T_I64 => GgufValue::I64(self.i64()?),
            T_F64 => GgufValue::F64(self.f64()?),
            T_ARRAY => {
                let et = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARRAY_ELEMS {
                    return Err(self.err(format!("数组元素数 {n} 越界")));
                }
                // 元素类型不得为 ARRAY(规范禁嵌套数组)
                if et == T_ARRAY {
                    return Err(self.err("数组元素类型不得为 ARRAY(规范禁嵌套)".into()));
                }
                let mut v = Vec::with_capacity(n.min(1 << 16) as usize);
                for _ in 0..n {
                    v.push(self.scalar(et, depth + 1)?);
                }
                GgufValue::Array(v)
            }
            other => return Err(self.err(format!("未知 KV 值类型 {other}"))),
        })
    }

    /// 数组元素(标量;按**已读出的**类型码,不再读判别符)
    fn scalar(&mut self, t: u32, depth: usize) -> Result<GgufValue, ModelError> {
        Ok(match t {
            T_U8 => GgufValue::U8(self.u8()?),
            T_I8 => GgufValue::I8(self.i8()?),
            T_U16 => GgufValue::U16(self.u16()?),
            T_I16 => GgufValue::I16(self.i16()?),
            T_U32 => GgufValue::U32(self.u32()?),
            T_I32 => GgufValue::I32(self.i32()?),
            T_F32 => GgufValue::F32(self.f32()?),
            T_BOOL => GgufValue::Bool(self.u8()? != 0),
            T_STRING => GgufValue::String(self.string()?),
            T_U64 => GgufValue::U64(self.u64()?),
            T_I64 => GgufValue::I64(self.i64()?),
            T_F64 => GgufValue::F64(self.f64()?),
            _ => {
                let _ = depth;
                return Err(self.err(format!("数组元素类型 {t} 非标量")));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_shared::testkit::{resource, resource_ok};
    use std::process::Command;

    /// 资源路径(清单优先;未列/不存在 ⇒ None ⇒ 用例 skip)
    fn res(key: &str) -> Option<PathBuf> {
        resource_ok(key).then(|| PathBuf::from(resource(key).expect("列了就有")))
    }

    // ---- canonical 编码(与 `gguf_dump.py` 逐字节一致;协议见该脚本头注)----

    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn fnv1a64(buf: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        for b in buf {
            h = (h ^ *b as u64).wrapping_mul(FNV_PRIME);
        }
        h
    }

    /// 标量 → 8 字节 LE(整数二补码位型 / 浮点 IEEE 位型 / 布尔 0-1)
    fn enc_scalar(v: &GgufValue, out: &mut Vec<u8>) {
        let bits: u64 = match v {
            GgufValue::F32(x) => x.to_bits() as u64,
            GgufValue::F64(x) => x.to_bits(),
            GgufValue::Bool(b) => *b as u64,
            GgufValue::U8(x) => *x as u64,
            GgufValue::I8(x) => *x as i64 as u64,
            GgufValue::U16(x) => *x as u64,
            GgufValue::I16(x) => *x as i64 as u64,
            GgufValue::U32(x) => *x as u64,
            GgufValue::I32(x) => *x as i64 as u64,
            GgufValue::U64(x) => *x,
            GgufValue::I64(x) => *x as u64,
            other => panic!("非标量不可入标量编码:{} {other:?}", other.type_name()),
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }

    /// KV 值 → (len, fnv) —— len 语义:STRING 标量 = 字节数;ARRAY = 元素数;其余 = 1
    fn canonical(v: &GgufValue) -> (u64, u64) {
        let mut buf = Vec::new();
        let len = match v {
            GgufValue::String(s) => {
                buf.extend_from_slice(s.as_bytes());
                s.len() as u64
            }
            GgufValue::Array(items) => {
                for it in items {
                    match it {
                        GgufValue::String(s) => {
                            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                            buf.extend_from_slice(s.as_bytes());
                        }
                        other => enc_scalar(other, &mut buf),
                    }
                }
                items.len() as u64
            }
            other => {
                enc_scalar(other, &mut buf);
                1
            }
        };
        (len, fnv1a64(&buf))
    }

    /// 类型名(与 gguf-py `GGUFValueType` 拼法一致)
    fn py_tyname(v: &GgufValue) -> &'static str {
        v.type_name()
    }

    /// **对拍门**:与 `llama.cpp/gguf-py` 逐字段比(独立第三源)。
    ///
    /// 协议:见 `src/formats/gguf_dump.py` 头注(TSV 三行式 + canonical 编码)。
    /// 比什么:容器版本 / 每键 KV(类型/长度/内容指纹)/ 每张量
    ///(名/ggml 型/`ne` 序形状/相对偏移/字节数)。
    /// 为什么值得:gguf-py 是 GGUF 生态的**参考实现**,与本解析器无血缘 ⇒
    /// 分歧必有一方错(设计 REQ-GGUF-01 的判定依据)。
    #[test]
    fn matches_gguf_py() {
        let Some(py) = owl_shared::env_reader::str("OWL_GGUF_PY")
            .or_else(|| Some("/home/div/Documents/codes/repos/llama.cpp/gguf-py".to_string()))
            .filter(|p| Path::new(p).is_dir())
        else {
            eprintln!("skip: gguf-py 源码树不在(OWL_GGUF_PY 可指路)");
            return;
        };
        let script = crate::testkit::manifest_dir().join("src/formats/gguf_dump.py");
        for key in ["gguf:4b", "gguf:27b"] {
            let Some(path) = res(key) else {
                eprintln!("skip: 资源清单缺 \"{key}\"");
                continue;
            };
            let g = Gguf::open(&path).unwrap_or_else(|e| panic!("[{key}] open: {e}"));
            let out = Command::new("uv")
                .arg("run")
                .arg("python")
                .arg(&script)
                .arg(&path)
                .env("PYTHONPATH", &py)
                .current_dir(crate::testkit::manifest_dir())
                .output()
                .expect("spawn uv(python 环境未就绪?先在 crates/models 下 `uv sync`)");
            assert!(
                out.status.success(),
                "[{key}] gguf-py dump 失败:\n--- stderr ---\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let text = String::from_utf8_lossy(&out.stdout).to_string();

            // ---- python 侧事实 ----
            let mut py_version = 0u32;
            let mut py_kv: std::collections::HashMap<String, (String, u64, u64)> = Default::default();
            let mut py_tensors: Vec<(String, String, String, u64, u64)> = Vec::new();
            for line in text.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                match f.first().copied() {
                    Some("version") => py_version = f[1].parse().expect("python version 数字"),
                    Some("kv") => {
                        let key = f[1].to_string();
                        let prev = py_kv.insert(
                            key.clone(),
                            (f[2].to_string(), f[3].parse().expect("len"), u64::from_str_radix(f[4], 16).expect("fnv")),
                        );
                        assert!(prev.is_none(), "[{key}] python 侧 KV 键重复 `{key}`");
                    }
                    Some("tensor") => py_tensors.push((
                        f[1].to_string(),
                        f[2].to_string(),
                        f[3].to_string(),
                        f[4].parse().expect("offset"),
                        f[5].parse().expect("nbytes"),
                    )),
                    other => panic!("[{key}] 未知 dump 行类型 {other:?}"),
                }
            }

            // ---- 版本 ----
            assert_eq!(g.version, py_version, "[{key}] 容器版本");

            // ---- KV:逐键(名集合相等 + 类型/长度/内容指纹)----
            let mut my_keys: Vec<&str> = g.kv.iter().map(|(k, _)| k).collect();
            let mut py_keys: Vec<&str> = py_kv.keys().map(|s| s.as_str()).collect();
            my_keys.sort();
            py_keys.sort();
            assert_eq!(my_keys, py_keys, "[{key}] KV 键面集合");
            for (k, v) in g.kv.iter() {
                let (pt, plen, pfnv) = &py_kv[k];
                assert_eq!(py_tyname(v), pt, "[{key}] KV `{k}` 类型");
                let (len, fnv) = canonical(v);
                assert_eq!(len, *plen, "[{key}] KV `{k}` 长度(实值 {})", v.render(60));
                assert_eq!(
                    fnv, *pfnv,
                    "[{key}] KV `{k}` 内容指纹(自算 {fnv:016x} vs gguf-py {pfnv:016x};实值 {})",
                    v.render(60)
                );
            }

            // ---- 张量表:逐条(顺序 = 文件序,也一并门住)----
            assert_eq!(g.tensors.len(), py_tensors.len(), "[{key}] 张量条数");
            for (i, (name, tyname, ne_csv, off, nb)) in py_tensors.iter().enumerate() {
                let t = &g.tensors[i];
                assert_eq!(t.name, *name, "[{key}] #{i} 张量名");
                assert_eq!(t.ty.name(), tyname, "[{key}] #{i} `{name}` ggml 型");
                let ne: Vec<String> = t.ne.iter().map(|d| d.to_string()).collect();
                assert_eq!(ne.join(","), *ne_csv, "[{key}] #{i} `{name}` ne 序形状");
                assert_eq!(t.offset, *off, "[{key}] #{i} `{name}` 相对偏移");
                assert_eq!(t.nbytes, *nb, "[{key}] #{i} `{name}` 字节数");
                // 内部自洽:行跨 × 行数 == 字节数;块轴整除已在 open 期门住
                assert_eq!(t.row_bytes() * t.rows(), t.nbytes, "[{key}] #{i} `{name}` 行跨账");
                assert_eq!(t.elems(), t.ne.iter().product::<u64>());
            }
            eprintln!("[{key}] 对拍绿:{} (KV {} 键)", g.summary(), g.kv.len());
        }
    }

    /// 响亮拒:非 GGUF 文件 / 空文件 / 目录 / 截断
    #[test]
    fn rejects_non_gguf() {
        let tmp = std::env::temp_dir().join("owl_gguf_reject_probe");
        std::fs::create_dir_all(&tmp).expect("mkdir");
        // 目录
        assert!(Gguf::open(&tmp).is_err(), "目录必须拒");
        // 空文件
        let empty = tmp.join("empty.gguf");
        std::fs::write(&empty, b"").expect("write");
        assert!(Gguf::open(&empty).is_err(), "空文件必须拒");
        // magic 不对
        let bad = tmp.join("bad.gguf");
        std::fs::write(&bad, b"NOPE\x02\x00\x00\x00").expect("write");
        assert!(Gguf::open(&bad).is_err(), "magic 不符必须拒");
        // 头对但计数撒谎(截断)
        let trunc = tmp.join("trunc.gguf");
        let mut b = MAGIC.to_le_bytes().to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes()); // 声称 1 张
        b.extend(0u64.to_le_bytes()); // 0 KV
        assert!(Gguf::open(&{
            std::fs::write(&trunc, &b).expect("write");
            trunc.clone()
        })
        .is_err(), "截断必须拒");
    }
}
