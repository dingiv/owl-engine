//! 客户端平面 —— 类型化调用框架(models 视角;两平面之一)。
//!
//! 每家族一文件:face = 名字常量/槽序单源/builder 校验/线格式。
//! models 只 `use` 本平面,不拼名字字符串、不摆参数槽。
//! 对偶面 = [`crate::server`](家族 runtime;DeviceRes 由后端实现)。

use crate::contract::{Arg, Bytes, LaunchMsg, OpError};
use crate::spec::{KernelMeta, ParamType};

/// 块引用(id + 块内字节偏移;`BlockSlice` 兼容)。
///
/// F3(2026-10-15)起 = **唯一住址**(原 cublas/flashinfer/marlin/xqa 各一份
/// 同构副本;各族 `pub use super::BlockRef` 保持旧路径可用)。
#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    pub id: u64,
    pub byte_offset: u64,
}

/// **按名取址的线格式视图**(F3:parse 与 builder **成对**按名取址)。
///
/// 病根:parse 此前按位置切片(`blocks[0]`/`scalars[0]`)+ 个数断言 ⇒
/// 「槽序」知识在**声明表**(`PARAMS`)与 parse 各存一份,同型槽对调时
/// L3 放行、解析器**静默错值**(gdn 两族、marlin、FI、XQA、cublas 全谱)。
/// 现在 parse 的每一次取材都先经 [`KernelMeta::slot_index`] 回到声明表 ——
/// **声明序 = 线序 = 唯一真源**;同型槽对调只能改声明,而声明序另有
/// `ops::tests::slot_order_lock_family_ops` 双锁(名字序列 + 类型签名)。
///
/// 边界:要求算子**已建表**(生产集 100% 已建表;未建表 = 结构化拒,
/// 不静默回落位置切片 —— 否则本纪律随时可被新算子架空)。
pub struct NamedSlots<'a> {
    meta: &'static KernelMeta,
    args: &'a [Arg],
}

impl<'a> NamedSlots<'a> {
    pub fn new(msg: &'a LaunchMsg) -> Result<Self, OpError> {
        let meta = crate::ops::meta(&msg.kernel.name).ok_or_else(|| OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "kernel.name",
            expect: "已建表算子(按名取址要读声明表 PARAMS)".into(),
            got: format!("未建表:{} 个槽", msg.args.len()),
        })?;
        if msg.args.len() != meta.params.len() {
            return Err(OpError::Contract {
                op: meta.entry.into(),
                field: "argc",
                expect: format!("{} 槽(声明表)", meta.params.len()),
                got: format!("{} 槽(线格式)", msg.args.len()),
            });
        }
        Ok(Self { meta, args: &msg.args })
    }

    /// 声明表(调用方要读声明侧属性时用)
    pub fn meta(&self) -> &'static KernelMeta {
        self.meta
    }

    /// 槽名 → 声明序下标(线序)
    fn at(&self, name: &str) -> Result<usize, OpError> {
        self.meta.slot_index(name).ok_or_else(|| OpError::Contract {
            op: self.meta.entry.into(),
            field: "slot",
            expect: "槽名在声明表 PARAMS".into(),
            got: name.to_string(),
        })
    }

    fn mismatch(&self, name: &str, expect: &str, got: &Arg) -> OpError {
        // 字段类型是 `&'static str` ⇒ 槽名不进 field(否则只能 Box::leak),
        // 进 expect 前缀(台面上仍旧指名道姓)
        OpError::Contract {
            op: self.meta.entry.into(),
            field: "slot",
            expect: format!("{name}: {expect}"),
            got: format!("{got:?}"),
        }
    }

    /// 块槽(Block 或任意偏移 BlockSlice;偏移语义归调用方)
    pub fn block(&self, name: &str) -> Result<BlockRef, OpError> {
        let a = &self.args[self.at(name)?];
        match a {
            Arg::Block { id, .. } => Ok(BlockRef { id: *id, byte_offset: 0 }),
            Arg::BlockSlice { id, byte_offset, .. } => {
                Ok(BlockRef { id: *id, byte_offset: *byte_offset })
            }
            other => Err(self.mismatch(name, "Block/BlockSlice", other)),
        }
    }

    /// 零偏移块槽(块句柄 + 线事实;`BlockSlice` 只认 `byte_offset == 0` ——
    /// eval `slice_view(0,..)` 的合法产物;非零偏移本面不适用)
    pub fn flat_block(&self, name: &str) -> Result<Bytes, OpError> {
        let a = &self.args[self.at(name)?];
        match a {
            Arg::Block { id, dtype, rank } => Ok(Bytes::new(*id, 0, *dtype, *rank)),
            Arg::BlockSlice { id, byte_offset, dtype, rank, .. } => {
                if *byte_offset != 0 {
                    return Err(OpError::Contract {
                        op: self.meta.entry.into(),
                        field: "args",
                        expect: "Block(零偏移)".into(),
                        got: format!("{name}: 非零偏移 BlockSlice({byte_offset})"),
                    });
                }
                Ok(Bytes::new(*id, 0, *dtype, *rank))
            }
            other => Err(self.mismatch(name, "Block(零偏移)", other)),
        }
    }

    /// 输出槽(回执句柄):必须是 Block(Out 槽禁 BlockSlice —— L3 同律),
    /// dtype/rank 取**声明**侧(单一真源;线侧事实由 L3 对齐)。
    pub fn out_block(&self, name: &str, elems: usize) -> Result<Bytes, OpError> {
        let i = self.at(name)?;
        let (dt, rk) = match self.meta.params[i].ty {
            ParamType::Tensor { dtype, rank } => (dtype, rank),
            ParamType::Scalar(s) => {
                return Err(OpError::Contract {
                    op: self.meta.entry.into(),
                    field: "slot",
                    expect: "张量槽".into(),
                    got: format!("{name} 标量槽({s:?})"),
                })
            }
        };
        match &self.args[i] {
            Arg::Block { id, .. } => Ok(Bytes::new(*id, elems, dt, rk)),
            other => Err(self.mismatch(name, "Block(Out 槽禁 BlockSlice)", other)),
        }
    }

    /// 标量槽(sz / F32Bits 位型走私 —— 线上一律 U64)
    pub fn u64(&self, name: &str) -> Result<u64, OpError> {
        let a = &self.args[self.at(name)?];
        match a {
            Arg::U64(v) => Ok(*v),
            other => Err(self.mismatch(name, "U64(sz 槽)", other)),
        }
    }
}

/// GDN chunked(face;零门)
pub mod gdn_chunked;
#[cfg(feature = "xqa")]
pub mod xqa;
/// GDN scalar(face;零门)
pub mod gdn_scalar;
/// cuBLAS GEMM face(feature=cublas,随底座)
#[cfg(feature = "cublas")]
pub mod cublas;
/// Marlin face(feature=marlin,随底座)
#[cfg(feature = "marlin")]
pub mod marlin;
/// FlashInfer 槽序签名 + scale 编码(ungated:models 零 feature 消费方
/// 也引用;解析面仍在 feature = "flashinfer" 的 flashinfer.rs)
pub mod fi_sig {
    /// 槽序签名(9 Block + 8 sz,O = 输出槽;槽序单源,models 禁止手撸 ——
    /// 2026-10-12 review H 案)
    pub const SIG: &str = "T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz";

    /// attention scale 位型编码(f32 bits 过 U64 槽,server 面 from_bits 还原;
    /// 编码单源 —— models 调用点禁止手撸 to_bits)
    pub fn scale_bits(hd: usize) -> u64 {
        (1.0f32 / (hd as f32).sqrt()).to_bits() as u64
    }
}

/// FlashInfer prefill 解析面(feature=flashinfer,随底座)
#[cfg(feature = "flashinfer")]
pub mod flashinfer;

#[cfg(test)]
mod tests {
    //! F3 双锁之一:**parse 取用的槽名集 == 声明表槽名集**(逐字)。
    //!
    //! 另一半锁在 `ops::tests::slot_order_lock_family_ops`(名字**序列** +
    //! 类型签名)—— 集合相等拦得住"改名/插槽/漏跟",顺序锁拦得住"同型槽对调"。
    //! 两把锁缺一:F3 之前"parse 按位置切片"正是两头都拦不住的缺口。
    use super::*;
    use crate::contract::{Arg, LaunchMsg};

    fn launch(name: &str, args: Vec<Arg>) -> LaunchMsg {
        LaunchMsg {
            kernel: crate::contract::KernelSource { name: name.into(), source: String::new() },
            args,
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: 0,
        }
    }

    /// 每个 parse 的名字表与声明表**集合逐字相等**(顺序另锁)
    #[test]
    fn parse_slots_are_exactly_declared() {
        let mut cases: Vec<(&str, &[&str])> = vec![
            (
                crate::family::gdn_chunked::GDN_CHUNKED_FWD,
                crate::client::gdn_chunked::PARSE_SLOTS,
            ),
            (
                crate::family::gdn_scalar::GDN_SCALAR_FWD,
                crate::client::gdn_scalar::PARSE_SLOTS,
            ),
        ];
        #[cfg(feature = "cublas")]
        cases.extend([
            (crate::family::cublas::GEMM_F16, crate::client::cublas::PARSE_SLOTS),
            (crate::family::cublas::GEMM_BF16, crate::client::cublas::PARSE_SLOTS),
        ]);
        #[cfg(feature = "marlin")]
        cases.extend([
            (crate::family::marlin::GEMM_W4A16, crate::client::marlin::PARSE_SLOTS),
            (crate::family::marlin::GEMM_W4A16_BF16, crate::client::marlin::PARSE_SLOTS),
            (
                crate::family::marlin::GEMM_W4A16_AWQ,
                crate::client::marlin::PARSE_SLOTS_AWQ,
            ),
        ]);
        #[cfg(feature = "flashinfer")]
        cases.extend([
            (crate::family::flashinfer::PREFILL_FI, crate::client::flashinfer::PARSE_SLOTS),
            (
                crate::family::flashinfer::PREFILL_FI_FP8KV,
                crate::client::flashinfer::PARSE_SLOTS,
            ),
        ]);
        #[cfg(feature = "xqa")]
        cases.extend([
            (crate::family::xqa::VERIFY_XQA, crate::client::xqa::PARSE_SLOTS),
            (crate::family::xqa::DECODE_XQA, crate::client::xqa::PARSE_SLOTS),
        ]);

        for (entry, want) in cases {
            let m = crate::ops::meta(entry).unwrap_or_else(|| panic!("{entry} 未建表"));
            assert_eq!(
                m.param_names(),
                *want,
                "{entry}: parse 取用槽名与声明表不一致 —— 改名/插槽必须**成对改**\
                 (只改一边 = 运行时才炸;只改声明 = 同型槽静默错位)"
            );
        }
    }

    /// 按名取址的错误面:未建表 / 名字打错 / 槽类型不符 / 槽数不符 —— 全部结构化拒
    #[test]
    fn named_slots_fail_loud() {
        let name = crate::family::gdn_chunked::GDN_CHUNKED_FWD;
        let blk = |id: u64| Arg::block(id, crate::contract::Dtype::F16, 3);
        let good: Vec<Arg> = vec![
            blk(1), blk(2), blk(3), blk(4), blk(5), blk(6), blk(7),
            Arg::U64(1), Arg::U64(0), Arg::U64(1), Arg::U64(1), Arg::U64(128), Arg::U64(0),
        ];

        // ① 未建表算子 ⇒ 拒(不回落位置切片)
        let e = NamedSlots::new(&launch("不存在的核", good.clone())).err().expect("未建表必拒");
        assert!(matches!(&e, OpError::Contract { field: "kernel.name", .. }), "{e}");

        // ② 槽数不符 ⇒ 拒(声明 13 槽)
        let e = NamedSlots::new(&launch(name, good[..12].to_vec())).err().expect("槽数不符必拒");
        assert!(matches!(&e, OpError::Contract { field: "argc", .. }), "{e}");

        // ③ 名字不在声明表 ⇒ 拒(台面上指名道姓:槽名在 expect)
        let m = launch(name, good.clone());
        let s = NamedSlots::new(&m).expect("建表");
        let e = s.u64("nv2").unwrap_err();
        match &e {
            OpError::Contract { field: "slot", expect, got, .. } => {
                assert!(got.contains("nv2"), "错语须指名槽名: {e}");
                assert!(expect.contains("PARAMS"), "错语须指向声明表: {e}");
            }
            other => panic!("期望 slot 拒,得 {other:?}"),
        }

        // ④ 槽类型不符 ⇒ 拒(对块槽按标量读 / 对标量槽按块读,两侧都指名)
        let e = s.u64("q").unwrap_err();
        assert!(matches!(&e, OpError::Contract { field: "slot", .. }), "{e}");
        let e = s.block("t").unwrap_err();
        assert!(matches!(&e, OpError::Contract { field: "slot", .. }), "{e}");

        // ⑤ Out 槽禁 BlockSlice(L3 同律;回执句柄必须整块)
        let mut sliced = good.clone();
        sliced[6] = Arg::BlockSlice {
            id: 7,
            byte_offset: 0,
            elems: 8,
            dtype: crate::contract::Dtype::F16,
            rank: 3,
        };
        let ms = launch(name, sliced);
        let e = NamedSlots::new(&ms)
            .expect("建表")
            .out_block("out", 8)
            .err()
            .expect("Out 槽 BlockSlice 必拒");
        assert!(matches!(&e, OpError::Contract { field: "slot", .. }), "{e}");
    }
}
