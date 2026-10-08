//! 语义算子胖算子样例(2026-10-12;用户设计:struct 即算子,值域校验
//! 内嵌,interpreter 强制)。
//!
//! **模式**(后续每个在用算子照此声明):
//! 1. struct 字段 = 全部标量/形状参数(张量槽经 `wire(&ins, out)` 注入,
//!    序 = 签名序);
//! 2. `validate()` = 值域/正交位检查(hd 档位、页宽、正性……);
//! 3. `KernelSpec` trait 实现(name/validate/out/wire);
//! 4. interpreter(Op::Spec 臂)发射前**强制** validate,不通过 =
//!    interpreter 层结构化报错。
//!
//! 值域依据:registry 注记(hd ∈ {128,256} = qwen3.5 两档;x ∈ {16,32}
//! = bs 契约,BLOCK∈{32,64} 的半段)。

use crate::contract::{Arg, Bytes, Dtype, KernelSpec, KernelSource, LaunchMsg, OpError, Shape};
use crate::native;

/// owl_narrow_strided_f16(narrow+stride 抽取;attention K 抽取用)
#[derive(Debug, Clone)]
pub struct NarrowStrided {
    pub t: usize,
    pub hd: usize,
    pub page: usize,
    /// 抽取半段宽(bs 契约;x = hd/x)
    pub x: usize,
    pub q: Bytes,
    pub kc: Bytes,
}

impl NarrowStrided {
    pub const NAME: &'static str = "owl_narrow_strided_f16";
}

impl KernelSpec for NarrowStrided {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn validate(&self) -> Result<(), OpError> {
        fn bad(field: &'static str, expect: &'static str, got: String) -> OpError {
            OpError::Contract { op: NarrowStrided::NAME.into(), field, expect: expect.into(), got }
        }
        if self.t == 0 {
            return Err(bad("t", "> 0", self.t.to_string()));
        }
        if self.hd != 128 && self.hd != 256 {
            return Err(bad("hd", "128 | 256(档位契约)", self.hd.to_string()));
        }
        if self.x != 16 && self.x != 32 {
            return Err(bad("x", "16 | 32(bs 半段契约)", self.x.to_string()));
        }
        if self.page == 0 || self.page > 4096 {
            return Err(bad("page", "1..=4096", self.page.to_string()));
        }
        Ok(())
    }

    fn out(&self) -> (Dtype, Shape) {
        (Dtype::F16, vec![self.t])
    }

    fn wire(&self, ins: &[Arg], out: &Bytes) -> LaunchMsg {
        LaunchMsg {
            kernel: KernelSource {
                name: Self::NAME.to_string(),
                // 登记表 = 源码唯一权威(名字拼错 = panic,构造期拦截)
                source: native::source(Self::NAME).to_string(),
            },
            args: vec![
                ins[0].clone(),
                Arg::U64(self.t as u64),
                Arg::U64(self.hd as u64),
                Arg::U64(self.page as u64),
                Arg::U64(self.x as u64),
                ins[1].clone(),
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: self.t,
        }
    }
}

#[cfg(test)]
mod locks {
    use super::*;

    fn base() -> NarrowStrided {
        NarrowStrided { t: 64, hd: 128, page: 32, x: 32, q: Bytes { id: 1, len: 0 }, kc: Bytes { id: 2, len: 0 } }
    }

    #[test]
    fn valid_passes() {
        base().validate().expect("合法参数应过");
    }

    #[test]
    fn hd_off_contract_rejected() {
        let mut s = base();
        s.hd = 96; // 非 {128,256}
        match s.validate() {
            Err(OpError::Contract { field: "hd", .. }) => {}
            other => panic!("期望 hd Contract,实得 {other:?}"),
        }
    }

    #[test]
    fn negative_guard() {
        // 负数(usize 下以 wrap 记)在值域门外:page=0 已拦;hd 档外拦
        let mut s = base();
        s.x = 8; // 非 {16,32}
        assert!(matches!(s.validate(), Err(OpError::Contract { field: "x", .. })));
    }

    #[test]
    fn wire_is_byte_compatible_with_legacy_chain() {
        // 与手摆 lower 序同构:[a, t, hd, page, x, b]
        let s = base();
        let out = Bytes { id: 9, len: 64 };
        let ins = [Arg::Block { id: 1 }, Arg::Block { id: 2 }];
        let msg = owl_kernels_spec_wire(&s, &ins, &out);
        assert_eq!(msg.args.len(), 6);
        assert!(matches!(&msg.args[0], Arg::Block { id: 1 }));
        assert!(matches!(&msg.args[1], Arg::U64(64)));
        assert!(matches!(&msg.args[2], Arg::U64(128)));
        assert!(matches!(&msg.args[3], Arg::U64(32)));
        assert!(matches!(&msg.args[4], Arg::U64(32)));
        assert!(matches!(&msg.args[5], Arg::Block { id: 2 }));
        assert_eq!(msg.kernel.name, NarrowStrided::NAME);
        assert_eq!(msg.kernel.source, crate::native::source(NarrowStrided::NAME));
    }

    fn owl_kernels_spec_wire(spec: &dyn crate::contract::KernelSpec, ins: &[Arg], out: &Bytes) -> LaunchMsg {
        spec.wire(ins, out)
    }
}
