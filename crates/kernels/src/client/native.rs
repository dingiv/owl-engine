//! 语义算子胖算子样例(2026-10-12;用户设计:struct 即算子,值域校验
//! 内嵌,interpreter 强制)。
//!
//! **模式**(后续每个在用算子照此声明):
//! 1. struct 字段 = 全部标量/形状参数(张量槽经 `wire(&ins, out)` 注入,
//!    序 = 签名序);
//! 2. `validate()` = 值域/正交位检查;
//! 3. [`OpSpec`] trait 实现(name/validate/out/wire);
//! 4. interpreter(Op::Spec 臂)发射前**强制** validate,不通过 =
//!    interpreter 层结构化报错(2026-10-12 review A 案已接线)。
//!
//! 值域依据:.cu 实语义(cu/text/attention.cu `owl_narrow_strided_f16`):
//! `dst[r·out_dim + d] = src[r·src_dim + start + d]`,total = outer·out_dim
//! —— 窄切窗口必须落在行内(start + out_dim ≤ src_dim)。

use crate::contract::{Arg, Bytes, Dtype, OpSpec, KernelSource, LaunchMsg, OpError, Shape};
use crate::native;

/// owl_narrow_strided_f16(非连续窄切物化拷贝;gate 切分/conv 三段切分用)
#[derive(Debug, Clone)]
pub struct NarrowStrided {
    pub outer: usize,
    pub src_dim: usize,
    pub start: usize,
    pub out_dim: usize,
    pub src: Bytes,
}

impl NarrowStrided {
    pub const NAME: &'static str = "owl_narrow_strided_f16";
}

impl OpSpec for NarrowStrided {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn validate(&self) -> Result<(), OpError> {
        fn bad(field: &'static str, expect: String, got: String) -> OpError {
            OpError::Contract { op: NarrowStrided::NAME.into(), field, expect, got }
        }
        if self.outer == 0 {
            return Err(bad("outer", "≥ 1".into(), self.outer.to_string()));
        }
        if self.out_dim == 0 {
            return Err(bad("out_dim", "≥ 1".into(), self.out_dim.to_string()));
        }
        if self.start + self.out_dim > self.src_dim {
            return Err(bad(
                "窗口",
                format!("start({}) + out_dim({}) ≤ src_dim", self.start, self.out_dim),
                format!("src_dim = {}", self.src_dim),
            ));
        }
        Ok(())
    }

    fn out(&self) -> (Dtype, Shape) {
        (Dtype::F16, vec![self.outer * self.out_dim])
    }

    fn wire(&self, ins: &[Arg], out: &Bytes) -> LaunchMsg {
        LaunchMsg {
            kernel: KernelSource {
                name: Self::NAME.to_string(),
                // 登记表 = 源码唯一权威(名字拼错 = panic,构造期拦截)
                source: native::source(Self::NAME).to_string(),
            },
            // 槽序 = "T,sz,sz,sz,sz,T"(输出末参,.cu ABI)
            args: vec![
                ins[0].clone(),
                Arg::U64(self.outer as u64),
                Arg::U64(self.src_dim as u64),
                Arg::U64(self.start as u64),
                Arg::U64(self.out_dim as u64),
                Arg::Block { id: out.id },
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: self.outer * self.out_dim,
        }
    }
}

#[cfg(test)]
mod locks {
    use super::*;
        /// 线格式与 legacy lower_kernel 产物逐字节同构(同一声明两条链互证:
    /// 按登记表 sig 手搓 lower 同款装配,与 wire 输出对位)
    #[test]
    fn wire_matches_legacy_lower_kernel() {
        use crate::contract::Arg;
        let s = base();
        let out = Bytes { id: 9, len: 8 * 16 };
        let ins = [Arg::Block { id: 1 }];
        let msg = s.wire(&ins, &out);

        // legacy 形态:按登记表 sig("T,sz,sz,sz,sz,T")走 lower_kernel 同款
        // 槽序对位(T → 父块 / sz → U64 / 末位 T → 输出块)
        let e = crate::native::lookup(NarrowStrided::NAME).expect("登记表条目");
        let toks: Vec<&str> = e.args.split(',').collect();
        assert_eq!(toks, vec!["T", "sz", "sz", "sz", "sz", "T"]);
        let legacy_args = vec![
            ins[0].clone(),
            Arg::U64(s.outer as u64),
            Arg::U64(s.src_dim as u64),
            Arg::U64(s.start as u64),
            Arg::U64(s.out_dim as u64),
            Arg::Block { id: out.id },
        ];
        assert_eq!(msg.args.len(), legacy_args.len());
        for (a, b) in msg.args.iter().zip(&legacy_args) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "槽序/位型漂移(legacy 链不识别)");
        }
        assert_eq!(msg.out_elems, s.outer * s.out_dim);
        // 源 = 登记表单源
        assert_eq!(msg.kernel.source, crate::native::source(NarrowStrided::NAME));
    }

    fn base() -> NarrowStrided {
        NarrowStrided { outer: 8, src_dim: 32, start: 16, out_dim: 16, src: Bytes { id: 1, len: 8 * 32 } }
    }

    #[test]
    fn valid_passes() {
        base().validate().expect("合法参数应过");
    }

    #[test]
    fn window_out_of_row_rejected() {
        let mut s = base();
        s.start = 20; // 20 + 16 > 32:窗口跨行越界
        match s.validate() {
            Err(OpError::Contract { field: "窗口", .. }) => {}
            other => panic!("期望窗口 Contract,实得 {other:?}"),
        }
    }

    #[test]
    fn zero_dims_rejected() {
        let mut s = base();
        s.outer = 0;
        assert!(matches!(s.validate(), Err(OpError::Contract { field: "outer", .. })));
        let mut s = base();
        s.out_dim = 0;
        assert!(matches!(s.validate(), Err(OpError::Contract { field: "out_dim", .. })));
    }
}
