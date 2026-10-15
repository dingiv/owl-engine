//! GDN scalar 客户端面 —— 单核臂(lmdeploy pre_sm90 port)类型化调用。
//!
//! 线格式:7 Block(q/k/v/g/beta **f32 直入** —— 层侧 CAST_F16_F32 SSA
//! 预铸,handler 零 cast-in)+ state(f32 槽块)+ out(f16)+ 7 sz
//! (T/slot/ns/hv/nk/kd/scale_bits)。ns=1 单序列(M2 并发批留待)。

use crate::contract::{Arg, Bytes, LaunchMsg, OpError};
use crate::family::gdn_scalar::GDN_SCALAR_FWD;

pub const GDN_SCALAR: &str = GDN_SCALAR_FWD;
pub const SIG: &str = "T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz";

/// `parse` 取用的槽名(声明序;⚠️ 线上第 3..6 个标量的真语义是
/// **ns/hv/nk/kd** —— 2026-10-15 F3 修正,旧表曾错标为 nv/nk/kd/vd) ——
/// F3 成对律,`client::tests::parse_slots_are_exactly_declared` 逐字锁死。
pub const PARSE_SLOTS: &[&str] = &[
    "q", "k", "v", "beta", "gate", "state", "out", "t", "slot", "ns", "nv", "nk", "kd",
    "scale_bits",
];

#[derive(Clone, Copy, Debug)]
pub struct GdnScalarShape {
    pub t: u32,
    pub ns: u32,
    pub nv: u32,
    pub nk: u32,
    pub kd: u32,
}

impl GdnScalarShape {
    pub fn scale_bits(self) -> u64 {
        (1.0 / (self.kd as f32).sqrt()).to_bits() as u64
    }
}

#[derive(Clone, Debug)]
pub struct GdnScalarCall {
    q: Bytes,
    k: Bytes,
    v: Bytes,
    g: Bytes,
    beta: Bytes,
    state: Bytes,
    slot: u32,
    shape: GdnScalarShape,
}

impl GdnScalarCall {
    pub fn builder(shape: GdnScalarShape) -> GdnScalarCallBuilder {
        GdnScalarCallBuilder { shape, f: [None, None, None, None, None], state: None, slot: None }
    }

    /// 输入槽(6 Block;q/k/v/g/beta/state;f32 直入)
    pub fn ins_args(&self) -> [Arg; 6] {
        [
            Arg::block(self.q.id, self.q.dtype, self.q.rank),
            Arg::block(self.k.id, self.k.dtype, self.k.rank),
            Arg::block(self.v.id, self.v.dtype, self.v.rank),
            Arg::block(self.g.id, self.g.dtype, self.g.rank),
            Arg::block(self.beta.id, self.beta.dtype, self.beta.rank),
            Arg::block(self.state.id, self.state.dtype, self.state.rank),
        ]
    }

    pub fn scalar_u64s(&self) -> [u64; 7] {
        let s = self.shape;
        [
            s.t as u64,
            self.slot as u64,
            s.ns as u64,
            s.nv as u64,
            s.nk as u64,
            s.kd as u64,
            s.scale_bits(),
        ]
    }

    /// 反序列化唯一点(7 Block + 7 sz;ns=1 单序列门)。**按名取址**(F3):
    /// 槽名/槽序全取自声明表(`ops::gdn_scalar`)—— 顺带定谳一处**声明标错**
    /// (旧表把 ns/hv/nk/kd 写成 nv/nk/kd/vd ⇒ `check` 的 slot 越界校验用的是
    /// 错维:stride 算成了 ns*nk*nk*4;2026-10-15 随 F3 一并修正)。
    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let s = crate::client::NamedSlots::new(msg)?;
        let ns = s.u64("ns")?;
        if ns != 1 {
            return Err(contract("ns", "1(单序列)", &ns.to_string()));
        }
        let out = s.out_block("out", msg.out_elems)?;
        Ok((
            Self {
                q: s.flat_block("q")?,
                k: s.flat_block("k")?,
                v: s.flat_block("v")?,
                g: s.flat_block("gate")?,
                beta: s.flat_block("beta")?,
                state: s.flat_block("state")?,
                slot: s.u64("slot")? as u32,
                shape: GdnScalarShape {
                    t: s.u64("t")? as u32,
                    ns: ns as u32,
                    nv: s.u64("nv")? as u32,
                    nk: s.u64("nk")? as u32,
                    kd: s.u64("kd")? as u32,
                },
            },
            out,
        ))
    }

    pub fn to_launch(&self, out: Bytes) -> LaunchMsg {
        let mut args: Vec<Arg> = Vec::with_capacity(14);
        args.extend(self.ins_args());
        args.push(Arg::block(out.id, out.dtype, out.rank));
        for v in self.scalar_u64s() {
            args.push(Arg::U64(v));
        }
        LaunchMsg {
            kernel: crate::contract::KernelSource { name: GDN_SCALAR.to_string(), source: String::new() },
            args,
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: self.shape.t as usize * self.shape.nv as usize * self.shape.kd as usize,
        }
    }

    pub fn q(&self) -> &Bytes { &self.q }
    pub fn k(&self) -> &Bytes { &self.k }
    pub fn v(&self) -> &Bytes { &self.v }
    pub fn g(&self) -> &Bytes { &self.g }
    pub fn beta(&self) -> &Bytes { &self.beta }
    pub fn state(&self) -> &Bytes { &self.state }
    pub fn slot(&self) -> u32 { self.slot }
    pub fn shape(&self) -> GdnScalarShape { self.shape }
}

fn contract(field: &'static str, expect: &str, got: &str) -> OpError {
    OpError::Contract { op: GDN_SCALAR.to_string(), field, expect: expect.into(), got: got.into() }
}

pub struct GdnScalarCallBuilder {
    shape: GdnScalarShape,
    f: [Option<Bytes>; 5],
    state: Option<Bytes>,
    slot: Option<u32>,
}

impl GdnScalarCallBuilder {
    fn check(&self, b: &Bytes, want: usize, field: &'static str) -> Result<(), OpError> {
        // 全局硬律 numel > 0(零长形状 ⇒ 结构化拒,带槽名 + 块 id)
        crate::contract::require_numel(want, field, GDN_SCALAR, b.id)?;
        if b.len != 0 && b.len != want {
            return Err(contract(field, &format!("{want} elems"), &format!("块 id={} len={}", b.id, b.len)));
        }
        Ok(())
    }
    pub fn q(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nk as usize * self.shape.kd as usize, "q")?;
        self.f[0] = Some(b.clone());
        Ok(self)
    }
    pub fn k(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nk as usize * self.shape.kd as usize, "k")?;
        self.f[1] = Some(b.clone());
        Ok(self)
    }
    pub fn v(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nv as usize * self.shape.kd as usize, "v")?;
        self.f[2] = Some(b.clone());
        Ok(self)
    }
    /// 门控 g(f32 直入;层侧 CAST_F16_F32 预铸)
    pub fn g(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nv as usize, "g")?;
        self.f[3] = Some(b.clone());
        Ok(self)
    }
    pub fn beta(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nv as usize, "beta")?;
        self.f[4] = Some(b.clone());
        Ok(self)
    }
    pub fn state(mut self, b: &Bytes, slot: u32) -> Result<Self, OpError> {
        self.state = Some(b.clone());
        self.slot = Some(slot);
        Ok(self)
    }
    pub fn build(self) -> Result<GdnScalarCall, OpError> {
        let [q, k, v, g, beta] = self.f;
        let (q, k, v, g, beta) = (
            q.ok_or_else(|| contract("builder", "q", "缺"))?,
            k.ok_or_else(|| contract("builder", "k", "缺"))?,
            v.ok_or_else(|| contract("builder", "v", "缺"))?,
            g.ok_or_else(|| contract("builder", "g", "缺"))?,
            beta.ok_or_else(|| contract("builder", "beta", "缺"))?,
        );
        let state = self.state.ok_or_else(|| contract("builder", "state", "缺"))?;
        let slot = self.slot.ok_or_else(|| contract("builder", "slot", "缺"))?;
        Ok(GdnScalarCall { q, k, v, g, beta, state, slot, shape: self.shape })
    }
}
