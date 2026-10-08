//! GDN chunked 客户端面 —— 类型化调用框架(M3 旁路样板,2026-10-12)。
//!
//! 替代 models/layers/gdn.rs 的 25 行手摆 `Vec<Arg>`(契约腐烂案现场):
//! builder 按 shape 校验,槽序/dtype 编码/标量位型由本模块单源;
//! models 调用点只剩「拿什么喂它」的语义决策。
//!
//! **线格式契约**(与现役 foreign 链逐字节同构,旁路互读):
//! ```text
//! args = [Block q, Block k, Block v, Block beta, Block gate, Block state,
//!         Block out,                       ← 解释器追加
//!         U64 t, U64 slot, U64 nv, U64 nk, U64 kd, U64 scale_bits]
//! sig = "T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz"
//! ```
//! dtype 配方(生产):q/k/v/beta = f16,gate = f16(raw,handler 内铸
//! f32/bf16),state = f32 槽块 [slots, HV, KD, VD],out = f16。
//!
//! ⚠️ g 语义律:本面只产 **raw gate**;cumsum 产物是运行时私有态,
//! 客户端不可达(类型面堵死 raw/cumsum 混喂)。

use crate::contract::{Arg, Bytes, LaunchMsg, OpError, KernelSource};

/// 家族名(全库唯一字面量住址 = contract::names;models 经本常量引用)
pub const GDN_CHUNKED: &str = crate::contract::names::GDN_CHUNKED;
/// 槽序签名(lower_kernel 对位;O = 输出槽)
pub const SIG: &str = "T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz";

/// raw 门控(f16;层侧 g 原块)—— 客户端面唯一产出的 g 形态
#[derive(Clone, Debug)]
pub struct GateRaw(pub Bytes);

/// GDN 状态槽寻址(state 池块 + slot 标量;池 [slots, HV, KD, VD] f32)
#[derive(Clone, Debug)]
pub struct StateSlot {
    pub pool: Bytes,
    pub slot: u32,
}

/// 形状 + 标量族(运行时常量;scale = kd^-0.5 由构造器推导,免手撸位型)
#[derive(Clone, Copy, Debug)]
pub struct GdnShape {
    pub t: u32,
    pub nv: u32,
    pub nk: u32,
    pub kd: u32,
}

impl GdnShape {
    /// attention scale(kd^-0.5;位型编码由本模块单源,调用点零手撸)
    pub fn scale(self) -> f32 {
        1.0 / (self.kd as f32).sqrt()
    }
    pub fn scale_bits(self) -> u64 {
        self.scale().to_bits() as u64
    }
    pub fn nt(self) -> u32 {
        self.t.div_ceil(64)
    }
}

/// 类型化调用(字段私有;builder 装配,parse 重建)
#[derive(Clone, Debug)]
pub struct GdnChunkedCall {
    q: Bytes,
    k: Bytes,
    v: Bytes,
    beta: Bytes,
    gate: GateRaw,
    state: StateSlot,
    shape: GdnShape,
}

impl GdnChunkedCall {
    pub fn builder(shape: GdnShape) -> GdnChunkedCallBuilder {
        GdnChunkedCallBuilder {
            shape,
            q: None,
            k: None,
            v: None,
            beta: None,
            gate: None,
            state: None,
        }
    }

    /// 输入槽(6 Block;顺序即契约)
    pub fn ins_args(&self) -> [Arg; 6] {
        [
            Arg::Block { id: self.q.id },
            Arg::Block { id: self.k.id },
            Arg::Block { id: self.v.id },
            Arg::Block { id: self.beta.id },
            Arg::Block { id: self.gate.0.id },
            Arg::Block { id: self.state.pool.id },
        ]
    }

    /// 标量值(models 侧 TensorOps::arg_usize 胶水用;序/编码单源)
    pub fn scalar_u64s(&self) -> [u64; 6] {
        let s = self.shape;
        [
            s.t as u64,
            self.state.slot as u64,
            s.nv as u64,
            s.nk as u64,
            s.kd as u64,
            s.scale_bits(),
        ]
    }

    /// 标量槽(6 sz;编码单源:scale = f32 位型入 U64)
    pub fn scalar_args(&self) -> [Arg; 6] {
        let s = self.shape;
        [
            Arg::U64(s.t as u64),
            Arg::U64(self.state.slot as u64),
            Arg::U64(s.nv as u64),
            Arg::U64(s.nk as u64),
            Arg::U64(s.kd as u64),
            Arg::U64(s.scale_bits()),
        ]
    }

    /// 反序列化唯一点:server 面从此读(错位/错数 = 结构化 Err,
    /// 不再按数量盲切)。out = 解释器追加的输出块(O 槽,第 7 块)。
    /// BlockSlice 仅认零偏移(eval slice_view(0,..) 合法产物;非零偏移
    /// 本面不适用 —— 与 gdn_scalar 面同一口径,2026-10-12 review I-6 统一)。
    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let mut blocks: Vec<Bytes> = Vec::new();
        let mut scalars: Vec<u64> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { id } => blocks.push(Bytes { id: *id, len: 0 }),
                Arg::BlockSlice { id, byte_offset, .. } => {
                    if *byte_offset != 0 {
                        return Err(contract_err("args", "Block(零偏移)", "非零偏移 BlockSlice"));
                    }
                    blocks.push(Bytes { id: *id, len: 0 });
                }
                Arg::U64(v) => scalars.push(*v),
                Arg::I32(v) => scalars.push(*v as u64),
                Arg::F32(_) => {
                    return Err(contract_err("args", "sz", "F32 标量(本面标量全 sz/U64)"));
                }
            }
        }
        if blocks.len() != 7 {
            return Err(contract_err("blocks", "7", &blocks.len().to_string()));
        }
        if scalars.len() != 6 {
            return Err(contract_err("scalars", "6", &scalars.len().to_string()));
        }
        let shape = GdnShape {
            t: scalars[0] as u32,
            nv: scalars[2] as u32,
            nk: scalars[3] as u32,
            kd: scalars[4] as u32,
        };
        let out = blocks[6].clone();
        let call = Self {
            q: blocks[0].clone(),
            k: blocks[1].clone(),
            v: blocks[2].clone(),
            beta: blocks[3].clone(),
            gate: GateRaw(blocks[4].clone()),
            state: StateSlot { pool: blocks[5].clone(), slot: scalars[1] as u32 },
            shape,
        };
        Ok((call, out))
    }

    /// 完整发射消息(旁路直发形态;解释器路径用 ins_args/scalar_args)。
    /// out 由调用方传入(直发面无解释器代分配)。
    pub fn to_launch(&self, out: Bytes) -> LaunchMsg {
        let mut args: Vec<Arg> = Vec::with_capacity(13);
        args.extend(self.ins_args());
        args.push(Arg::Block { id: out.id });
        args.extend(self.scalar_args());
        LaunchMsg {
            kernel: KernelSource { name: GDN_CHUNKED.to_string(), source: String::new() },
            args,
            grid: (0, 0, 0), // foreign 面:grid/block 由 runtime 单源(核内常量)
            block: (0, 0, 0),
            shared_mem: 0,
            // vd = kd(GDN 家族律,server 同式推导;零字面量 —— 2026-10-12 review G 案)
            out_elems: self.shape.t as usize * self.shape.nv as usize * self.shape.kd as usize,
        }
    }

    // ── 访问器(runtime 只读消费;字段私有防漂移)──
    pub fn q(&self) -> &Bytes { &self.q }
    pub fn k(&self) -> &Bytes { &self.k }
    pub fn v(&self) -> &Bytes { &self.v }
    pub fn beta(&self) -> &Bytes { &self.beta }
    pub fn gate(&self) -> &GateRaw { &self.gate }
    pub fn state(&self) -> &StateSlot { &self.state }
    pub fn shape(&self) -> GdnShape { self.shape }
}

fn contract_err(field: &'static str, expect: &str, got: &str) -> OpError {
    OpError::Contract {
        op: GDN_CHUNKED.to_string(),
        field,
        expect: expect.to_string(),
        got: got.to_string(),
    }
}

/// builder(字段名即契约;类型不对编译不过,长度不对构造期结构化报错)
pub struct GdnChunkedCallBuilder {
    shape: GdnShape,
    q: Option<Bytes>,
    k: Option<Bytes>,
    v: Option<Bytes>,
    beta: Option<Bytes>,
    gate: Option<GateRaw>,
    state: Option<StateSlot>,
}

impl GdnChunkedCallBuilder {
    fn check(&self, b: &Bytes, want_elems: usize, field: &'static str) -> Result<(), OpError> {
        // Bytes.len = f32 元素口径账长;f16 块账长 = 元素数(C1 同律)。
        // len=0 = 叶子占位(裸 id),放行(测试域/特殊流)。
        if b.len != 0 && b.len != want_elems {
            return Err(OpError::Contract {
                op: GDN_CHUNKED.to_string(),
                field,
                expect: format!("{want_elems} elems (t·dim)"),
                got: b.len.to_string(),
            });
        }
        Ok(())
    }

    pub fn q(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nk as usize * self.shape.kd as usize, "q")?;
        self.q = Some(b.clone());
        Ok(self)
    }

    pub fn k(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nk as usize * self.shape.kd as usize, "k")?;
        self.k = Some(b.clone());
        Ok(self)
    }

    pub fn v(mut self, b: &Bytes) -> Result<Self, OpError> {
        // vd = kd(GDN 家族律;零字面量,变架构随 GdnShape 单源)
        self.check(b, self.shape.t as usize * self.shape.nv as usize * self.shape.kd as usize, "v")?;
        self.v = Some(b.clone());
        Ok(self)
    }

    pub fn beta(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nv as usize, "beta")?;
        self.beta = Some(b.clone());
        Ok(self)
    }

    /// raw 门控(f16;[T, NV])
    pub fn gate_raw(mut self, b: &Bytes) -> Result<Self, OpError> {
        self.check(b, self.shape.t as usize * self.shape.nv as usize, "gate")?;
        self.gate = Some(GateRaw(b.clone()));
        Ok(self)
    }

    pub fn state(mut self, pool: &Bytes, slot: u32) -> Result<Self, OpError> {
        // 池账长 = slots·HV·KD·VD(f32 口径;VD = kd 家族律)。slots 不入
        // 契约,以整除性 + slot 越界两道校验替代(2026-10-12 review G 案:
        // 原空体 if 为死校验,静默放行已删除)。
        let slot_stride = self.shape.nv as usize * self.shape.kd as usize * self.shape.kd as usize;
        if pool.len != 0 {
            if pool.len % slot_stride != 0 {
                return Err(OpError::Contract {
                    op: GDN_CHUNKED.to_string(),
                    field: "state",
                    expect: format!("池长为 slot_stride({slot_stride}) 的整倍数"),
                    got: pool.len.to_string(),
                });
            }
            if slot as usize >= pool.len / slot_stride {
                return Err(OpError::Contract {
                    op: GDN_CHUNKED.to_string(),
                    field: "slot",
                    expect: format!("< {}(池长/stride)", pool.len / slot_stride),
                    got: slot.to_string(),
                });
            }
        }
        self.state = Some(StateSlot { pool: pool.clone(), slot });
        Ok(self)
    }

    pub fn build(self) -> Result<GdnChunkedCall, OpError> {
        let missing = [
            ("q", self.q.is_none()),
            ("k", self.k.is_none()),
            ("v", self.v.is_none()),
            ("beta", self.beta.is_none()),
            ("gate", self.gate.is_none()),
            ("state", self.state.is_none()),
        ]
        .into_iter()
        .filter(|(_, miss)| *miss)
        .map(|(n, _)| n)
        .collect::<Vec<_>>()
        .join(",");
        if !missing.is_empty() {
            return Err(OpError::Contract {
                op: GDN_CHUNKED.to_string(),
                field: "builder",
                expect: "全部字段就位".into(),
                got: format!("缺 {missing}"),
            });
        }
        Ok(GdnChunkedCall {
            q: self.q.unwrap(),
            k: self.k.unwrap(),
            v: self.v.unwrap(),
            beta: self.beta.unwrap(),
            gate: self.gate.unwrap(),
            state: self.state.unwrap(),
            shape: self.shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(t: u32) -> GdnShape {
        GdnShape { t, nv: 48, nk: 16, kd: 128 }
    }

    fn dummy(id: u64, len: usize) -> Bytes {
        Bytes { id, len }
    }

    fn sample(t: u32) -> GdnChunkedCall {
        let s = shape(t);
        GdnChunkedCall::builder(s)
            .q(&dummy(1, (s.t as usize) * 16 * 128)).unwrap()
            .k(&dummy(2, (s.t as usize) * 16 * 128)).unwrap()
            .v(&dummy(3, (s.t as usize) * 48 * 128)).unwrap()
            .beta(&dummy(4, (s.t as usize) * 48)).unwrap()
            .gate_raw(&dummy(5, (s.t as usize) * 48)).unwrap()
            .state(&dummy(6, 48 * 128 * 128), 0).unwrap()
            .build().unwrap()
    }

    /// roundtrip property:parse(to_launch(x)) == x(契约单源的结构锁)
    #[test]
    fn roundtrip_preserves_call() {
        let call = sample(128);
        let out = dummy(99, 128 * 48 * 128);
        let msg = call.to_launch(out.clone());
        let (back, out2) = GdnChunkedCall::parse(&msg).unwrap();
        assert_eq!(back.q().id, call.q().id);
        assert_eq!(back.k().id, call.k().id);
        assert_eq!(back.v().id, call.v().id);
        assert_eq!(back.beta().id, call.beta().id);
        assert_eq!(back.gate().0.id, call.gate().0.id);
        assert_eq!(back.state().pool.id, call.state().pool.id);
        assert_eq!(back.state().slot, call.state().slot);
        assert_eq!(back.shape().t, call.shape().t);
        assert_eq!(back.shape().scale_bits(), call.shape().scale_bits());
        assert_eq!(out2.id, out.id);
    }

    /// 线格式兼容:与现役 lower_kernel 产物逐字节对位
    /// (旁路战略:新旧链互读,M4 一键切换的前提)
    #[test]
    fn wire_compatible_with_legacy_lower() {
        let call = sample(64);
        let msg = call.to_launch(dummy(99, 64 * 48 * 128));
        assert_eq!(msg.kernel.name, GDN_CHUNKED);
        // 7 块(q/k/v/beta/gate/state/out)+ 6 标量
        let mut blocks = 0;
        let mut scalars: Vec<u64> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { .. } => blocks += 1,
                Arg::U64(v) => scalars.push(*v),
                other => panic!("线格式外元素: {other:?}"),
            }
        }
        assert_eq!(blocks, 7);
        assert_eq!(
            scalars,
            vec![64, 0, 48, 16, 128, shape(64).scale_bits()],
            "标量序/位型漂移 = 旧链不识别"
        );
    }

    /// builder 长度校验:错块在构造期结构化报错(不进 GPU)
    #[test]
    fn builder_rejects_wrong_len() {
        let s = shape(64);
        let r = GdnChunkedCall::builder(s)
            .q(&dummy(1, 999)); // 长度错:构造期即拒,链条在此断
        match r {
            Err(OpError::Contract { field: "q", .. }) => {}
            Err(e) => panic!("期望 q 长度 Contract 错,实得 {e}"),
            Ok(_) => panic!("期望 q 长度 Contract 错,实得 Ok"),
        }
    }
}
