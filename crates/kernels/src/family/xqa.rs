//! XQA(kNHD fp8 KV 的 decode/verify 注意力;foreign-kernel 通道)。
//!
//! 溯源:vendor `crates/kernels/cu/xqa/`(移植自
//! repos/vllm-ampere-optimized/flashampere/xqa/csrc,Ampere 闭包;上游 =
//! FlashInfer csrc/xqa)。**构建期 cubin 资产 + Rust host 发射器**
//! (2026-10-13 C3a:`-DGENERATE_CUBIN` 编两个宏变体 fatbin;原 `.a` 已删),
//! feature `xqa`。
//!
//! **为什么用它**(长 ctx verify 立案 D3):现役 v2 伪序列臂 = 8 条伪序列
//! 各扫一遍全 ctx(元素数 ×8);XQA 是 **decode 形状**(KV-split +
//! warp-specialized + GQA 内共享 K/V 读),`SPEC_DEC` 模式 q=1+K 的 q 维扩展
//! 近乎免费。kernels crate 单算子实测(ctx 8192,q=8):**0.495ms vs v2
//! 2.673ms = 5.40×**,数值 max|XQA−host| 4.5e-4(v2 4.6e-4)同级。
//!
//! **布局契约**(与 owl kNHD 主池逐字对齐;XQA 文档名 NHD):
//! - KV:`[num_pages, tokens_per_page, Hkv, head_dim]` **fp8 e4m3**;
//!   strides 以**元素**过线(host 契约)
//! - q/out:`[batch, 1, q_seq_len, Hq, head_dim]` f16(owl verify 的
//!   `[T, hq*hd]` 是它的 free view)
//! - page_table **int32** `[batch, max_pages]`(owl 现有 bt 是 f32 → 需 i32 孪生)
//! - seq_lens **uint32** `[batch]` = **整请求总 KV 长**(不是逐行前缀长!
//!   块内 q 行的因果**只由 mask 表达**;踩坑留档见 cu/xqa/README.md)
//! - mask **uint32** `[batch, q_seq_len, ceil(q/32)]`:行 i = 块内第 i 个
//!   token,bit k = 允许看块内第 k 个 key,因果 = `bits 0..i`(含自身)
//!   ⇒ `mask[i] = (1<<(i+1))-1`
//!
//! **workspace**:semaphores 与 scratch 由 runtime 持久持有(图内安全:
//! 热路径零分配;参考实现 scratch 给 256MiB)。
/// foreign-kernel 虚拟核名(server 分派臂;**字面量住址 = 本模块**)
pub const VERIFY_XQA: &str = "owl_xqa_verify";
/// 第二份实例化的虚拟核名(裸 decode q=1;见 cu/xqa/README.md 双实例化节)
pub const DECODE_XQA: &str = "owl_xqa_decode";
/// prefill(chunked q=T)虚拟核名 —— **同一份 `M_TILESIZE=32` 实例化**,
/// 差别只在 host 侧几何(本臂强制单块 ⇒ KV-split 关)。
/// 自 `feat/prefill-graph` 合入(2026-10-15);它只作 **FI 缺口交接臂**
/// (`t < fi_min_tokens` 尾块 / `ctx.fi` 缺席),CUPTI 实测主路径换它净亏 4.1×
/// ⇒ `dispatch.xqa_prefill` 缺省 **false**。
pub const PREFILL_XQA: &str = "owl_xqa_prefill";
/// 槽序契约(LaunchMsg.args;6 Block + 10 sz):
/// `[T q, T k_cache, T v_cache, T page_table(i32), T seq_lens(u32),
///   T mask(u32), T semaphores(u32), T scratch, T wr, O out, sz q_seq_len, sz hkv,
///   sz hd, sz page, sz max_seq_len,
///   sz q_scale_bits, sz sm_count, sz stride_page, sz stride_token,
///   sz stride_head]`
/// q_scale 以 `f32::to_bits` 过线(= attn_scale·sqrt(hd);核内再除 sqrt(hd))。
pub const VERIFY_XQA_SLOTS: &str =
    "[T q, T k_cache, T v_cache, T page_table, T seq_lens, T mask, \
     T semaphores, T scratch, T wr, O out, sz q_seq_len, sz hkv, sz hd, sz page, \
     sz max_seq_len, sz q_scale_bits, sz sm_count, sz stride_page, \
     sz stride_token, sz stride_head]";
/// **线槽序权威**(plain type-list;`lower_kernel`/`foreign_sig` 消费,
/// 必须与层侧 `.arg` 链同序:9 入参 T + O 输出 + 10 sz)
pub const VERIFY_XQA_SIG: &str =
    "T,T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz,sz,sz";
/// **decode 臂线槽序**:与 verify **逐字同型**(两份实例化签名一致,q=1 时
/// mask 仍走 SPEC_DEC 掩码路径 —— 只需 bit0=1)。分列常量只为"臂/名成对律"
/// 与将来若真分叉时留单点改口。
pub const DECODE_XQA_SIG: &str =
    "T,T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz,sz,sz";
/// **prefill 臂线槽序**:与 verify/decode **逐字同型**(第三份"臂"共用同一份
/// 实例化与资产 —— 差别只在 host 侧几何(prefill 强制单块)与表件来源)。
pub const PREFILL_XQA_SIG: &str =
    "T,T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz,sz,sz";
/// XQA 工作区(参考实现给 256MiB;持久复用)
pub const XQA_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

/// **scratch 尺寸下限**(F2 自证门;不是精确需求)。
///
/// 精确需求随 geometry(nbSubSeq/nbVChunks)变,且**住内核
/// `MemSegmenter`**;host 侧重推 = 复制内核常量 ⇒ 违本仓 `零常量复制` 律。
/// 故只守量级下限:实测本仓生产形状(batch=1/hkv=4/hd256/ctx≤16k)的
/// KV-split partial ≈0.7MB,生产定档 **32MiB**(`engine/state.rs`),
/// 测试 64MiB —— 下限取 1MiB(≈实测需求 1.4×):抓 `量级错`(如 1 word /
/// 4KiB 块),不抓 1.5× 少分配。
pub const XQA_SCRATCH_FLOOR_BYTES: usize = 1024 * 1024;

/// F2 判决(纯):scratch 尺寸(`have` = 块账长;下限语义见上)。
pub fn scratch_size_check(have: usize) -> Result<(), String> {
    if have < XQA_SCRATCH_FLOOR_BYTES {
        return Err(format!(
            "scratch 块账长 {have} B < 下限 {XQA_SCRATCH_FLOOR_BYTES} B(参考实现 {XQA_SCRATCH_BYTES} B;本仓生产定档 32MiB)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 够长放行 / 短一截拒(负例只差 1 字节 ⇒ 边界语义证明)
    #[test]
    fn scratch_floor_is_a_floor() {
        assert!(scratch_size_check(XQA_SCRATCH_FLOOR_BYTES).is_ok());
        assert!(scratch_size_check(32 << 20).is_ok(), "生产定档 32MiB 应放行");
        let e = scratch_size_check(XQA_SCRATCH_FLOOR_BYTES - 1).unwrap_err();
        assert!(e.contains("scratch 块账长"), "拒语须指名槽位与账长: {e}");
        assert!(scratch_size_check(4).is_err(), "零容量/1 word 块应拒");
    }
}
// ============================================================================
// 资产与发射面(2026-10-13 C3a:`.a` → 构建期 fatbin + **Rust host 发射器**)
// ============================================================================
//
// 形态:`cu/xqa/mha.cu` 用 `-DGENERATE_CUBIN`(上游自带的"只出设备码"模式)
// 编出**两个宏变体** fatbin(`xqa_verify` / `xqa_decode`,见 build.rs);host 发射器
// 从 C 搬到 Rust(本模块 + `server/xqa.rs`)。⇒ 无 `.a`、无 `libstdc++` 链接、
// 无符号改名 hack(两份实例化是**两个模块**,同名不撞)。
//
// **资产自描述**(关键设计):单 TU cubin 会把 `__global__` 之外的符号全内部化
// (实测 `smemSize` 等设备变量落 STB_LOCAL,`cuModuleGetGlobal` 取不到)⇒ 内核
// 侧另发一个 1 线程探针核 `owl_xqa_launch_consts`,把 host 需要的**全部编译期
// 常量**写进缓冲:dynamic smem 字节数 / ctaTile / headGrpSize / cta 形状 /
// rowsPerBlock / allowMultiBlock / tokensPerPage / validElemsPerHead。
// **Rust 侧零常量复制**(复制常量 = "两处对不上"的经典温床)。
/// 变体名(资产表键;= `$OUT_DIR/xqa/xqa_<name>.fatbin`)
pub const ASSET_VERIFY: &str = "verify";
/// 裸 decode 臂变体(`M_TILESIZE=16`;见 cu/xqa/README.md 双实例化节)
pub const ASSET_DECODE: &str = "decode";
/// 内核符号名(两个变体同名 —— 分属两个模块,不冲突)
pub const SYM_KERNEL: &str = "kernel_mha";
/// 资产自描述探针核符号名
pub const SYM_CONSTS: &str = "owl_xqa_launch_consts";
/// 探针核输出槽数(`owl_xqa_launch_consts` 写入的 u32 个数)
pub const CONSTS_SLOTS: usize = 12;
/// 资产自描述常量(由探针核在装配期报回;**唯一真源 = `mha.cu` 的 constexpr**)
#[derive(Debug, Clone, Copy)]
pub struct XqaConsts {
    /// dynamic smem 字节数(`sizeof(SharedMem)`)
    pub smem: u32,
    /// `ctaTile.x`(grid 推导:`nbSubSeqPerSeq` 上限)
    pub cta_tile_x: u32,
    /// `ctaTile.y`
    pub cta_tile_y: u32,
    /// `headGrpSize`(grid.y 推导)
    pub head_grp: u32,
    /// cta 形状(线程):x = warp_size × ctaShapeInWarps.x,y/z = ctaShapeInWarps.y/z
    pub cta_x: u32,
    pub cta_y: u32,
    pub cta_z: u32,
    /// `rowsPerBlock`(= `M_TILESIZE`;grid.y 推导)
    pub rows_per_block: u32,
    /// `allowMultiBlockMode`(0/1)
    pub allow_multi_block: u32,
    /// `tokensPerPage`
    pub tokens_per_page: u32,
    /// `validElemsPerHead`(KV stride 元素序 → head 序换算)
    pub valid_elems_per_head: u32,
    /// **`sizeof(KVCacheList<true>)`**(内核侧真值)⇒ 与 Rust 镜像 size 比对,
    /// 布局漂移**装配期**即拦(否则是静默错值)
    pub cache_list_bytes: u32,
}
impl XqaConsts {
    /// 从探针核回读的 11 个 u32 组装(长度不足 = 资产/代码代际失配 ⇒ Err)
    pub fn from_slots(v: &[u32]) -> Result<Self, crate::contract::OpError> {
        if v.len() < CONSTS_SLOTS {
            return Err(crate::contract::OpError::Asset {
                op: "xqa".into(),
                detail: format!(
                    "资产自描述槽数 {} < {CONSTS_SLOTS}(探针核与 host 期望失配)",
                    v.len()
                ),
            });
        }
        Ok(Self {
            smem: v[0],
            cta_tile_x: v[1],
            cta_tile_y: v[2],
            head_grp: v[3],
            cta_x: v[4],
            cta_y: v[5],
            cta_z: v[6],
            rows_per_block: v[7],
            allow_multi_block: v[8],
            tokens_per_page: v[9],
            valid_elems_per_head: v[10],
            cache_list_bytes: v[11],
        })
        .and_then(|c| {
            // 布局硬门:Rust 镜像 vs 内核形参**逐字节一致**
            let want = std::mem::size_of::<KVCacheList>() as u32;
            if c.cache_list_bytes != want {
                return Err(crate::contract::OpError::Asset {
                    op: "xqa".into(),
                    detail: format!(
                        "KVCacheList 布局失配:内核 {} B vs Rust 镜像 {} B —— 镜像必须逐字节同构",
                        c.cache_list_bytes, want
                    ),
                });
            }
            Ok(c)
        })
    }
    /// grid / block / dynamic smem(`mha.cu:launchMHAFlashInfer` 逐式同构)
    pub fn launch_geometry(
        &self,
        sm_count: u32,
        batch: u32,
        nb_k_heads: u32,
        max_seq_len: u32,
        q_seq_len: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32), u32) {
        let div_up = |a: u32, b: u32| a.div_ceil(b.max(1));
        let nb_sub_seq = if self.allow_multi_block == 0 {
            1
        } else {
            let by_sm = (sm_count / (batch.max(1) * nb_k_heads.max(1))).max(1);
            by_sm.min(div_up(max_seq_len, self.cta_tile_x))
        };
        self.geometry(nb_sub_seq, nb_k_heads, q_seq_len)
    }
    /// **单块几何**(KV-split 关;prefill 臂契约;自 `feat/prefill-graph` 合入)。
    ///
    /// 为什么 prefill 不要 KV-split:prefill 的 `grid.y` =
    /// `nbKHeads · ceil(q·headGrp / rowsPerBlock)` 已经把 SM 喂饱
    /// (t=512 ⇒ 384 CTAs),`grid.x` 再加分割**实测时间一字不变**
    /// (2026-10-14 探针 `XQA_PF_NOSPLIT`:ctx512 0.882 vs 0.886 ms;
    /// ctx16384 21.244 vs 21.564 ms)⇒ 白拿 multi-block 归约的全部代价:
    /// scratch(16k ctx = 129MB)+ semaphores。
    ///
    /// ⚠️ **semaphores 尺寸雷**(同批挖出,见 `cu/xqa/README.md`):
    /// `mha.cu` 按 `semaphores[gridDim.y·idxReq + blockIdx.y]` 索引,
    /// SPEC_DEC 下 prefill 需要 `grid.y = 384` 槽,而引擎现分配 12 槽
    /// ⇒ 开 split 会越界写。本入口把 `grid.x` 钉成 1 ⇒ 该数组**结构上不解引用**。
    pub fn launch_geometry_single_block(
        &self,
        nb_k_heads: u32,
        q_seq_len: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32), u32) {
        self.geometry(1, nb_k_heads, q_seq_len)
    }
    /// 几何公共体(sub_seq 由调用方钉;两条入口只差这一个形参)
    fn geometry(
        &self,
        nb_sub_seq: u32,
        nb_k_heads: u32,
        q_seq_len: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32), u32) {
        let nb_token_blocks = q_seq_len
            .max(1)
            .saturating_mul(self.head_grp.max(1))
            .div_ceil(self.rows_per_block.max(1));
        let block = (self.cta_x, self.cta_y, self.cta_z);
        ((nb_sub_seq, nb_k_heads * nb_token_blocks, 1), block, self.smem)
    }
    /// `maxNbPagesPerSeq = maxSeqLen / tokensPerPage`(exactDiv)
    pub fn max_nb_pages_per_seq(&self, max_seq_len: u32) -> u32 {
        max_seq_len / self.tokens_per_page.max(1)
    }
    /// KV stride:元素序 → head 序(`launchMHAFlashInfer` 同式)
    pub fn stride_in_heads(&self, elems: u64) -> u32 {
        (elems / self.valid_elems_per_head.max(1) as u64) as u32
    }
}
/// **`KVCacheList<true>` 的 Rust 镜像**(by-value 内核参数;`ENABLE_4BIT_KV_CACHE=0`
/// 时的字段集)。布局律见 `device::launch_vals` 头注:5 个 u64 槽与 40 字节
/// 结构体逐字节同布局(自然对齐 + 紧密排布)。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KVCacheList {
    pub k_cache: u64,
    pub v_cache: u64,
    pub page_list: u64,
    pub seq_len_list: u64,
    pub max_nb_pages_per_seq: u32,
    /// 填充到 8 字节对齐(C++ 结构体同款)
    pub _pad: u32,
}
// 布局硬门:改了字段就必须同步改展槽代码(否则编译期红)
const _: () = assert!(std::mem::size_of::<KVCacheList>() == 40);
impl KVCacheList {
    /// 结构体镜像 → 参数槽(`LaunchVal::Blob`;**整块**给驱动,不拆槽 —— 见其注)
    pub fn as_launch_val(&self) -> crate::contract::LaunchVal {
        let p = self as *const Self as *const u8;
        // SAFETY: size_of == 40 已由上面的 const 断言锁死;只读
        crate::contract::LaunchVal::blob(unsafe { std::slice::from_raw_parts(p, std::mem::size_of::<Self>()) })
    }
}
/// 一次发射的全部实参(与 `kernel_mha` 形参逐位对位;`SPEC_DEC=1` 实例化)
#[derive(Debug, Clone, Copy)]
pub struct XqaArgs {
    pub q_seq_len: u32,
    pub nb_k_heads: u32,
    /// `qCuSeqLens`(本实例化恒 nullptr —— varlen 未启用)
    pub q_cu_seq_lens: u64,
    pub q_scale: f32,
    /// `qScalePtr`(恒 nullptr —— 无 per-tensor scale)
    pub q_scale_ptr: u64,
    pub output: u64,
    pub q: u64,
    pub mask: u64,
    /// `attentionSinks`(恒 nullptr)
    pub attention_sinks: u64,
    pub cache: KVCacheList,
    pub batch_size: u32,
    pub kv_scale: f32,
    pub kv_scale_ptr: u64,
    pub stride_page: u32,
    pub stride_token: u32,
    pub stride_head: u32,
    pub semaphores: u64,
    pub scratch: u64,
}
impl XqaArgs {
    /// 值表(与 `kernel_mha` 形参**逐位同序**;`SPEC_DEC=1`/`SLIDING_WINDOW=0`/
    /// `LOW_PREC_OUTPUT=0`/`BEAM_WIDTH=1` 的展开)
    pub fn vals(&self, head_grp: u32) -> Vec<crate::contract::LaunchVal> {
        use crate::contract::LaunchVal as V;
        let mut v = vec![
            V::U32(self.q_seq_len),
            V::U32(self.nb_k_heads),
            V::U32(head_grp),
            V::U64(self.q_cu_seq_lens),
            V::F32(self.q_scale),
            V::U64(self.q_scale_ptr),
            V::Ptr(self.output),
            V::Ptr(self.q),
            V::Ptr(self.mask),
            V::Ptr(self.attention_sinks),
        ];
        v.push(self.cache.as_launch_val());
        v.extend([
            V::U32(self.batch_size),
            V::F32(self.kv_scale),
            V::U64(self.kv_scale_ptr),
            V::U32(self.stride_page),
            V::U32(self.stride_token),
            V::U32(self.stride_head),
            V::Ptr(self.semaphores),
            V::Ptr(self.scratch),
        ]);
        v
    }
}

// ============================================================================
// 测试/诊断面(不经 DeviceRes/Exec 的直发;**生产走 server/xqa.rs**)
// ============================================================================
//
// 探针(单算子对拍)需要"最小可判据"地发射一次;生产面要走 `Exec`(模块缓存 +
// 捕获窗登记)。两条路**共用** [`XqaArgs::vals`] 与 [`crate::device::launch_vals`]
// ⇒ 压槽逐位同构("测试绿 ⇒ 生产绿")。

/// 从构建期资产载模块 + 内核 + 常量(探针面)
#[cfg(feature = "device")]
pub fn load_for_probe(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    variant: &str,
) -> Result<(cudarc::driver::CudaFunction, XqaConsts), crate::contract::OpError> {
    let bytes = crate::family::cubins::xqa_asset(variant).ok_or_else(|| crate::contract::OpError::Asset {
        op: "xqa".into(),
        detail: format!("资产 {variant} 不存在(feature `xqa`?)"),
    })?;
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(bytes.to_vec()))
        .map_err(|e| crate::contract::OpError::Asset { op: "xqa".into(), detail: format!("load {variant}: {e:?}") })?;
    let probe = module
        .load_function(SYM_CONSTS)
        .map_err(|e| crate::contract::OpError::Asset { op: "xqa".into(), detail: format!("{SYM_CONSTS}: {e:?}") })?;
    let stream = ctx.default_stream();
    let d = stream.alloc_zeros::<u32>(CONSTS_SLOTS).map_err(|e| crate::contract::OpError::Asset {
        op: "xqa".into(),
        detail: format!("consts alloc: {e:?}"),
    })?;
    let (p, _g) = cudarc::driver::DevicePtr::device_ptr(&d, &stream);
    let vals = [crate::contract::LaunchVal::Ptr(p)];
    crate::device::launch_vals(&stream, &probe, (1, 1, 1), (32, 1, 1), 0, &vals)
        .map_err(|e| crate::contract::OpError::Asset { op: "xqa".into(), detail: format!("consts probe: {e:?}") })?;
    stream.synchronize().map_err(|e| crate::contract::OpError::Asset { op: "xqa".into(), detail: format!("sync: {e:?}") })?;
    let slots: Vec<u32> = stream.clone_dtoh(&d).map_err(|e| crate::contract::OpError::Asset {
        op: "xqa".into(),
        detail: format!("consts d2h: {e:?}"),
    })?;
    let consts = XqaConsts::from_slots(&slots)?;
    let kernel = module
        .load_function(SYM_KERNEL)
        .map_err(|e| crate::contract::OpError::Asset { op: "xqa".into(), detail: format!("{SYM_KERNEL}: {e:?}") })?;
    crate::device::Exec::set_shared_limit(&kernel, consts.smem)?;
    Ok((kernel, consts))
}

/// 直发一次(探针面;几何/值表与生产同源)
#[cfg(feature = "device")]
pub fn launch_on_stream(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    consts: &XqaConsts,
    args: &XqaArgs,
    sm_count: u32,
) -> Result<(), crate::contract::OpError> {
    let (grid, block, smem) = consts.launch_geometry(
        sm_count,
        args.batch_size,
        args.nb_k_heads,
        args.cache.max_nb_pages_per_seq * consts.tokens_per_page,
        args.q_seq_len,
    );
    crate::device::launch_vals(stream, kernel, grid, block, smem, &args.vals(consts.head_grp)).map_err(|e| {
        crate::contract::OpError::Launch {
            op: "xqa".into(),
            stage: crate::contract::Stage::Compute,
            detail: format!("{e:?}"),
        }
    })
}
