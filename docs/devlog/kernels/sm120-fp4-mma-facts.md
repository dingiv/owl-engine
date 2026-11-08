# sm120 FP4 张量核 —— **裸 PTX 事实测定**（2026-10-29，“正向”地基）

> 目标：摆脱 CUTLASS/cute，让 NVFP4 臂用自己的内核（只针对 sm_120 / 50 系）。
> 这条路的唯一高风险是 `mma.sync...mxf4nvf4.block_scale` 的**片段/缩放因子寄存器
> 约定** —— CUTLASS 用两三千行布局代数把它包起来。本文是**实测答案**，
> 由 `cu/probe/fp4_mma_sm120_probe.cu` 与 `..._sf_probe.cu` 两个探针得出
> （探针不引任何 CUTLASS 头）。

## 0. 指令与算子尺寸

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
    .row.col.f32.e2m1.e2m1.f32.ue4m3
  {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3},
  {sfa}, {bidA,tidA}, {sfb}, {bidB,tidB};
```
- `m16n8k64`：A 16×64、B **8**×64（N 只有 8，靠多次发行拼 tile）、C/D 16×8 f32。
- 操作数寄存器：A = `u32[4]`（32 个 nibble/lane）、B = `u32[2]`（16 个）、
  SF = **每个一个 u32（4×e4m3）**，`bid/tid` **恒传 0**（CUTLASS 也这么传）。
- 编译目标必须 `-gencode arch=compute_120a,code=sm_120a`（`-arch=sm_120a`
  会被 nvcc 退化成 `compute_120`，ptxas 直接拒 `kind::mxf4nvf4`）。

## 1. A/B 数据片段（实测 `max|err| = 0`，128/128 输出逐位相等）

lane `L`（`t0 = L&3`、`t1 = L>>2`）的第 `j` 个 nibble 取自：

```
A: nibble j = v0 + 8*v1 + 16*v2     (v0 = j&7, v1 = (j>>3)&1, v2 = j>>4)
   row m = t1 + 8*v1
   col k = 8*t0 + v0 + 32*v2
B: nibble j = v0 + 8*v1             (v0 = j&7, v1 = (j>>3)&1)
   col n = t1
   col k = 8*t0 + v0 + 32*v1
```
即：**行（A）/列（B）由 `t1` 决定，k 由 `t0` + 寄存器内位置决定**。一个 u32 内
按 nibble 序号从低到高排（`w |= v << 4*b`，b = 该 u32 内的第几个 nibble）。

C/D 片段 = `SM80_16x8_Row`：`group = L>>2`、`tid = L&3`，
`d0 → (group, 2*tid)`、`d1 → (group, 2*tid+1)`、`d2/d3 → (group+8, …)`。

## 2. 缩放因子（SF）寄存器约定（黑盒差分实测）

SF 是 **e4m3、每 16 个 k 一个**（k=64 ⇒ 4 个）。每个 lane 的 u32 **4 个字节
= k-block 0..3**；**哪一行/哪一列**由 lane 号决定：

```
A(SFA): 只有 (L & 2) == 0 的 lane 有效  ⇒  行 m = (L>>2) + 8*(L&1)
B(SFB): 只有 (L & 3) == 0 的 lane 有效  ⇒  列 n = (L>>2)
其余 lane 的 SF 寄存器被硬件忽略（广播模式）。
```
即：**byte 下标 = k-block 下标**，一个 u32 就够（不需要 `bid/tid`）。

实测依据（`_sf_probe`，数据只在某个 16 宽 k-block 里为 1，把一个 (lane,byte)
的 SF 改成 3.0，看哪一 (行/列) 的输出 ×3）：A 侧 16 个 lane 各命中一行、
`byte == k-block`；B 侧 8 个 lane 各命中一列、同样 `byte == k-block`。

## 3. 这对“正向重构”意味着什么（关键的解放）

我们自己同时掌控 **打包器 + 内核**，所以**不需要沿用 CUTLASS 的 SF 全局布局**
（那个 `tile_atom_to_shape_SFB` 的交错 + 5 个仿射步长 `sfb_s[5]` 全部可以扔掉）。
可以自定义成对内核最友好的形状，例如：

```
SFB: [n/128][g][128]        // g = k/16,  128 = 一个 tile 的行块
SFA: [m/128][g][128]
```
—— 一个 128 行块 × 一个 k=64 步长 = `4 × 128` 字节连续，正好一次 512B 载入，
`lane → (row, byte=kblock)` 的映射直接对上（无需再算仿射偏移）。

## 4. 还没做（下一刀的清单）

1. `nvfp4_mma_sm120.cuh`：把上面两条映射写成 `__device__` 装载片段的小工具；
2. 主循环：tile 128×128、K-step 64、`cp.async` 双/三级缓冲（先不用 TMA），
   A/B 的 fp4 与 SF 一起进 smem；
3. epilogue：f32 累加 → f16 输出 + `alpha_ptr`（保持现有可捕获语义）；
4. 打包器改成 §3 的自定义 SF 布局（设备打包 + host 回落两条都要）；
5. 与现有 CUTLASS 实现 A/B：装配期 oracle 5 档 `max|err|` 必须仍为 0，
   decode/prefill ±3%；达标后删 `cu/cutlass/`（FA2 仍需 `cute`）。

## 5. 复现

```bash
cd crates/kernels
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
     cu/probe/fp4_mma_sm120_probe.cu -o /tmp/p1 -lcudart && /tmp/p1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
     cu/probe/fp4_mma_sm120_sf_probe.cu -o /tmp/p2 -lcudart && /tmp/p2 a && /tmp/p2 b
```
（`p1` 打印 `max|err| = 0`；`p2` 打印 SF 的 (lane,byte) → 行/列 归属表。）
