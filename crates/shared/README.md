# owl-testkit —— 单算子对拍测试框架

> 立项:2026-09-23。教训驱动:GDN 递推 NaN 立案(roadmap.local/phase2-qwen35-0.8b.md §六)
> 暴露的基建缺口 = 算子对拍代码各测试手抄(setup/htod/dtoh/参考公式/断言五件套),
> 无统一容差、无形状契约、无失配报告。本框架是后续所有算子
> (GDN/attention 全族、engine dry 核)的验收基建。

## 定位

单算子 = 纯函数。框架提供一条龙:

```
输入声明(seed 确定性生成) → 设备发射(闭包) → host 参考直译 → 容差对比 + 形状契约
```

支持两类发射器(框架只管「指针进/指针出 + stream」,不感知发射器内部):

- **owl-nn kernels**(nvrtc 独立模块,如 `GdnKernels`、后续 K 系);
- **engine dry_kernels**(owl-engine 的 naive 核,同指针形态)。

## 核心 API

| 件 | 签名摘要 | 说明 |
|---|---|---|
| Rig | `Rig::acquire() -> Arc<Rig>` | 进程级 OnceLock;设备序号读 `OWL_TEST_DEVICE`(缺省回落 `owl_cuda::test_device_ordinal`);持 Weights 池(默认 64 MiB,`Rig::acquire_sized(bytes)` 可调) |
| 输入 | `Rng::new(seed)`;`f32_uniform(lo,hi)` / `f32_normal(mu,sigma)` / `u32_range` | splitmix64 确定性;同 seed 同序列,跨机器跨进程可复现 |
| 设备缓冲 | `rig.htod_f32(&[f32]) -> DevBuf` / `htod_u32` / 通用 `htod<T>` | DevBuf 持有缓冲(活性随行,防悬空指针——gdn 测试踩过的坑);`.f32_ptr()/.u8_ptr()/.u32_ptr()` 位型视图 |
| 回读 | `rig.dtoh_f32(&DevBuf) -> Vec<f32>` / `dtoh_ptr_f32` | 内部先 synchronize(非默认流竞速教训,2026-09-23 safetensors flake 同根) |
| 对比 | `allclose(got, want, rtol, atol, nan_eq: bool) -> Result<(), DiffReport>` | 逐元素;失配报首个坏点(index/got/want/rel);NaN 恒不相等,`nan_eq=true` 才允许双方同 NaN;另有 `assert_finite` |
| 契约 | `expect_shape(got, want, op, in_shapes)` | 失败信息携带全部输入形状(防 narrow 丢 after / 模板序错位一类坑在边界暴露) |
| 用例 | `Case::new(name, seed).in_f32(name, shape, gen).out_f32(name, shape)` + `run(rtol, atol, device_fn, host_fn)` | 见下 |

## Case 用例形态

```rust
Case::new("gdn_fused_gating_f32", 20260923)
    .in_f32("a_log", &[8],   Gen::uniform(-1.0, 0.0))
    .in_f32("a",     &[64],  Gen::uniform(-3.0, 3.0))
    .out_f32("g",    &[64])
    .out_f32("beta", &[64])
    .run(1e-5, 1e-6,
        |b| { /* 设备闭包:b.in("a").f32_ptr() 取指针 → kernel launch → Ok(()) */ },
        |h| { /* host 参考:返回 HashMap<String, Vec<f32>>(键 = 输出名) */ })?;
```

框架负责:输入落池(同名 `b.in(name)`)、输出缓冲预清零(`b.out(name)`)、
执行后 synchronize、回读、形状契约(输出 shape/dtype 与声明一致)、
逐元素对比 + DiffReport;host 参考一律纯 Rust f32 直译公式(注释标公式出处)。

## 纪律

- 测试钉卡:`OWL_TEST_DEVICE` 环境变量(框架内已接);
- 独立构建目录:`CARGO_TARGET_DIR=/tmp/owl-tgt-testkit`;
- 参考公式来源必须注释(xinfer 原实现 / HF 公式 / 数学定义);
- **只迁移示范用例,不改被测代码**(owl-nn 现有手写对拍保持原样并存)。

## 示范用例

`tests/gdn_exemplars.rs`:fused_gating / l2_norm / rmsnorm_act 三例,
输入公式与 owl-nn 手写对拍逐字一致(可交叉验证结果一致性),
host 参考公式出处见各用例注释(gdn_shim 定谳语义 + 数学定义)。
