# VENDOR 依赖/参考源地图

> ⚠️ **本文件当前是占位**(2026-10-17 审查:此前为 **0 字节** —— "名义存在实为空",
> 比不存在更坏:它让 charter/AGENTS 里的引用看起来已被满足)。
>
> 现役权威落点(改依赖前先读这两处):
> - `crates/backends/README.md` —— **第三方 CUDA 准入铁律**(能触 CUDA 的库必须
>   源码审核账外分配;会私自 `cudaMalloc` 的库不用,或 copy 进来改造);
> - `docs/arch/charter.md` A4(**vendor 所有权**:FROZEN/OWNED/REFERENCE/BINARY 四分类)。
>
> 本机(不入库)的细目地图:建议放 `VENDOR.local.md`(`*.local.md` 已被
> `.gitignore` 收口)。要进版本时按 A4 四分类逐条核 Cargo.toml / `repos/` 后
> 替换本占位,再拆掉这段说明。
