# FlashInfer 头 vendor(owl 树内)

**来源**:`guoqingbao/flashinfer` fork @ `0f06c2305a276bcb`(本地检出路径
`~/.cudaforge/git/checkouts/flashinfer-0f06c2305a276bcb/include`,Apache-2.0)。

**为什么 vendor**:原 build.rs 的 `FLASHINFER_FORCE_BUILD=1` 路径直接读**树外**
`~/.cudaforge/.../include` ⇒ **clone 里重建不出 `libowl_flashinfer.a`**
(违反设计 §15.1.1 律 1"所有族的源必须在树")。本目录 = 该 `.a` 的**头闭包**
(实测 `nvcc -M` 全量依赖列表,22 个文件,~700 KB;**不含 cutlass**)。
`include/flashinfer/**` 相对布局保持原样(头里用 `../x.cuh` 相对包含)。

**闭包复现方式**(改 `owl_fi_attention.cu` 的 include 后重取):

```bash
nvcc -M -std=c++17 -arch=sm_86 -I<flashinfer>/include \
     cu/flashinfer/owl_fi_attention.cu | grep -o '<flashinfer>/include/[^ \\]*' \
     | sed 's|<flashinfer>/include/||' | sort -u
```

**纪律**:
- 只放**头**(`.cuh`/`.h`),不放 `csrc/*.cu`/py/测试 —— 那是 vendor 上游的另一半,
  当前不需要(owl 只用 `flashinfer/attention/prefill.cuh` 的模板实例化);
- 升级 = 换 fork rev 后按上式重取闭包 + 重编 `.a`(`FLASHINFER_FORCE_BUILD=1`),
  并更新本文件的 rev;
- 构建期默认 `-I cu/flashinfer/include`(`FLASHINFER_INCLUDE` 仍可覆盖)。
