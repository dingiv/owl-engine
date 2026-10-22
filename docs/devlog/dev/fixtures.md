# 夹具与门控(写测试前必读)

owl 的测试分三层,**别混用**:单元(纯函数/CPU)→ 设备门(`*_gpu_*`,要卡)→ 端到端(e2e,真 server/真模型)。

## 1. 门控约定(env 只在入口读一次)

| env | 作用 | 缺省行为 |
|---|---|---|
| `OWL_TEST_DEVICE=<BDF>` | GPU 门的**唯一开关**:值 = **PCI BDF**(如 `41:00.0`;CUDA 序号**已禁用** —— 序会漂)。解析走 `owl_cuda::resolve_ordinal_by_bdf`(设备表 + `DeviceSpec` 归一) | **未设 = 逐字 skip**(不是失败);纯数字 = **响亮 panic** |
| `OWL_E2E_ONLY=<子串>` | 只跑名字含该子串的场景 | 未设 = 全跑 |
| `OWL_E2E_SEQ=<n>` | 0.8B 场景的 `max_seq`(预算类门的旋钮) | 4096 |
| `testdata/resources.toml` | **资源清单**(取代 `OWL_AWQ27B_DIR` / `OWL_DFLASH2_DIR` / `OWL_VLM_CKPT` 等 17 键) | 缺 = 相关用例 skip(点名 key);**旧 env 设了 = 响亮 panic** |
| `OWL_TEST_CONF` | 测试环境档(整轮口径;可含 `flavor`) | 未设 = 各用例走 `OwlConfig::default()` + overlay |

**历史文档里的数字写法已失效**:旧卷/旧脚本中的 `OWL_TEST_DEVICE=<数字>` 现在会被
**响亮拒**(`CUDA 序号已禁用`);BDF 用 `nvidia-smi --query-gpu=pci.bus_id` 查(如 `00000000:41:00.0`,可简写 `41:00.0`)。

**GPU 必须串行**:`cargo test ... -- --test-threads=1`。并行 = 多 CUDA 上下文同卡并发图捕获 ⇒ **假挂**(不是失败,是卡住)。

## 2. 夹具(fixtures)

| 夹具 | 位置 | 用途 |
|---|---|---|
| models testkit | `crates/models/src/testkit.rs`(`f32b` / `harvest` / `gpu_client`) | 设备门的取值/喂值/取回 |
| `#[owl_test]` / `#[owl_test(gpu)]` | `crates/macros`(过程宏) | 用例配置注入 + GPU 门控样板(未设 env ⇒ 与旧文案逐字相同的 skip) |
| 金标 manifest | `testdata/gdn_fla/cases/*.manifest.json`(**进 git**) | FLA/conv 族的金标输入输出 |
| 0.8B 开发模型 | `crates/models/assets/Qwen3.5-0.8B`(进 git) | e2e 默认档、图捕获门 |
| 探针测试 | `crates/kernels/tests/{fi_graph_replay_probe,gdn_bucket_pad_probe}.rs` | 图回放/桶 padding 的**定向取证**(含反向档,如 `OWL_FI_NO_PLAN_RESTORE=1` 让机制"露馅"的红绿实验) |
| port 测试 | `crates/models/src/port_tests.rs` | 迁移面回归(算子/call 口径变更时) |

## 3. e2e 两个入口(**每 crate 唯一入口**)

```bash
# ① engine e2e:0.8B/27B 全场景,一次装载共跑(17 场景 ≈ 89s)
OWL_TEST_DEVICE=41:00.0 OWL_AWQ27B_DIR=<27B-AWQ> OWL_DFLASH2_DIR=<draft> \
  CARGO_BUILD_JOBS=8 cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1 --nocapture
# 过滤单场景:OWL_E2E_ONLY=longctx / schedule_decision / fuse_text_parity ...

# ② server e2e:真 server 黑盒(L1 协议 / L2 语义 / L3 答案),端口 8135
cargo test -p server --test e2e -- --test-threads=1 --nocapture
```
- 场景体在 `crates/engine/src/tests.rs`(`pub async fn`,**不挂 `#[test]`**,由入口顺序调用);
  **新场景一律并入共享装载组**,自带装载只留给 boot 期配置不同的(草稿类型/池 dtype/图档)。
- server e2e 配置 = `testdata/args/e2e-27b.toml`(**本地 gitignore 文件**,含本机绝对路径;换机要重写)。
  prefill 桶已于 2026-10-21 退役；本门只覆盖 eager prefill + decode/verify 图。
- ⚠️ **起 server 前先确认卡空**:端口被残留实例占用会让门"打到别人身上"= 假绿(门已内置占用探测 fail-fast)。

## 4. lint 门(人眼不查的规矩,机器查)

| 门 | 位置 | 规矩 |
|---|---|---|
| `facedir_lint` | `crates/models/tests/facedir_lint.rs` | `layers/` **不得**直引 `owl_kernels`(走 `crate::ops` 动作表 / `crate::kernel` 垫子);models 不得触服务端面 |
| `ops_whitelist` | `apps/server/tests/ops_whitelist.rs` | 生产 flavor 的算子白名单 ⇔ kernels 实测台账一致 |
| 名字级互证门 | kernels 侧(声明表 ⇔ `.cu` 形参名) | 121 张表全绿才算过 |
| 壳脚本门 | `apps/server/tests/{agent_gates.sh,lag_probe.py,prefix_reuse_probe.py}` | 真机多请求内容正确性 / 前缀复用真值(spec-on 用 content 级,spec-off 用 token 级) |

## 5. 加一个新测试/场景的步骤

1. 选层:纯逻辑 → 单元;要卡 → 设备门(挂 `#[owl_test(gpu)]`,用 `OWL_TEST_DEVICE` 门控);
   要真 server/真模型 → 并入 `engine_e2e` 共享组或 `apps/server/tests/e2e.rs`。
2. **判据写硬**:宁可 fail 也不要"看起来没失败" —— `-> bool`/`-> ()` 的可失败函数是禁止项(律 #1/#2,
   `apps/server/tests/e2e.rs` 头注有病史:吞错误导致整轮假绿)。
3. 跑**单场景**验证,再跑整档;GPU 门记得 `--test-threads=1`。
4. 判据与原始输出记进 roadmap/设计文档(别人要能复现同样结论)。
