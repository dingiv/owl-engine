//! owl-cpu —— CPU 后端:`DeviceClient` 契约的**进程内同步实现**。
//!
//! 与 owl-cuda 的形态对照(同一契约,两种执行体):
//!
//! ```text
//! owl-cuda  客户端 ──mpsc 管道──► GPU actor 线程(绑卡/三流/账房)──► 硬件
//! owl-cpu   客户端 ──函数调用──►  直接执行(朴素算子,Vec<f32>)    ──► CPU
//! ```
//!
//! **无 server**:CPU 没有异构执行域,launch/memcpy 本身就是同步完成的,
//! 不需要 actor/管道/回执桥 —— `submit` 即执行,回执即返回。
//! 这就是"后端是否需要 server"的分界:server 的存在理由是跨执行域
//! 序列化设备访问,CPU 后端没有这个跨域问题。
//!
//! 模块地图:
//! - [`value`]:f32 值块(CPU 后端的数据平面;host 真值)
//! - [`ops`][]:朴素算子实现(matmul/add/silu/rmsnorm;对拍锚的独立副本)
//! - [`face`]:CpuFace(DeviceClient 实现;客户端直接持有的后端句柄)
//!
//! 纪律:本 crate 只依赖 owl-iface(契约),**不依赖任何前端 crate**
//! (2026-09-25 后端不依赖前端裁决;与 owl-cuda 同律)。
//!
//! # 能力边界(继承 `DeviceClient` 默认面的那部分 —— 2026-10-17 审查 §三轮-C2 成文)
//!
//! 测试作者必须知道"哪些面在 CPU 上**不是**生产语义":
//!
//! | 面 | CPU 档行为 | 与真卡的差别 |
//! |---|---|---|
//! | `swap` | **host 兜底**(逐片 `dtoh_at` + `upload_pinned`) | 真卡是**设备侧单命令**(`Command::Swap` + connector)⇒ store 在 CPU 面测的是兜底路径,不是生产路径 |
//! | `launch_and_harvest` | `Err("未实现")` | 真卡有实现;CPU 面不可跑该面(响亮拒,不静默) |
//! | `alloc_uninit` | 默认 = `alloc`(**清零**);`MockLimits::poison_uninit` 置位才填 NaN | 真卡"未初始化"无零保证 ⇒ 默认档对"忘 memset"类缺陷是**乐观的** |
//! | `op_env` / `face_clone` / `loader_faces` | `None` | 无设备属性/无多 face ⇒ 装载自动回退 CPU 打包与串行 |
//! | `sm_count` | `Err`(引擎回退常量 + **响亮告警**) | 真卡取 `multiProcessorCount` |
//! | `htod_batch` / `dtoh_batch` / `copy_batch` / `write_blocks_f32` / `htod_f32` / `upload_pinned_on` | 逐片循环(等价) | 无流概念,语义等价 ✓ |
//!
//! 判据:**以 CPU 面跑绿推断生产绿之前,先看该面是否在上表**。

mod face;
mod ops;
mod value;

pub use face::{bench_decode_f16, bench_read, bench_write, mock_stats, mock_stats_reset, CpuFace, MockLimits};
pub use value::Value;
