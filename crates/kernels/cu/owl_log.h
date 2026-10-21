/* owl_log.h —— 算子侧(CUDA host / C++)的**唯一**日志出口。
 *
 * 与 Rust 侧 `owl_kernels::sink` 共用同一条 `LogSink`（一个 fn 指针包）：
 * C/C++ 不许自己 `fprintf`，只许走本头文件的宏。
 *
 * ── 三道门（与 owl_shared::log / metrics 逐条对齐）────────────────────────
 *
 * 1. 编译期：`OWL_LOG_ENABLED`（由 build.rs 按 cargo feature `log` 定义）。
 *    未定义 ⇒ 全部宏展开为 `((void)0)`，**实参不求值** ⇒ 逐字节零成本。
 * 2. 运行期分级：`owl_log_on(level, tag)` —— logger 阈值
 *    （`--log-level` / `/debug/log/level/<l>`）。
 * 3. tag 过滤：同一个 `owl_log_on` 里还带 `/debug/log/tags/a,b/-x`
 *    白黑名单 ⇒ C++ 与 Rust 的可见性判据**只有一套**。
 *
 * 组合效果：
 *   - 不带 feature 的 release：宏不在，连 call 都没有；
 *   - 带 feature 但 level=warn：每次调用 = 一次 `owl_log_on`（Rust 侧
 *     一次原子读 + tag 前缀比对），**不格式化**；
 *   - 放行：`vsnprintf` 进 1024B 栈缓冲 → `owl_log_emit` → logger。
 *
 * ── 纪律 ────────────────────────────────────────────────────────────────
 *
 * - **tag 必须是字面量**（准入先行；tag 参与前缀匹配，不许运行时拼）；
 * - **错误码路径**（返回非 0 之前的上下文）用 `OWL_LOG_ERROR`：level=error
 *   恒在阈值内（除非显式 `off`），符合"拒启/致命不静默"；
 * - **设备侧** `__device__` 里的 printf 走 CUDA printf 缓冲、无法回宿主 ⇒
 *   只受 `OWL_LOG_DEVICE` 编译门约束（默认关，且只许用于一次性取证）；
 * - **不要**在 `__device__` 函数里用本文件的宿主宏。
 */
#ifndef OWL_LOG_H
#define OWL_LOG_H

#include <stddef.h>

#if defined(__cplusplus)
#include <cstdarg>
#include <cstdio>
#else
#include <stdarg.h>
#include <stdio.h>
#endif

/* ── 级别常量：与 crates/sink/src/lib.rs 的 LVL_* 逐值一致 ──────────────
 * 漂移由 `crates/kernels/src/sink.rs::c_header_levels_match` 机器判据拦住
 * （Rust 读本头文件比对），改一处必须改两处。 */
#define OWL_LVL_ERROR 0u
#define OWL_LVL_WARN 1u
#define OWL_LVL_INFO 2u
#define OWL_LVL_DEBUG 3u
#define OWL_LVL_TRACE 4u

#if defined(__cplusplus)
extern "C" {
#endif

/* Rust 侧出口（crates/kernels/src/sink.rs；只读，不写状态）。
 * `owl_log_on` 返回 1 = 放行；`owl_log_emit` 的 msg 是 len 字节 UTF-8。 */
int owl_log_on(unsigned level, const char *tag);
void owl_log_emit(unsigned level, const char *tag, const char *msg, size_t len);

#if defined(__cplusplus)
}
#endif

#if defined(OWL_LOG_ENABLED)

/* 准入后才格式化：不过门时连 vsnprintf 都不付。缓冲固定 1024B（栈），
 * 超长截断 —— 日志面宁可截断也不许分配/失败。 */
#if defined(__GNUC__)
__attribute__((format(printf, 3, 4)))
#endif
static inline void owl_logf_impl(unsigned level, const char *tag, const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1;
  owl_log_emit(level, tag, buf, len);
}

/* 主宏：`do{}while(0)` ⇒ 可当语句；if 分支只付一次准入 */
#define OWL_LOGF(level, tag, ...)                          \
  do {                                                     \
    if (owl_log_on((level), (tag)))                        \
      owl_logf_impl((level), (tag), __VA_ARGS__);           \
  } while (0)

/* 热路径手写循环用：先问一次，再决定要不要组装参数 */
#define OWL_LOG_ON(level, tag) (owl_log_on((level), (tag)) != 0)

#else /* !OWL_LOG_ENABLED */

/* 编译期剔除：实参不求值（`((void)0)` 的展开体里没有 __VA_ARGS__） */
#define OWL_LOGF(level, tag, ...) ((void)0)
#define OWL_LOG_ON(level, tag) 0

#endif /* OWL_LOG_ENABLED */

#define OWL_LOG_ERROR(tag, ...) OWL_LOGF(OWL_LVL_ERROR, tag, __VA_ARGS__)
#define OWL_LOG_WARN(tag, ...) OWL_LOGF(OWL_LVL_WARN, tag, __VA_ARGS__)
#define OWL_LOG_INFO(tag, ...) OWL_LOGF(OWL_LVL_INFO, tag, __VA_ARGS__)
#define OWL_LOG_DEBUG(tag, ...) OWL_LOGF(OWL_LVL_DEBUG, tag, __VA_ARGS__)
#define OWL_LOG_TRACE(tag, ...) OWL_LOGF(OWL_LVL_TRACE, tag, __VA_ARGS__)

/* ── 设备侧（__device__ 内的 printf）：只受编译门，运行时不可控 ─────────
 * 默认关。开启 = 一次性的取证构建（`OWL_LOG_DEVICE=1` 进 build），
 * 输出走 CUDA printf 缓冲（需 CUDA_LAUNCH_BLOCKING 或 compute-sanitizer
 * 才可靠可见），不是长期机制。 */
#if defined(OWL_LOG_DEVICE) && defined(__CUDACC__)
#define OWL_DEV_PRINTF(...) printf(__VA_ARGS__)
#else
#define OWL_DEV_PRINTF(...) ((void)0)
#endif

#endif /* OWL_LOG_H */
