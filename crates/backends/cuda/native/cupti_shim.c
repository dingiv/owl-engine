/* owl × CUPTI 进程内归因 shim(2026-10-14 PoC)
 *
 * 为什么用 C shim: CUPTI 的 activity 结构体**逐 CUDA 版本演进**
 * (13.2 = `CUpti_ActivityKernel11`),在 Rust 侧手抄 ABI 是错值陷阱。
 * shim 在编译期吃真头文件,把记录压成**自包含 plain struct**(核名内联
 * 64B,不留悬垂指针)交给 Rust —— 与项目 marlin/flashinfer 的 foreign
 * shim 同一形态(A4:能触 CUDA 的第三方一律经 shim 收口)。
 *
 * 纪律:
 * - 回调跑在**驱动线程**,只做 memcpy + 推入无锁环(Rust 侧负责排空),
 *   绝不在此做分配/日志;
 * - `graph_id == 0` = 图外(直接发射);非 0 = **捕获图回放内的节点**,
 *   这是 nsys/gpu_prof 都拿不到的那个维度(见 perf-framework §十一)。
 */

#include <cupti_activity.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OWL_CUPTI_NAME_CAP 64

typedef struct {
    char name[OWL_CUPTI_NAME_CAP]; /* 内联核名(NUL 结尾;超长截断) */
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t graph_node_id; /* 0 = 图外 */
    uint32_t graph_id;      /* 0 = 图外 */
    uint32_t stream_id;
} owl_cupti_kern;

typedef void (*owl_cupti_cb_fn)(const owl_cupti_kern *rec);

static owl_cupti_cb_fn g_cb = NULL;
static volatile int g_on = 0;
static char g_err[256] = {0};

static void set_err(const char *what, CUptiResult r) {
    const char *s = NULL;
    cuptiGetResultString(r, &s);
    snprintf(g_err, sizeof(g_err), "%s: %s(%d)", what, s ? s : "?", (int)r);
}

const char *owl_cupti_last_error(void) { return g_err; }

static void CUPTIAPI on_buffer_requested(uint8_t **buffer, size_t *size, size_t *max_num_records) {
    *size = 1u << 21; /* 2 MiB */
    *buffer = (uint8_t *)malloc(*size);
    *max_num_records = 0; /* 0 = 只看字节数,不限条数 */
    if (*buffer == NULL) {
        *size = 0;
    }
}

static void CUPTIAPI on_buffer_completed(CUcontext context, uint32_t stream_id, uint8_t *buffer,
                                         size_t size, size_t valid_size) {
    (void)context;
    (void)size;
    if (buffer == NULL) {
        return;
    }
    CUpti_Activity *rec = NULL;
    for (;;) {
        CUptiResult r = cuptiActivityGetNextRecord(buffer, valid_size, &rec);
        if (r != CUPTI_SUCCESS) {
            /* CUPTI_ERROR_MAX_LIMIT_REACHED / SUCCESS 都是正常收尾 */
            break;
        }
        if (rec->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
            rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            continue;
        }
        CUpti_ActivityKernel11 *k = (CUpti_ActivityKernel11 *)rec;
        owl_cupti_kern e;
        memset(&e, 0, sizeof(e));
        if (k->name != NULL) {
            strncpy(e.name, k->name, OWL_CUPTI_NAME_CAP - 1);
        }
        e.start_ns = k->start;
        e.end_ns = k->end;
        e.graph_node_id = k->graphNodeId;
        e.graph_id = k->graphId;
        e.stream_id = (stream_id != 0) ? stream_id : k->streamId;
        if (g_cb != NULL) {
            g_cb(&e); /* 只投递:不做任何 CUPTI/日志/分配动作 */
        }
    }
    free(buffer);
}

int owl_cupti_start(owl_cupti_cb_fn cb) {
    if (g_on) {
        return 0; /* 幂等 */
    }
    g_cb = cb;
    CUptiResult r = cuptiActivityRegisterCallbacks(on_buffer_requested, on_buffer_completed);
    if (r != CUPTI_SUCCESS) {
        set_err("cuptiActivityRegisterCallbacks", r);
        g_cb = NULL;
        return 1;
    }
    /* 现代 kind(10)优先:nsys 走的也是它;legacy KERNEL(3) 对**图内节点**
     * 只报名字不报时间戳(2026-10-14 实测:gpu.graph.<gid>.* 的 n 对、
     * total=0 —— 而 CONCURRENT_KERNEL 记录带真时间)。两者都收,去重在
     * Rust 侧不做(实测不会同时到)。 */
    CUptiResult rk = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (rk != CUPTI_SUCCESS) {
        r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
        if (r != CUPTI_SUCCESS) {
            set_err("cuptiActivityEnable(KERNEL)", r);
            g_cb = NULL;
            return 2;
        }
    }
    /* device-launched 图(若有)的节点记录;必须在 CUDA 初始化前调,
     * 这里已晚 ⇒ 只记结果不致命(host-launched 图不需要它) */
    (void)cuptiActivityEnableDeviceGraph(1);
    g_on = 1;
    return 0;
}

void owl_cupti_flush(void) {
    if (g_on) {
        (void)cuptiActivityFlushAll(0);
    }
}

void owl_cupti_stop(void) {
    if (!g_on) {
        return;
    }
    (void)cuptiActivityFlushAll(0);
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    g_cb = NULL;
    g_on = 0;
}
