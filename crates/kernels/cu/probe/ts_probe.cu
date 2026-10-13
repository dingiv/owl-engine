// 刀D 取证:层间时间戳内核(OWL_TS_PROBE)。拷贝 in→out(链式接续)+
// thread(0,0) 写 clock64 到 ts_buf[idx]。发射 = 哨兵自动网格。
// (2026-10-12 自 models::model.rs 逃生舱内联源收编 —— Kernel 节点消灭,
//  非注册内核的合法入口 = 加源 + 登记表 + 语义变体 + driver 臂)
#include <cuda_fp16.h>
extern "C" __global__ void probe_ts_f16(
    const __half* in, unsigned long long* ts, const size_t idx, const size_t n,
    __half* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) { out[i] = in[i]; }
    if (blockIdx.x == 0 && threadIdx.x == 0) { ts[idx] = clock64(); }
}
