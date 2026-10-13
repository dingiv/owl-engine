// GDN 状态布局转置(2026-10-11 fork-bf16 全家桶配套;2026-10-13 C 轮自
// cu/owl/cast.cu 按**算子族**拆出 —— 与 cast 族分家,各自成文件)。
//
// owl 池 [HV,K,V] ↔ fork h0/ht [HV,V,K](f32;KD=VD=128)。grid (v_dim, hv),
// block (k_dim):写侧合并,读侧列距 —— 半保守转置,3.1MB 双向 ~30µs 量级
// (672 次/请求 ≈ 20ms,对应省下的 5.5× GDN 核时间可忽略)。
//
// 槽序契约:in/out 两块 + i32 k_dim + i32 v_dim(输出**非末参**,见声明表)。

extern "C" __global__ void owl_state_kv_to_vk(const float *__restrict__ in,
                                            float *__restrict__ out,
                                            int k_dim, int v_dim) {
    // out[(hv*V + v)*K + k] = in[(hv*K + k)*V + v]
    out[(blockIdx.y * v_dim + blockIdx.x) * k_dim + threadIdx.x] =
        in[(blockIdx.y * k_dim + threadIdx.x) * v_dim + blockIdx.x];
}

extern "C" __global__ void owl_state_vk_to_kv(const float *__restrict__ in,
                                            float *__restrict__ out,
                                            int k_dim, int v_dim) {
    // 逆:in [HV,V,K] → out [HV,K,V]
    out[(blockIdx.y * k_dim + threadIdx.x) * v_dim + blockIdx.x] =
        in[(blockIdx.y * v_dim + blockIdx.x) * k_dim + threadIdx.x];
}
