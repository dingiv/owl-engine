// marlin-ffi: extern "C" thin wrapper over the original IST-DASLab Marlin kernel.
//
// ported from repos/marlin/marlin/marlin_cuda.cpp @ IST-DASLab/marlin (Apache-2.0)
// upstream host `mul()` used torch::Tensor; here we take raw pointers + explicit
// dims so the kernel is usable from Rust without any torch dependency.
// The kernel body (marlin_cuda_kernel.cu) is used verbatim, unmodified.

#include <cuda_runtime.h>
#include <cstdint>

// Defined in marlin_cuda_kernel.cu (C++ linkage, name-mangled).
int marlin_cuda(
  const void* A,      // row-major fp16, (prob_m, prob_k)
  const void* B,      // marlin-packed int32, (prob_k/16, prob_n*16/8)
        void* C,      // row-major fp16, (prob_m, prob_n)
        void* s,      // fp16 scales, (prob_k/groupsize, prob_n)
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,    // zeroed int32, >= prob_n/128 * max_par
  int groupsize,      // -1 (per-channel) or 128
  int dev,            // cuda device ordinal (attribute lookup only)
  cudaStream_t stream,
  int thread_k,
  int thread_n,
  int sms,
  int max_par
);

extern "C" {

// err: 0 ok, 1 ERR_PROB_SHAPE, 2 ERR_KERN_SHAPE
int marlin_mul_ffi(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize,      // -1 or 128
  int dev,            // device ordinal for SM-count lookup (context is the caller's current device)
  uintptr_t stream,   // cudaStream_t as usize; 0 = legacy default stream
  int thread_k,       // -1 = auto
  int thread_n,       // -1 = auto
  int sms,            // -1 = auto (cudaDeviceGetAttribute)
  int max_par         // 16 = upstream default
) {
  return marlin_cuda(A, B, C, s, prob_m, prob_n, prob_k, workspace,
                     groupsize, dev, (cudaStream_t)stream,
                     thread_k, thread_n, sms, max_par);
}

int marlin_stream_sync_ffi(uintptr_t stream) {
  return (int)cudaStreamSynchronize((cudaStream_t)stream);
}

int marlin_device_sync_ffi(void) {
  return (int)cudaDeviceSynchronize();
}

const char* marlin_cuda_error_str_ffi(int err) {
  return cudaGetErrorString((cudaError_t)err);
}

int marlin_malloc_ffi(size_t bytes, void** out) {
  return (int)cudaMalloc(out, bytes);
}

int marlin_memcpy_h2d_ffi(void* dst, const void* src, size_t bytes) {
  return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

int marlin_memcpy_d2h_ffi(void* dst, const void* src, size_t bytes) {
  return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

int marlin_memset_ffi(void* p, int value, size_t bytes) {
  return (int)cudaMemset(p, value, bytes);
}

int marlin_free_ffi(void* p) {
  return (int)cudaFree(p);
}

} // extern "C"
