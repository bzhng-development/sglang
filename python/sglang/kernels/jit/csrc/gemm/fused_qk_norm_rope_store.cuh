#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstddef>
#include <cstdint>

// Fused decode-time pre-attention chain for GQA models with per-head QK
// RMSNorm and rotate-half RoPE (Qwen3 family), page_size-1 paged caches:
//
//   q       = rope(rmsnorm_head(q_in))                      -> q_out
//   k_cache = rmsnorm_head(k_in)          (un-roped)        -> k_cache[slot]
//   k_hot   = rope(rmsnorm_head(k_in))                      -> k_hot[slot]
//   v_cache = v_in                                          -> v_cache[slot]
//   v_hot   = v_in                                          -> v_hot[slot]
//
// One CTA per (token, head); replaces the norm / rope / cat / 4x index_copy
// chain (~8 dispatches per layer per token) in latency-bound decode.
// `cos_sin` is the caller-gathered position row (mrope-combined), so the
// kernel is position-encoding agnostic. Dual K stores serve streaming designs
// that keep an un-roped cache (slot-shifted between windows) plus a roped
// hot buffer for length-limited attention; pass the same tensor twice when a
// single store suffices. Inspired by the dsv4 fused indexer kernels.
// Arch-generic: sm90 / sm100 / sm120.

namespace {

constexpr size_t kBlockSize = 128;

template <typename DType>
__global__ void fused_qk_norm_rope_store_kernel(
    const DType* __restrict__ qkv,        // (tokens, (Hq+2*Hkv)*D) packed q|k|v
    DType* __restrict__ q_out,            // (tokens, Hq*D) roped
    DType* __restrict__ k_cache,          // strided un-roped normed K store
    DType* __restrict__ k_hot,            // (pages, Hkv, D) roped normed K
    DType* __restrict__ v_cache,          // strided V store
    DType* __restrict__ v_hot,            // (pages, Hkv, D)
    const DType* __restrict__ q_norm_w,   // (D,)
    const DType* __restrict__ k_norm_w,   // (D,)
    const float* __restrict__ cos_sin,    // (tokens, 2, D) gathered rows
    const int64_t* __restrict__ slots,    // (tokens,) page index per token
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const float eps,
    // Cache-store element strides: hot buffers are page-major contiguous, but
    // the (un-roped) caches may live head-major, e.g. HF StaticCache (H, L, D).
    const int64_t cache_page_stride,
    const int64_t cache_head_stride) {
  using namespace device;

  const uint32_t token = blockIdx.x;
  const uint32_t head = blockIdx.y;  // [0, Hq + Hkv): q heads first, then kv
  const uint32_t tid = threadIdx.x;
  const uint32_t half = head_dim / 2;
  const bool is_q = head < num_q_heads;

  const uint32_t qkv_stride = (num_q_heads + 2 * num_kv_heads) * head_dim;
  const DType* src = qkv + token * qkv_stride +
      (is_q ? head * head_dim : (num_q_heads + (head - num_q_heads)) * head_dim);
  const float* cos_row = cos_sin + token * 2 * head_dim;
  const float* sin_row = cos_row + head_dim;
  const int64_t slot = slots[token];

  // Per-head RMS over head_dim (<= kBlockSize; verified host-side).
  const float value = tid < head_dim ? static_cast<float>(src[tid]) : 0.0f;
  __shared__ float sum_smem[kBlockSize / kWarpThreads];
  const uint32_t warp_id = tid / kWarpThreads;
  sum_smem[warp_id] = warp::reduce_sum(value * value);
  __syncthreads();
  __shared__ float rms_inv;
  if (tid == 0) {
    float total = 0.0f;
    for (uint32_t w = 0; w < kBlockSize / kWarpThreads; ++w) {
      total += sum_smem[w];
    }
    rms_inv = rsqrtf(total / static_cast<float>(head_dim) + eps);
  }
  __syncthreads();

  // All threads stay live through the block syncs; lanes past head_dim only
  // skip the loads/stores.
  const bool lane_active = tid < head_dim;
  const DType* norm_w = is_q ? q_norm_w : k_norm_w;
  const float normed =
      lane_active ? value * rms_inv * static_cast<float>(norm_w[tid]) : 0.0f;

  // rotate-half RoPE: out[i] = x[i]*cos[i] + rot(x)[i]*sin[i],
  // rot(x) = [-x[half:], x[:half]].
  const uint32_t pair = tid < half ? tid + half : tid - half;
  __shared__ float normed_smem[kBlockSize];
  normed_smem[tid] = normed;
  __syncthreads();
  if (!lane_active) {
    return;
  }
  const float pair_normed = normed_smem[pair];
  const float rotated = tid < half ? -pair_normed : pair_normed;
  const float roped = normed * cos_row[tid] + rotated * sin_row[tid];

  if (is_q) {
    q_out[token * num_q_heads * head_dim + head * head_dim + tid] = static_cast<DType>(roped);
    return;
  }

  const uint32_t kv_head = head - num_q_heads;
  const int64_t hot_offset = (slot * num_kv_heads + kv_head) * head_dim + tid;
  const int64_t cache_offset = slot * cache_page_stride + kv_head * cache_head_stride + tid;
  k_cache[cache_offset] = static_cast<DType>(normed);
  k_hot[hot_offset] = static_cast<DType>(roped);

  // V passes through untouched; the v rows sit after all K rows in qkv.
  const DType* v_src =
      qkv + token * qkv_stride + (num_q_heads + num_kv_heads + kv_head) * head_dim;
  const DType v_val = v_src[tid];
  v_cache[cache_offset] = v_val;
  v_hot[hot_offset] = v_val;
}

template <typename DType>
void fused_qk_norm_rope_store(
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView q_out,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView k_hot,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView v_hot,
    tvm::ffi::TensorView q_norm_w,
    tvm::ffi::TensorView k_norm_w,
    tvm::ffi::TensorView cos_sin,
    tvm::ffi::TensorView slots,
    double eps) {
  using namespace host;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto T = SymbolicSize{"num_tokens"};
  TensorMatcher({T, -1}).with_dtype<DType>().with_device(device).verify(qkv);
  TensorMatcher({T, -1}).with_dtype<DType>().with_device(device).verify(q_out);
  TensorMatcher({T, 2, -1}).with_dtype<float>().with_device(device).verify(cos_sin);
  TensorMatcher({T}).with_dtype<int64_t>().with_device(device).verify(slots);

  const int64_t num_tokens = T.unwrap();
  const int64_t head_dim = q_norm_w.size(0);
  const int64_t num_kv_heads = k_hot.size(1);
  const int64_t num_q_heads = q_out.size(1) / head_dim;
  // Cache-store layout from the (possibly head-major) k_cache view; k/v caches
  // must share it, hot buffers must be page-major contiguous.
  const int64_t cache_page_stride = k_cache.stride(0);
  const int64_t cache_head_stride = k_cache.stride(1);
  RuntimeCheck(k_cache.stride(2) == 1, "k_cache innermost stride must be 1");
  RuntimeCheck(head_dim % 2 == 0, "head_dim must be even, got ", head_dim);
  RuntimeCheck(
      head_dim <= static_cast<int64_t>(kBlockSize),
      "head_dim must be <= ",
      kBlockSize,
      ", got ",
      head_dim);

  const dim3 grid(num_tokens, num_q_heads + num_kv_heads);
  LaunchKernel(grid, kBlockSize, device.unwrap())(
      fused_qk_norm_rope_store_kernel<DType>,
      static_cast<const DType*>(qkv.data_ptr()),
      static_cast<DType*>(q_out.data_ptr()),
      static_cast<DType*>(k_cache.data_ptr()),
      static_cast<DType*>(k_hot.data_ptr()),
      static_cast<DType*>(v_cache.data_ptr()),
      static_cast<DType*>(v_hot.data_ptr()),
      static_cast<const DType*>(q_norm_w.data_ptr()),
      static_cast<const DType*>(k_norm_w.data_ptr()),
      static_cast<const float*>(cos_sin.data_ptr()),
      static_cast<const int64_t*>(slots.data_ptr()),
      static_cast<uint32_t>(num_q_heads),
      static_cast<uint32_t>(num_kv_heads),
      static_cast<uint32_t>(head_dim),
      static_cast<float>(eps),
      cache_page_stride,
      cache_head_stride);
}

}  // namespace
