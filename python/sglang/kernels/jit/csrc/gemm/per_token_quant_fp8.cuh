#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstddef>
#include <cstdint>

// Per-token dynamic FP8 (e4m3) quantization, small-batch schedule: one token
// per CTA with a block max-reduction. JIT port of the sgl-kernel AOT
// `per_token_quant_fp8_small_batch_kernel`, bit-identical semantics
// (`scale = rowmax/448`, no epsilon). Decode-time M is 1..O(1k); the
// warp-batched large-M schedule can join later behind the same entry.
// Arch-generic elementwise code: sm90/sm100/sm120 all supported.

namespace {

constexpr size_t kBlockSize = 256;
constexpr float kFP8E4M3Max = 448.0f;

template <typename DType, uint32_t kVecSize>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const DType* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  using namespace device;

  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const uint32_t tid = threadIdx.x;

  const DType* token_input = input + token_idx * hidden_dim;
  auto* token_output = output_q + token_idx * hidden_dim;

  using vec_t = AlignedVector<DType, kVecSize>;
  const auto gmem_in = tile::Memory<vec_t>::thread();
  const int64_t num_vec_elems = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (int64_t i = tid; i < num_vec_elems; i += blockDim.x) {
    const auto input_vec = gmem_in.load(token_input, i);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
  }

  __shared__ float reduce_smem[kBlockSize / kWarpThreads];
  cta::reduce_max(max_value, reduce_smem);
  __syncthreads();

  __shared__ float scale;
  if (tid == 0) {
    scale = reduce_smem[0] / kFP8E4M3Max;
    output_s[token_idx] = scale;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale;

  const auto gmem_out = tile::Memory<AlignedVector<fp8_e4m3_t, kVecSize>>::thread();
  for (int64_t i = tid; i < num_vec_elems; i += blockDim.x) {
    const auto input_vec = gmem_in.load(token_input, i);
    AlignedVector<fp8_e4m3_t, kVecSize> output_vec;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float value =
          math::max(math::min(static_cast<float>(input_vec[j]) * scale_inv, kFP8E4M3Max), -kFP8E4M3Max);
      output_vec[j] = static_cast<fp8_e4m3_t>(value);
    }
    gmem_out.store(token_output, output_vec, i);
  }
}

template <typename DType>
void per_token_quant_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden_dim"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, H})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);
  TensorMatcher({M, H})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(output_q);
  TensorMatcher({M})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(output_s);

  const int64_t num_tokens = M.unwrap();
  const int64_t hidden_dim = H.unwrap();
  RuntimeCheck(hidden_dim % 4 == 0, "hidden_dim must be divisible by 4, got ", hidden_dim);

  const auto launch = [&](auto kernel) {
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())(
        kernel,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  };

  if (hidden_dim % 16 == 0) {
    launch(per_token_quant_fp8_small_batch_kernel<DType, 16>);
  } else if (hidden_dim % 8 == 0) {
    launch(per_token_quant_fp8_small_batch_kernel<DType, 8>);
  } else {
    launch(per_token_quant_fp8_small_batch_kernel<DType, 4>);
  }
}

}  // namespace
