import argparse
from typing import Tuple

import torch
from triton.testing import do_bench_cudagraph


def generate_inputs(
    m: int, n: int, k: int, dtype: torch.dtype, device: torch.device, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn(m, k, dtype=dtype, device=device)
    # For sgl-kernel GEMM we will pass B as [K, N] with column-major layout.
    # Construct a row-major buffer [N, K] first; we will transpose to [K, N]
    # for kernels that require column-major B.
    b_row = torch.randn(n, k, dtype=dtype, device=device)
    return a, b_row


def quantize_per_tensor_fp8_sglkernel_for_cutlass(
    a: torch.Tensor, b_row: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor FP8 quantization for both A and B, then expand the scalar scales
    to vectors to satisfy sgl-kernel fp8_scaled_mm's interface:
    - A scales: repeat scalar to length M
    - B scales: repeat scalar to length N

    B is provided as row-major [N, K]; we quantize that and transpose to [K, N]
    (column-major view) for the sgl-kernel GEMM.
    """
    from sgl_kernel import sgl_per_tensor_quant_fp8

    # Quantize A per-tensor
    a_q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    a_s_scalar = torch.zeros(1, dtype=torch.float32, device=a.device)
    sgl_per_tensor_quant_fp8(a.contiguous(), a_q, a_s_scalar, False)
    a_s_vec = a_s_scalar.expand(a.shape[0]).contiguous()  # (M,)

    # Quantize B (row-major [N, K]) per-tensor
    b_row_q = torch.empty_like(b_row, dtype=torch.float8_e4m3fn)
    b_s_scalar = torch.zeros(1, dtype=torch.float32, device=b_row.device)
    sgl_per_tensor_quant_fp8(b_row.contiguous(), b_row_q, b_s_scalar, False)
    b_s_vec = b_s_scalar.expand(b_row.shape[0]).contiguous()  # (N,)

    # Use column-major [K, N] view for sgl-kernel
    b_cm = b_row_q.t()

    return a_q, b_cm, a_s_vec, b_s_vec


def quantize_per_tensor_fp8_sglkernel(
    a: torch.Tensor, b_kn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize per-tensor using sgl-kernel per-tensor quant op to get scalar scales
    for the FlashInfer path. b_kn must be [K, N] row-major for flashinfer.
    """
    from sgl_kernel import sgl_per_tensor_quant_fp8

    # A per-tensor
    a_q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    a_s = torch.zeros(1, dtype=torch.float32, device=a.device)
    sgl_per_tensor_quant_fp8(a.contiguous(), a_q, a_s, False)

    # B per-tensor
    b_q = torch.empty_like(b_kn, dtype=torch.float8_e4m3fn)
    b_s = torch.zeros(1, dtype=torch.float32, device=b_kn.device)
    sgl_per_tensor_quant_fp8(b_kn.contiguous(), b_q, b_s, False)

    return a_q, b_q, a_s, b_s


def quantize_per_tensor_fp8_for_flashinfer(
    a: torch.Tensor, b_row: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor quantization for FlashInfer bmm_fp8:
    - A_q: [M, K]
    - B_q_cm: [K, N] as a column-major view (transpose of row-major [N, K])
    - a_s_scalar, b_s_scalar: scalars
    """
    from sgl_kernel import sgl_per_tensor_quant_fp8

    a_q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    a_s = torch.zeros(1, dtype=torch.float32, device=a.device)
    sgl_per_tensor_quant_fp8(a.contiguous(), a_q, a_s, False)

    b_row_q = torch.empty_like(b_row, dtype=torch.float8_e4m3fn)
    b_s = torch.zeros(1, dtype=torch.float32, device=b_row.device)
    sgl_per_tensor_quant_fp8(b_row.contiguous(), b_row_q, b_s, False)

    b_q_cm = b_row_q.t()  # column-major view for [K, N]
    return a_q, b_q_cm, a_s, b_s


def bench_sglkernel_fp8_mm(
    a_q: torch.Tensor,
    b_cm: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: torch.dtype,
) -> float:
    from sgl_kernel import fp8_scaled_mm

    def run():
        # a_q: [M, K], b_cm: [K, N] (column-major), a_s: (M,), b_s: (N,)
        return fp8_scaled_mm(a_q, b_cm, a_s, b_s, out_dtype, bias=None)

    ms = do_bench_cudagraph(run)
    return ms


def bench_flashinfer_fp8_mm(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_s_scalar: torch.Tensor,
    b_s_scalar: torch.Tensor,
    out_dtype: torch.dtype,
) -> float | None:
    """Benchmark FlashInfer bmm_fp8 directly. Returns None if import fails."""
    try:
        from flashinfer.gemm import bmm_fp8 as fi_bmm_fp8
    except Exception:
        return None

    def run():
        # bmm_fp8 expects (B, M, K) and (B, K, N) with column-major for B; batch=1
        return fi_bmm_fp8(
            a_q.unsqueeze(0),
            b_q.unsqueeze(0),
            a_s_scalar,
            b_s_scalar,
            out_dtype,
            backend="auto",
        )

    ms = do_bench_cudagraph(run)
    return ms


def to_tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) * 1e-12 / (ms * 1e-3)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 GEMM: sgl-kernel (CUTLASS) vs FlashInfer (SM100)"
    )
    parser.add_argument("--m", type=int, default=16, help="Tokens (M)")
    parser.add_argument("--n", type=int, default=16384, help="Output features (N)")
    parser.add_argument("--k", type=int, default=8192, help="Hidden size (K)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Output dtype",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-flashinfer", action="store_true", help="Skip FlashInfer path"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # Alignment constraints
    if args.k % 16 != 0:
        raise ValueError("K must be a multiple of 16 for FP8 GEMM kernels")

    device = torch.device(args.device)
    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.half

    a, b_row = generate_inputs(args.m, args.n, args.k, out_dtype, device, args.seed)

    # Prepare inputs for sgl-kernel fp8_scaled_mm (per-tensor quant, expanded scales)
    a_q_sgl, b_cm_sgl, a_s_vec, b_s_vec = quantize_per_tensor_fp8_sglkernel_for_cutlass(
        a, b_row
    )

    # Prepare inputs for FlashInfer (per-tensor, with column-major B view)
    a_q_fi, b_q_fi_cm, a_s_fi, b_s_fi = quantize_per_tensor_fp8_for_flashinfer(a, b_row)

    # Run sgl-kernel
    ms_sgl = bench_sglkernel_fp8_mm(a_q_sgl, b_cm_sgl, a_s_vec, b_s_vec, out_dtype)
    tflops_sgl = to_tflops(args.m, args.n, args.k, ms_sgl)

    ms_fi = None
    tflops_fi = None
    if not args.skip_flashinfer:
        ms_fi = bench_flashinfer_fp8_mm(a_q_fi, b_q_fi_cm, a_s_fi, b_s_fi, out_dtype)
        if ms_fi is not None:
            tflops_fi = to_tflops(args.m, args.n, args.k, ms_fi)

    print("===== FP8 GEMM Bench (SM100 assumed) =====")
    print(f"Shape M={args.m} N={args.n} K={args.k} dtype={args.dtype}")
    print(f"sgl-kernel fp8_scaled_mm: {ms_sgl:.3f} ms, {tflops_sgl:.2f} TFLOPs")
    if ms_fi is not None:
        print(f"FlashInfer bmm_fp8:      {ms_fi:.3f} ms, {tflops_fi:.2f} TFLOPs")
    else:
        if not args.skip_flashinfer:
            print("FlashInfer not available; skipped FI bench.")


if __name__ == "__main__":
    main()
