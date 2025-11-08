import argparse
import time
from typing import List, Tuple

import torch
from triton.testing import do_bench_cudagraph


def generate_inputs(
    m: int, n: int, k: int, dtype: torch.dtype, device: torch.device, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn(m, k, dtype=dtype, device=device)
    # Create B as row-major [N, K]; we will transpose to [K, N] for column-major use
    b_row = torch.randn(n, k, dtype=dtype, device=device)
    return a, b_row


def quantize_per_token_fp8(
    a: torch.Tensor, b_row: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-token FP8 quantization using sgl-kernel ops:
    - A_q: [M, K] with scales_a: (M,)
    - B_q_cm: [K, N] (column-major view) with scales_b: (N,)
    """
    from sgl_kernel import sgl_per_token_quant_fp8

    # Quantize A per-token (per-row)
    a_q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    a_s = torch.zeros(a.shape[0], dtype=torch.float32, device=a.device)
    sgl_per_token_quant_fp8(a.contiguous(), a_q, a_s)

    # Quantize B per-token on row-major [N, K]
    b_row_q = torch.empty_like(b_row, dtype=torch.float8_e4m3fn)
    b_s = torch.zeros(b_row.shape[0], dtype=torch.float32, device=b_row.device)
    sgl_per_token_quant_fp8(b_row.contiguous(), b_row_q, b_s)

    # Column-major view for [K, N]
    b_q_cm = b_row_q.t()

    return a_q, b_q_cm, a_s, b_s


def bench_fp8_mm(
    a_q: torch.Tensor,
    b_cm: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: torch.dtype,
    warmup: int,
    iters: int,
    use_cuda_graph: bool = True,
) -> float:
    from sgl_kernel import fp8_scaled_mm

    def run():
        return fp8_scaled_mm(a_q, b_cm, a_s, b_s, out_dtype, bias=None)

    if use_cuda_graph:
        ms = do_bench_cudagraph(run)
    else:
        # Manual timing path (exposes errors without CUDA Graphs)
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3 / iters
    return ms


def to_tflops(m: int, n: int, k: int, ms: float) -> float:
    # 2*M*N*K FLOPs for GEMM
    return (2.0 * m * n * k) * 1e-12 / (ms * 1e-3)


def parse_m_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Simple SGLang CUTLASS FP8 GEMM benchmark"
    )
    parser.add_argument("--m", type=int, default=16, help="Tokens (M)")
    parser.add_argument(
        "--m_list",
        type=str,
        default=None,
        help="Comma-separated list of M sizes (e.g. '1,2,4,8,16,32,64,128,256')",
    )
    parser.add_argument("--n", type=int, default=16384, help="Output features (N)")
    parser.add_argument("--k", type=int, default=8192, help="Hidden size (K)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Output dtype",
    )
    parser.add_argument("--iters", type=int, default=100, help="Timing iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph in timing to surface kernel errors directly",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    if args.k % 16 != 0:
        raise ValueError("K must be a multiple of 16 for FP8 GEMM kernels")

    device = torch.device("cuda")
    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.half
    in_dtype = out_dtype

    m_vals = [args.m]
    if args.m_list:
        m_vals = parse_m_list(args.m_list)

    print("===== SGLang CUTLASS FP8 GEMM =====")
    print(
        f"N={args.n} K={args.k} dtype={args.dtype} iters={args.iters} warmup={args.warmup}"
    )
    header = "{:>6}  {:>10}  {:>10}".format("M", "ms", "TFLOPs")
    print(header)

    for m in m_vals:
        a, b_row = generate_inputs(m, args.n, args.k, in_dtype, device, args.seed)
        a_q, b_cm_q, a_s, b_s = quantize_per_token_fp8(a, b_row)
        ms = bench_fp8_mm(
            a_q,
            b_cm_q,
            a_s,
            b_s,
            out_dtype,
            args.warmup,
            args.iters,
            use_cuda_graph=not args.disable_cuda_graph,
        )
        tflops = to_tflops(m, args.n, args.k, ms)
        print(f"{m:6d}  {ms:10.3f}  {tflops:10.2f}")


if __name__ == "__main__":
    main()
