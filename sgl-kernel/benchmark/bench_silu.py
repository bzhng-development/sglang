import argparse
from typing import Iterable, List

import torch
import triton
import triton.language as tl
import triton.testing
from transformers.activations import SiLUActivation


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = silu(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def run_triton(
    x: torch.Tensor, out: torch.Tensor, block_size: int = 1024
) -> torch.Tensor:
    assert x.is_cuda and out.is_cuda
    assert x.is_contiguous() and out.is_contiguous()
    n_elements = x.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out


hf_silu = SiLUActivation()


def run_huggingface(x: torch.Tensor) -> torch.Tensor:
    return hf_silu(x)


def time_it(fn) -> float:
    triton.testing.do_bench_cudagraph(fn, warmup=10)
    median_ms, _, _ = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return median_ms * 1000


def benchmark_once(n_elements: int, dtype: torch.dtype, block_size: int) -> None:
    device = torch.device("cuda")
    x = torch.randn(n_elements, device=device, dtype=dtype)
    out = torch.empty_like(x)

    run_triton(x, out, block_size)
    run_huggingface(x)
    torch.cuda.synchronize()

    def triton_call():
        run_triton(x, out, block_size)
        return out

    def huggingface_call():
        return run_huggingface(x)

    triton_us = time_it(triton_call)
    hf_us = time_it(huggingface_call)

    ref = huggingface_call()
    torch.cuda.synchronize()
    if not torch.allclose(out, ref, rtol=1e-3, atol=1e-5):
        raise AssertionError("Triton and Hugging Face SiLU mismatch")

    speedup = hf_us / triton_us if triton_us else float("nan")
    bytes_processed = n_elements * x.element_size()
    triton_gbps = bytes_processed / (triton_us / 1e6) / 1e9
    hf_gbps = bytes_processed / (hf_us / 1e6) / 1e9

    print(
        f"N={n_elements:>12d} | dtype={str(dtype):>9s} | "
        f"triton={triton_us:8.2f} µs | hf={hf_us:8.2f} µs | "
        f"speedup={speedup:5.2f}× | bw_triton={triton_gbps:6.2f} GB/s | bw_hf={hf_gbps:6.2f} GB/s"
    )


def parse_num_elements(arg: str) -> List[int]:
    values: Iterable[str] = (x.strip() for x in arg.split(","))
    result: List[int] = []
    for value in values:
        if not value:
            continue
        if value.lower().endswith("m"):
            result.append(int(float(value[:-1]) * 1_000_000))
        elif value.lower().endswith("k"):
            result.append(int(float(value[:-1]) * 1_000))
        else:
            result.append(int(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton SiLU vs Hugging Face SiLU"
    )
    parser.add_argument(
        "--numel",
        type=parse_num_elements,
        default=[1 << 12, 1 << 16, 1 << 20, 1 << 24],
        help="Comma separated list of element counts (supports suffix k/m)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Tensor dtype for the benchmark",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size used by the Triton kernel",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("Benchmarking Triton SiLU kernel against Hugging Face SiLU activation")
    print(f"Sizes: {args.numel} | dtype={dtype} | block_size={args.block_size}\n")

    for n in args.numel:
        benchmark_once(n, dtype, args.block_size)


if __name__ == "__main__":
    main()
