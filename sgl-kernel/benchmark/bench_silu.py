import argparse
from typing import Iterable, List

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing


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


def time_it_events(fn, iters=200):
    # warmup
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters  # ms/op


def benchmark_once(n_elements: int, dtype: torch.dtype, block_size: int) -> None:
    device = torch.device("cuda")
    x = torch.randn(n_elements, device=device, dtype=dtype)
    out_triton = torch.empty_like(x)
    out_torch = torch.empty_like(x)

    run_triton(x, out_triton, block_size)
    F.silu(x, out=out_torch)
    torch.cuda.synchronize()

    triton_ms = time_it_events(lambda: run_triton(x, out_triton, block_size))
    torch_ms = time_it_events(lambda: F.silu(x, out=out_torch))

    ref = F.silu(x)
    torch.cuda.synchronize()
    assert torch.allclose(out_triton, ref, rtol=1e-3, atol=1e-5)

    bytes_moved = 2 * n_elements * x.element_size()  # read + write
    triton_gibps = (bytes_moved / (triton_ms / 1e3)) / (1024**3)
    torch_gibps = (bytes_moved / (torch_ms / 1e3)) / (1024**3)
    speedup = torch_ms / triton_ms

    print(
        f"N={n_elements:>12d} | dtype={str(dtype):>9s} | "
        f"triton={triton_ms:8.3f} ms | torch={torch_ms:8.3f} ms | "
        f"speedup={speedup:5.2f}× | RW_bw_triton={triton_gibps:6.2f} GiB/s | "
        f"RW_bw_torch={torch_gibps:6.2f} GiB/s"
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
        description="Benchmark Triton SiLU kernel against torch.nn.functional.silu"
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

    print("Benchmarking Triton SiLU kernel against torch.nn.functional.silu")
    print(f"Sizes: {args.numel} | dtype={dtype} | block_size={args.block_size}\n")

    for n in args.numel:
        benchmark_once(n, dtype, args.block_size)


if __name__ == "__main__":
    main()
