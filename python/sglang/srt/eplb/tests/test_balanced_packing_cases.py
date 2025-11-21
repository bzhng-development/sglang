import math
import time
from typing import Iterable, Tuple

import torch

from sglang.srt.eplb.eplb_algorithms.deepseek import balanced_packing


def _validate_case(
    weight: torch.Tensor, num_packs: int, description: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    pack_index, rank_in_pack = balanced_packing(weight, num_packs)

    assert pack_index.shape == weight.shape, f"{description}: pack_index shape mismatch"
    assert (
        rank_in_pack.shape == weight.shape
    ), f"{description}: rank_in_pack shape mismatch"

    num_layers, num_groups = weight.shape
    groups_per_pack = num_groups // num_packs

    for layer in range(num_layers):
        pack_counts = [0] * num_packs
        pack_weights = [0.0] * num_packs

        for group in range(num_groups):
            pack = int(pack_index[layer, group])
            rank = int(rank_in_pack[layer, group])

            assert 0 <= pack < num_packs, f"{description}: invalid pack id"
            assert (
                0 <= rank < groups_per_pack
            ), f"{description}: invalid rank within pack"

            pack_counts[pack] += 1
            pack_weights[pack] += float(weight[layer, group])

        assert all(
            count == groups_per_pack for count in pack_counts
        ), f"{description}: uneven object distribution in layer {layer}"

        max_weight = max(pack_weights)
        min_weight = min(pack_weights)
        assert math.isfinite(
            max_weight
        ), f"{description}: non-finite max weight detected"
        assert (
            max_weight - min_weight
        ) >= 0, f"{description}: weight ordering violated"

    return pack_index, rank_in_pack


def _make_test_cases() -> Iterable[Tuple[str, torch.Tensor, int]]:
    base_dtype = torch.float32

    # Simple uniform weights, small size.
    yield (
        "uniform_balanced",
        torch.ones((2, 8), dtype=base_dtype),
        4,
    )

    # Adversarial: one extremely heavy expert, others light.
    yield (
        "single_heavy_expert",
        torch.tensor(
            [
                [1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [500.0, 400.0, 300.0, 200.0, 100.0, 50.0, 25.0, 10.0],
            ],
            dtype=base_dtype,
        ),
        4,
    )

    # Adversarial: descending weights to stress greedy order.
    yield (
        "descending_weights",
        torch.arange(32, 0, -1, dtype=base_dtype).view(1, -1),
        8,
    )

    # Adversarial: repeating high/low pattern.
    alternating = torch.tensor(
        [[10.0 if i % 2 == 0 else 0.1 for i in range(24)]], dtype=base_dtype
    )
    yield ("alternating_high_low", alternating, 6)

    # Random-ish but deterministic values spread across layers.
    grid = torch.tensor(
        [[i * j % 17 + 0.5 for j in range(12)] for i in range(3)],
        dtype=base_dtype,
    )
    yield ("modulated_grid", grid, 3)


def run_cases():
    for description, weight, num_packs in _make_test_cases():
        _validate_case(weight, num_packs, description)
        print(f"[balanced_packing] case '{description}' passed.")


def benchmark_cases(
    num_warmup: int = 5,
    num_runs: int = 200,
):
    print(f"[balanced_packing] benchmark start (warmup={num_warmup}, runs={num_runs})")
    for description, weight, num_packs in _make_test_cases():
        # warmup
        for _ in range(num_warmup):
            balanced_packing(weight, num_packs)

        start = time.perf_counter()
        for _ in range(num_runs):
            balanced_packing(weight, num_packs)
        elapsed = time.perf_counter() - start
        per_run_ms = (elapsed / num_runs) * 1000
        print(
            f"[balanced_packing] benchmark '{description}': {per_run_ms:.4f} ms/run over {num_runs} runs"
        )
    print("[balanced_packing] benchmark end")


if __name__ == "__main__":
    run_cases()
    benchmark_cases()
