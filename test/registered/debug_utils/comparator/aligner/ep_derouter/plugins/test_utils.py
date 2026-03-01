import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins._utils import (
    compute_within_group_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _naive_within_group_indices(token_ids: torch.Tensor) -> torch.Tensor:
    """Reference implementation using a Python loop."""
    n: int = token_ids.shape[0]
    counter: dict[int, int] = {}
    result: torch.Tensor = torch.zeros(n, dtype=torch.long, device=token_ids.device)
    for i in range(n):
        tid: int = int(token_ids[i].item())
        result[i] = counter.get(tid, 0)
        counter[tid] = counter.get(tid, 0) + 1
    return result


class TestComputeWithinGroupIndices:
    """Test vectorized k-counter."""

    def test_empty(self) -> None:
        result: torch.Tensor = compute_within_group_indices(
            torch.tensor([], dtype=torch.long)
        )
        assert result.shape == (0,)

    def test_all_unique(self) -> None:
        token_ids: torch.Tensor = torch.tensor([3, 1, 4, 0, 2], dtype=torch.long)
        result: torch.Tensor = compute_within_group_indices(token_ids)
        expected: torch.Tensor = torch.zeros(5, dtype=torch.long)
        assert torch.equal(result, expected)

    def test_all_same(self) -> None:
        token_ids: torch.Tensor = torch.tensor([7, 7, 7, 7], dtype=torch.long)
        result: torch.Tensor = compute_within_group_indices(token_ids)
        expected: torch.Tensor = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_mixed(self) -> None:
        token_ids: torch.Tensor = torch.tensor([0, 1, 0, 0, 1], dtype=torch.long)
        result: torch.Tensor = compute_within_group_indices(token_ids)
        expected: torch.Tensor = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_single_element(self) -> None:
        token_ids: torch.Tensor = torch.tensor([42], dtype=torch.long)
        result: torch.Tensor = compute_within_group_indices(token_ids)
        expected: torch.Tensor = torch.tensor([0], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_fuzz_vs_naive(self) -> None:
        """Fuzz test: compare vectorized impl against naive loop."""
        generator: torch.Generator = torch.Generator().manual_seed(42)
        for _ in range(20):
            n: int = int(torch.randint(1, 100, (1,), generator=generator).item())
            num_unique: int = int(torch.randint(1, max(2, n), (1,), generator=generator).item())
            token_ids: torch.Tensor = torch.randint(
                0, num_unique, (n,), dtype=torch.long, generator=generator
            )

            result: torch.Tensor = compute_within_group_indices(token_ids)
            expected: torch.Tensor = _naive_within_group_indices(token_ids)
            assert torch.equal(result, expected), (
                f"Mismatch for token_ids={token_ids.tolist()}: "
                f"got {result.tolist()}, expected {expected.tolist()}"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
