"""Layer 1: Pure logic tests for per-token relative difference computation.

No matplotlib dependency — tests calc_per_token_rel_diff() and its integration
with _compute_diff() / DiffInfo.
"""

import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    _compute_diff,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import DiffInfo
from sglang.srt.debug_utils.comparator.utils import calc_per_token_rel_diff
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="default", nightly=True)


class TestCalcPerTokenRelDiff:
    def test_identical_tensors(self) -> None:
        """Identical tensors → per-token diff all zero."""
        x: torch.Tensor = torch.randn(8, 16)
        result: torch.Tensor = calc_per_token_rel_diff(x, x, seq_dim=0)

        assert result.shape == (8,)
        assert torch.allclose(result, torch.zeros(8), atol=1e-6)

    def test_different_tensors(self) -> None:
        """Single token position differs → that position has higher diff."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x.clone()
        y[3, :] += 10.0

        result: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=0)

        assert result.shape == (8,)
        assert result[3] > result[0]
        assert result[3] > result[7]
        for i in [0, 1, 2, 4, 5, 6, 7]:
            assert result[i] < 1e-6

    def test_seq_dim_selection(self) -> None:
        """Different seq_dim values produce correct output shapes."""
        x: torch.Tensor = torch.randn(4, 8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        result_dim0: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=0)
        assert result_dim0.shape == (4,)

        result_dim1: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=1)
        assert result_dim1.shape == (8,)

        result_dim2: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=2)
        assert result_dim2.shape == (16,)

    def test_1d_tensor(self) -> None:
        """1D tensor with seq_dim=0 returns per-element diff."""
        x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
        y: torch.Tensor = torch.tensor([1.0, 2.0, 4.0])

        result: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=0)

        assert result.shape == (3,)
        assert result[0] < 1e-6
        assert result[1] < 1e-6
        assert result[2] > 0.01


class TestComputeDiffWithSeqDim:
    def test_with_seq_dim(self) -> None:
        """_compute_diff with seq_dim fills per_token_rel_diff as list[float]."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = _compute_diff(
            x_baseline=x, x_target=y, diff_threshold=1e-3, seq_dim=0
        )

        assert diff.per_token_rel_diff is not None
        assert isinstance(diff.per_token_rel_diff, list)
        assert len(diff.per_token_rel_diff) == 8
        assert all(isinstance(v, float) for v in diff.per_token_rel_diff)

    def test_without_seq_dim(self) -> None:
        """_compute_diff without seq_dim leaves per_token_rel_diff as None."""
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = _compute_diff(x_baseline=x, x_target=y, diff_threshold=1e-3)

        assert diff.per_token_rel_diff is None

    def test_seq_dim_none_explicit(self) -> None:
        """Explicit seq_dim=None behaves same as omitting it."""
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = _compute_diff(
            x_baseline=x, x_target=y, diff_threshold=1e-3, seq_dim=None
        )

        assert diff.per_token_rel_diff is None

    def test_json_serializable(self) -> None:
        """DiffInfo with per_token_rel_diff can be serialized to JSON."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(4, 8)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = _compute_diff(
            x_baseline=x, x_target=y, diff_threshold=1e-3, seq_dim=0
        )

        json_str: str = diff.model_dump_json()
        assert "per_token_rel_diff" in json_str

        roundtripped: DiffInfo = DiffInfo.model_validate_json(json_str)
        assert roundtripped.per_token_rel_diff is not None
        assert len(roundtripped.per_token_rel_diff) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
