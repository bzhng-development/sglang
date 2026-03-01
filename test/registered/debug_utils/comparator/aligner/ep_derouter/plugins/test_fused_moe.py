import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.fused_moe import (
    FusedMoEDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestFusedMoEDeRouter:
    """Test inverse scatter using sorted_token_ids from moe_align_block_size."""

    def test_basic_inverse_scatter(self) -> None:
        """sorted_token_ids maps dispatch positions to original flatten indices."""
        num_tokens: int = 4
        top_k: int = 2
        hidden_dim: int = 3
        total_slots: int = num_tokens * top_k  # 8

        # Simulate: sorted_token_ids[dispatch_pos] = original_flatten_idx
        # Dispatch positions 0-7 map to flatten indices [3, 0, 7, 1, 5, 2, 4, 6]
        sorted_token_ids: torch.Tensor = torch.tensor(
            [3, 0, 7, 1, 5, 2, 4, 6], dtype=torch.int64
        )

        # Build routed tensor: dispatch_pos i has value original_flatten_idx + 10
        routed_tensor: torch.Tensor = torch.zeros(total_slots, hidden_dim)
        for i in range(total_slots):
            routed_tensor[i] = float(sorted_token_ids[i].item() + 10)

        plugin: FusedMoEDeRouter = FusedMoEDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={"sorted_token_ids": sorted_token_ids},
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        # result[flatten_idx] should have value flatten_idx + 10
        for flatten_idx in range(total_slots):
            expected: float = float(flatten_idx + 10)
            assert torch.allclose(
                result[flatten_idx], torch.full((hidden_dim,), expected)
            ), f"Mismatch at flatten_idx={flatten_idx}"

    def test_padding_positions_ignored(self) -> None:
        """Padding positions (sorted_token_ids >= total_slots) are skipped."""
        num_tokens: int = 2
        top_k: int = 1
        hidden_dim: int = 2
        total_slots: int = 2

        # 4 dispatch positions, but only 2 are valid; last 2 are padding
        sorted_token_ids: torch.Tensor = torch.tensor(
            [1, 0, 999, 999], dtype=torch.int64
        )
        routed_tensor: torch.Tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0], [99.0, 99.0]]
        )

        plugin: FusedMoEDeRouter = FusedMoEDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={"sorted_token_ids": sorted_token_ids},
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        assert torch.allclose(result[0], torch.tensor([3.0, 4.0]))
        assert torch.allclose(result[1], torch.tensor([1.0, 2.0]))

    def test_identity_permutation(self) -> None:
        """Identity sorted_token_ids (no permutation) returns same tensor."""
        num_tokens: int = 3
        top_k: int = 1
        hidden_dim: int = 4

        sorted_token_ids: torch.Tensor = torch.arange(num_tokens, dtype=torch.int64)
        routed_tensor: torch.Tensor = torch.randn(num_tokens, hidden_dim)

        plugin: FusedMoEDeRouter = FusedMoEDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={"sorted_token_ids": sorted_token_ids},
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert torch.allclose(result, routed_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
