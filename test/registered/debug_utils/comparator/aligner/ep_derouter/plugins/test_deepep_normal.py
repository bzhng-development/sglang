import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal import (
    DeepEPNormalDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestDeepEPNormalDeRouter:
    """Test reconstruction from rank_prefix_matrix."""

    def test_single_rank(self) -> None:
        """Single source rank: tokens arrive in order within that rank."""
        num_tokens: int = 4
        top_k: int = 2
        hidden_dim: int = 3
        total_slots: int = num_tokens * top_k

        # Single rank contributes all tokens
        rank_prefix_matrix: torch.Tensor = torch.tensor([0], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.arange(
            total_slots * hidden_dim, dtype=torch.float
        ).reshape(total_slots, hidden_dim)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "rank_prefix_matrix": rank_prefix_matrix,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)

    def test_two_ranks(self) -> None:
        """Two source ranks: verify tokens are correctly mapped to global indices."""
        num_tokens: int = 4
        top_k: int = 1
        hidden_dim: int = 2
        total_slots: int = num_tokens * top_k

        # Rank 0 contributes 2 tokens, rank 1 contributes 2 tokens
        rank_prefix_matrix: torch.Tensor = torch.tensor([0, 2], dtype=torch.long)

        # routed_tensor: positions 0,1 from rank 0; positions 2,3 from rank 1
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "rank_prefix_matrix": rank_prefix_matrix,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        # Rank 0 tokens: global token 0 and 1
        # Rank 1 tokens: global token 2 and 3
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0]))
        assert torch.allclose(result[1], torch.tensor([20.0, 21.0]))
        assert torch.allclose(result[2], torch.tensor([30.0, 31.0]))
        assert torch.allclose(result[3], torch.tensor([40.0, 41.0]))

    def test_output_shape(self) -> None:
        """Output always has shape [num_tokens * top_k, hidden_dim]."""
        num_tokens: int = 6
        top_k: int = 2
        hidden_dim: int = 8
        total_slots: int = num_tokens * top_k

        rank_prefix_matrix: torch.Tensor = torch.tensor([0], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.randn(total_slots, hidden_dim)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "rank_prefix_matrix": rank_prefix_matrix,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
