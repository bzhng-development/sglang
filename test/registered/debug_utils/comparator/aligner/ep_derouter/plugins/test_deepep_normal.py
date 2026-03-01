import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal import (
    DeepEPNormalDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestDeepEPNormalDeRouter:
    """Test compute_forward_permutation from rank_prefix_matrix."""

    def test_single_rank(self) -> None:
        """Single source rank: tokens arrive in order within that rank."""
        num_tokens: int = 4
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        rank_prefix_matrix: torch.Tensor = torch.tensor([0], dtype=torch.long)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_rank_prefix_matrix": rank_prefix_matrix},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)
        # Single rank, identity ordering: position i maps to flatten_idx i
        assert torch.equal(perm, torch.arange(total_slots, dtype=torch.long))

    def test_two_ranks(self) -> None:
        """Two source ranks: verify tokens are correctly mapped to global indices."""
        num_tokens: int = 4
        top_k: int = 1

        # Rank 0 contributes 2 tokens (positions 0,1), rank 1 contributes 2 (positions 2,3)
        rank_prefix_matrix: torch.Tensor = torch.tensor([0, 2], dtype=torch.long)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_rank_prefix_matrix": rank_prefix_matrix},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # Rank 0: global tokens 0,1 → flatten 0,1
        # Rank 1: global tokens 2,3 → flatten 2,3
        assert torch.equal(perm, torch.tensor([0, 1, 2, 3], dtype=torch.long))

    def test_output_shape(self) -> None:
        """Output always has shape [num_routed]."""
        num_tokens: int = 6
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        rank_prefix_matrix: torch.Tensor = torch.tensor([0], dtype=torch.long)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_rank_prefix_matrix": rank_prefix_matrix},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
