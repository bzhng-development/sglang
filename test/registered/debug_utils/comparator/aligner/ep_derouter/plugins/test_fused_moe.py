import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.fused_moe import (
    FusedMoEDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestFusedMoEDeRouter:
    """Test compute_forward_permutation using sorted_token_ids from moe_align_block_size."""

    def test_basic_permutation(self) -> None:
        """sorted_token_ids maps dispatch positions to original flatten indices."""
        num_tokens: int = 4
        top_k: int = 2
        total_slots: int = num_tokens * top_k  # 8

        sorted_token_ids: torch.Tensor = torch.tensor(
            [3, 0, 7, 1, 5, 2, 4, 6], dtype=torch.int64
        )

        plugin: FusedMoEDeRouter = FusedMoEDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"sorted_token_ids": sorted_token_ids},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)
        assert torch.equal(perm, torch.tensor([3, 0, 7, 1, 5, 2, 4, 6], dtype=torch.long))

    def test_padding_positions_marked_negative(self) -> None:
        """Padding positions (sorted_token_ids >= total_slots) become -1."""
        num_tokens: int = 2
        top_k: int = 1

        sorted_token_ids: torch.Tensor = torch.tensor(
            [1, 0, 999, 999], dtype=torch.int64
        )

        plugin: FusedMoEDeRouter = FusedMoEDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"sorted_token_ids": sorted_token_ids},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        assert torch.equal(perm, torch.tensor([1, 0, -1, -1], dtype=torch.long))

    def test_identity_permutation(self) -> None:
        """Identity sorted_token_ids produces identity forward permutation."""
        num_tokens: int = 3
        top_k: int = 1

        sorted_token_ids: torch.Tensor = torch.arange(num_tokens, dtype=torch.int64)

        plugin: FusedMoEDeRouter = FusedMoEDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"sorted_token_ids": sorted_token_ids},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=num_tokens,
        )

        assert torch.equal(perm, torch.arange(num_tokens, dtype=torch.long))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
