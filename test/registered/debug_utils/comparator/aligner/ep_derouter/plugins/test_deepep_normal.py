import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal import (
    DeepEPNormalDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestDeepEPNormalDeRouter:
    """Test compute_forward_permutation from src2dst."""

    def test_identity(self) -> None:
        """src2dst is identity → forward_perm is identity."""
        num_tokens: int = 4
        top_k: int = 1
        total_slots: int = num_tokens * top_k

        src2dst: torch.Tensor = torch.arange(total_slots, dtype=torch.long)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_src2dst": src2dst},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)
        assert torch.equal(perm, torch.arange(total_slots, dtype=torch.long))

    def test_reversed(self) -> None:
        """src2dst reverses positions → forward_perm is the inverse."""
        num_tokens: int = 3
        top_k: int = 1
        total_slots: int = num_tokens * top_k

        # canonical 0→dispatch 2, canonical 1→dispatch 1, canonical 2→dispatch 0
        src2dst: torch.Tensor = torch.tensor([2, 1, 0], dtype=torch.long)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_src2dst": src2dst},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        # forward_perm[dispatch_pos] = canonical_pos
        # dispatch 0 ← canonical 2, dispatch 1 ← canonical 1, dispatch 2 ← canonical 0
        assert torch.equal(perm, torch.tensor([2, 1, 0], dtype=torch.long))

    def test_partial_ep_rank(self) -> None:
        """EP rank only handles subset of tokens (others have negative src2dst)."""
        num_tokens: int = 4
        top_k: int = 1
        total_slots: int = num_tokens * top_k

        # This rank handles canonical 1 and 3, dispatched to positions 0 and 1
        src2dst: torch.Tensor = torch.tensor([-1, 0, -1, 1], dtype=torch.long)
        num_routed: int = 2

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_src2dst": src2dst},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=num_routed,
        )

        assert perm.shape == (num_routed,)
        # dispatch 0 ← canonical 1, dispatch 1 ← canonical 3
        assert torch.equal(perm, torch.tensor([1, 3], dtype=torch.long))

    def test_output_shape(self) -> None:
        """Output always has shape [num_routed]."""
        num_tokens: int = 6
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        src2dst: torch.Tensor = torch.arange(total_slots, dtype=torch.long)

        plugin: DeepEPNormalDeRouter = DeepEPNormalDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_src2dst": src2dst},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
