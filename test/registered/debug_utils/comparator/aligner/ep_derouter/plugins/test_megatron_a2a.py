import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.megatron_a2a import (
    MegatronA2ADeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestMegatronA2ADeRouter:
    """Test compute_forward_permutation from reversed_local_input_permutation_mapping."""

    def test_basic_token_level_permutation(self) -> None:
        """Token-level sorted_indices maps back to original token order."""
        num_tokens: int = 4
        top_k: int = 1

        # sorted_indices[i] = original token at permuted position i
        # Permuted order: [token2, token0, token3, token1]
        sorted_indices: torch.Tensor = torch.tensor([2, 0, 3, 1], dtype=torch.long)

        plugin: MegatronA2ADeRouter = MegatronA2ADeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"reversed_local_input_permutation_mapping": sorted_indices},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # token2→slot2, token0→slot0, token3→slot3, token1→slot1
        assert torch.equal(perm, torch.tensor([2, 0, 3, 1], dtype=torch.long))

    def test_top_k_expansion(self) -> None:
        """When top_k > 1, each token appears multiple times with incrementing k-index."""
        num_tokens: int = 2
        top_k: int = 2

        # Token 0 appears twice (expert 0 and expert 1), token 1 appears twice
        # Permuted: [token0, token1, token0, token1]
        sorted_indices: torch.Tensor = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        plugin: MegatronA2ADeRouter = MegatronA2ADeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"reversed_local_input_permutation_mapping": sorted_indices},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # token0,k=0→slot0, token1,k=0→slot2, token0,k=1→slot1, token1,k=1→slot3
        assert torch.equal(perm, torch.tensor([0, 2, 1, 3], dtype=torch.long))

    def test_identity_permutation(self) -> None:
        """Identity permutation returns identity forward perm."""
        num_tokens: int = 3
        top_k: int = 1

        sorted_indices: torch.Tensor = torch.arange(num_tokens, dtype=torch.long)

        plugin: MegatronA2ADeRouter = MegatronA2ADeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"reversed_local_input_permutation_mapping": sorted_indices},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=num_tokens,
        )

        assert torch.equal(perm, torch.arange(num_tokens, dtype=torch.long))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
