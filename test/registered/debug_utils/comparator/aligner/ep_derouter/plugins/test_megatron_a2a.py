import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.megatron_a2a import (
    MegatronA2ADeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestMegatronA2ADeRouter:
    """Test reversed_local_input_permutation_mapping inverse transform."""

    def test_basic_token_level_inverse(self) -> None:
        """Token-level sorted_indices maps back to original token order."""
        num_tokens: int = 4
        top_k: int = 1
        hidden_dim: int = 3
        total_slots: int = num_tokens * top_k

        # sorted_indices[i] = original token at permuted position i
        # Permuted order: [token2, token0, token3, token1]
        sorted_indices: torch.Tensor = torch.tensor(
            [2, 0, 3, 1], dtype=torch.long
        )
        tokens_per_expert: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        # routed[0] has token 2's data, routed[1] has token 0's data, etc.
        routed_tensor: torch.Tensor = torch.zeros(total_slots, hidden_dim)
        routed_tensor[0] = torch.tensor([20.0, 21.0, 22.0])  # token 2
        routed_tensor[1] = torch.tensor([0.0, 1.0, 2.0])  # token 0
        routed_tensor[2] = torch.tensor([30.0, 31.0, 32.0])  # token 3
        routed_tensor[3] = torch.tensor([10.0, 11.0, 12.0])  # token 1

        plugin: MegatronA2ADeRouter = MegatronA2ADeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "reversed_local_input_permutation_mapping": sorted_indices,
                "tokens_per_expert": tokens_per_expert,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        # token 0 → flatten_idx 0
        assert torch.allclose(result[0], torch.tensor([0.0, 1.0, 2.0]))
        # token 1 → flatten_idx 1
        assert torch.allclose(result[1], torch.tensor([10.0, 11.0, 12.0]))
        # token 2 → flatten_idx 2
        assert torch.allclose(result[2], torch.tensor([20.0, 21.0, 22.0]))
        # token 3 → flatten_idx 3
        assert torch.allclose(result[3], torch.tensor([30.0, 31.0, 32.0]))

    def test_top_k_expansion(self) -> None:
        """When top_k > 1, each token appears multiple times in sorted_indices."""
        num_tokens: int = 2
        top_k: int = 2
        hidden_dim: int = 2
        total_slots: int = num_tokens * top_k

        # Token 0 appears twice (expert 0 and expert 1), token 1 appears twice
        # Permuted: [token0, token1, token0, token1]
        sorted_indices: torch.Tensor = torch.tensor(
            [0, 1, 0, 1], dtype=torch.long
        )
        tokens_per_expert: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )

        plugin: MegatronA2ADeRouter = MegatronA2ADeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "reversed_local_input_permutation_mapping": sorted_indices,
                "tokens_per_expert": tokens_per_expert,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        # token 0, k=0 → flatten 0: first occurrence (pos 0)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        # token 0, k=1 → flatten 1: second occurrence (pos 2)
        assert torch.allclose(result[1], torch.tensor([5.0, 6.0]))
        # token 1, k=0 → flatten 2: first occurrence (pos 1)
        assert torch.allclose(result[2], torch.tensor([3.0, 4.0]))
        # token 1, k=1 → flatten 3: second occurrence (pos 3)
        assert torch.allclose(result[3], torch.tensor([7.0, 8.0]))

    def test_identity_permutation(self) -> None:
        """Identity permutation returns same order."""
        num_tokens: int = 3
        top_k: int = 1
        hidden_dim: int = 4

        sorted_indices: torch.Tensor = torch.arange(num_tokens, dtype=torch.long)
        tokens_per_expert: torch.Tensor = torch.tensor([num_tokens], dtype=torch.long)
        routed_tensor: torch.Tensor = torch.randn(num_tokens, hidden_dim)

        plugin: MegatronA2ADeRouter = MegatronA2ADeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "reversed_local_input_permutation_mapping": sorted_indices,
                "tokens_per_expert": tokens_per_expert,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert torch.allclose(result, routed_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
