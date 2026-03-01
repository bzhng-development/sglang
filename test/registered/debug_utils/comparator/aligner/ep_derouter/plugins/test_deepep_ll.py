import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_ll import (
    DeepEPLLDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestDeepEPLLDeRouter:
    """Test flatten_routed_tensor and compute_forward_permutation."""

    def test_flatten_extracts_valid_rows(self) -> None:
        """flatten_routed_tensor extracts valid rows from 3D tensor."""
        num_experts: int = 2
        expected_m: int = 4
        hidden_dim: int = 3

        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = torch.tensor([10.0, 11.0, 12.0])
        routed_tensor[0, 1] = torch.tensor([20.0, 21.0, 22.0])
        routed_tensor[1, 0] = torch.tensor([30.0, 31.0, 32.0])
        routed_tensor[1, 1] = torch.tensor([40.0, 41.0, 42.0])

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        flat: torch.Tensor = plugin.flatten_routed_tensor(
            routed_tensor=routed_tensor,
            aux_tensors={
                "packed_recv_src_info": torch.zeros(2, 4),
                "masked_m": masked_m,
            },
        )

        assert flat.shape == (4, hidden_dim)
        assert torch.allclose(flat[0], torch.tensor([10.0, 11.0, 12.0]))
        assert torch.allclose(flat[1], torch.tensor([20.0, 21.0, 22.0]))
        assert torch.allclose(flat[2], torch.tensor([30.0, 31.0, 32.0]))
        assert torch.allclose(flat[3], torch.tensor([40.0, 41.0, 42.0]))

    def test_basic_permutation(self) -> None:
        """Decode packed_recv_src_info to forward permutation."""
        num_tokens: int = 4
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 4

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0  # token 0
        packed_recv_src_info[0, 1] = 1  # token 1
        packed_recv_src_info[1, 0] = 2  # token 2
        packed_recv_src_info[1, 1] = 3  # token 3

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={
                "packed_recv_src_info": packed_recv_src_info,
                "masked_m": masked_m,
            },
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # token0,k=0→slot0; token1,k=0→slot2; token2,k=0→slot4; token3,k=0→slot6
        assert torch.equal(perm, torch.tensor([0, 2, 4, 6], dtype=torch.long))

    def test_padding_rows_ignored(self) -> None:
        """Only masked_m valid rows contribute to the permutation."""
        num_tokens: int = 2
        top_k: int = 1
        num_experts: int = 2
        expected_m: int = 4

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0  # valid: token 0
        packed_recv_src_info[0, 1] = 999  # padding
        packed_recv_src_info[1, 0] = 1  # valid: token 1
        packed_recv_src_info[1, 1] = 999  # padding

        masked_m: torch.Tensor = torch.tensor([1, 1], dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={
                "packed_recv_src_info": packed_recv_src_info,
                "masked_m": masked_m,
            },
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=2,
        )

        assert perm.shape == (2,)
        assert torch.equal(perm, torch.tensor([0, 1], dtype=torch.long))

    def test_top_k_assignment(self) -> None:
        """When a token appears in multiple experts, k-index increments."""
        num_tokens: int = 2
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 2

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0  # token 0, first appearance
        packed_recv_src_info[0, 1] = 1  # token 1, first appearance
        packed_recv_src_info[1, 0] = 0  # token 0, second appearance
        packed_recv_src_info[1, 1] = 1  # token 1, second appearance

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={
                "packed_recv_src_info": packed_recv_src_info,
                "masked_m": masked_m,
            },
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # token0,k=0→slot0; token1,k=0→slot2; token0,k=1→slot1; token1,k=1→slot3
        assert torch.equal(perm, torch.tensor([0, 2, 1, 3], dtype=torch.long))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
