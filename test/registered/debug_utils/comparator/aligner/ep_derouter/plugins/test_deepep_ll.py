import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_ll import (
    DeepEPLLDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestDeepEPLLDeRouter:
    """Test packed_recv_src_info decoding and 3D→1D reshaping."""

    def test_basic_decoding(self) -> None:
        """Decode packed_recv_src_info to token indices and assign k-slots."""
        num_tokens: int = 4
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 4
        hidden_dim: int = 3
        total_slots: int = num_tokens * top_k

        # packed_recv_src_info: token_index = value % num_tokens
        # Expert 0 has 2 valid tokens (from token 0 and token 1)
        # Expert 1 has 2 valid tokens (from token 2 and token 3)
        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0  # token 0
        packed_recv_src_info[0, 1] = 1  # token 1
        packed_recv_src_info[1, 0] = 2  # token 2
        packed_recv_src_info[1, 1] = 3  # token 3

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        # 3D routed tensor: (num_experts, expected_m, hidden_dim)
        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = torch.tensor([10.0, 11.0, 12.0])
        routed_tensor[0, 1] = torch.tensor([20.0, 21.0, 22.0])
        routed_tensor[1, 0] = torch.tensor([30.0, 31.0, 32.0])
        routed_tensor[1, 1] = torch.tensor([40.0, 41.0, 42.0])

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "packed_recv_src_info": packed_recv_src_info,
                "masked_m": masked_m,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        # token 0, k=0 → flatten_idx=0
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0, 12.0]))
        # token 1, k=0 → flatten_idx=2
        assert torch.allclose(result[2], torch.tensor([20.0, 21.0, 22.0]))
        # token 2, k=0 → flatten_idx=4
        assert torch.allclose(result[4], torch.tensor([30.0, 31.0, 32.0]))
        # token 3, k=0 → flatten_idx=6
        assert torch.allclose(result[6], torch.tensor([40.0, 41.0, 42.0]))

    def test_padding_rows_ignored(self) -> None:
        """Rows beyond masked_m[e] are padding and should be ignored."""
        num_tokens: int = 2
        top_k: int = 1
        num_experts: int = 2
        expected_m: int = 4
        hidden_dim: int = 2
        total_slots: int = num_tokens * top_k

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0  # valid: token 0
        packed_recv_src_info[0, 1] = 999  # padding
        packed_recv_src_info[1, 0] = 1  # valid: token 1
        packed_recv_src_info[1, 1] = 999  # padding

        masked_m: torch.Tensor = torch.tensor([1, 1], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = torch.tensor([1.0, 2.0])
        routed_tensor[0, 1] = torch.tensor([99.0, 99.0])  # padding, should not appear
        routed_tensor[1, 0] = torch.tensor([3.0, 4.0])
        routed_tensor[1, 1] = torch.tensor([99.0, 99.0])  # padding

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "packed_recv_src_info": packed_recv_src_info,
                "masked_m": masked_m,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[1], torch.tensor([3.0, 4.0]))

    def test_top_k_assignment(self) -> None:
        """When a token appears in multiple experts, k-index increments."""
        num_tokens: int = 2
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 2
        hidden_dim: int = 1
        total_slots: int = num_tokens * top_k

        # Token 0 goes to expert 0 and expert 1, token 1 goes to expert 0 and expert 1
        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0  # token 0, first appearance
        packed_recv_src_info[0, 1] = 1  # token 1, first appearance
        packed_recv_src_info[1, 0] = 0  # token 0, second appearance
        packed_recv_src_info[1, 1] = 1  # token 1, second appearance

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = 10.0  # token 0 in expert 0
        routed_tensor[0, 1] = 20.0  # token 1 in expert 0
        routed_tensor[1, 0] = 30.0  # token 0 in expert 1
        routed_tensor[1, 1] = 40.0  # token 1 in expert 1

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        result: torch.Tensor = plugin.de_route(
            routed_tensor=routed_tensor,
            aux_tensors={
                "packed_recv_src_info": packed_recv_src_info,
                "masked_m": masked_m,
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert result.shape == (total_slots, hidden_dim)
        # token 0, k=0 → flatten_idx 0
        assert result[0].item() == pytest.approx(10.0)
        # token 0, k=1 → flatten_idx 1
        assert result[1].item() == pytest.approx(30.0)
        # token 1, k=0 → flatten_idx 2
        assert result[2].item() == pytest.approx(20.0)
        # token 1, k=1 → flatten_idx 3
        assert result[3].item() == pytest.approx(40.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
