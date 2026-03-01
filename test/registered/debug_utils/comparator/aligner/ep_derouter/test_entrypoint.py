import sys

import pydantic
import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.entrypoint import (
    execute_de_router_plan,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestExecuteDeRouterPlan:
    """Test plugin selection and aux tensor resolution."""

    def test_fused_moe_dispatch(self) -> None:
        """Plan with dispatch_path='fused_moe' selects FusedMoEDeRouter."""
        num_tokens: int = 2
        top_k: int = 1
        hidden_dim: int = 3

        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="fused_moe",
            num_tokens=num_tokens,
            top_k=top_k,
        )

        sorted_token_ids: torch.Tensor = torch.tensor([1, 0], dtype=torch.int64)
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        )
        aux_tensors: dict[str, torch.Tensor] = {
            "fused_moe_sorted_token_ids": sorted_token_ids,
        }

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_tensors=aux_tensors
        )

        assert result.shape == (num_tokens * top_k, hidden_dim)
        # sorted_token_ids = [1, 0] → pos 0 maps to flatten_idx 1, pos 1 maps to 0
        assert torch.allclose(result[0], torch.tensor([20.0, 21.0, 22.0]))
        assert torch.allclose(result[1], torch.tensor([10.0, 11.0, 12.0]))

    def test_unknown_dispatch_path_raises(self) -> None:
        """Unknown dispatch_path is rejected by Pydantic validation."""
        with pytest.raises(pydantic.ValidationError):
            DeRouterPlan(
                dispatch_path="unknown_path",
                num_tokens=4,
                top_k=2,
            )

    def test_missing_aux_tensor_raises(self) -> None:
        """Missing required auxiliary tensor raises ValueError."""
        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="fused_moe",
            num_tokens=2,
            top_k=1,
        )
        with pytest.raises(ValueError, match="not found"):
            execute_de_router_plan(
                plan=plan,
                tensor=torch.randn(2, 3),
                aux_tensors={},
            )

    def test_megatron_a2a_dispatch(self) -> None:
        """Plan with dispatch_path='megatron_a2a' selects MegatronA2ADeRouter."""
        num_tokens: int = 3
        top_k: int = 1
        hidden_dim: int = 2

        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="megatron_a2a",
            num_tokens=num_tokens,
            top_k=top_k,
        )

        # Identity permutation
        aux_tensors: dict[str, torch.Tensor] = {
            "megatron_a2a_reversed_local_input_permutation_mapping": torch.arange(
                num_tokens, dtype=torch.long
            ),
        }
        routed_tensor: torch.Tensor = torch.randn(num_tokens, hidden_dim)

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_tensors=aux_tensors
        )

        assert torch.allclose(result, routed_tensor)

    def test_deepep_normal_dispatch(self) -> None:
        """End-to-end: deepep_normal with 2-rank scenario."""
        num_tokens: int = 4
        top_k: int = 1
        hidden_dim: int = 2

        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="deepep_normal",
            num_tokens=num_tokens,
            top_k=top_k,
        )

        # Rank 0 contributes 2 tokens (positions 0,1), rank 1 contributes 2 (positions 2,3)
        aux_tensors: dict[str, torch.Tensor] = {
            "deepep_normal_rank_prefix_matrix": torch.tensor(
                [0, 2], dtype=torch.long
            ),
        }
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_tensors=aux_tensors
        )

        assert result.shape == (num_tokens * top_k, hidden_dim)
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0]))
        assert torch.allclose(result[1], torch.tensor([20.0, 21.0]))
        assert torch.allclose(result[2], torch.tensor([30.0, 31.0]))
        assert torch.allclose(result[3], torch.tensor([40.0, 41.0]))

    def test_deepep_ll_dispatch(self) -> None:
        """End-to-end: deepep_ll with 2-expert 3D tensor, verifies flatten + scatter."""
        num_tokens: int = 2
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 2
        hidden_dim: int = 3

        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="deepep_ll",
            num_tokens=num_tokens,
            top_k=top_k,
        )

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        # Expert 0: token 0 and token 1
        packed_recv_src_info[0, 0] = 0
        packed_recv_src_info[0, 1] = 1
        # Expert 1: token 0 and token 1 (second appearance → k=1)
        packed_recv_src_info[1, 0] = 0
        packed_recv_src_info[1, 1] = 1

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = torch.tensor([10.0, 11.0, 12.0])  # token 0, k=0
        routed_tensor[0, 1] = torch.tensor([20.0, 21.0, 22.0])  # token 1, k=0
        routed_tensor[1, 0] = torch.tensor([30.0, 31.0, 32.0])  # token 0, k=1
        routed_tensor[1, 1] = torch.tensor([40.0, 41.0, 42.0])  # token 1, k=1

        aux_tensors: dict[str, torch.Tensor] = {
            "deepep_ll_packed_recv_src_info": packed_recv_src_info,
            "deepep_ll_masked_m": masked_m,
        }

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_tensors=aux_tensors
        )

        total_slots: int = num_tokens * top_k
        assert result.shape == (total_slots, hidden_dim)
        # token 0, k=0 → slot 0
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0, 12.0]))
        # token 0, k=1 → slot 1
        assert torch.allclose(result[1], torch.tensor([30.0, 31.0, 32.0]))
        # token 1, k=0 → slot 2
        assert torch.allclose(result[2], torch.tensor([20.0, 21.0, 22.0]))
        # token 1, k=1 → slot 3
        assert torch.allclose(result[3], torch.tensor([40.0, 41.0, 42.0]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
