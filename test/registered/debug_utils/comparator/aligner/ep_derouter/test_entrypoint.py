import sys

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
            aux_tensor_refs={"sorted_token_ids": "my_sorted_token_ids"},
            num_tokens=num_tokens,
            top_k=top_k,
        )

        sorted_token_ids: torch.Tensor = torch.tensor([1, 0], dtype=torch.int64)
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        )
        aux_tensors: dict[str, torch.Tensor] = {"my_sorted_token_ids": sorted_token_ids}

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_tensors=aux_tensors
        )

        assert result.shape == (num_tokens * top_k, hidden_dim)
        # sorted_token_ids = [1, 0] → pos 0 maps to flatten_idx 1, pos 1 maps to 0
        assert torch.allclose(result[0], torch.tensor([20.0, 21.0, 22.0]))
        assert torch.allclose(result[1], torch.tensor([10.0, 11.0, 12.0]))

    def test_unknown_dispatch_path_raises(self) -> None:
        """Unknown dispatch_path raises ValueError."""
        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="unknown_path",
            aux_tensor_refs={},
            num_tokens=4,
            top_k=2,
        )
        with pytest.raises(ValueError, match="Unknown dispatch_path"):
            execute_de_router_plan(
                plan=plan,
                tensor=torch.randn(8, 3),
                aux_tensors={},
            )

    def test_missing_aux_tensor_raises(self) -> None:
        """Missing required auxiliary tensor raises ValueError."""
        plan: DeRouterPlan = DeRouterPlan(
            dispatch_path="fused_moe",
            aux_tensor_refs={"sorted_token_ids": "missing_tensor"},
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
            aux_tensor_refs={
                "reversed_local_input_permutation_mapping": "perm_map",
                "tokens_per_expert": "tpe",
            },
            num_tokens=num_tokens,
            top_k=top_k,
        )

        # Identity permutation
        aux_tensors: dict[str, torch.Tensor] = {
            "perm_map": torch.arange(num_tokens, dtype=torch.long),
            "tpe": torch.tensor([num_tokens], dtype=torch.long),
        }
        routed_tensor: torch.Tensor = torch.randn(num_tokens, hidden_dim)

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_tensors=aux_tensors
        )

        assert torch.allclose(result, routed_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
