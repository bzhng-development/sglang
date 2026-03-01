from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_ll import (
    DeepEPLLDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal import (
    DeepEPNormalDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.fused_moe import (
    FusedMoEDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.megatron_a2a import (
    MegatronA2ADeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan

_PLUGIN_REGISTRY: dict[str, type[DeRouterPlugin]] = {
    "fused_moe": FusedMoEDeRouter,
    "deepep_normal": DeepEPNormalDeRouter,
    "deepep_ll": DeepEPLLDeRouter,
    "megatron_a2a": MegatronA2ADeRouter,
}


def execute_de_router_plan(
    plan: DeRouterPlan,
    tensor: torch.Tensor,
    aux_tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Execute a de-router plan on a single tensor.

    Resolves the plugin from ``plan.dispatch_path``, gathers the required
    auxiliary tensors, then calls ``flatten_routed_tensor`` +
    ``compute_forward_permutation`` and applies the unified scatter.
    """
    plugin_cls: type[DeRouterPlugin] | None = _PLUGIN_REGISTRY.get(plan.dispatch_path)
    if plugin_cls is None:
        raise ValueError(
            f"Unknown dispatch_path {plan.dispatch_path!r}. "
            f"Available: {sorted(_PLUGIN_REGISTRY.keys())}"
        )

    plugin: DeRouterPlugin = plugin_cls()

    resolved_aux: dict[str, torch.Tensor] = {}
    for ref_key, dump_name in plan.aux_tensor_refs.items():
        if dump_name not in aux_tensors:
            raise ValueError(
                f"De-router plugin {plan.dispatch_path!r} requires auxiliary tensor "
                f"{ref_key!r} (dump name {dump_name!r}), but it was not found. "
                f"Available: {sorted(aux_tensors.keys())}"
            )
        resolved_aux[ref_key] = aux_tensors[dump_name]

    flat_tensor: torch.Tensor = plugin.flatten_routed_tensor(
        routed_tensor=tensor, aux_tensors=resolved_aux
    )
    forward_perm: torch.Tensor = plugin.compute_forward_permutation(
        aux_tensors=resolved_aux,
        num_tokens=plan.num_tokens,
        top_k=plan.top_k,
        num_routed=flat_tensor.shape[0],
    )

    return _apply_forward_permutation(
        flat_routed=flat_tensor,
        forward_perm=forward_perm,
        total_slots=plan.num_tokens * plan.top_k,
    )


def _apply_forward_permutation(
    flat_routed: torch.Tensor,
    forward_perm: torch.Tensor,
    *,
    total_slots: int,
) -> torch.Tensor:
    """Scatter flat_routed into output using forward_perm indices.

    Positions where ``forward_perm[i] == -1`` are discarded (padding).
    """
    valid_mask: torch.Tensor = forward_perm >= 0
    trailing_shape: list[int] = list(flat_routed.shape[1:])

    output: torch.Tensor = torch.zeros(
        [total_slots] + trailing_shape,
        dtype=flat_routed.dtype,
        device=flat_routed.device,
    )
    output[forward_perm[valid_mask]] = flat_routed[valid_mask]

    return output
