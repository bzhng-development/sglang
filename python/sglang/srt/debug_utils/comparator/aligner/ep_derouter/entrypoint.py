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
    auxiliary tensors by their ref keys, and calls the plugin's ``de_route``.
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

    return plugin.de_route(
        routed_tensor=tensor,
        aux_tensors=resolved_aux,
        num_tokens=plan.num_tokens,
        top_k=plan.top_k,
    )
