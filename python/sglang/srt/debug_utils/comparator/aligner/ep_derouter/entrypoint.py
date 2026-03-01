from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.raw_aux_loader import RawAuxLoader

_PLUGIN_REGISTRY: dict[str, type[DeRouterPlugin]] = {
    "fused_moe": FusedMoEDeRouter,
    "deepep_normal": DeepEPNormalDeRouter,
    "deepep_ll": DeepEPLLDeRouter,
    "megatron_a2a": MegatronA2ADeRouter,
}

ALL_EP_AUX_DUMP_NAMES: frozenset[str] = frozenset().union(
    *(cls().required_aux_dump_names for cls in _PLUGIN_REGISTRY.values())
)


def get_required_aux_dump_names(dispatch_path: str) -> frozenset[str]:
    """Return the set of dump tensor names required by a given dispatch path."""
    plugin_cls: type[DeRouterPlugin] | None = _PLUGIN_REGISTRY.get(dispatch_path)
    if plugin_cls is None:
        raise ValueError(
            f"Unknown dispatch_path {dispatch_path!r}. "
            f"Available: {sorted(_PLUGIN_REGISTRY.keys())}"
        )
    return plugin_cls().required_aux_dump_names


def execute_de_router_plan(
    plan: DeRouterPlan,
    tensor: torch.Tensor,
    *,
    aux_loader: Optional[RawAuxLoader],
    meta: dict[str, Any],
) -> torch.Tensor:
    """Execute a de-router plan on a single tensor.

    Resolves the plugin from ``plan.dispatch_path``, loads the required
    auxiliary tensors via *aux_loader*, then calls
    ``flatten_routed_tensor`` + ``compute_forward_permutation`` and
    applies the unified scatter.
    """
    plugin_cls: type[DeRouterPlugin] | None = _PLUGIN_REGISTRY.get(plan.dispatch_path)
    if plugin_cls is None:
        raise ValueError(
            f"Unknown dispatch_path {plan.dispatch_path!r}. "
            f"Available: {sorted(_PLUGIN_REGISTRY.keys())}"
        )

    plugin: DeRouterPlugin = plugin_cls()

    aux_tensors: dict[str, torch.Tensor] = _load_aux_tensors(
        required_names=plugin.required_aux_dump_names,
        aux_loader=aux_loader,
        meta=meta,
    )

    flat_tensor: torch.Tensor = plugin.flatten_routed_tensor(
        routed_tensor=tensor, aux_tensors=aux_tensors
    )
    forward_perm: torch.Tensor = plugin.compute_forward_permutation(
        aux_tensors=aux_tensors,
        num_tokens=plan.num_tokens,
        top_k=plan.top_k,
        num_routed=flat_tensor.shape[0],
    )

    return _apply_forward_permutation(
        flat_routed=flat_tensor,
        forward_perm=forward_perm,
        total_slots=plan.num_tokens * plan.top_k,
    )


def _load_aux_tensors(
    *,
    required_names: frozenset[str],
    aux_loader: Optional[RawAuxLoader],
    meta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Load auxiliary tensors required by a de-router plugin."""
    if not required_names:
        return {}

    if aux_loader is None:
        raise ValueError(
            f"De-router requires auxiliary tensors {sorted(required_names)}, "
            f"but no aux_loader was provided."
        )

    result: dict[str, torch.Tensor] = aux_loader.load(
        step=int(meta["step"]),
        rank=int(meta["rank"]),
        layer_id=int(meta["layer_id"]),
        aux_names=required_names,
    )

    for name in required_names:
        if name not in result:
            raise ValueError(
                f"De-router requires auxiliary tensor {name!r}, "
                f"but it was not found in the dump. "
                f"Available: {sorted(result.keys())}"
            )

    return result


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
