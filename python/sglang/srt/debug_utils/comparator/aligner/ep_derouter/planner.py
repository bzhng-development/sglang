from __future__ import annotations

from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.entrypoint import (
    _PLUGIN_REGISTRY,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan
from sglang.srt.debug_utils.comparator.dims_spec import DimsSpec, ParallelAxis


def maybe_compute_de_router_plan(
    *,
    bundle_name: str,
    dims_spec: Optional[DimsSpec],
) -> list[DeRouterPlan]:
    """Generate a DeRouterPlan if dims contain ``(ep)`` and the name matches a plugin."""
    if dims_spec is None:
        return []

    has_ep: bool = any(
        mod.axis == ParallelAxis.EP
        for dim in dims_spec.dims
        for mod in dim.parallel_modifiers
    )
    if not has_ep:
        return []

    dispatch_path: Optional[str] = _infer_dispatch_path(bundle_name)
    if dispatch_path is None:
        return []

    return [DeRouterPlan(dispatch_path=dispatch_path)]


def _infer_dispatch_path(bundle_name: str) -> Optional[str]:
    """Infer dispatch path from the tensor name prefix."""
    for path in _PLUGIN_REGISTRY:
        if bundle_name.startswith(path + "_"):
            return path
    return None
