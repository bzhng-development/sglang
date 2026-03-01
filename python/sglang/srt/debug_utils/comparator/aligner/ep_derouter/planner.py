from __future__ import annotations

from typing import Any, Optional

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan


def maybe_compute_de_router_plan(
    metas: list[dict[str, Any]],
) -> list[DeRouterPlan]:
    """Generate a DeRouterPlan if the first meta has ``ep_dispatch_path``."""
    if not metas:
        return []

    meta: dict[str, Any] = metas[0]
    dispatch_path: Optional[str] = meta.get("ep_dispatch_path")
    if dispatch_path is None:
        return []

    num_tokens: int = int(meta["ep_num_tokens"])
    top_k: int = int(meta["ep_top_k"])
    aux_tensor_refs: dict[str, str] = meta.get("ep_aux_tensor_refs", {})

    return [
        DeRouterPlan(
            dispatch_path=dispatch_path,
            aux_tensor_refs=aux_tensor_refs,
            num_tokens=num_tokens,
            top_k=top_k,
        )
    ]
