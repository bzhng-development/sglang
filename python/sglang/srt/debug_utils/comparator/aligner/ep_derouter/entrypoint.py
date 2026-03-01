from __future__ import annotations

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
