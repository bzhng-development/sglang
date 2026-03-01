from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims_spec import ParallelAxis

_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")

# MOE sub-axes that are redundant when identical to their parent axis.
# When moe_tp has the same rank+size as tp, it means _MOE_TP = _TP
# (they share the same process group), so reporting both is redundant.
_REDUNDANT_CHILD_TO_PARENT: dict[ParallelAxis, ParallelAxis] = {
    ParallelAxis.MOE_TP: ParallelAxis.TP,
    ParallelAxis.MOE_EP: ParallelAxis.EP,
}


def _is_error_sentinel(value: dict) -> bool:
    """Check if a parallel_info dict is an error sentinel (e.g. {'megatron_error': True})."""
    return any(k.endswith("_error") for k in value)


def normalize_parallel_info(meta: dict) -> dict[ParallelAxis, AxisInfo]:
    """Extract unified parallel info from dump meta."""
    info: Optional[dict] = None
    for key in _PARALLEL_INFO_KEYS:
        value = meta.get(key)
        if isinstance(value, dict) and value and not _is_error_sentinel(value):
            if info is not None:
                raise ValueError(
                    f"Meta contains multiple parallel_info keys among {_PARALLEL_INFO_KEYS}"
                )
            info = value

    if info is None:
        info = {}

    result: dict[ParallelAxis, AxisInfo] = {}
    for axis in ParallelAxis:
        axis_rank = info.get(f"{axis.value}_rank")
        axis_size = info.get(f"{axis.value}_size")

        # Recompute pseudo-axis lives at top-level meta, not inside parallel_info
        if axis_rank is None:
            axis_rank = meta.get(f"{axis.value}_rank")
            axis_size = meta.get(f"{axis.value}_size")

        if axis_rank is not None and axis_size is not None and axis_size > 1:
            result[axis] = AxisInfo(
                axis_rank=axis_rank,
                axis_size=axis_size,
            )

    _remove_redundant_child_axes(result)

    return result


def _remove_redundant_child_axes(
    result: dict[ParallelAxis, AxisInfo],
) -> None:
    """Drop MOE sub-axes that are identical to their parent (same process group)."""
    for child, parent in _REDUNDANT_CHILD_TO_PARENT.items():
        if child in result and parent in result and result[child] == result[parent]:
            del result[child]
