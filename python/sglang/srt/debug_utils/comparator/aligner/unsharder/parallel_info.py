from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims_spec import ParallelAxis

_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")

_MOE_ALIASES: list[tuple[str, str]] = [
    ("moe_tp", "etp"),
    ("moe_dp", "edp"),
]


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

    _apply_moe_aliases(info)

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

    return result


def _apply_moe_aliases(info: dict) -> None:
    """Rewrite sglang moe_tp/moe_dp keys to canonical etp/edp names in-place."""
    for src_prefix, dst_prefix in _MOE_ALIASES:
        for suffix in ("_rank", "_size"):
            src_key: str = f"{src_prefix}{suffix}"
            dst_key: str = f"{dst_prefix}{suffix}"
            if src_key in info:
                if dst_key in info:
                    raise ValueError(
                        f"Both {src_key!r} and {dst_key!r} present in parallel_info"
                    )
                info[dst_key] = info.pop(src_key)
