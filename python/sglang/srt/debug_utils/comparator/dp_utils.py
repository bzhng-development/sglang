"""Shared utilities for extracting DP (Data Parallel) info from dump metadata."""

from __future__ import annotations

from typing import Any

_PARALLEL_INFO_KEYS: tuple[str, ...] = (
    "sglang_parallel_info",
    "megatron_parallel_info",
)


def extract_dp_rank_from_meta(meta: dict[str, Any]) -> int:
    """Extract dp_rank from embedded metadata. Returns 0 if not found.

    Scans all known parallel_info keys for dp_rank or attn_dp_rank.
    """
    for key in _PARALLEL_INFO_KEYS:
        parallel_info: Any = meta.get(key)
        if not isinstance(parallel_info, dict):
            continue

        if "dp_rank" in parallel_info:
            return int(parallel_info["dp_rank"])
        if "attn_dp_rank" in parallel_info:
            return int(parallel_info["attn_dp_rank"])

    return 0


def extract_dp_size_from_meta(meta: dict[str, Any]) -> int:
    """Extract dp_size from embedded metadata. Returns 1 if not found.

    Scans all known parallel_info keys for dp_size or attn_dp_size.
    """
    for key in _PARALLEL_INFO_KEYS:
        parallel_info: Any = meta.get(key)
        if not isinstance(parallel_info, dict):
            continue

        if "dp_size" in parallel_info:
            return int(parallel_info["dp_size"])
        if "attn_dp_size" in parallel_info:
            return int(parallel_info["attn_dp_size"])

    return 1


def extract_dp_rank_from_meta_with_plugin(
    meta: dict[str, Any], plugin_name: str
) -> int:
    """Extract dp_rank from metadata for a specific plugin. Returns 0 if not found."""
    pi_key: str = f"{plugin_name}_parallel_info"
    parallel_info: dict[str, Any] = meta.get(pi_key, {})
    if not isinstance(parallel_info, dict):
        return 0

    if plugin_name == "megatron":
        return int(parallel_info.get("dp_rank", 0))
    elif plugin_name == "sglang":
        return int(parallel_info.get("attn_dp_rank", 0))

    return 0
