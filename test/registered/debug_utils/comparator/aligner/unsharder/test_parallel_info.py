import sys

import pytest

from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims_spec import ParallelAxis
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestNormalizeParallelInfo:
    def test_sglang_info(self) -> None:
        meta = {
            "sglang_parallel_info": {
                "tp_rank": 2,
                "tp_size": 4,
                "pp_rank": 0,
                "pp_size": 1,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {ParallelAxis.TP: AxisInfo(axis_rank=2, axis_size=4)}

    def test_megatron_info(self) -> None:
        meta = {
            "megatron_parallel_info": {
                "tp_rank": 1,
                "tp_size": 2,
                "cp_rank": 0,
                "cp_size": 4,
                "dp_rank": 0,
                "dp_size": 1,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=4),
        }

    def test_no_parallel_info(self) -> None:
        assert normalize_parallel_info({}) == {}
        assert normalize_parallel_info({"other_key": 42}) == {}

    def test_both_present_raises(self) -> None:
        meta = {
            "sglang_parallel_info": {"tp_rank": 0, "tp_size": 2},
            "megatron_parallel_info": {"tp_rank": 0, "tp_size": 2},
        }
        with pytest.raises(ValueError, match="multiple parallel_info"):
            normalize_parallel_info(meta)

    def test_megatron_with_sp(self) -> None:
        """Megatron SP reuses TP group: sp_rank==tp_rank, sp_size==tp_size."""
        meta = {
            "megatron_parallel_info": {
                "tp_rank": 1,
                "tp_size": 4,
                "sp_rank": 1,
                "sp_size": 4,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=4),
            ParallelAxis.SP: AxisInfo(axis_rank=1, axis_size=4),
        }

    def test_size_1_filtered(self) -> None:
        meta = {
            "sglang_parallel_info": {
                "tp_rank": 0,
                "tp_size": 1,
                "cp_rank": 0,
                "cp_size": 1,
            }
        }
        assert normalize_parallel_info(meta) == {}

    def test_recompute_pseudo_from_top_level_meta(self) -> None:
        """recompute_pseudo_rank/size at top-level meta is extracted alongside TP."""
        meta = {
            "recompute_pseudo_rank": 1,
            "recompute_pseudo_size": 2,
            "sglang_parallel_info": {"tp_rank": 0, "tp_size": 2},
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.RECOMPUTE_PSEUDO: AxisInfo(axis_rank=1, axis_size=2),
            ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
        }

    def test_sglang_moe_tp(self) -> None:
        """sglang moe_tp_rank/moe_tp_size maps to ParallelAxis.MOE_TP."""
        meta = {
            "sglang_parallel_info": {
                "tp_rank": 0,
                "tp_size": 2,
                "moe_tp_rank": 1,
                "moe_tp_size": 4,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            ParallelAxis.MOE_TP: AxisInfo(axis_rank=1, axis_size=4),
        }

    def test_sglang_moe_dp(self) -> None:
        """sglang moe_dp_rank/moe_dp_size maps to ParallelAxis.MOE_DP."""
        meta = {
            "sglang_parallel_info": {
                "ep_rank": 0,
                "ep_size": 2,
                "moe_dp_rank": 1,
                "moe_dp_size": 4,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.EP: AxisInfo(axis_rank=0, axis_size=2),
            ParallelAxis.MOE_DP: AxisInfo(axis_rank=1, axis_size=4),
        }

    def test_megatron_etp(self) -> None:
        """Megatron etp_rank/etp_size maps to ParallelAxis.ETP."""
        meta = {
            "megatron_parallel_info": {
                "tp_rank": 0,
                "tp_size": 2,
                "etp_rank": 1,
                "etp_size": 4,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            ParallelAxis.ETP: AxisInfo(axis_rank=1, axis_size=4),
        }

    def test_sglang_moe_ep(self) -> None:
        """sglang moe_ep_rank/moe_ep_size maps to ParallelAxis.MOE_EP."""
        meta = {
            "sglang_parallel_info": {
                "tp_rank": 0,
                "tp_size": 2,
                "moe_ep_rank": 1,
                "moe_ep_size": 4,
            }
        }
        result = normalize_parallel_info(meta)
        assert result == {
            ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            ParallelAxis.MOE_EP: AxisInfo(axis_rank=1, axis_size=4),
        }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
