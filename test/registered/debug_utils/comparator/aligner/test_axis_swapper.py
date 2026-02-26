import sys
from typing import Optional

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.axis_swapper import (
    AxisSwapperPlan,
    compute_axis_swapper_plan,
    execute_axis_swapper_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestComputeAxisSwapperPlan:
    def test_no_dims_returns_none(self) -> None:
        assert compute_axis_swapper_plan(x_dims_str=None, y_dims_str=None) is None
        assert compute_axis_swapper_plan(x_dims_str="t h d", y_dims_str=None) is None
        assert compute_axis_swapper_plan(x_dims_str=None, y_dims_str="t h d") is None

    def test_same_order_returns_none(self) -> None:
        result: Optional[AxisSwapperPlan] = compute_axis_swapper_plan(
            x_dims_str="t h d", y_dims_str="t h d"
        )
        assert result is None

    def test_different_order(self) -> None:
        result: Optional[AxisSwapperPlan] = compute_axis_swapper_plan(
            x_dims_str="t h d", y_dims_str="t d h"
        )
        assert result is not None
        assert result.pattern == "t h d -> t d h"

    def test_name_mismatch_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        result: Optional[AxisSwapperPlan] = compute_axis_swapper_plan(
            x_dims_str="t h d", y_dims_str="t h e"
        )
        assert result is None
        assert "dim name sets differ" in caplog.text

    def test_modifiers_ignored_for_name_extraction(self) -> None:
        result: Optional[AxisSwapperPlan] = compute_axis_swapper_plan(
            x_dims_str="t h(tp) d", y_dims_str="t d h(tp)"
        )
        assert result is not None
        assert result.pattern == "t h d -> t d h"


class TestExecuteAxisSwapperPlan:
    def test_rearrange(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16)
        plan = AxisSwapperPlan(pattern="t h d -> t d h")

        result: torch.Tensor = execute_axis_swapper_plan(tensor=tensor, plan=plan)

        assert result.shape == (4, 16, 8)
        for i in range(4):
            assert torch.equal(
                result[i],
                tensor[i].T,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
