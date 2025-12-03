# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FP8 GEMM backend configuration.

This module provides a server argument-based configuration for FP8 GEMM backends,
replacing the previous environment variable-based approach.

The supported backends are:
    - auto: Automatically select the best backend based on hardware and availability
    - flashinfer: Use FlashInfer kernels (optimal for Blackwell GPUs)
    - cutlass: Use CUTLASS kernels (optimal for Hopper/Blackwell GPUs)
    - deep_gemm: Use DeepGEMM kernels (JIT-compiled, good for block-wise FP8)
    - triton: Use Triton kernels (fallback, widely compatible)
    - aiter: Use AITER kernels (AMD GPUs only)

Usage:
    Launch the server with --fp8-gemm-runner-backend <backend>:
        python -m sglang.launch_server --model-path <model> --fp8-gemm-runner-backend flashinfer

Note:
    The environment variables SGLANG_ENABLE_FLASHINFER_FP8_GEMM and
    SGLANG_SUPPORT_CUTLASS_BLOCK_FP8 are deprecated. Please use the
    --fp8-gemm-runner-backend server argument instead.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class Fp8GemmRunnerBackend(Enum):
    """Enum for FP8 GEMM runner backend selection.

    This enum mirrors the structure of MoeRunnerBackend for consistency.
    """

    AUTO = "auto"
    FLASHINFER = "flashinfer"
    CUTLASS = "cutlass"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    AITER = "aiter"

    def is_auto(self) -> bool:
        return self == Fp8GemmRunnerBackend.AUTO

    def is_flashinfer(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER

    def is_cutlass(self) -> bool:
        return self == Fp8GemmRunnerBackend.CUTLASS

    def is_deep_gemm(self) -> bool:
        return self == Fp8GemmRunnerBackend.DEEP_GEMM

    def is_triton(self) -> bool:
        return self == Fp8GemmRunnerBackend.TRITON

    def is_aiter(self) -> bool:
        return self == Fp8GemmRunnerBackend.AITER


# Global configuration state
FP8_GEMM_RUNNER_BACKEND: Optional[Fp8GemmRunnerBackend] = None


def initialize_fp8_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP8 GEMM configuration from server arguments.

    Args:
        server_args: Server arguments containing fp8_gemm_runner_backend setting.
    """
    global FP8_GEMM_RUNNER_BACKEND

    FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend(server_args.fp8_gemm_runner_backend)

    logger.info(f"FP8 GEMM runner backend set to: {FP8_GEMM_RUNNER_BACKEND.value}")


def get_fp8_gemm_runner_backend() -> Fp8GemmRunnerBackend:
    """Get the current FP8 GEMM runner backend.

    Returns:
        The configured FP8 GEMM runner backend. Defaults to AUTO if not initialized.
    """
    global FP8_GEMM_RUNNER_BACKEND

    if FP8_GEMM_RUNNER_BACKEND is None:
        logger.warning(
            "FP8_GEMM_RUNNER_BACKEND is not initialized, using auto backend. "
            "Consider setting --fp8-gemm-runner-backend explicitly."
        )
        FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend.AUTO

    return FP8_GEMM_RUNNER_BACKEND


# Choices for server argument validation
FP8_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "flashinfer",
    "cutlass",
    "deep_gemm",
    "triton",
    "aiter",
]


def add_fp8_gemm_runner_backend_choices(choices: list) -> None:
    """Allow external code to add more backend choices.

    Args:
        choices: List of additional backend choices to add.
    """
    FP8_GEMM_RUNNER_BACKEND_CHOICES.extend(choices)
