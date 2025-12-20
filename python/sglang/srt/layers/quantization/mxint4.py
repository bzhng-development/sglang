# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MX-INT4 (Microscaling INT4) quantization for MoE layers.

This module provides support for MX-INT4 quantized MoE models using the
TRT-LLM Gen kernels via FlashInfer. MX-INT4 uses:
- INT4 weights packed into uint8 (2 values per byte)
- bfloat16 scales with 32-element block size
- BlockMajorK weight layout optimized for SM100+ (Blackwell)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import (
    is_flashinfer_available,
    log_info_on_rank0,
    next_power_of_2,
    round_up,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

# MX-INT4 block size (elements per scale)
MXINT4_BLOCK_SIZE = 32


class MxInt4Config(QuantizationConfig):
    """Configuration for MX-INT4 quantization.

    MX-INT4 (Microscaling INT4) is a block-scaled INT4 format optimized for
    NVIDIA Blackwell (SM100+) GPUs. It uses:
    - Symmetric INT4 quantization (range [-8, 7])
    - bfloat16 scales with 32-element blocks
    - BlockMajorK weight layout for tensor core efficiency
    """

    def __init__(
        self,
        is_checkpoint_mxint4_serialized: bool = True,
    ):
        super().__init__()
        self.is_checkpoint_mxint4_serialized = is_checkpoint_mxint4_serialized
        self.block_size = MXINT4_BLOCK_SIZE

    @classmethod
    def get_name(cls) -> str:
        return "mxint4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        """Auto-detect MX-INT4 from compressed-tensors config.

        MX-INT4 can be detected from compressed-tensors config when:
        - quant_method is "compressed-tensors" or "compressed_tensors"
        - config_groups contain weights with:
          - num_bits == 4
          - group_size == 32 (MX-INT4 block size)
          - type == "int"
          - symmetric == True
        """
        if hf_quant_cfg is None:
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        # Check if it's compressed-tensors format
        if quant_method not in ("compressed-tensors", "compressed_tensors"):
            return None

        # Check config_groups for MX-INT4 characteristics
        config_groups = hf_quant_cfg.get("config_groups", {})
        if not config_groups:
            return None

        # Check if any config group matches MX-INT4 specs
        for group_name, group_config in config_groups.items():
            if not isinstance(group_config, dict):
                continue

            weights_config = group_config.get("weights")
            if not isinstance(weights_config, dict):
                continue

            # Check MX-INT4 characteristics
            num_bits = weights_config.get("num_bits")
            group_size = weights_config.get("group_size")
            weight_type = weights_config.get("type", "").lower()
            symmetric = weights_config.get("symmetric", False)

            # MX-INT4 uses: INT4, group_size=32, symmetric=True
            if (
                num_bits == 4
                and group_size == MXINT4_BLOCK_SIZE
                and weight_type == "int"
                and symmetric is True
            ):
                return cls.get_name()

        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MxInt4Config":
        """Create MX-INT4 config from quantization config.

        Supports both:
        1. Direct MX-INT4 config: {"quant_method": "mxint4", ...}
        2. Compressed-tensors config: {"quant_method": "compressed-tensors", "config_groups": {...}}
        """
        # Try to get quant_method, but handle compressed-tensors format
        quant_method = config.get("quant_method", "").lower()

        # Check if it's explicitly MX-INT4
        if "mxint4" in quant_method:
            is_checkpoint_mxint4_serialized = True
        # Check if it's compressed-tensors format that matches MX-INT4
        elif quant_method in ("compressed-tensors", "compressed_tensors"):
            # Verify it matches MX-INT4 characteristics
            config_groups = config.get("config_groups", {})
            is_mxint4_format = False
            for group_config in config_groups.values():
                if not isinstance(group_config, dict):
                    continue
                weights_config = group_config.get("weights", {})
                if (
                    isinstance(weights_config, dict)
                    and weights_config.get("num_bits") == 4
                    and weights_config.get("group_size") == MXINT4_BLOCK_SIZE
                    and weights_config.get("type", "").lower() == "int"
                    and weights_config.get("symmetric") is True
                ):
                    is_mxint4_format = True
                    break
            is_checkpoint_mxint4_serialized = is_mxint4_format
        else:
            is_checkpoint_mxint4_serialized = False

        return cls(is_checkpoint_mxint4_serialized=is_checkpoint_mxint4_serialized)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            # MX-INT4 currently only supports MoE layers
            # Linear layers fall back to unquantized
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            if not get_moe_runner_backend().is_flashinfer_trtllm():
                raise ValueError(
                    "MX-INT4 quantization requires flashinfer_trtllm backend. "
                    "Please set --moe-runner-backend flashinfer_trtllm"
                )
            return MxInt4MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class MxInt4MoEMethod(FusedMoEMethodBase):
    """MoE method for MX-INT4 quantization using TRT-LLM Gen kernels.

    This method handles:
    1. Weight creation with proper shapes for MX-INT4 format
    2. Weight processing (shuffling, scale interleaving) after loading
    3. Forward pass using trtllm_mxint4_block_scale_moe kernel
    """

    def __init__(self, quant_config: MxInt4Config):
        super().__init__()
        self.quant_config = quant_config
        self._cache_permute_indices: Dict[str, torch.Tensor] = {}

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight tensors for MX-INT4 MoE.

        Weight shapes:
        - w13_weight: [num_experts, 2*intermediate_size, hidden_size//2] uint8
        - w13_weight_scale: [num_experts, 2*intermediate_size, hidden_size//32] bf16
        - w2_weight: [num_experts, hidden_size, intermediate_size//2] uint8
        - w2_weight_scale: [num_experts, hidden_size, intermediate_size//32] bf16
        """
        block_size = self.quant_config.block_size

        # Pad sizes to be compatible with kernel requirements
        # MX-INT4 requires alignment to 256 for hidden_size and intermediate_size
        hidden_size_padded = round_up(hidden_size, 256)
        intermediate_size_padded = round_up(intermediate_size_per_partition, 256)

        self.hidden_size = hidden_size
        self.hidden_size_padded = hidden_size_padded
        self.intermediate_size = intermediate_size_per_partition
        self.intermediate_size_padded = intermediate_size_padded
        self.num_experts = num_experts

        weight_dtype = torch.uint8
        scale_dtype = torch.bfloat16

        # GEMM1: [num_experts, 2*intermediate_size, hidden_size//2] (packed INT4)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                2 * intermediate_size_padded,
                hidden_size_padded // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # GEMM1 scales: [num_experts, 2*intermediate_size, hidden_size//32] bf16
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                2 * intermediate_size_padded,
                hidden_size_padded // block_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        # GEMM2: [num_experts, hidden_size, intermediate_size//2] (packed INT4)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                hidden_size_padded,
                intermediate_size_padded // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # GEMM2 scales: [num_experts, hidden_size, intermediate_size//32] bf16
        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                hidden_size_padded,
                intermediate_size_padded // block_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process MX-INT4 weights after loading.

        This involves:
        1. Reordering W1/W3 rows for fused gated activation (swap gate/up order)
        2. Shuffling weights for transposed MMA output
        3. Interleaving scales for SM100 tensor cores
        4. Converting weights to BlockMajorK layout
        """
        if not is_flashinfer_available():
            raise ImportError(
                "FlashInfer is required for MX-INT4 weight processing. "
                "Please install flashinfer."
            )

        from flashinfer import block_scale_interleave
        from flashinfer.utils import (
            get_shuffle_matrix_a_row_indices,
            get_shuffle_matrix_sf_a_row_indices,
        )

        log_info_on_rank0(
            logger,
            "Processing MX-INT4 MoE weights for TRT-LLM Gen kernel...",
        )

        block_size = self.quant_config.block_size
        epilogue_tile_m = 128
        block_k = 128

        w13_weight = layer.w13_weight.data
        w13_weight_scale = layer.w13_weight_scale.data
        w2_weight = layer.w2_weight.data
        w2_weight_scale = layer.w2_weight_scale.data

        num_experts = w13_weight.shape[0]

        # Swap W1 and W3 rows for TRT-LLM Gen swiglu definition
        # TRT-LLM expects [W3, W1] order for gated activation
        def swap_gate_up_rows(x, axis=-2):
            """Swap every pair of rows: [W1, W3] -> [W3, W1]"""
            shape = list(x.shape)
            if axis < 0:
                axis = len(shape) + axis
            new_shape = shape.copy()
            new_shape[axis] = shape[axis] // 2
            new_shape.insert(axis + 1, 2)
            x = x.reshape(*new_shape)
            x = x.flip(axis + 1)
            return x.reshape(*shape)

        w13_weight = swap_gate_up_rows(w13_weight, axis=-2)
        w13_weight_scale = swap_gate_up_rows(w13_weight_scale, axis=-2)

        # Process each expert's weights
        gemm1_weights_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_shuffled = []
        gemm2_scales_shuffled = []

        for i in range(num_experts):
            # GEMM1 weight shuffling
            permute_indices = get_shuffle_matrix_a_row_indices(
                w13_weight[i].view(torch.uint8), epilogue_tile_m
            )
            w13_shuffled = (
                w13_weight[i]
                .view(torch.uint8)[permute_indices.to(w13_weight.device)]
                .contiguous()
            )

            # GEMM1 scale shuffling and interleaving
            permute_sf_indices = get_shuffle_matrix_sf_a_row_indices(
                w13_weight_scale[i].view(torch.bfloat16),
                epilogue_tile_m,
                num_elts_per_sf=block_size,
            )
            w13_scale_shuffled = (
                w13_weight_scale[i]
                .view(torch.bfloat16)[permute_sf_indices.to(w13_weight_scale.device)]
                .contiguous()
            )
            w13_scale_interleaved = block_scale_interleave(w13_scale_shuffled)

            # GEMM2 weight shuffling
            permute_indices_w2 = get_shuffle_matrix_a_row_indices(
                w2_weight[i].view(torch.uint8), epilogue_tile_m
            )
            w2_shuffled = (
                w2_weight[i]
                .view(torch.uint8)[permute_indices_w2.to(w2_weight.device)]
                .contiguous()
            )

            # GEMM2 scale shuffling and interleaving
            permute_sf_indices_w2 = get_shuffle_matrix_sf_a_row_indices(
                w2_weight_scale[i].view(torch.bfloat16),
                epilogue_tile_m,
                num_elts_per_sf=block_size,
            )
            w2_scale_shuffled = (
                w2_weight_scale[i]
                .view(torch.bfloat16)[permute_sf_indices_w2.to(w2_weight_scale.device)]
                .contiguous()
            )
            w2_scale_interleaved = block_scale_interleave(w2_scale_shuffled)

            # Convert to BlockMajorK layout
            w13_block_layout = self._convert_to_block_layout(w13_shuffled, block_k)
            w2_block_layout = self._convert_to_block_layout(w2_shuffled, block_k)

            gemm1_weights_shuffled.append(w13_block_layout)
            gemm1_scales_shuffled.append(w13_scale_interleaved)
            gemm2_weights_shuffled.append(w2_block_layout)
            gemm2_scales_shuffled.append(w2_scale_interleaved)

        # Stack and store processed weights
        layer.w13_weight = Parameter(
            torch.stack(gemm1_weights_shuffled), requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            torch.stack(gemm1_scales_shuffled).view(torch.bfloat16), requires_grad=False
        )
        layer.w2_weight = Parameter(
            torch.stack(gemm2_weights_shuffled), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            torch.stack(gemm2_scales_shuffled).view(torch.bfloat16), requires_grad=False
        )

        torch.cuda.empty_cache()

    def _convert_to_block_layout(
        self, weight: torch.Tensor, block_k: int
    ) -> torch.Tensor:
        """Convert weight tensor to BlockMajorK layout.

        Args:
            weight: Input tensor of shape [N, K//2] (packed INT4)
            block_k: Block size for K dimension

        Returns:
            Tensor in BlockMajorK layout
        """
        n, k_packed = weight.shape
        k = k_packed * 2  # Unpacked K dimension

        # Reshape to blocks
        num_k_blocks = k // block_k
        block_k_packed = block_k // 2

        # [N, K//2] -> [N, num_k_blocks, block_k//2]
        weight = weight.reshape(n, num_k_blocks, block_k_packed)
        # [N, num_k_blocks, block_k//2] -> [num_k_blocks, N, block_k//2]
        weight = weight.permute(1, 0, 2)
        # Flatten back
        weight = weight.reshape(-1)

        return weight.contiguous()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        """Store MoE runner config for use in apply."""
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Apply MX-INT4 MoE using standard dispatch interface."""
        return self.apply_with_router_logits(layer, dispatch_output)

    def apply_with_router_logits(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> torch.Tensor:
        """Apply MX-INT4 MoE with router logits (FlashInfer TRT-LLM Gen path).

        This method:
        1. Extracts routing info from dispatch_output
        2. Calls trtllm_mxint4_block_scale_moe kernel
        3. Returns the MoE output
        """
        from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.srt.layers.moe.utils import RoutingMethodType

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        assert TopKOutputChecker.format_is_bypassed(
            topk_output
        ), "MX-INT4 MoE requires bypassed TopK format"

        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported for MX-INT4 MoE"

        # Note: MX-INT4 MoE does not support routing_bias
        if topk_config.correction_bias is not None:
            logger.warning_once(
                "MX-INT4 MoE does not support routing_bias/correction_bias. "
                "It will be ignored."
            )

        routing_method_type = getattr(
            layer, "routing_method_type", RoutingMethodType.Renormalize
        )

        # DeepSeekV3 routing requires float32 router logits
        if routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)

        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            num_tokens = hidden_states.shape[0]
            hidden_size = hidden_states.shape[1]
            output = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

            result = trtllm_mxint4_block_scale_moe(
                routing_logits=router_logits,
                hidden_states=hidden_states,
                gemm1_weights=layer.w13_weight,
                gemm1_weights_scale=layer.w13_weight_scale,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.w2_weight,
                gemm2_weights_scale=layer.w2_weight_scale,
                num_experts=layer.num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=self.intermediate_size_padded,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
                routing_method_type=routing_method_type,
                output=output,
                tune_max_num_tokens=next_power_of_2(num_tokens),
            )

        return result
