import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.utils import (
    direct_register_custom_op,
    is_flashinfer_available,
    supports_custom_op,
)

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_workspace_manager = None

if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        _flashinfer_comm = comm
    except ImportError:
        logger.warning(
            "flashinfer.comm is not available, falling back to standard "
            "implementation"
        )


class FlashInferWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
        self.ipc_handles = None
        self.world_size = None
        self.rank = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        group=None,
        use_fp32_lamport: bool = False,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        if _flashinfer_comm is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace " "initialization"
            )
            return

        self.cleanup()

        self.ipc_handles, self.workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                max_token_num,
                hidden_dim,
                group=group,
                use_fp32_lamport=use_fp32_lamport,
            )
        )

        self.world_size = world_size
        self.rank = rank
        self.initialized = True

        logger.info(
            f"FlashInfer workspace initialized for rank {rank}, "
            f"world_size {world_size}"
        )

    def cleanup(self):
        """Clean up workspace"""
        if self.initialized and self.ipc_handles is not None:
            try:
                _flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                    self.ipc_handles, group=dist.group.WORLD
                )
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.ipc_handles = None
                self.initialized = False


_workspace_manager = FlashInferWorkspaceManager()


def ensure_workspace_initialized(
    max_token_num: int = 2048, hidden_dim: int = 4096, use_fp32_lamport: bool = False
):
    """Ensure workspace is initialized"""
    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if (
        not _workspace_manager.initialized
        or _workspace_manager.world_size != world_size
    ):
        _workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=use_fp32_lamport,
        )

    return _workspace_manager.initialized


def flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's fused allreduce + residual + RMS norm operation

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug(
            "FlashInfer not available, falling back to standard " "implementation"
        )
        return None, None

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    token_num, hidden_dim = input_tensor.shape

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    _flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        world_size=world_size,
        world_rank=dist.get_rank(),
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=True,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=(_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm),
        allreduce_out=None,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=None,
        scale_out=None,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=None,
    )

    return norm_out, residual_out


def fake_flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)
    return norm_out, residual_out


if supports_custom_op():
    direct_register_custom_op(
        "flashinfer_allreduce_residual_rmsnorm",
        flashinfer_allreduce_residual_rmsnorm,
        mutates_args=["input_tensor", "residual", "weight"],
        fake_impl=fake_flashinfer_allreduce_residual_rmsnorm,
    )


def cleanup_flashinfer_workspace():
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()


def flashinfer_allreduce_residual_rmsnorm_fp4_quant(
    input_tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    input_global_scale: torch.Tensor,
    *,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
):
    """
    FlashInfer fused allreduce + residual add + RMSNorm + FP4 quantization.

    Requirements:
    - world_size > 1
    - flashinfer.comm available and workspace initialized
    - input_tensor: [tokens, hidden_dim]
    - residual: same shape as input_tensor (or None if no residual path)
    - quant_out: uint8 with shape [tokens, hidden_dim // 2]
    - output_scale: NVFP4 block scale buffer (flashinfer-compatible layout)
    - input_global_scale: scalar fp32 tensor (static scale)
    """
    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug("FlashInfer not available for FP4 fused path")
        return None, None

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        logger.debug("Single GPU, skip FP4 fused allreduce")
        return None, None

    # Basic shape/type validations
    assert (
        input_tensor.dim() == 2
    ), f"input_tensor must be 2D, got {input_tensor.dim()}D"
    token_num, hidden_dim = input_tensor.shape
    if residual is not None:
        assert (
            residual.shape == input_tensor.shape
        ), f"residual shape {residual.shape} must match input {input_tensor.shape}"

    assert (
        quant_out.dtype == torch.uint8
    ), f"quant_out must be uint8, got {quant_out.dtype}"
    assert quant_out.shape == (
        token_num,
        hidden_dim // 2,
    ), f"quant_out shape {quant_out.shape} must be (tokens, hidden_dim//2)"

    if input_global_scale.numel() != 1:
        raise ValueError("input_global_scale must be a scalar tensor")
    if input_global_scale.dtype != torch.float32:
        # FlashInfer expects fp32 scale factor
        input_global_scale = input_global_scale.to(torch.float32)

    # Ensure workspace is ready
    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        logger.debug("FlashInfer workspace not available for FP4 fused path")
        return None, None

    # Prepare output buffers for residual/norm to satisfy the API
    # We do not return norm_out here; fused path primarily consumes quant_out.
    residual_out = input_tensor if residual is None else torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    _flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        world_size=world_size,
        world_rank=dist.get_rank(),
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=True,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=(
            _flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant
        ),
        allreduce_out=None,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=output_scale,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=input_global_scale,
        layout_code=None,
    )

    # Return outputs useful to the caller
    if residual is not None:
        return quant_out, residual_out
    else:
        # In no-residual path, residual_out acts as allreduce_out
        return quant_out, residual_out


def fake_flashinfer_allreduce_residual_rmsnorm_fp4_quant(
    input_tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    input_global_scale: torch.Tensor,
    *,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
):
    # No-op fake impl: just return provided buffers
    if residual is not None:
        return quant_out, residual
    else:
        return quant_out, input_tensor


if supports_custom_op():
    direct_register_custom_op(
        "flashinfer_allreduce_residual_rmsnorm_fp4_quant",
        flashinfer_allreduce_residual_rmsnorm_fp4_quant,
        mutates_args=[
            "input_tensor",
            "residual",
            "weight",
            "input_global_scale",
            "quant_out",
            "output_scale",
        ],
        fake_impl=fake_flashinfer_allreduce_residual_rmsnorm_fp4_quant,
    )
