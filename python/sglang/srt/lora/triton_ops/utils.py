from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.utils import is_cuda, is_sm90_supported


@lru_cache
def supports_pdl(device: Optional[torch.device] = None) -> bool:
    # PDL requires Hopper (SM90) or newer with compatible CUDA
    return bool(is_cuda() and is_sm90_supported(device))
