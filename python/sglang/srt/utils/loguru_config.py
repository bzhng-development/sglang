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
"""Loguru configuration for SGLang."""

import os
import sys
from typing import Set

import orjson
from loguru import logger

from sglang.srt.environ import envs

# Store seen messages for once-only logging
_seen_warnings: Set[str] = set()
_seen_infos: Set[str] = set()


def warning_once(msg: str) -> None:
    """Log a warning message only once."""
    if msg not in _seen_warnings:
        _seen_warnings.add(msg)
        logger.warning(msg)


def info_once(msg: str) -> None:
    """Log an info message only once."""
    if msg not in _seen_infos:
        _seen_infos.add(msg)
        logger.info(msg)


def _rank_zero_filter(record):
    """Filter to only log on rank 0 in distributed settings."""
    try:
        from sglang.srt.distributed import (
            get_tensor_model_parallel_rank,
            model_parallel_is_initialized,
        )

        if model_parallel_is_initialized() and get_tensor_model_parallel_rank() != 0:
            return False
    except ImportError:
        pass
    return True


def configure_loguru(server_args, prefix: str = ""):
    """Configure loguru logger for SGLang.

    Args:
        server_args: Server arguments containing log_level.
        prefix: Optional prefix to add to log messages.
    """
    # Check for custom logging config
    if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
            raise Exception(
                "Setting SGLANG_LOGGING_CONFIG_PATH from env with "
                f"{SGLANG_LOGGING_CONFIG_PATH} but it does not exist!"
            )
        with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = orjson.loads(file.read())
        # For custom configs, we still need basic loguru setup
        # Users can customize via the JSON config for standard logging interception
        logger.remove()
        logger.add(
            sys.stderr,
            level=custom_config.get("root", {}).get("level", "INFO"),
            format="[{time:YYYY-MM-DD HH:mm:ss"
            + (".SSS" if envs.SGLANG_LOG_MS.get() else "")
            + prefix
            + "}] {message}",
        )
        return

    # Build format string
    maybe_ms = ".SSS" if envs.SGLANG_LOG_MS.get() else ""
    format_str = "[{time:YYYY-MM-DD HH:mm:ss" + maybe_ms + prefix + "}] {message}"

    # Remove default handler and add our configured one
    logger.remove()
    logger.add(
        sys.stderr,
        level=server_args.log_level.upper(),
        format=format_str,
        colorize=True,
    )


def get_logger():
    """Get the loguru logger instance."""
    return logger
