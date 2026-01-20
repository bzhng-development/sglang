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
"""Logging interception to redirect standard logging to loguru."""

import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    """Handler to intercept standard logging and redirect to loguru.

    This allows third-party libraries (vLLM, Uvicorn, etc.) that use
    standard logging to have their logs formatted by loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging_interception(
    intercept_libraries: list[str] | None = None,
    level: int = logging.DEBUG,
):
    """Set up interception of standard logging to redirect to loguru.

    Args:
        intercept_libraries: List of library names to intercept logging from.
            If None, intercepts root logger only.
        level: Minimum logging level to intercept.
    """
    # Intercept root logger
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)

    # Intercept specific libraries if requested
    if intercept_libraries:
        for lib_name in intercept_libraries:
            lib_logger = logging.getLogger(lib_name)
            lib_logger.handlers = [InterceptHandler()]
            lib_logger.propagate = False
