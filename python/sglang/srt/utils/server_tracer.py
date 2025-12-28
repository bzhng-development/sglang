"""
Server-wide CPU Tracing for FastAPI with Perfetto/Chrome Trace Export

This module provides comprehensive CPU profiling of the entire FastAPI server
using torch.profiler to capture actual Python function calls. Output is in
Chrome Trace format which can be viewed in Perfetto UI (https://ui.perfetto.dev).

Usage:
    1. Call /start_server_trace to begin tracing
    2. Make requests to the server
    3. Call /stop_server_trace to stop and get the trace file path
    4. Open the trace file in Perfetto UI
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class ServerTracer:
    """
    Server-wide CPU tracer that uses torch.profiler to capture actual function calls.

    This wraps torch.profiler.profile() to capture CPU activity including:
    - Python function calls with stack traces
    - PyTorch operations
    - All code paths during request handling
    """

    _instance: Optional["ServerTracer"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self._enabled = False
        self._profiler: Optional[torch.profiler.profile] = None
        self._output_dir = "/tmp"
        self._trace_id: Optional[str] = None
        self._with_stack = True
        self._record_shapes = False
        self._request_count = 0
        self._start_time: Optional[float] = None

    @classmethod
    def get_instance(cls) -> "ServerTracer":
        return cls()

    def enable(
        self,
        output_dir: Optional[str] = None,
        trace_id: Optional[str] = None,
        with_stack: bool = True,
        record_shapes: bool = False,
    ):
        """Start server-wide CPU profiling using torch.profiler.

        Args:
            output_dir: Directory to save trace file
            trace_id: Custom trace identifier
            with_stack: Capture Python stack traces (recommended for seeing function calls)
            record_shapes: Record tensor shapes (adds overhead)
        """
        if self._enabled:
            logger.warning("Server tracing already enabled")
            return

        self._output_dir = output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        self._trace_id = trace_id or f"server-trace-{int(time.time())}"
        self._with_stack = with_stack
        self._record_shapes = record_shapes
        self._request_count = 0
        self._start_time = time.time()

        # Create and start the torch profiler
        self._profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_memory=False,
        )
        self._profiler.__enter__()
        self._enabled = True

        logger.info(
            f"Server CPU tracing started. Trace ID: {self._trace_id}, "
            f"with_stack={with_stack}, record_shapes={record_shapes}"
        )

    def disable(self) -> Optional[str]:
        """Stop tracing and export to Chrome trace file. Returns the output path."""
        if not self._enabled or self._profiler is None:
            return None

        self._enabled = False

        # Stop the profiler
        self._profiler.__exit__(None, None, None)

        # Export to Chrome trace format
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(self._output_dir, f"{self._trace_id}.trace.json.gz")

        self._profiler.export_chrome_trace(output_path)

        duration = time.time() - self._start_time if self._start_time else 0
        logger.info(
            f"Server CPU tracing stopped. Duration: {duration:.1f}s, "
            f"Requests: {self._request_count}, Trace: {output_path}"
        )

        self._profiler = None
        return output_path

    def is_enabled(self) -> bool:
        return self._enabled

    def increment_request_count(self):
        """Called by middleware to track request count"""
        if self._enabled:
            self._request_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current tracing statistics"""
        return {
            "enabled": self._enabled,
            "trace_id": self._trace_id,
            "output_dir": self._output_dir,
            "request_count": self._request_count,
            "with_stack": self._with_stack,
            "record_shapes": self._record_shapes,
            "duration_s": (
                time.time() - self._start_time
                if self._start_time and self._enabled
                else None
            ),
        }


# Simple middleware that just tracks request count
# The actual profiling is done by torch.profiler which captures everything
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class ServerTracingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks requests while torch.profiler captures function calls"""

    def __init__(self, app, tracer: Optional[ServerTracer] = None):
        super().__init__(app)
        self.tracer = tracer or ServerTracer.get_instance()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self.tracer.is_enabled():
            self.tracer.increment_request_count()
            # Add a record_function marker so requests are visible in the trace
            with torch.profiler.record_function(
                f"HTTP {request.method} {request.url.path}"
            ):
                return await call_next(request)
        return await call_next(request)


def get_server_tracer() -> ServerTracer:
    """Get the global server tracer instance"""
    return ServerTracer.get_instance()
