"""
Server-wide Tracing for FastAPI with Perfetto/Chrome Trace Export

This module provides comprehensive tracing of the entire FastAPI server,
capturing all HTTP requests and their internal processing to identify
performance bottlenecks. Output is in Chrome Trace format which can be
viewed in Perfetto UI (https://ui.perfetto.dev).

Usage:
    1. Call /start_server_trace to begin tracing
    2. Make requests to the server
    3. Call /stop_server_trace to stop and get the trace file path
    4. Open the trace file in Perfetto UI
"""

import gzip
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Chrome Trace Event format

    See: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
    """

    name: str
    cat: str  # category
    ph: str  # phase: "X" complete, "B" begin, "E" end, "i" instant, "M" metadata
    pid: int  # process id
    tid: int  # thread id
    ts: float  # timestamp in microseconds
    dur: Optional[float] = None  # duration in microseconds (for "X" events)
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts,
        }
        if self.dur is not None:
            d["dur"] = self.dur
        if self.args:
            d["args"] = self.args
        return d


class ServerTracer:
    """
    Server-wide tracer that captures all HTTP requests and exports to Chrome Trace format.

    Supports:
    - All HTTP endpoints (not just chat completions)
    - Nested spans for internal processing stages
    - Thread-safe collection from async handlers
    - Export to gzipped Chrome trace JSON
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
        self._events: List[TraceEvent] = []
        self._events_lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._start_time_perf: Optional[float] = None

        # Track active spans per thread for nesting
        self._active_spans: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

        # Request ID to thread ID mapping for async correlation
        self._request_threads: Dict[str, int] = {}

        # Thread ID counter for virtual threads (async tasks)
        self._next_tid = 1000
        self._tid_lock = threading.Lock()

        # Output settings
        self._output_dir = "/tmp"
        self._trace_id: Optional[str] = None

        # Process info
        self._pid = os.getpid()

    @classmethod
    def get_instance(cls) -> "ServerTracer":
        return cls()

    def enable(self, output_dir: Optional[str] = None, trace_id: Optional[str] = None):
        """Start tracing all server requests"""
        with self._events_lock:
            self._events.clear()
            self._active_spans.clear()
            self._request_threads.clear()

        self._output_dir = output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        self._trace_id = trace_id or f"server-trace-{int(time.time())}"
        self._start_time = time.time()
        self._start_time_perf = time.perf_counter()
        self._enabled = True

        # Add metadata event
        self._add_metadata_event("process_name", {"name": "sglang-server"})
        self._add_metadata_event("thread_name", {"name": "main"}, tid=0)

        logger.info(f"Server tracing started. Trace ID: {self._trace_id}")

    def disable(self) -> Optional[str]:
        """Stop tracing and export to file. Returns the output file path."""
        if not self._enabled:
            return None

        self._enabled = False
        output_path = self._export_trace()
        logger.info(f"Server tracing stopped. Trace exported to: {output_path}")
        return output_path

    def is_enabled(self) -> bool:
        return self._enabled

    def _get_timestamp_us(self) -> float:
        """Get current timestamp in microseconds relative to trace start"""
        if self._start_time_perf is None:
            return 0
        return (time.perf_counter() - self._start_time_perf) * 1_000_000

    def _get_tid(self, request_id: Optional[str] = None) -> int:
        """Get thread ID, creating a virtual one for async requests if needed"""
        real_tid = threading.current_thread().ident or 0

        if request_id:
            # For request-scoped operations, use a consistent virtual TID
            if request_id not in self._request_threads:
                with self._tid_lock:
                    self._request_threads[request_id] = self._next_tid
                    self._next_tid += 1
            return self._request_threads[request_id]

        return real_tid

    def _add_metadata_event(self, name: str, args: Dict[str, Any], tid: int = 0):
        """Add a metadata event (process/thread name)"""
        event = TraceEvent(
            name=name,
            cat="__metadata",
            ph="M",
            pid=self._pid,
            tid=tid,
            ts=0,
            args=args,
        )
        with self._events_lock:
            self._events.append(event)

    def trace_request_start(
        self,
        request_id: str,
        method: str,
        path: str,
        args: Optional[Dict[str, Any]] = None,
    ):
        """Start tracing an HTTP request"""
        if not self._enabled:
            return

        tid = self._get_tid(request_id)
        ts = self._get_timestamp_us()

        # Add thread name metadata for this request
        self._add_metadata_event(
            "thread_name",
            {"name": f"{method} {path}"},
            tid=tid,
        )

        # Begin event
        event = TraceEvent(
            name=f"{method} {path}",
            cat="http",
            ph="B",
            pid=self._pid,
            tid=tid,
            ts=ts,
            args=args or {},
        )

        with self._events_lock:
            self._events.append(event)
            self._active_spans[tid].append((f"{method} {path}", ts))

    def trace_request_end(
        self,
        request_id: str,
        status_code: int,
        args: Optional[Dict[str, Any]] = None,
    ):
        """End tracing an HTTP request"""
        if not self._enabled:
            return

        tid = self._get_tid(request_id)
        ts = self._get_timestamp_us()

        event_args = {"status_code": status_code}
        if args:
            event_args.update(args)

        event = TraceEvent(
            name="",  # End events don't need name
            cat="http",
            ph="E",
            pid=self._pid,
            tid=tid,
            ts=ts,
            args=event_args,
        )

        with self._events_lock:
            self._events.append(event)
            if self._active_spans[tid]:
                self._active_spans[tid].pop()

    def trace_span_start(
        self,
        name: str,
        request_id: Optional[str] = None,
        category: str = "function",
        args: Optional[Dict[str, Any]] = None,
    ):
        """Start a span within a request"""
        if not self._enabled:
            return

        tid = self._get_tid(request_id)
        ts = self._get_timestamp_us()

        event = TraceEvent(
            name=name,
            cat=category,
            ph="B",
            pid=self._pid,
            tid=tid,
            ts=ts,
            args=args or {},
        )

        with self._events_lock:
            self._events.append(event)
            self._active_spans[tid].append((name, ts))

    def trace_span_end(
        self,
        name: str,
        request_id: Optional[str] = None,
        category: str = "function",
        args: Optional[Dict[str, Any]] = None,
    ):
        """End a span within a request"""
        if not self._enabled:
            return

        tid = self._get_tid(request_id)
        ts = self._get_timestamp_us()

        event = TraceEvent(
            name=name,
            cat=category,
            ph="E",
            pid=self._pid,
            tid=tid,
            ts=ts,
            args=args or {},
        )

        with self._events_lock:
            self._events.append(event)
            if self._active_spans[tid]:
                self._active_spans[tid].pop()

    def trace_complete_event(
        self,
        name: str,
        duration_us: float,
        request_id: Optional[str] = None,
        category: str = "function",
        args: Optional[Dict[str, Any]] = None,
    ):
        """Add a complete event (with known duration)"""
        if not self._enabled:
            return

        tid = self._get_tid(request_id)
        ts = self._get_timestamp_us() - duration_us  # Start time

        event = TraceEvent(
            name=name,
            cat=category,
            ph="X",
            pid=self._pid,
            tid=tid,
            ts=ts,
            dur=duration_us,
            args=args or {},
        )

        with self._events_lock:
            self._events.append(event)

    def trace_instant_event(
        self,
        name: str,
        request_id: Optional[str] = None,
        category: str = "instant",
        scope: str = "t",  # "g" global, "p" process, "t" thread
        args: Optional[Dict[str, Any]] = None,
    ):
        """Add an instant event (marker)"""
        if not self._enabled:
            return

        tid = self._get_tid(request_id)
        ts = self._get_timestamp_us()

        event = TraceEvent(
            name=name,
            cat=category,
            ph="i",
            pid=self._pid,
            tid=tid,
            ts=ts,
            args={"s": scope, **(args or {})},
        )

        with self._events_lock:
            self._events.append(event)

    def _export_trace(self) -> str:
        """Export collected events to Chrome trace JSON format"""
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        output_path = os.path.join(self._output_dir, f"{self._trace_id}.trace.json.gz")

        with self._events_lock:
            trace_data = {
                "traceEvents": [e.to_dict() for e in self._events],
                "displayTimeUnit": "ms",
                "metadata": {
                    "trace_id": self._trace_id,
                    "start_time": self._start_time,
                    "hostname": (
                        os.uname().nodename if hasattr(os, "uname") else "unknown"
                    ),
                },
            }

        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(trace_data, f)

        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Get current tracing statistics"""
        with self._events_lock:
            return {
                "enabled": self._enabled,
                "trace_id": self._trace_id,
                "event_count": len(self._events),
                "active_requests": len(self._request_threads),
                "output_dir": self._output_dir,
            }


class ServerTracingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that traces all HTTP requests"""

    def __init__(self, app, tracer: Optional[ServerTracer] = None):
        super().__init__(app)
        self.tracer = tracer or ServerTracer.get_instance()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.tracer.is_enabled():
            return await call_next(request)

        # Generate request ID
        request_id = f"req-{id(request)}-{time.perf_counter_ns()}"

        # Extract request info
        method = request.method
        path = request.url.path

        # Start request trace
        self.tracer.trace_request_start(
            request_id=request_id,
            method=method,
            path=path,
            args={
                "query": str(request.query_params) if request.query_params else None,
            },
        )

        # Store request_id in request state for downstream use
        request.state.trace_request_id = request_id

        try:
            response = await call_next(request)

            self.tracer.trace_request_end(
                request_id=request_id,
                status_code=response.status_code,
            )

            return response
        except Exception as e:
            self.tracer.trace_request_end(
                request_id=request_id,
                status_code=500,
                args={"error": str(e)},
            )
            raise


def get_server_tracer() -> ServerTracer:
    """Get the global server tracer instance"""
    return ServerTracer.get_instance()


# Context manager for tracing spans
class trace_span:
    """Context manager for tracing a span within a request

    Usage:
        with trace_span("my_operation", request_id):
            do_something()
    """

    def __init__(
        self,
        name: str,
        request_id: Optional[str] = None,
        category: str = "function",
        args: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.request_id = request_id
        self.category = category
        self.args = args
        self.tracer = get_server_tracer()

    def __enter__(self):
        self.tracer.trace_span_start(
            self.name,
            self.request_id,
            self.category,
            self.args,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracer.trace_span_end(
            self.name,
            self.request_id,
            self.category,
        )
        return False
