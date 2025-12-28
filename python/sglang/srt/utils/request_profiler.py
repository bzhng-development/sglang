"""
Request CPU Profiler for HTTP Request Handling

This module profiles the CPU time spent in forming/processing chat completion requests,
separate from the GPU inference time. It tracks stages like:
- Request validation
- Message processing (chat template application)
- Tokenization
- Request conversion
"""

import cProfile
import io
import logging
import pstats
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class StageTimings:
    """Stores timing information for a single profiling session"""

    request_id: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    stages: Dict[str, float] = field(default_factory=dict)
    stage_start_times: Dict[str, float] = field(default_factory=dict)
    cprofile_stats: Optional[str] = None

    @property
    def total_time_ms(self) -> float:
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_time_ms": self.total_time_ms,
            "stages_ms": {k: v * 1000 for k, v in self.stages.items()},
            "cprofile_stats": self.cprofile_stats,
        }


class RequestProfiler:
    """
    Profiles CPU time spent in HTTP request handling for chat completions.

    Usage:
        profiler = RequestProfiler()
        profiler.enable()

        # In request handler:
        with profiler.profile_request("req-123") as session:
            with session.stage("validation"):
                validate_request()
            with session.stage("message_processing"):
                process_messages()

        # Get results
        results = profiler.get_results()
    """

    _instance: Optional["RequestProfiler"] = None
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
        self._use_cprofile = False
        self._results: Dict[str, StageTimings] = {}
        self._results_lock = threading.Lock()
        self._max_results = 1000  # Keep last N results

    @classmethod
    def get_instance(cls) -> "RequestProfiler":
        """Get or create the singleton instance"""
        return cls()

    def enable(self, use_cprofile: bool = False):
        """Enable request profiling

        Args:
            use_cprofile: If True, also collect cProfile statistics for detailed
                         function-level profiling. More overhead but more detail.
        """
        self._enabled = True
        self._use_cprofile = use_cprofile
        logger.info(f"Request CPU profiler enabled (cprofile={use_cprofile})")

    def disable(self):
        """Disable request profiling"""
        self._enabled = False
        logger.info("Request CPU profiler disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def clear_results(self):
        """Clear all stored profiling results"""
        with self._results_lock:
            self._results.clear()

    def get_results(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling results

        Args:
            request_id: If specified, return results for this request only.
                       Otherwise return all results.
        """
        with self._results_lock:
            if request_id:
                if request_id in self._results:
                    return self._results[request_id].to_dict()
                return {}
            return {k: v.to_dict() for k, v in self._results.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate statistics across all profiled requests"""
        with self._results_lock:
            if not self._results:
                return {"count": 0}

            all_stages: Dict[str, List[float]] = {}
            total_times: List[float] = []

            for timing in self._results.values():
                total_times.append(timing.total_time_ms)
                for stage, duration in timing.stages.items():
                    if stage not in all_stages:
                        all_stages[stage] = []
                    all_stages[stage].append(duration * 1000)

            def calc_stats(values: List[float]) -> Dict[str, float]:
                if not values:
                    return {}
                sorted_vals = sorted(values)
                return {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": sorted_vals[len(sorted_vals) // 2],
                    "p90": sorted_vals[int(len(sorted_vals) * 0.9)],
                    "p99": sorted_vals[int(len(sorted_vals) * 0.99)],
                }

            return {
                "count": len(self._results),
                "total_time_ms": calc_stats(total_times),
                "stages_ms": {
                    stage: calc_stats(times) for stage, times in all_stages.items()
                },
            }

    @contextmanager
    def profile_request(self, request_id: str):
        """Context manager for profiling a single request

        Args:
            request_id: Unique identifier for this request

        Yields:
            ProfileSession object with stage() method
        """
        if not self._enabled:
            yield _NoOpProfileSession()
            return

        session = _ProfileSession(
            request_id=request_id,
            use_cprofile=self._use_cprofile,
        )

        try:
            yield session
        finally:
            session.finish()
            self._store_result(session.timings)

    def _store_result(self, timings: StageTimings):
        """Store profiling result, evicting old results if necessary"""
        with self._results_lock:
            self._results[timings.request_id] = timings

            # Evict oldest results if we exceed max
            if len(self._results) > self._max_results:
                oldest_keys = sorted(
                    self._results.keys(),
                    key=lambda k: self._results[k].start_time,
                )[: len(self._results) - self._max_results]
                for key in oldest_keys:
                    del self._results[key]


class _ProfileSession:
    """Active profiling session for a single request"""

    def __init__(self, request_id: str, use_cprofile: bool = False):
        self.timings = StageTimings(request_id=request_id)
        self.use_cprofile = use_cprofile
        self._cprofile: Optional[cProfile.Profile] = None

        if use_cprofile:
            self._cprofile = cProfile.Profile()
            self._cprofile.enable()

    @contextmanager
    def stage(self, name: str):
        """Context manager for timing a specific stage

        Args:
            name: Name of the stage (e.g., "validation", "tokenization")
        """
        start = time.perf_counter()
        self.timings.stage_start_times[name] = start
        try:
            yield
        finally:
            end = time.perf_counter()
            self.timings.stages[name] = end - start

    def finish(self):
        """Finalize the profiling session"""
        self.timings.end_time = time.perf_counter()

        if self._cprofile:
            self._cprofile.disable()
            stream = io.StringIO()
            stats = pstats.Stats(self._cprofile, stream=stream)
            stats.sort_stats("cumulative")
            stats.print_stats(30)  # Top 30 functions
            self.timings.cprofile_stats = stream.getvalue()


class _NoOpProfileSession:
    """No-op session when profiling is disabled"""

    @contextmanager
    def stage(self, name: str):
        yield


# Decorator for profiling specific functions
def profile_stage(stage_name: str):
    """Decorator to profile a function as a named stage

    Usage:
        @profile_stage("tokenization")
        def tokenize_input(text):
            ...
    """

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            profiler = RequestProfiler.get_instance()
            if not profiler.is_enabled():
                return func(*args, **kwargs)

            # Try to get request_id from kwargs or first arg
            request_id = kwargs.get("request_id", "unknown")

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger.debug(f"[{request_id}] {stage_name}: {duration*1000:.2f}ms")

        return wrapper  # type: ignore

    return decorator


# Convenience function to get the global profiler
def get_request_profiler() -> RequestProfiler:
    """Get the global request profiler instance"""
    return RequestProfiler.get_instance()
