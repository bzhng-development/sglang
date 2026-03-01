from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sglang.srt.debug_utils.comparator.output_types import GeneralWarning


class WarningSink:
    def __init__(self) -> None:
        self._stack: list[list[GeneralWarning]] = []

    @contextmanager
    def context(self) -> Generator[list[GeneralWarning], None, None]:
        bucket: list[GeneralWarning] = []
        self._stack.append(bucket)
        try:
            yield bucket
        finally:
            popped = self._stack.pop()
            assert popped is bucket

    def add(self, warning: GeneralWarning) -> None:
        if self._stack:
            self._stack[-1].append(warning)


warning_sink = WarningSink()
