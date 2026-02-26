import types
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EditSpec(_StrictBase):
    match: str
    replacement: str


class PatchSpec(_StrictBase):
    target: str
    edits: list[EditSpec]


class PatchState:
    def __init__(
        self, *, target_fn: Callable[..., Any], original_code: types.CodeType
    ) -> None:
        self.target_fn = target_fn
        self.original_code = original_code

    def restore(self) -> None:
        self.target_fn.__code__ = self.original_code
