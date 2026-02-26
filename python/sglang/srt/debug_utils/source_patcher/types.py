import types
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class PatchApplicationError(Exception):
    """match text not found or not unique in source."""


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EditSpec(_StrictBase):
    """Specify one edit: either replace the matched text or append lines after it.

    Use ``replacement`` to substitute the matched text (empty string = delete).
    Use ``append`` to keep the matched text and add lines after it.
    These two fields are mutually exclusive.
    """

    match: str
    replacement: str = ""
    append: str = ""

    @model_validator(mode="after")
    def _check_replacement_and_append_exclusive(self) -> "EditSpec":
        if self.replacement.strip() and self.append.strip():
            raise ValueError("'replacement' and 'append' are mutually exclusive")
        return self


class PatchSpec(_StrictBase):
    target: str
    edits: list[EditSpec]
    preamble: str = ""


class PatchConfig(_StrictBase):
    patches: list[PatchSpec]


class PatchState:
    def __init__(
        self, *, target_fn: Callable[..., Any], original_code: types.CodeType
    ) -> None:
        self.target_fn = target_fn
        self.original_code = original_code

    def restore(self) -> None:
        self.target_fn.__code__ = self.original_code
