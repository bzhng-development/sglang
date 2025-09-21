"""Beam search data structures for SGLang."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.srt.managers.schedule_batch import BaseFinishReason


@dataclass
class BeamSearchSequence:
    """Beam search seq: It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user."""

    tokens: List[int]
    last_token: int
    cum_logprob: float
    finish: Optional[BaseFinishReason] = None
    text: Optional[str] = None

    last_req_pool_idx: int = -1
    prev_req_pool_idx: int = -1
    prefix_len: int = 0

    def finished(self) -> bool:
        return self.finish is not None

    def check_prefix(self, req) -> bool:
        if len(req.output_ids) < self.prefix_len:
            return False
        prefix_tokens = self.tokens[: self.prefix_len]
        req_prefix = req.output_ids[: self.prefix_len]
        return prefix_tokens == req_prefix


class BeamSearchList:
    """The temporary status of beam search."""

    def __init__(self, beam_width: int, req_pool_start_idx: int):
        self.beam_width = beam_width
        self.req_pool_start_idx = req_pool_start_idx
        self.completed: List[BeamSearchSequence] = []
        self.incompleted: List[BeamSearchSequence] = []
        # Track the req_to_token_pool rows and associated metadata for active beams.
        self.active_row_indices: List[int] = []
        self.row_info: dict[int, tuple[int, int]] = {}

    def empty(self) -> bool:
        return len(self.completed) == 0 and len(self.incompleted) == 0


@dataclass
class BeamSearchOutput:
    sequences: List[BeamSearchSequence]


def sort_by_beam_search_score(
    seq: BeamSearchSequence, length_penalty: float = 1.0
) -> float:
    effective_length = len(seq.tokens)
    if seq.finished():
        effective_length -= 1
    if effective_length <= 0:
        effective_length = 1
    return seq.cum_logprob / (effective_length**length_penalty)


def beam_search_sequence_to_dict(seq: BeamSearchSequence) -> Dict[str, Any]:
    return {
        "tokens": list(seq.tokens),
        "last_token": seq.last_token,
        "cum_logprob": seq.cum_logprob,
        "finish_reason": seq.finish.to_json() if seq.finish else None,
        "text": seq.text,
        "prefix_len": seq.prefix_len,
    }


def beam_search_output_to_dict(output: BeamSearchOutput) -> Dict[str, Any]:
    return {
        "sequences": [beam_search_sequence_to_dict(seq) for seq in output.sequences]
    }
