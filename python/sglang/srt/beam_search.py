from dataclasses import dataclass, field
from typing import List, Optional

"""Utilities for beam search support.

Based on vLLM's implementation:
https://github.com/vllm-project/vllm/blob/5f0ec3935a0118fee8cf2764728f765c8cc53d2a/vllm/beam_search.py
"""


@dataclass
class BeamSearchSequence:
    """Beam search sequence state."""

    last_token: int
    tokens: List[int]
    finish: Optional[object] = None
    cum_logprob: float = 0.0
    text: Optional[str] = None

    last_req_pool_idx: int = -1
    prefix: List[int] = field(default_factory=list)
    prefix_len: int = 0
    vid: Optional[int] = None

    def finished(self) -> bool:
        return self.finish is not None

    def check_prefix(self, req) -> bool:
        for j, (beam_token, parent_token) in enumerate(
            zip(self.tokens, req.output_ids)
        ):
            if beam_token == parent_token and j >= len(self.prefix):
                return False
        return True


@dataclass
class BeamSearchList:
    """Tracks completed and in-progress beam sequences."""

    beam_width: int = 0
    req_pool_start_idx: int = -1
    completed: List[BeamSearchSequence] = field(default_factory=list)
    incompleted: List[BeamSearchSequence] = field(default_factory=list)

    def empty(self) -> bool:
        return len(self.completed) + len(self.incompleted) == 0


@dataclass
class BeamSearchOutput:
    """Finalized beam search result."""

    sequences: List[BeamSearchSequence]


def sort_by_beam_search_score(
    sequence: BeamSearchSequence, length_penalty: float = 1.0
) -> float:
    seq_len = len(sequence.tokens)
    if sequence.finished() and seq_len > 1:
        seq_len -= 1
    return sequence.cum_logprob / (seq_len ** length_penalty)
