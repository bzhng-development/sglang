"""Temporary unit coverage for early beam-search plumbing.

These tests exercise the beam payload hand-off without depending on the yet-to-be
finished scheduler beam expansion, so they should be replaced once the final
beam integration lands.
"""

from types import SimpleNamespace

from sglang.srt.beam_search import BeamSearchOutput, BeamSearchSequence
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # type: ignore
from sglang.srt.managers.io_struct import BatchStrOut, BatchTokenIDOut
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _DummyTokenizer:
    def __init__(self):
        self.calls: list[list[list[int]]] = []

    def batch_decode(
        self,
        token_batches,
        *,
        skip_special_tokens=None,
        spaces_between_special_tokens=None,
    ):
        # Record every decode call so we can verify the beam path is decoded once.
        self.calls.append(token_batches)
        return [f"decoded:{tokens}" for tokens in token_batches]


class _DummyEvent:
    def __init__(self):
        self.triggered = False

    def set(self):
        self.triggered = True


def _make_token_batch(**overrides) -> BatchTokenIDOut:
    defaults = dict(
        rids=["req-0"],
        finished_reasons=[None],
        decoded_texts=[""],
        decode_ids=[[101]],
        read_offsets=[0],
        output_ids=[[101]],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[False],
        no_stop_trim=[True],
        prompt_tokens=[1],
        completion_tokens=[0],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[[]],
        input_token_logprobs_idx=[[]],
        output_token_logprobs_val=[[]],
        output_token_logprobs_idx=[[]],
        input_top_logprobs_val=[[]],
        input_top_logprobs_idx=[[]],
        output_top_logprobs_val=[[]],
        output_top_logprobs_idx=[[]],
        input_token_ids_logprobs_val=[[]],
        input_token_ids_logprobs_idx=[[]],
        output_token_ids_logprobs_val=[[]],
        output_token_ids_logprobs_idx=[[]],
        output_hidden_states=[[]],
        placeholder_tokens_idx=[None],
        placeholder_tokens_val=[None],
        beam_search_output=None,
    )
    defaults.update(overrides)
    return BatchTokenIDOut(**defaults)


def _make_str_batch(**overrides) -> BatchStrOut:
    defaults = dict(
        rids=["req-0"],
        finished_reasons=[None],
        output_strs=["chunk"],
        output_ids=[[201]],
        prompt_tokens=[1],
        completion_tokens=[1],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[[]],
        input_token_logprobs_idx=[[]],
        output_token_logprobs_val=[[]],
        output_token_logprobs_idx=[[]],
        input_top_logprobs_val=[[]],
        input_top_logprobs_idx=[[]],
        output_top_logprobs_val=[[]],
        output_top_logprobs_idx=[[]],
        input_token_ids_logprobs_val=[[]],
        input_token_ids_logprobs_idx=[[]],
        output_token_ids_logprobs_val=[[]],
        output_token_ids_logprobs_idx=[[]],
        output_hidden_states=[[]],
        placeholder_tokens_idx=[None],
        placeholder_tokens_val=[None],
        beam_search_output=None,
    )
    defaults.update(overrides)
    return BatchStrOut(**defaults)


def test_detokenizer_populates_beam_text():
    dummy_tokenizer = _DummyTokenizer()
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager.tokenizer = dummy_tokenizer
    manager.decode_status = {}
    manager.is_tool_call_parser_gpt_oss = False

    beam_seq = BeamSearchSequence(tokens=[10, 11], last_token=11, cum_logprob=0.0)
    batch = _make_token_batch(
        beam_search_output=[BeamSearchOutput(sequences=[beam_seq])]
    )

    result = manager.handle_batch_token_id_out(batch)

    assert result.beam_search_output is not None
    populated = result.beam_search_output[0].sequences[0]
    assert populated.text == "decoded:[10, 11]"
    # Ensure the beam tokens were decoded in a single batched call.
    assert dummy_tokenizer.calls[-1] == [[10, 11]]


def test_tokenizer_manager_meta_includes_beams():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(
        tokenizer_worker_num=1,
        weight_version="test-weight",
        stream_output=False,
        speculative_algorithm=False,
        enable_lora=False,
    )
    manager.rid_to_state = {}
    manager.enable_metrics = False
    manager.dump_requests_folder = None
    manager.crash_dump_folder = None
    manager.lora_registry = SimpleNamespace(release=lambda *_: None)

    state = SimpleNamespace(
        obj=SimpleNamespace(
            stream=False,
            return_logprob=False,
            lora_path=None,
            log_metrics=False,
        ),
        text="",
        output_ids=[],
        last_output_offset=0,
        finished=False,
        created_time=0.0,
        out_list=[],
        event=_DummyEvent(),
    )
    manager.rid_to_state["req-0"] = state

    beam_output = BeamSearchOutput(
        sequences=[
            BeamSearchSequence(tokens=[1], last_token=1, cum_logprob=0.0, text="beam")
        ]
    )
    batch = _make_str_batch(beam_search_output=[beam_output])

    manager._handle_batch_output(batch)

    assert state.event.triggered is True
    out_dict = state.out_list[-1]
    assert out_dict["meta_info"]["beam_search_outputs"] is beam_output


def test_handle_output_by_index_keeps_single_beam_slice():
    beam_outputs = [
        BeamSearchOutput(
            sequences=[
                BeamSearchSequence(tokens=[5], last_token=5, cum_logprob=0.0, text="a")
            ]
        ),
        BeamSearchOutput(
            sequences=[
                BeamSearchSequence(tokens=[6], last_token=6, cum_logprob=0.0, text="b")
            ]
        ),
    ]
    batch = _make_str_batch(
        rids=["req-0", "req-1"],
        finished_reasons=[None, None],
        output_strs=["zero", "one"],
        output_ids=[[100], [200]],
        prompt_tokens=[1, 1],
        completion_tokens=[1, 1],
        cached_tokens=[0, 0],
        spec_verify_ct=[0, 0],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_hidden_states=[[], []],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        beam_search_output=beam_outputs,
    )

    sliced = _handle_output_by_index(batch, 1)

    assert sliced.beam_search_output == [beam_outputs[1]]
    assert sliced.beam_search_output[0] is beam_outputs[1]
    assert sliced.output_strs == ["one"]
