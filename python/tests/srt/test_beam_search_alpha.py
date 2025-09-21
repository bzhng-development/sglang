"""Temporary unit coverage for early beam-search plumbing.

These tests exercise the beam payload hand-off without depending on the yet-to-be
finished scheduler beam expansion, so they should be replaced once the final
beam integration lands.
"""

from types import SimpleNamespace
from typing import List

import torch

from sglang.srt.beam_search import (
    BeamSearchList,
    BeamSearchOutput,
    BeamSearchSequence,
    beam_search_output_to_dict,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # type: ignore
from sglang.srt.managers.io_struct import BatchStrOut, BatchTokenIDOut
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams


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
    assert out_dict["meta_info"]["beam_search_outputs"] == beam_search_output_to_dict(
        beam_output
    )


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


class _DummyPenalizer:
    is_required = False

    def cumulate_output_tokens(self, *_args, **_kwargs):
        pass


class _DummySamplingInfo:
    def __init__(self):
        self.penalizer_orchestrator = _DummyPenalizer()

    def filter_batch(self, *_args, **_kwargs):
        pass

    def merge_batch(self, *_args, **_kwargs):
        pass


class _DummySpecAlgorithm:
    def is_eagle(self):
        return False

    def is_standalone(self):
        return False

    def is_lookahead(self):
        return False

    def is_none(self):
        return True


class _StubTreeCache:
    def __init__(self):
        self.finished = []
        self.unfinished = []
        self.beam_batches = []

    def cache_finished_req(self, req):
        self.finished.append(req)

    def cache_unfinished_req(self, req, *_, **__):
        self.unfinished.append(req)

    def cache_finished_beam_search(self, batch):
        self.beam_batches.append(batch)


def _make_scheduler_stub():
    sched = Scheduler.__new__(Scheduler)
    sched.tree_cache = _StubTreeCache()
    sched.enable_overlap = False
    sched.page_size = 1
    sched.abort_requests = []

    def _abort(abort_req):
        sched.abort_requests.append(abort_req)

    sched.abort_request = _abort  # type: ignore[attr-defined]
    return sched


class _DummyReqToTokenPool:
    def __init__(self, rows: int, cols: int):
        self.req_to_token = torch.zeros((rows, cols), dtype=torch.int32)
        self._next_row = 1
        self.freed_rows: List[int] = []

    def write(self, indices, values):
        rows, cols = indices
        self.req_to_token[rows, cols] = values

    def alloc(self, need_rows: int, *_reqs):
        rows = list(range(self._next_row, self._next_row + need_rows))
        self._next_row += need_rows
        return rows

    def free(self, rows):
        if isinstance(rows, int):
            self.freed_rows.append(rows)
        else:
            self.freed_rows.extend(int(r) for r in rows)


class _DummyTokenAllocator:
    page_size = 1

    def __init__(self):
        self._next = 0
        self._capacity = 10_000
        self.freed: List[int] = []
        self.device = "cpu"

    def alloc(self, num_tokens: int):
        start = self._next
        out = torch.arange(start, start + num_tokens, dtype=torch.int64)
        self._next += num_tokens
        return out

    def available_size(self) -> int:
        return max(self._capacity - self._next, 0)

    def free(self, free_index):
        if isinstance(free_index, torch.Tensor):
            self.freed.extend(int(x) for x in free_index.cpu().tolist())
        elif isinstance(free_index, (list, tuple)):
            self.freed.extend(int(x) for x in free_index)
        else:
            self.freed.append(int(free_index))


class _DummyReq:
    def __init__(self, origin_ids, output_ids, beam_list=None):
        self.origin_input_ids = origin_ids
        self.output_ids = output_ids
        self.stream = False
        self.grammar = None
        self.return_logprob = False
        self.return_hidden_states = False
        self.beam_list = beam_list
        self.finished_reason = None
        self.sampling_params = SimpleNamespace(
            max_new_tokens=32,
            stop_token_ids=None,
            stop_strs=[],
            stop_str_max_len=0,
            ignore_eos=False,
        )

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def finished(self) -> bool:
        return self.finished_reason is not None


def _make_schedule_batch(reqs, beam_width, pool_rows=4, pool_cols=8):
    pool = _DummyReqToTokenPool(pool_rows, pool_cols)
    pool._next_row = len(reqs)
    token_allocator = _DummyTokenAllocator()
    model_config = SimpleNamespace(is_encoder_decoder=False, attention_chunk_size=1)
    seq_lens_list = [len(req.origin_input_ids) + len(req.output_ids) for req in reqs]
    if seq_lens_list:
        output_tail = [
            (req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1])
            for req in reqs
        ]
    else:
        output_tail = []
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=token_allocator,
        tree_cache=None,
        is_hybrid=False,
        model_config=model_config,
        forward_mode=ForwardMode.DECODE,
        enable_overlap=False,
        batch_is_full=False,
        beam_width=beam_width,
        sampling_info=_DummySamplingInfo(),
        next_batch_sampling_info=None,
        input_ids=None,
        input_embeds=None,
        token_type_ids=None,
        req_pool_indices=torch.arange(len(reqs), dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens_list, dtype=torch.int64),
        out_cache_loc=None,
        output_ids=(
            torch.tensor(output_tail, dtype=torch.int64)
            if output_tail
            else torch.empty(0, dtype=torch.int64)
        ),
        multimodal_inputs=None,
        seq_lens_sum=sum(seq_lens_list),
        orig_seq_lens=torch.tensor(seq_lens_list, dtype=torch.int32),
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        is_extend_in_batch=False,
        can_run_dp_cuda_graph=False,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        temp_scaled_logprobs=False,
        top_p_normalized_logprobs=False,
        prefix_lens=None,
        extend_lens=None,
        extend_num_tokens=None,
        decoding_reqs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        encoder_cached=None,
        encoder_lens=None,
        encoder_lens_cpu=None,
        encoder_out_cache_loc=None,
        has_stream=False,
        has_grammar=False,
        device="cpu",
        spec_algorithm=_DummySpecAlgorithm(),
        spec_info=None,
        return_hidden_states=False,
        is_prefill_only=False,
        hicache_consumer_index=-1,
        chunked_req=None,
        launch_done=None,
    )
    return batch


def test_schedule_batch_regular_decode_when_beams_missing():
    beam_list = SimpleNamespace(beam_width=1, incompleted=[], req_pool_start_idx=-1)
    req = _DummyReq([1, 2, 3], [4], beam_list)
    batch = _make_schedule_batch([req], beam_width=1)

    batch.prepare_for_decode()

    assert batch.input_ids.tolist() == [4]
    # Original 3-token prompt + 1 generated token -> seq_len 4; decode adds one more -> 5.
    assert batch.seq_lens.tolist() == [5]
    assert batch.req_pool_indices.tolist() == [0]
    assert batch.seq_lens_sum == sum(batch.seq_lens.tolist())


def test_schedule_batch_prepares_beam_rows_when_ready():
    beam = BeamSearchSequence(
        tokens=[1, 2, 3, 6], last_token=6, cum_logprob=0.0, prefix_len=3
    )
    beam.prev_req_pool_idx = 0
    beam_list = SimpleNamespace(beam_width=1, incompleted=[beam], req_pool_start_idx=-1)
    req = _DummyReq([11, 12, 13, 14], [21], beam_list)
    batch = _make_schedule_batch([req], beam_width=1, pool_rows=3, pool_cols=8)
    batch.req_to_token_pool.req_to_token[0, :4] = torch.tensor(
        [100, 101, 102, 103], dtype=torch.int32
    )

    batch.prepare_for_decode()

    assert batch.input_ids.tolist() == [21, 6]
    assert batch.seq_lens.tolist() == [6, 4]
    assert batch.req_pool_indices.tolist() == [0, 1]
    assert batch.seq_lens_sum == sum(batch.seq_lens.tolist())
    assert req.beam_list.req_pool_start_idx == 0
    assert req.beam_list.active_row_indices == [1]
    assert beam.last_req_pool_idx == 1
    assert torch.equal(
        batch.req_to_token_pool.req_to_token[1, : beam.prefix_len],
        batch.req_to_token_pool.req_to_token[0, : beam.prefix_len],
    )


def test_filter_beam_search_batch_remaps_rows():
    beam_width = 2

    def _make_beam(path_prefix: List[int]) -> BeamSearchSequence:
        return BeamSearchSequence(
            tokens=path_prefix,
            last_token=path_prefix[-1],
            cum_logprob=float(len(path_prefix)),
        )

    beams_req0 = [_make_beam([9, 10, 11]), _make_beam([9, 12, 13])]
    beams_req1 = [_make_beam([21, 22, 23]), _make_beam([21, 24, 25])]

    beam_list0 = BeamSearchList(beam_width=beam_width, req_pool_start_idx=-1)
    beam_list0.incompleted = beams_req0
    beam_list1 = BeamSearchList(beam_width=beam_width, req_pool_start_idx=-1)
    beam_list1.incompleted = beams_req1

    req0 = _DummyReq([1, 2, 3], [30], beam_list0)
    req1 = _DummyReq([4, 5, 6], [40], beam_list1)

    batch = _make_schedule_batch([req0, req1], beam_width=beam_width, pool_rows=10)
    for idx, req in enumerate(batch.reqs):
        req.req_pool_idx = idx

    batch.prepare_for_decode()

    assert req0.beam_list.req_pool_start_idx == 0
    assert req1.beam_list.req_pool_start_idx == 3

    req0.finished_reason = FINISH_LENGTH(length=0)
    batch.filter_batch()

    assert len(batch.reqs) == 1
    kept_req = batch.reqs[0]
    assert kept_req is req1
    assert kept_req.beam_list.req_pool_start_idx == 0
    assert [
        beam.last_req_pool_idx for beam in kept_req.beam_list.incompleted[:beam_width]
    ] == [1, 2]

    assert batch.req_pool_indices.tolist() == [1, 4, 5]
    actual_pool_rows = [
        batch.req_pool_indices[idx]
        for idx in [kept_req.beam_list.req_pool_start_idx]
        + [
            beam.last_req_pool_idx
            for beam in kept_req.beam_list.incompleted[:beam_width]
        ]
    ]
    expected_pool_rows = [kept_req.req_pool_idx] + kept_req.beam_list.active_row_indices
    assert actual_pool_rows == expected_pool_rows


def test_process_batch_result_beam_search_updates_incompleted():
    sched = _make_scheduler_stub()
    sampling_params = SamplingParams(max_new_tokens=4)
    sampling_params.normalize(None)
    req = Req(
        "req-1",
        "",
        [1, 2],
        sampling_params,
        return_logprob=True,
        top_logprobs_num=4,
        vocab_size=1000,
    )
    req.output_ids = [3]

    beam_list = BeamSearchList(beam_width=2, req_pool_start_idx=0)
    beam_list.incompleted = [
        BeamSearchSequence(
            tokens=[1, 2, 5],
            last_token=5,
            cum_logprob=-0.5,
            prefix_len=2,
        ),
        BeamSearchSequence(
            tokens=[1, 2, 6],
            last_token=6,
            cum_logprob=-0.3,
            prefix_len=2,
        ),
    ]
    req.beam_list = beam_list

    next_token_ids = [7, 8, 9]
    next_token_logprobs = [-0.2, -0.1, -0.05]
    logits_output = LogitsProcessorOutput(
        next_token_logits=torch.zeros((3, 1)),
        next_token_logprobs=torch.tensor(next_token_logprobs),
        next_token_top_logprobs_val=[
            [-0.2, -0.4],
            [-0.1, -0.5],
            [-0.05, -0.45],
        ],
        next_token_top_logprobs_idx=[[7, 11], [8, 12], [9, 13]],
    )

    batch = SimpleNamespace(return_logprob=True, reqs=[req], decoding_reqs=None)

    Scheduler.process_batch_result_beam_search(
        sched, batch, logits_output, next_token_ids, next_token_logprobs
    )

    assert req.output_ids[-1] == 7
    assert req.output_token_logprobs_val[-1] == -0.2
    assert len(req.beam_list.incompleted) == 2
    assert {beam.last_token for beam in req.beam_list.incompleted} == {8, 9}
    assert not req.beam_list.completed
    assert sched.tree_cache.unfinished[-1] is req
    assert req.beam_search_output is None


def test_process_batch_result_beam_search_finalizes_request():
    sched = _make_scheduler_stub()
    sampling_params = SamplingParams(max_new_tokens=3, stop_token_ids=[0])
    sampling_params.normalize(None)
    req = Req(
        "req-2",
        "",
        [4, 5],
        sampling_params,
        return_logprob=True,
        top_logprobs_num=2,
        vocab_size=100,
    )
    req.output_ids = [6]

    beam_list = BeamSearchList(beam_width=1, req_pool_start_idx=0)
    beam_list.incompleted = [
        BeamSearchSequence(
            tokens=[4, 5, 6],
            last_token=6,
            cum_logprob=-0.3,
            prefix_len=2,
        )
    ]
    req.beam_list = beam_list

    next_token_ids = [7, 0]
    next_token_logprobs = [-0.4, -0.05]
    logits_output = LogitsProcessorOutput(
        next_token_logits=torch.zeros((2, 1)),
        next_token_logprobs=torch.tensor(next_token_logprobs),
        next_token_top_logprobs_val=[[-0.4], [-0.05]],
        next_token_top_logprobs_idx=[[7], [0]],
    )

    batch = SimpleNamespace(return_logprob=True, reqs=[req], decoding_reqs=None)

    Scheduler.process_batch_result_beam_search(
        sched, batch, logits_output, next_token_ids, next_token_logprobs
    )

    assert req.finished()
    assert isinstance(req.finished_reason, FINISH_MATCHED_TOKEN)
    assert req.beam_list.incompleted == []
    assert req.beam_list.completed
    assert req.beam_list.completed[0].last_token == 0
    assert req.beam_search_output is not None
    assert req.beam_search_output.sequences[0].last_token == 0
    assert sched.tree_cache.finished[-1] is req
    assert batch in sched.tree_cache.beam_batches
    assert req.time_stats.completion_time > 0


def test_radix_cache_cache_finished_beam_search_releases_rows():
    pool = _DummyReqToTokenPool(rows=4, cols=6)
    allocator = _DummyTokenAllocator()
    cache = RadixCache(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
    )

    req = _DummyReq(
        [1, 2, 3], [4], beam_list=BeamSearchList(beam_width=1, req_pool_start_idx=0)
    )
    req.req_pool_idx = 0

    pool.req_to_token[0, :5] = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    pool.req_to_token[1, :5] = torch.tensor([10, 11, 12, 30, 31], dtype=torch.int32)

    req.beam_list.row_info = {1: (3, 5)}
    req.beam_list.req_pool_start_idx = 0

    batch = SimpleNamespace(
        reqs=[req],
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([5, 5], dtype=torch.int64),
    )

    cache.cache_finished_beam_search(batch)

    assert sorted(allocator.freed) == [30, 31]
    assert pool.freed_rows == [1]
