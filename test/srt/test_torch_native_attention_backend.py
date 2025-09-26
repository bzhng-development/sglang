"""
Usage:
python3 -m unittest test_triton_attention_backend.TestTritonAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTorchNativeAttnBackend(CustomTestCase):
    def test_mmlu(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "torch_native"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)


class TorchNativeCudaGraphMetadataTest(unittest.TestCase):
    def setUp(self):
        self.backend = TorchNativeAttnBackend(SimpleNamespace(device="cpu"))

    def test_capture_and_replay_metadata(self):
        backend = self.backend
        backend.init_cuda_graph_state(max_bs=4, max_num_tokens=4)

        req_pool_indices = torch.arange(4, dtype=torch.int32)
        seq_lens = torch.arange(4, dtype=torch.int32) + 1

        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            num_tokens=4,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        self.assertTrue(torch.equal(metadata.req_pool_indices, req_pool_indices[:4]))
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens[:4]))
        self.assertTrue(
            torch.equal(
                metadata.seq_lens_cpu,
                seq_lens[:4].to(device="cpu", dtype=torch.int32),
            )
        )

        # Update the underlying tensors to mimic runtime padding updates
        req_pool_indices[:] = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
        seq_lens[:] = torch.tensor([8, 9, 10, 11], dtype=torch.int32)

        backend.init_forward_metadata_replay_cuda_graph(
            bs=4,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=int(seq_lens.sum().item()),
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=None,
        )

        metadata = backend.forward_metadata
        self.assertTrue(torch.equal(metadata.req_pool_indices, req_pool_indices[:4]))
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens[:4]))
        self.assertTrue(
            torch.equal(
                metadata.seq_lens_cpu,
                seq_lens[:4].to(device="cpu", dtype=torch.int32),
            )
        )

    def test_unsupported_forward_modes(self):
        backend = self.backend
        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2)

        req_pool_indices = torch.zeros(2, dtype=torch.int32)
        seq_lens = torch.ones(2, dtype=torch.int32)

        with self.assertRaises(NotImplementedError):
            backend.init_forward_metadata_capture_cuda_graph(
                bs=2,
                num_tokens=2,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.EXTEND,
                spec_info=None,
            )

        backend._graph_metadata.clear()

        backend.init_forward_metadata_capture_cuda_graph(
            bs=2,
            num_tokens=2,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )

        with self.assertRaises(NotImplementedError):
            backend.init_forward_metadata_replay_cuda_graph(
                bs=2,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_sum=int(seq_lens.sum().item()),
                encoder_lens=None,
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=None,
                seq_lens_cpu=None,
            )


class TorchNativeSdpaDecodeTest(unittest.TestCase):
    def setUp(self):
        self.backend = TorchNativeAttnBackend(SimpleNamespace(device="cpu"))

    def _reference_decode(
        self,
        query,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        num_tokens, num_heads, head_dim = query.shape
        output = torch.zeros_like(query)

        query_permute = query.movedim(0, 1)

        for seq_idx, seq_len in enumerate(seq_lens.tolist()):
            seq_len = int(seq_len)
            if seq_len <= 0:
                continue
            per_req_query = query_permute[:, seq_idx : seq_idx + 1, :]
            per_req_tokens = req_to_token[req_pool_indices[seq_idx], :seq_len]
            per_req_key = k_cache[per_req_tokens].movedim(0, 1)
            per_req_value = v_cache[per_req_tokens].movedim(0, 1)

            attn = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(0, 1)
            )
            output[seq_idx : seq_idx + 1] = attn
        return output

    def test_decode_matches_reference(self):
        torch.manual_seed(0)
        device = torch.device("cpu")

        batch = 4
        max_ctx = 6
        num_heads = 3
        head_dim = 5

        query = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.float32)
        output = torch.empty_like(query)
        k_cache = torch.randn(max_ctx * batch, num_heads, head_dim, device=device, dtype=torch.float32)
        v_cache = torch.randn_like(k_cache)

        req_to_token = torch.arange(
            0, max_ctx * batch, device=device, dtype=torch.int32
        ).view(batch, max_ctx)
        req_pool_indices = torch.arange(batch, device=device, dtype=torch.int32)

        seq_lens = torch.tensor([0, 1, 3, max_ctx], dtype=torch.int32, device=device)

        expected = self._reference_decode(
            query,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            scaling=None,
            enable_gqa=False,
            causal=False,
        )

        self.backend._run_sdpa_forward_decode(
            query.clone(),
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            scaling=None,
            enable_gqa=False,
            causal=False,
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
