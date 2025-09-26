"""
Usage:
python3 -m unittest test_triton_attention_backend.TestTritonAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

import torch

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
        self.assertEqual(
            metadata.req_pool_indices.data_ptr(), req_pool_indices[:4].data_ptr()
        )
        self.assertEqual(metadata.seq_lens.data_ptr(), seq_lens[:4].data_ptr())

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


if __name__ == "__main__":
    unittest.main()
