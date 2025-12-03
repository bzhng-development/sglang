# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tests for the --fp8-gemm-runner-backend server argument.

=============================================================================
WHY THIS TEST IS DUPLICATED FROM test_deepseek_r1_fp8_trtllm_backend.py
=============================================================================

This test file is intentionally duplicated to verify that the new
--fp8-gemm-runner-backend server argument works identically to the deprecated
environment variable-based configuration (SGLANG_ENABLE_FLASHINFER_FP8_GEMM).

The duplication serves the following purposes:

1. **Backward Compatibility Verification**: Ensures that migrating from the
   deprecated environment variable (SGLANG_ENABLE_FLASHINFER_FP8_GEMM=1) to
   the new server argument (--fp8-gemm-runner-backend=flashinfer) produces
   the same results.

2. **Parallel Testing**: Both the old (env var) and new (server arg) methods
   can be tested in CI to verify they work identically during the deprecation
   transition period.

3. **Migration Guide**: This test serves as an example for users migrating
   from the deprecated environment variable approach to the new server
   argument approach.

4. **Test Coverage During Transition**: While the environment variable is
   deprecated but still supported, having both tests ensures we catch any
   regressions in either code path.

DEPRECATED APPROACH (test_deepseek_r1_fp8_trtllm_backend.py):
    env={"SGLANG_ENABLE_FLASHINFER_FP8_GEMM": "1"}

NEW APPROACH (this file):
    other_args=["--fp8-gemm-runner-backend", "flashinfer"]

Once the deprecated environment variables are removed in a future release,
this test file should become the canonical test and the duplicated test
using environment variables can be removed.
=============================================================================
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

FULL_DEEPSEEK_V3_MODEL_PATH = "deepseek-ai/DeepSeek-V3-0324"


class TestFp8GemmRunnerBackendFlashinfer(CustomTestCase):
    """Test --fp8-gemm-runner-backend=flashinfer.

    This test is equivalent to TestDeepseekR1Fp8Flashinfer in
    test_deepseek_r1_fp8_trtllm_backend.py but uses the new server argument
    instead of the deprecated SGLANG_ENABLE_FLASHINFER_FP8_GEMM environment
    variable.

    Both tests should produce the same results (accuracy > 0.92 on GSM8K).
    """

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(FULL_DEEPSEEK_V3_MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--max-running-requests",
            "512",
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.9",
            "--cuda-graph-max-bs",
            "128",
            "--max-prefill-tokens",
            "8192",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--quantization",
            "fp8",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "1",
            "--expert-parallel-size",
            "1",
            "--scheduler-recv-interval",
            "10",
            "--stream-interval",
            "10",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--enable-symm-mem",
            # NEW: Use server argument instead of environment variable
            "--fp8-gemm-runner-backend",
            "flashinfer",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            # NOTE: No SGLANG_ENABLE_FLASHINFER_FP8_GEMM env var needed!
            # The --fp8-gemm-runner-backend=flashinfer argument replaces it.
            env={**os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """Test GSM8K accuracy with FlashInfer FP8 GEMM backend.

        This test verifies that using --fp8-gemm-runner-backend=flashinfer
        produces the same results as the deprecated environment variable
        SGLANG_ENABLE_FLASHINFER_FP8_GEMM=1.

        Expected: accuracy > 0.92 (same as the deprecated env var test)
        """
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=512,
            parallel=512,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(
            f"Eval accuracy of GSM8K with --fp8-gemm-runner-backend=flashinfer: {metrics=}"
        )

        # Same accuracy threshold as the deprecated env var test
        self.assertGreater(metrics["accuracy"], 0.92)


if __name__ == "__main__":
    unittest.main()
