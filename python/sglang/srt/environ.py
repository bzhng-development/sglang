import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from functools import cache
from typing import Any

# Global flag to control whether environment variable reads should be cached
_CACHING_ENABLED = False

# Registry to track all EnvField instances for cache management
_ENV_FIELD_REGISTRY = []


class EnvField:
    def __init__(self, default: Any):
        self.default = default
        # NOTE: we use None to indicate whether the value is set or not
        # If the value is manually set to None, we need mark it as _set_to_none.
        # Always use clear() to reset the value, which leads to the default fallback.
        self._set_to_none = False
        # Register this instance for cache management
        _ENV_FIELD_REGISTRY.append(self)
        # Create a cached version of the getter for this specific instance
        self._cached_get = cache(self._get_uncached)

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _get_uncached(self) -> Any:
        """Internal method that performs the actual environment variable lookup."""
        value = os.getenv(self.name)
        if self._set_to_none:
            assert value is None
            return None

        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def get(self) -> Any:
        """Get the environment variable value, using cache if enabled."""
        if _CACHING_ENABLED:
            return self._cached_get()
        else:
            return self._get_uncached()

    def _clear_cache(self):
        """Clear the cache for this field."""
        self._cached_get.cache_clear()

    def is_set(self):
        # NOTE: If None is manually set, it is considered as set.
        return self.name in os.environ or self._set_to_none

    def get_set_value_or(self, or_value: Any):
        # NOTE: Ugly usage, but only way to get custom default value.
        return self.get() if self.is_set() else or_value

    def set(self, value: Any):
        if value is None:
            self._set_to_none = True
            os.environ.pop(self.name, None)
        else:
            self._set_to_none = False
            os.environ[self.name] = str(value)
        # Invalidate cache when value is modified
        self._clear_cache()

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none
        # Invalidate cache after override context exits
        self._clear_cache()

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False
        # Invalidate cache when value is cleared
        self._clear_cache()

    @property
    def value(self):
        return self.get()


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)

    # Test & Debug
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_AMD_CI = EnvBool(False)
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_GC_LOG = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("multinomial")
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)

    # Constrained Decoding
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(True)
    SGLANG_GRAMMAR_TIMEOUT = EnvFloat(300)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)

    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvBool(False)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_ENABLE_FLASHINFER_GEMM = EnvBool(False)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # vLLM dependencies
    USE_VLLM_CUSTOM_ALLREDUCE = EnvBool(False)
    USE_VLLM_CUTLASS_W8A8_FP8_KERNEL = EnvBool(False)

    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SUPPORT_CUTLASS_BLOCK_FP8 = EnvBool(False)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # fmt: on


envs = Envs()


def enable_env_caching():
    """
    Enable caching for all environment variable reads.

    This should be called after server startup is complete to optimize
    hot path performance by caching environment variable lookups.
    After this is called, environment variables are assumed to be static
    and not modified at runtime.
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = True


def disable_env_caching():
    """
    Disable caching for environment variable reads.

    This is useful in tests where environment variables may be modified
    dynamically. When caching is disabled, each get() call will read
    from os.environ directly.
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = False


def clear_all_env_caches():
    """
    Clear all cached environment variable values.

    This should be called when environment variables have been modified
    and caches need to be invalidated.
    """
    for field in _ENV_FIELD_REGISTRY:
        field._clear_cache()


def warmup_env_caches():
    """
    Pre-populate all environment variable caches.

    This reads all environment variables once to warm up the caches.
    Should be called after enable_env_caching() and before serving requests.
    """
    if not _CACHING_ENABLED:
        return

    for field in _ENV_FIELD_REGISTRY:
        # Trigger a read to populate the cache
        field.get()


def enable_and_warmup_env_caching():
    """
    Enable caching and pre-populate all caches in one step.

    This is a convenience function that combines enable_env_caching()
    and warmup_env_caches(). It should be called after server startup
    is complete to optimize runtime performance.
    """
    enable_env_caching()
    warmup_env_caches()


def _convert_SGL_to_SGLANG():
    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.value is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.value is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.value is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.value is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.value is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.value is None
        )

    assert envs.SGLANG_TEST_RETRACT.value is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.value is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.value is None

    example_with_exit_stack()
    example_with_subprocess()


if __name__ == "__main__":
    examples()
