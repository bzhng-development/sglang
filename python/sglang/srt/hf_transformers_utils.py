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
"""Utilities for Huggingface Transformers."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import torch
from huggingface_hub import snapshot_download

# Only import InternVL config since it's needed by the processor registration
from sglang.srt.configs.internvl import InternVLChatConfig
from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, logger, lru_cache_frozenset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import (
        AutoConfig as _AutoConfig,
        AutoProcessor as _AutoProcessor,
        AutoTokenizer as _AutoTokenizer,
        GenerationConfig as _GenerationConfig,
        PretrainedConfig as _PretrainedConfig,
        PreTrainedTokenizer as _PreTrainedTokenizer,
        PreTrainedTokenizerBase as _PreTrainedTokenizerBase,
        PreTrainedTokenizerFast as _PreTrainedTokenizerFast,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as _MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )


_TRANSFORMERS_MODULE = None
_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = None


def _ensure_transformers():
    global _TRANSFORMERS_MODULE
    if _TRANSFORMERS_MODULE is None:
        import transformers

        _TRANSFORMERS_MODULE = transformers
    return _TRANSFORMERS_MODULE


def _get_transformers_attr(name: str):
    return getattr(_ensure_transformers(), name)


def _get_model_for_causal_lm_mapping_names():
    global _MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    if _MODEL_FOR_CAUSAL_LM_MAPPING_NAMES is None:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as mapping,
        )

        _MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = mapping
    return _MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# Lazy config registry - maps model_type to (module, class_name)
_CONFIG_REGISTRY: Dict[str, tuple[str, str]] = {
    "chatglm": ("sglang.srt.configs.chatglm", "ChatGLMConfig"),
    "dbrx": ("sglang.srt.configs.dbrx", "DbrxConfig"),
    "exaone": ("sglang.srt.configs.exaone", "ExaoneConfig"),
    "deepseek_vl_v2": ("sglang.srt.configs.deepseekvl2", "DeepseekVL2Config"),
    "multi_modality": ("sglang.srt.configs.janus_pro", "MultiModalityConfig"),
    "kimi_vl": ("sglang.srt.configs.kimi_vl", "KimiVLConfig"),
    "internvl_chat": ("sglang.srt.configs.internvl", "InternVLChatConfig"),
    "step3_vl": ("sglang.srt.configs.step3_vl", "Step3VLConfig"),
    "longcat_flash": ("sglang.srt.configs.longcat_flash", "LongcatFlashConfig"),
    "qwen3_next": ("sglang.srt.configs.qwen3_next", "Qwen3NextConfig"),
    "dots_vlm": ("sglang.srt.configs.dots_vlm", "DotsVLMConfig"),
}


def _get_auto_config_cls():
    return _get_transformers_attr("AutoConfig")


def _get_auto_tokenizer_cls():
    return _get_transformers_attr("AutoTokenizer")


def _get_auto_processor_cls():
    return _get_transformers_attr("AutoProcessor")


def _get_generation_config_cls():
    return _get_transformers_attr("GenerationConfig")


def _get_pretrained_tokenizer_base_cls():
    return _get_transformers_attr("PreTrainedTokenizerBase")


def _get_pretrained_tokenizer_fast_cls():
    return _get_transformers_attr("PreTrainedTokenizerFast")


def _get_siglip_vision_config_cls():
    return _get_transformers_attr("SiglipVisionConfig")


def _get_config_class(model_type: str) -> Optional[Type["PretrainedConfig"]]:
    """Lazily load and return a config class for the given model type."""
    if model_type == "internvl_chat":
        # Already imported
        return InternVLChatConfig
    
    if model_type not in _CONFIG_REGISTRY:
        return None
    
    module_path, class_name = _CONFIG_REGISTRY[model_type]
    try:
        module = importlib.import_module(module_path)
        config_class = getattr(module, class_name)
        return config_class
    except (ImportError, AttributeError):
        logger.warning(f"Failed to import {class_name} from {module_path}")
        return None


def download_from_hf(
    model_path: str,
    allow_patterns: Optional[Union[str, list]] = None,
):
    if os.path.exists(model_path):
        return model_path

    if not allow_patterns:
        allow_patterns = ["*.json", "*.bin", "*.model"]

    return snapshot_download(model_path, allow_patterns=allow_patterns)


def get_hf_text_config(config: "PretrainedConfig"):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if config.architectures is not None:
        class_name = config.architectures[0]
        if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
            # We support non-hf version of llava models, so we do not want to
            # read the wrong values from the unused default text_config.
            # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as
            # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.
            setattr(config, "torch_dtype", torch.float16)
            return config

    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            setattr(
                thinker_config.text_config,
                "torch_dtype",
                getattr(thinker_config, "torch_dtype", None),
            )
            return thinker_config.text_config
        return thinker_config
    else:
        return config


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if is_remote_url(model):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    auto_config_cls = _get_auto_config_cls()
    config = auto_config_cls.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )
    if (
        config.architectures is not None
        and config.architectures[0] == "Phi4MMForCausalLM"
    ):
        # Phi4MMForCausalLM uses a hard-coded vision_config. See:
        # https://github.com/vllm-project/vllm/blob/6071e989df1531b59ef35568f83f7351afb0b51e/vllm/model_executor/models/phi4mm.py#L71
        # We set it here to support cases where num_attention_heads is not divisible by the TP size.
        siglip_vision_config_cls = _get_siglip_vision_config_cls()

        vision_config = {
            "hidden_size": 1152,
            "image_size": 448,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 26,  # Model is originally 27-layer, we only need the first 26 layers for feature extraction.
            "patch_size": 14,
        }
        config.vision_config = siglip_vision_config_cls(**vision_config)
    text_config = get_hf_text_config(config=config)

    if isinstance(model, str) and text_config is not None:
        for key, val in text_config.__dict__.items():
            if not hasattr(config, key) and getattr(text_config, key, None) is not None:
                setattr(config, key, val)

    # Use lazy loading for custom configs
    config_class = _get_config_class(config.model_type)
    if config_class is not None:
        config = config_class.from_pretrained(model, revision=revision)
        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        setattr(config, "_name_or_path", model)

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        config.update({"architectures": ["MultiModalityCausalLM"]})

    if model_override_args:
        config.update(model_override_args)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        mapping = _get_model_for_causal_lm_mapping_names()
        if config.model_type not in mapping:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = mapping[config.model_type]
        config.update({"architectures": [model_type]})

    return config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    **kwargs,
):
    try:
        generation_config_cls = _get_generation_config_cls()
        return generation_config_cls.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except OSError as e:
        return None


# Qwen-1M related
def get_sparse_attention_config(
    model: str,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> Dict[str, Any]:
    is_local = os.path.isdir(model)
    if not is_local:
        # Download the config files.
        model = download_from_hf(model, allow_patterns=["*.json"])

    config_file = os.path.join(model, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union["_PreTrainedTokenizer", "_PreTrainedTokenizerFast"]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    # TODO(Xinyuan): Remove this once we have a proper tokenizer for Devstral
    if tokenizer_name == "mistralai/Devstral-Small-2505":
        tokenizer_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_remote_url(tokenizer_name):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    try:
        auto_tokenizer_cls = _get_auto_tokenizer_cls()
        tokenizer = auto_tokenizer_cls.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    pretrained_tokenizer_fast_cls = _get_pretrained_tokenizer_fast_cls()
    if not isinstance(tokenizer, pretrained_tokenizer_fast_cls):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


# Some models doesn't have an available processor, e.g.: InternVL
def get_tokenizer_from_processor(processor):
    pretrained_tokenizer_base_cls = _get_pretrained_tokenizer_base_cls()
    if isinstance(processor, pretrained_tokenizer_base_cls):
        return processor
    return processor.tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    use_fast: Optional[bool] = True,
    **kwargs,
):
    # pop 'revision' from kwargs if present.
    revision = kwargs.pop("revision", tokenizer_revision)

    auto_config_cls = _get_auto_config_cls()
    config = auto_config_cls.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )

    # fix: for Qwen2-VL and Sarashina2Vision models, inject default 'size' if not provided.
    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:
        if "size" not in kwargs:
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

    if config.model_type not in {"llava", "clip"}:
        kwargs["use_fast"] = use_fast
    try:
        if "InternVL3_5" in tokenizer_name:
            auto_tokenizer_cls = _get_auto_tokenizer_cls()
            processor = auto_tokenizer_cls.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:
            auto_processor_cls = _get_auto_processor_cls()
            processor = auto_processor_cls.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )

    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                f"Processor {tokenizer_name} does not have a slow version. Automatically use fast version"
            )
            kwargs["use_fast"] = True
            auto_processor_cls = _get_auto_processor_cls()
            processor = auto_processor_cls.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:
            raise e
    tokenizer = get_tokenizer_from_processor(processor)

    attach_additional_stop_token_ids(tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
