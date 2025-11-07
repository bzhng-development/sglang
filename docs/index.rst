SGLang Documentation
====================

SGLang is a high-performance serving framework for large language models and vision-language models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), and reward models (Skywork), with easy extensibility for integrating new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, supporting chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 300,000 GPUs worldwide.

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   get_started/index

.. toctree::
   :maxdepth: 2
   :caption: Basic Usage

   basic_usage/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   advanced_features/index

.. toctree::
   :maxdepth: 2
   :caption: Supported Models

   supported_models/index

.. toctree::
   :maxdepth: 2
   :caption: Hardware Platforms

   platforms/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/index

.. toctree::
   :maxdepth: 2
   :caption: References

   references/index

.. toctree::
   :maxdepth: 2
   :caption: Security Acknowledgement

   security/acknowledgements.md
