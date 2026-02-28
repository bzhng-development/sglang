from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "raw": [
        "--grouping-skip-keys",
    ],
    "sglang_dev": [
        "--grouping-skip-keys",
        "rank",
        "recompute_status",
    ],
    "sglang_megatron": [
        "--grouping-skip-keys",
        "rank",
        "recompute_status",
        "step",
        "--token-aligner",
        "concat_steps",
    ],
}

DEFAULT_PRESET: str = "sglang_dev"


def expand_preset(argv: list[str], presets: dict[str, list[str]]) -> list[str]:
    """Expand ``--preset <name>`` into the corresponding argv fragment.

    If ``--preset`` is absent **and** ``--grouping-skip-keys`` is also absent,
    the DEFAULT_PRESET is applied automatically.
    """
    if "--preset" in argv:
        idx: int = argv.index("--preset")
        preset_name: str = argv[idx + 1]
        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
            )
        preset_args: list[str] = presets[preset_name]
        return argv[:idx] + preset_args + argv[idx + 2 :]

    if "--grouping-skip-keys" not in argv:
        return presets[DEFAULT_PRESET] + argv

    return argv
