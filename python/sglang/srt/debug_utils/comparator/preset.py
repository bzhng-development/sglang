from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "raw": [
        "--grouping-skip-keys",
    ],
    "sglang_dev": [
        "--grouping-skip-keys",
        "rank",
    ],
    "sglang_megatron": [
        "--grouping-skip-keys",
        "rank",
        "step",
        "--token-aligner",
        "concat_steps",
    ],
}

DEFAULT_PRESET: str = "sglang_dev"


def _expand_flag(
    argv: list[str], flag: str, mapping: dict[str, list[str]]
) -> list[str]:
    """Replace ``flag <name>`` in *argv* with the corresponding argv fragment from *mapping*."""
    if flag not in argv:
        return argv

    idx: int = argv.index(flag)
    name: str = argv[idx + 1]
    if name not in mapping:
        raise ValueError(f"Unknown value for {flag}: {name}. Available: {list(mapping.keys())}")

    return argv[:idx] + mapping[name] + argv[idx + 2 :]


def expand_preset(argv: list[str], presets: dict[str, list[str]]) -> list[str]:
    """Expand ``--preset <name>`` into the corresponding argv fragment.

    If ``--preset`` is absent **and** ``--grouping-skip-keys`` is also absent,
    the DEFAULT_PRESET is applied automatically.
    """
    has_preset: bool = "--preset" in argv
    argv = _expand_flag(argv, "--preset", presets)

    if not has_preset and "--grouping-skip-keys" not in argv:
        return presets[DEFAULT_PRESET] + argv

    return argv
