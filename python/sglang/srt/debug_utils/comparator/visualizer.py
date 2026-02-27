"""Heatmap / histogram / scatter visualization for tensor comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_DOWNSAMPLE_THRESHOLD: int = 10_000_000
_SCATTER_SAMPLE_SIZE: int = 10_000


def generate_comparison_figure(
    *,
    baseline: torch.Tensor,
    target: torch.Tensor,
    name: str,
    output_path: Path,
) -> None:
    """Generate a multi-panel comparison PNG for a baseline/target tensor pair.

    Panels (5 rows x 2 cols, left=normal, right=log10):
      Row 0: Baseline heatmap
      Row 1: Target heatmap
      Row 2: Abs Diff heatmap
      Row 3: Abs Diff histogram
      Row 4: Hist2D scatter (baseline vs target density)
      Row 5: Sampled scatter (10k sampled mini-heatmap)
    """
    import matplotlib.pyplot as plt

    baseline_f: torch.Tensor = baseline.detach().cpu().float()
    target_f: torch.Tensor = target.detach().cpu().float()

    can_diff: bool = baseline_f.shape == target_f.shape

    baseline_2d: torch.Tensor = _preprocess_tensor(baseline_f)
    target_2d: torch.Tensor = _preprocess_tensor(target_f)

    nrows: int = 6 if can_diff else 2
    ncols: int = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    stats_lines: list[str] = []

    _draw_heatmap_pair(axes, row_idx=0, t=baseline_2d, title=f"{name} Baseline")
    stats_lines.append(_format_stats(f"Baseline", baseline_2d))

    _draw_heatmap_pair(axes, row_idx=1, t=target_2d, title=f"{name} Target")
    stats_lines.append(_format_stats(f"Target", target_2d))

    if can_diff:
        diff: torch.Tensor = (baseline_2d - target_2d).abs()
        _draw_heatmap_pair(axes, row_idx=2, t=diff, title=f"{name} Abs Diff")
        stats_lines.append(_format_stats("Abs Diff", diff))

        _draw_histogram_pair(axes, row_idx=3, diff=diff, label=f"{name} Abs Diff")
        _draw_scatter_hist2d(
            axes, row_idx=4, baseline=baseline_2d, target=target_2d, label=name
        )
        _draw_scatter_sampled(
            axes, row_idx=5, baseline=baseline_2d, target=target_2d, label=name
        )

    num_stats: int = len(stats_lines)
    title_height: float = 0.015 * num_stats + 0.015
    fig.suptitle(
        "\n".join(stats_lines),
        fontsize=9,
        family="monospace",
        y=1 - title_height / 2,
    )
    plt.tight_layout(rect=[0, 0, 1, 1 - title_height])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────── preprocessing ──────────────────────────


def _preprocess_tensor(tensor: torch.Tensor) -> torch.Tensor:
    t: torch.Tensor = tensor.squeeze()

    while t.ndim < 2:
        t = t.unsqueeze(0)
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-1])

    t = _reshape_to_balanced_aspect(t)
    return t


def _reshape_to_balanced_aspect(
    t: torch.Tensor, max_ratio: float = 5.0
) -> torch.Tensor:
    assert t.ndim == 2

    h, w = t.shape
    ratio: float = h / w if w > 0 else float("inf")

    if 1 / max_ratio <= ratio <= max_ratio:
        return t

    total: int = h * w
    target_side: int = int(math.sqrt(total))

    for new_h in range(target_side, 0, -1):
        if total % new_h == 0:
            new_w: int = total // new_h
            new_ratio: float = new_h / new_w
            if 1 / max_ratio <= new_ratio <= max_ratio:
                return t.reshape(new_h, new_w)

    return t.reshape(1, -1)


# ────────────────────────── drawing helpers ──────────────────────────


def _draw_heatmap_pair(
    axes: np.ndarray,
    *,
    row_idx: int,
    t: torch.Tensor,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_normal = axes[row_idx, 0]
    ax_log = axes[row_idx, 1]

    im = ax_normal.imshow(t.numpy(), aspect="auto", cmap="viridis")
    ax_normal.set_title(title)
    plt.colorbar(im, ax=ax_normal)

    im_log = ax_log.imshow(_to_log10(t).numpy(), aspect="auto", cmap="viridis")
    ax_log.set_title(f"{title} (Log10)")
    cbar = plt.colorbar(im_log, ax=ax_log)
    _format_log_ticks(cbar.ax, axis="y")


def _draw_histogram_pair(
    axes: np.ndarray,
    *,
    row_idx: int,
    diff: torch.Tensor,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_normal = axes[row_idx, 0]
    ax_log = axes[row_idx, 1]

    diff_flat: np.ndarray = _maybe_downsample_numpy(diff.flatten())

    ax_normal.hist(diff_flat, bins=100, edgecolor="none")
    ax_normal.set_title(f"{label} Histogram")
    ax_normal.set_xlabel("Abs Diff")
    ax_normal.set_ylabel("Count")

    log_flat: np.ndarray = np.log10(np.abs(diff_flat) + 1e-10)
    ax_log.hist(log_flat, bins=100, edgecolor="none")
    ax_log.set_title(f"{label} Histogram (Log10)")
    ax_log.set_xlabel("Abs Diff")
    ax_log.set_ylabel("Count")
    _format_log_ticks(ax_log, axis="x")


def _draw_scatter_hist2d(
    axes: np.ndarray,
    *,
    row_idx: int,
    baseline: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_normal = axes[row_idx, 0]
    ax_log = axes[row_idx, 1]

    b_flat: np.ndarray = _maybe_downsample_numpy(baseline.flatten())
    t_flat: np.ndarray = _maybe_downsample_numpy(target.flatten())
    min_len: int = min(len(b_flat), len(t_flat))
    b_flat = b_flat[:min_len]
    t_flat = t_flat[:min_len]

    # Normal scale
    lim: float = float(max(np.abs(b_flat).max(), np.abs(t_flat).max())) * 1.05
    if lim == 0:
        lim = 1.0
    _h, _xe, _ye, im = ax_normal.hist2d(
        b_flat,
        t_flat,
        bins=200,
        range=[[-lim, lim], [-lim, lim]],
        cmap="viridis",
        norm="log",
    )
    ax_normal.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.5)
    ax_normal.set_title(f"{label} Hist2D")
    ax_normal.set_xlabel("Baseline")
    ax_normal.set_ylabel("Target")
    ax_normal.set_aspect("equal")
    plt.colorbar(im, ax=ax_normal)

    # Log scale
    b_log: np.ndarray = np.log10(np.abs(b_flat) + 1e-10)
    t_log: np.ndarray = np.log10(np.abs(t_flat) + 1e-10)
    vmin: float = float(min(b_log.min(), t_log.min())) - 0.5
    vmax: float = float(max(b_log.max(), t_log.max())) + 0.5
    _h2, _xe2, _ye2, im2 = ax_log.hist2d(
        b_log,
        t_log,
        bins=200,
        range=[[vmin, vmax], [vmin, vmax]],
        cmap="viridis",
        norm="log",
    )
    ax_log.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=0.5)
    ax_log.set_title(f"{label} Hist2D (Log10 Abs)")
    ax_log.set_xlabel("Baseline")
    ax_log.set_ylabel("Target")
    ax_log.set_aspect("equal")
    plt.colorbar(im2, ax=ax_log)
    _format_log_ticks(ax_log, axis="both")


def _draw_scatter_sampled(
    axes: np.ndarray,
    *,
    row_idx: int,
    baseline: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_baseline = axes[row_idx, 0]
    ax_target = axes[row_idx, 1]

    b_flat: np.ndarray = baseline.flatten().numpy()
    t_flat: np.ndarray = target.flatten().numpy()

    n_samples: int = min(_SCATTER_SAMPLE_SIZE, len(b_flat))
    rng: np.random.Generator = np.random.default_rng(seed=42)
    indices: np.ndarray = np.sort(rng.choice(len(b_flat), n_samples, replace=False))
    b_sampled: np.ndarray = b_flat[indices]
    t_sampled: np.ndarray = t_flat[indices]

    side: int = int(np.sqrt(n_samples))
    n_use: int = side * side
    b_2d: np.ndarray = b_sampled[:n_use].reshape(side, side)
    t_2d: np.ndarray = t_sampled[:n_use].reshape(side, side)

    vmin: float = float(min(b_2d.min(), t_2d.min()))
    vmax: float = float(max(b_2d.max(), t_2d.max()))

    im_b = ax_baseline.imshow(b_2d, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_baseline.set_title(f"{label} Baseline (10k sampled)")
    plt.colorbar(im_b, ax=ax_baseline)

    im_t = ax_target.imshow(t_2d, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_target.set_title(f"{label} Target (10k sampled)")
    plt.colorbar(im_t, ax=ax_target)


# ────────────────────────── utility ──────────────────────────


def _to_log10(t: torch.Tensor) -> torch.Tensor:
    return t.abs().clamp(min=1e-10).log10()


def _format_log_ticks(ax: object, axis: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(
        lambda x, _: f"1e{int(x)}" if x == int(x) else f"1e{x:.1f}"
    )
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def _format_stats(name: str, t: torch.Tensor) -> str:
    return (
        f"{name}: shape={tuple(t.shape)}, "
        f"min={t.min().item():.4g}, max={t.max().item():.4g}, "
        f"mean={t.mean().item():.4g}, std={t.std().item():.4g}"
    )


def _maybe_downsample_numpy(
    t: torch.Tensor,
    max_elements: int = _DOWNSAMPLE_THRESHOLD,
) -> np.ndarray:
    if t.numel() <= max_elements:
        return t.numpy()

    rng: np.random.Generator = np.random.default_rng(seed=0)
    indices: np.ndarray = rng.choice(t.numel(), max_elements, replace=False)
    return t.numpy()[indices]
