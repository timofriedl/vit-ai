from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def latent_axis_limits(
    x: np.ndarray, y: np.ndarray, *, pad: float = 0.12
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = max(
        0.5 * max(xmax - xmin, 1e-6) * (1 + pad),
        0.5 * max(ymax - ymin, 1e-6) * (1 + pad),
        0.25,
    )
    return (xc - half, xc + half), (yc - half, yc + half), xc, yc


def sample_plot_indices(
    indices: list[int], *, n_samples: int, seed: int
) -> np.ndarray:
    plot_n = min(n_samples, len(indices))
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=plot_n, replace=False)


def sample_digit_indices(
    dataset: Dataset, indices: list[int], *, seed: int
) -> list[int]:
    by_digit: dict[int, list[int]] = {d: [] for d in range(10)}
    for idx in indices:
        by_digit[int(dataset[int(idx)][1])].append(int(idx))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for digit in range(10):
        pool = by_digit[digit]
        if not pool:
            raise ValueError(f"no digit {digit} in pool")
        chosen.append(int(rng.choice(pool)))
    return chosen


@torch.inference_mode()
def reconstruct_digits(
    model: nn.Module,
    dataset: Dataset,
    example_indices: list[int],
    device: torch.device,
) -> np.ndarray:
    imgs = torch.stack(
        [dataset[int(i)][0] for i in example_indices], dim=0
    ).to(device)
    return model(imgs).cpu().numpy()


@torch.inference_mode()
def collect_latents(
    model: nn.Module,
    dataset: Dataset,
    indices: np.ndarray | list[int],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, labels = [], [], []
    for i in range(0, len(indices), batch_size):
        batch = indices[i : i + batch_size]
        imgs = torch.stack([dataset[int(j)][0] for j in batch], dim=0).to(device)
        z = model.encode(imgs).cpu().numpy()
        xs.append(z[:, 0])
        ys.append(z[:, 1])
        labels.extend(int(dataset[int(j)][1]) for j in batch)
    return np.concatenate(xs), np.concatenate(ys), np.asarray(labels)


def render_latent_ax(
    ax,
    x_all: np.ndarray,
    y_all: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "latent",
) -> tuple[float, float]:
    xlim, ylim, cx, cy = latent_axis_limits(x_all, y_all)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    cmap = plt.get_cmap("tab10")
    for x, y, lab in zip(x_all, y_all, labels, strict=False):
        ax.text(
            x, y, str(lab), color=cmap(lab % 10), fontsize=6, ha="center", va="center"
        )
    return cx, cy


def render_digit_recon_grid(
    fig,
    grid_spec,
    recons: np.ndarray,
    *,
    title: str = "recon",
) -> None:
    sub = grid_spec.subgridspec(
        6, 2, height_ratios=[0.12, 1, 1, 1, 1, 1], hspace=0.12, wspace=0.02
    )
    ax_title = fig.add_subplot(sub[0, :])
    ax_title.set_title(title, fontsize=10)
    ax_title.set_axis_off()
    for digit in range(10):
        row, col = divmod(digit, 2)
        ax = fig.add_subplot(sub[row + 1, col])
        ax.imshow(recons[digit, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_axis_off()


def save_latent_distribution(
    path: Path,
    model: nn.Module,
    dataset: Dataset,
    indices: list[int],
    device: torch.device,
    *,
    batch_size: int = 64,
    n_samples: int = 1000,
    seed: int = 42,
) -> None:
    plot_idx = sample_plot_indices(indices, n_samples=n_samples, seed=seed)
    x_all, y_all, labels = collect_latents(
        model, dataset, plot_idx, device, batch_size=batch_size
    )
    digit_idx = sample_digit_indices(dataset, indices, seed=seed + 1)  # fixed recon panel
    recons = reconstruct_digits(model, dataset, digit_idx, device)

    fig = plt.figure(figsize=(11, 5.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.25)
    ax_latent = fig.add_subplot(outer[0, 0])
    render_latent_ax(ax_latent, x_all, y_all, labels)
    render_digit_recon_grid(fig, outer[0, 1], recons)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_training_gif(
    frame_paths: list[Path], gif_path: Path, *, duration_ms: int = 500
) -> None:
    if not frame_paths:
        return
    from PIL import Image

    frames = [Image.open(p) for p in frame_paths]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for img in frames:
        img.close()
    for path in frame_paths:
        path.unlink()  # keep only training.gif
