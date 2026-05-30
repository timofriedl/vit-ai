from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vit_ai import ViTAutoencoder
from vit_ai.latent_plot import collect_latents, render_latent_ax, sample_plot_indices
from vit_ai.vit_autoencoder import (
    config_from_checkpoint,
    default_device,
    load_checkpoint,
    load_ckpt_dict,
)
from vit_ai.mnist import build_memmap_mnist, load_train_indices, split_indices

CACHE_DIR = Path.home() / ".cache" / "vit-ai"


def _local_file(path: Path) -> Path:
    # copy checkpoint off Desktop/iCloud before torch.load
    dst = CACHE_DIR / path.name
    if dst.is_file() and dst.stat().st_size == path.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst


@torch.inference_mode()
def _decode_image(model: ViTAutoencoder, z: torch.Tensor) -> np.ndarray:
    return model.decode(z)[0, 0].cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/mnist_ae/best.pt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_ae"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()
    device = default_device() if args.device == "auto" else torch.device(args.device)

    ckpt_path = _local_file(args.checkpoint.resolve())
    ckpt = load_ckpt_dict(ckpt_path)
    model = ViTAutoencoder(config_from_checkpoint(ckpt, out_dir=args.out_dir)).to(
        device
    )
    load_checkpoint(model, ckpt_path)
    model.eval()

    meta = json.loads((args.out_dir / "run_meta.json").read_text())
    split_seed = int(meta["seed"])
    full = build_memmap_mnist(args.data_dir)
    train_pool, _, _ = split_indices(len(full), seed=split_seed)
    train_idx = load_train_indices(args.out_dir) or train_pool
    plot_idx = sample_plot_indices(
        train_idx, n_samples=args.n_samples, seed=split_seed
    )

    x_all, y_all, labels = collect_latents(
        model, full, plot_idx, device, batch_size=64
    )

    fig, (ax_latent, ax_img) = plt.subplots(1, 2, figsize=(10, 5))
    cx, cy = render_latent_ax(ax_latent, x_all, y_all, labels)
    ax_img.set_axis_off()
    ax_img.set_title("decode")
    im = ax_img.imshow(
        _decode_image(
            model,
            torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    def on_move(event):
        if event.x is None or event.y is None:
            return
        if not ax_latent.get_window_extent().contains(event.x, event.y):
            return
        # text labels don't set event.xdata; use display coords
        xdata, ydata = ax_latent.transData.inverted().transform((event.x, event.y))
        z = torch.tensor([[xdata, ydata]], dtype=torch.float32, device=device)
        im.set_data(_decode_image(model, z))
        ax_img.set_title(f"z=({xdata:.2f}, {ydata:.2f})")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    main()
