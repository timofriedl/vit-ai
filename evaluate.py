from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vit_ai.vit_autoencoder import config_from_checkpoint, default_device, load_ckpt_dict
from vit_ai.mnist import build_full_mnist, load_test_indices, make_loader, recon_mse
from vit_ai import ViTAutoencoder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("runs/mnist_ae/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/mnist_ae"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    device = default_device() if args.device == "auto" else torch.device(args.device)
    full = build_full_mnist(args.data_dir)
    loader = make_loader(
        full,
        load_test_indices(args.out_dir),  # fixed 1k held-out set
        batch_size=args.batch_size,
        device=device,
        shuffle=False,
    )
    ckpt = load_ckpt_dict(args.checkpoint)
    model = ViTAutoencoder(config_from_checkpoint(ckpt, out_dir=args.out_dir)).to(
        device
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"test_mse={recon_mse(model, loader, device):.6f}")
    return 0


if __name__ == "__main__":
    main()
