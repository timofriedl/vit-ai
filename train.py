from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # allow `python train.py` without install
from vit_ai import ViTAutoencoder
from vit_ai.vit_autoencoder import (
    add_model_args,
    config_from_args,
    default_device,
    to_jsonable,
)
from vit_ai.latent_plot import build_training_gif, save_latent_distribution
from vit_ai.mnist import (
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    build_full_mnist,
    make_loader,
    save_test_indices,
    save_train_indices,
    split_indices,
    subsample_indices,
)


class EarlyStopping:

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        losses.append(torch.mean((model(x) - x) ** 2).item())
    return float(sum(losses) / max(1, len(losses)))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pbar: tqdm,
) -> float:
    model.train()
    total_loss, num_batches = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        loss = torch.mean((model(x) - x) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        num_batches += 1
        total_loss += float(loss.item())
        pbar.update(1)
        pbar.set_postfix(loss=f"{total_loss / num_batches:.6f}", refresh=False)
    return total_loss / max(1, num_batches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_ae"))
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    add_model_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = default_device() if args.device == "auto" else torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full = build_full_mnist(args.data_dir)
    train_pool, val_idx, test_idx = split_indices(len(full), seed=args.seed)
    train_idx = (
        subsample_indices(train_pool, args.train_size, args.seed)
        if args.train_size > 0
        else train_pool  # 0 = use all indices left after val/test
    )
    save_test_indices(args.out_dir, test_idx)
    save_train_indices(args.out_dir, train_idx)
    train_loader = make_loader(
        full, train_idx, batch_size=args.batch_size, device=device, shuffle=True
    )
    val_loader = make_loader(
        full, val_idx, batch_size=args.batch_size, device=device, shuffle=False
    )
    cfg = config_from_args(args)
    model = ViTAutoencoder(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    run_meta = {
        "device": str(device),
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": TEST_SIZE,
        },
        "seed": args.seed,
        "args": to_jsonable(vars(args)),
        "model_config": asdict(cfg),
    }
    (args.out_dir / "run_meta.json").write_text(
        json.dumps(to_jsonable(run_meta), indent=2) + "\n"
    )
    best_val = math.inf
    best_path = args.out_dir / "best.pt"
    distribution_path = args.out_dir / "distribution.png"
    training_gif_path = args.out_dir / "training.gif"
    best_path.unlink(missing_ok=True)
    for old_frame in args.out_dir.glob("distribution_*.png"):
        old_frame.unlink()
    training_gif_path.unlink(missing_ok=True)
    distribution_frames: list[Path] = []
    plot_kw = dict(
        batch_size=args.batch_size,
        n_samples=min(1000, len(train_idx)),
        seed=args.seed,
    )
    init_frame = args.out_dir / "distribution_0.png"
    save_latent_distribution(  # untrained weights
        init_frame, model, full, train_idx, device, **plot_kw
    )
    shutil.copy2(init_frame, distribution_path)
    distribution_frames.append(init_frame)
    start = time.time()
    pbar = tqdm(
        total=len(train_loader) * args.epochs,
        desc="train",
        unit="batch",
        dynamic_ncols=True,
    )
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, pbar)
            val_loss = evaluate(model, val_loader, device)
            frame_path = args.out_dir / f"distribution_{epoch}.png"
            save_latent_distribution(
                frame_path,
                model,
                full,
                train_idx,
                device,
                **plot_kw,
            )
            shutil.copy2(frame_path, distribution_path)
            distribution_frames.append(frame_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_config": asdict(cfg),
                        "args": to_jsonable(vars(args)),
                    },
                    best_path,
                )
            pbar.set_postfix(
                epoch=f"{epoch}/{args.epochs}",
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val:.4f}",
                elapsed=_format_seconds(time.time() - start),
            )
            if stopper.step(val_loss):
                pbar.write(f"early stop ({args.patience} epochs)")
                break
    finally:
        pbar.close()
    build_training_gif(distribution_frames, training_gif_path)  # deletes distribution_*.png
    print(best_path)
    if training_gif_path.is_file():
        print(training_gif_path)
    return 0


if __name__ == "__main__":
    main()
