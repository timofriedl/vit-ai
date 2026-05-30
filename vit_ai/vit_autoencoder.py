from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _init_trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
        tensor.clamp_(min=-2.0 * std, max=2.0 * std)
    return tensor


@dataclass(frozen=True)
class ViTAutoencoderConfig:
    image_size: int = 28
    in_channels: int = 1
    patch_size: int = 4
    embed_dim: int = 192
    encoder_depth: int = 6
    decoder_depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    stochastic_depth: float = 0.0
    use_cls_token: bool = False

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})."
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.encoder_depth <= 0:
            raise ValueError("encoder_depth must be positive.")
        if self.decoder_depth <= 0:
            raise ValueError("decoder_depth must be positive.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError("attention_dropout must be in [0, 1].")
        if not 0.0 <= self.stochastic_depth <= 1.0:
            raise ValueError("stochastic_depth must be in [0, 1].")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        g = self.grid_size
        return g * g


class DropPath(nn.Module):

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random / keep_prob


class MLP(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiheadSelfAttention(nn.Module):

    def __init__(
        self, dim: int, num_heads: int, attn_dropout: float, proj_dropout: float
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = float(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )
        else:
            scale = self.head_dim ** (-0.5)
            attn = q * scale @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if self.attn_drop and self.training:
                attn = F.dropout(attn, p=self.attn_drop, training=True)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.attn = MultiheadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, embed_dim: int
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Expected input size {(self.image_size, self.image_size)}, got {(h, w)}."
            )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTAutoencoder(nn.Module):

    def __init__(self, cfg: ViTAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim = 2  # hardcoded 2D bottleneck
        self.patch_embed = PatchEmbed(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        num_tokens = cfg.num_patches + (1 if cfg.use_cls_token else 0)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            if cfg.use_cls_token
            else None
        )
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, num_tokens, cfg.embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, num_tokens, cfg.embed_dim))
        self.dec_tokens = nn.Parameter(torch.zeros(1, num_tokens, cfg.embed_dim))
        self.drop = nn.Dropout(cfg.dropout)
        enc_dpr = torch.linspace(0.0, cfg.stochastic_depth, cfg.encoder_depth).tolist()
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attention_dropout,
                    drop_path=enc_dpr[i],
                )
                for i in range(cfg.encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-06)
        dec_dpr = torch.linspace(0.0, cfg.stochastic_depth, cfg.decoder_depth).tolist()
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attention_dropout,
                    drop_path=dec_dpr[i],
                )
                for i in range(cfg.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-06)
        self.to_latent = nn.Linear(cfg.embed_dim, self.latent_dim, bias=True)
        self.latent_to_embed = nn.Linear(self.latent_dim, cfg.embed_dim, bias=True)
        patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size
        self.to_patch = nn.Linear(cfg.embed_dim, patch_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.cls_token is not None:
            _init_trunc_normal_(self.cls_token, std=0.02)
        _init_trunc_normal_(self.pos_embed_enc, std=0.02)
        _init_trunc_normal_(self.pos_embed_dec, std=0.02)
        _init_trunc_normal_(self.dec_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    @property
    def num_patches(self) -> int:
        return self.cfg.num_patches

    def unpatchify(self, patches: Tensor) -> Tensor:
        p = self.cfg.patch_size
        b, n, d = patches.shape
        c = self.cfg.in_channels
        if d != c * p * p:
            raise ValueError(f"Expected patch_dim {c * p * p}, got {d}.")
        g = self.cfg.grid_size
        if n != g * g:
            raise ValueError(f"Expected {g * g} patches, got {n}.")
        x = patches.view(b, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        img = x.view(b, c, g * p, g * p)
        return img

    def _add_tokens_and_position(self, tokens: Tensor, pos_embed: Tensor) -> Tensor:
        if self.cfg.use_cls_token:
            assert self.cls_token is not None
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + pos_embed
        tokens = self.drop(tokens)
        return tokens

    def encode(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x)
        tokens = self._add_tokens_and_position(tokens, self.pos_embed_enc)
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)
        if self.cfg.use_cls_token:
            pooled = tokens[:, 0, :]
        else:
            pooled = tokens.mean(dim=1)
        z2 = self.to_latent(pooled)
        return z2

    def decode(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z of shape (B, {self.latent_dim}), got {tuple(z.shape)}."
            )
        cond = self.latent_to_embed(z)[:, None, :]
        tokens = self.dec_tokens.expand(z.shape[0], -1, -1)
        tokens = tokens + cond + self.pos_embed_dec  # z biases all patch tokens
        tokens = self.drop(tokens)
        for blk in self.decoder_blocks:
            tokens = blk(tokens)
        tokens = self.decoder_norm(tokens)
        if self.cfg.use_cls_token:
            tokens = tokens[:, 1:, :]
        patches = self.to_patch(tokens)
        recon = self.unpatchify(patches)
        return recon

    def forward(
        self, x: Tensor, *, return_latents: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return (recon, z) if return_latents else recon


def default_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--enc-depth", type=int, default=4)
    parser.add_argument("--dec-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)


def config_from_args(args: argparse.Namespace) -> ViTAutoencoderConfig:
    return ViTAutoencoderConfig(
        image_size=28,
        in_channels=1,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.enc_depth,
        decoder_depth=args.dec_depth,
        num_heads=args.heads,
        use_cls_token=False,
    )


def load_ckpt_dict(path: Path) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        ckpt = torch.load(path, map_location="cpu")
    return ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}


def load_checkpoint(model: ViTAutoencoder, path: Path) -> dict:
    ckpt = load_ckpt_dict(path)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt


def _config_dict(raw: dict) -> dict:
    known = {f.name for f in fields(ViTAutoencoderConfig)}
    return {k: v for k, v in raw.items() if k in known}


def config_from_checkpoint(ckpt: dict, *, out_dir: Path) -> ViTAutoencoderConfig:
    if cfg := ckpt.get("model_config"):
        return ViTAutoencoderConfig(**_config_dict(cfg))
    meta = json.loads((out_dir / "run_meta.json").read_text())
    return ViTAutoencoderConfig(**_config_dict(meta["model_config"]))
