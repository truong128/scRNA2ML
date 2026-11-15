"""Tabular neural-network architectures (FT-Transformer, ResNet, RealMLP-style MLP). Definitions only; training lives in
deep_models.py. RealMLP here is a compact re-implementation, not the official library."""
from __future__ import annotations
import math,warnings
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

ARCH_CATALOG: dict[str, dict[str, Any]] = {
    "ft_transformer": {
        "model_type": "ft_transformer",
        "d_token": 96,
        "n_blocks": 4,
        "n_heads": 8,
        "d_ffn": 192,
        "dropout": 0.15,
        "lr": 2e-4,
    },
    "ft_transformer_large": {
        "model_type": "ft_transformer",
        "d_token": 128,
        "n_blocks": 5,
        "n_heads": 8,
        "d_ffn": 256,
        "dropout": 0.12,
        "lr": 1.5e-4,
    },
    "tabular_resnet": {
        "model_type": "resnet",
        "d_hidden": 256,
        "n_blocks": 5,
        "dropout": 0.20,
        "lr": 5e-4,
    },
    "realmlp": {
        "model_type": "realmlp",
        "d_hidden": 256,
        "n_blocks": 4,
        "dropout": 0.20,
        "lr": 7e-4,
    },
}

class _FeatureTokenizerMixed(nn.Module):
    """One token per feature: Embedding for cats, Linear(1→d) for continuous."""

    def __init__(self, cat_cards: list[int], n_cont: int, d_token: int):
        super().__init__()
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cards]
        )
        self.cont_weight = nn.Parameter(torch.empty(n_cont, d_token)) if n_cont else None
        self.cont_bias = nn.Parameter(torch.empty(n_cont, d_token)) if n_cont else None
        if self.cont_weight is not None:
            nn.init.kaiming_uniform_(self.cont_weight, a=math.sqrt(5))
            nn.init.zeros_(self.cont_bias)
        for emb in self.cat_embs:
            nn.init.normal_(emb.weight, std=0.02)
            # unseen/pad index 0 near zero
            with torch.no_grad():
                emb.weight[0].fill_(0.0)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, emb in enumerate(self.cat_embs):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))
        if self.cont_weight is not None and x_cont.shape[1] > 0:
            # [B, C, D] = bias + weight * x
            cont_tok = self.cont_bias.unsqueeze(0) + self.cont_weight.unsqueeze(0) * x_cont.unsqueeze(-1)
            tokens.append(cont_tok)
        if not tokens:
            raise RuntimeError("No features to tokenize")
        return torch.cat(tokens, dim=1)

class FTTransformerMixed(nn.Module):
    def __init__(
        self,
        cat_cards: list[int],
        n_cont: int,
        d_token: int = 96,
        n_blocks: int = 4,
        n_heads: int = 8,
        d_ffn: int = 192,
        dropout: float = 0.12,
    ):
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token={d_token} not divisible by n_heads={n_heads}")
        self.tokenizer = _FeatureTokenizerMixed(cat_cards, n_cont, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor is True", category=UserWarning)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(inplace=True), nn.Linear(d_token, 1))

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer(x_cat, x_cont)
        cls = self.cls.expand(tok.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, tok], dim=1))
        return self.head(h[:, 0, :]).squeeze(-1)

class _CatContEmbed(nn.Module):
    """Flattened categorical embeddings + continuous → vector."""

    def __init__(self, cat_cards: list[int], n_cont: int, d_emb: int = 16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d_emb) for c in cat_cards])
        for emb in self.embs:
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)
        self.n_cont = n_cont
        self.out_dim = d_emb * len(cat_cards) + n_cont

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        if self.n_cont:
            parts.append(x_cont)
        return torch.cat(parts, dim=1) if parts else x_cont

class _ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.norm(x)), inplace=True)
        h = self.drop(h)
        h = self.drop(self.fc2(h))
        return x + h

class TabularResNetMixed(nn.Module):
    def __init__(
        self,
        cat_cards: list[int],
        n_cont: int,
        d_hidden: int = 256,
        n_blocks: int = 5,
        dropout: float = 0.15,
        d_emb: int = 16,
    ):
        super().__init__()
        self.stem = _CatContEmbed(cat_cards, n_cont, d_emb=d_emb)
        self.first = nn.Linear(self.stem.out_dim, d_hidden)
        self.blocks = nn.ModuleList([_ResBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.head_norm = nn.BatchNorm1d(d_hidden)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(self.stem(x_cat, x_cont)), inplace=True)
        for b in self.blocks:
            x = b(x)
        return self.head(F.relu(self.head_norm(x), inplace=True)).squeeze(-1)

class RealMLPMixed(nn.Module):
    """Modern residual MLP with SELU-ish style blocks (compact RealMLP spirit)."""

    def __init__(
        self,
        cat_cards: list[int],
        n_cont: int,
        d_hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.15,
        d_emb: int = 16,
    ):
        super().__init__()
        self.stem = _CatContEmbed(cat_cards, n_cont, d_emb=d_emb)
        self.first = nn.Linear(self.stem.out_dim, d_hidden)
        layers = []
        for _ in range(n_blocks):
            layers.extend(
                [
                    nn.BatchNorm1d(d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(dropout),
                ]
            )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(nn.BatchNorm1d(d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, 1))

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(self.stem(x_cat, x_cont)), inplace=True)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x).squeeze(-1)

def build_model(
    model_type: str,
    cat_cards: list[int],
    n_cont: int,
    cfg: dict[str, Any],
) -> nn.Module:
    if model_type == "ft_transformer":
        return FTTransformerMixed(
            cat_cards,
            n_cont,
            d_token=int(cfg.get("d_token", 96)),
            n_blocks=int(cfg.get("n_blocks", 4)),
            n_heads=int(cfg.get("n_heads", 8)),
            d_ffn=int(cfg.get("d_ffn", 192)),
            dropout=float(cfg.get("dropout", 0.12)),
        )
    if model_type == "resnet":
        return TabularResNetMixed(
            cat_cards,
            n_cont,
            d_hidden=int(cfg.get("d_hidden", 256)),
            n_blocks=int(cfg.get("n_blocks", 5)),
            dropout=float(cfg.get("dropout", 0.15)),
        )
    if model_type == "realmlp":
        return RealMLPMixed(
            cat_cards,
            n_cont,
            d_hidden=int(cfg.get("d_hidden", 256)),
            n_blocks=int(cfg.get("n_blocks", 4)),
            dropout=float(cfg.get("dropout", 0.15)),
        )
    raise ValueError(f"Unknown model_type: {model_type}")
