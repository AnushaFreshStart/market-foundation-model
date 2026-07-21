"""
models.py — Neural Network Architectures for Market Foundation Model
=====================================================================
Builds the model requested by TrainingConfig.architecture:
  - patchtst:      PatchTST (patch-based time-series transformer)
  - tft:           Temporal Fusion Transformer
  - hybrid:        PatchTST encoder + TFT gating + multi-task heads
  - lightweight:   Thin transformer for fast experiments
  - lstm_baseline: LSTM baseline

Output heads (market domain):
  Stage 1 (pretrain):  logits  (B, T, vocab_size)
  Stage 2 (joint):     mu, log_sigma, point_pred  (B, T, 1)
  Stage 3 (finetune):  direction_logit, vol_logit  (B, 1)

FlashAttention-3 is used when available (H100 native); falls back
to nn.MultiheadAttention gracefully for CPU/older GPU dev.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainingConfig

# Try to import Transformer Engine for FP8 Linear layers
try:
    import transformer_engine.pytorch as te
    _HAS_TE = True
except ImportError:
    _HAS_TE = False


class TECompatLinear(nn.Module):
    """
    Linear layer wrapper. Drops in te.Linear when transformer_engine is
    available to enable native FP8 Hopper Tensor Core acceleration.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if _HAS_TE:
            self.linear = te.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# FlashAttention wrapper (transparent fallback)
# ---------------------------------------------------------------------------

def make_attn(embed_dim: int, n_heads: int, dropout: float) -> nn.Module:
    """Return FlashAttention-backed MHA if available, else standard MHA."""
    try:
        from flash_attn.modules.mha import MHA
        return MHA(
            embed_dim, n_heads, dropout=dropout,
            use_flash_attn=True, causal=False,
        )
    except ImportError:
        return nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)


class _FlashAttnWrapper(nn.Module):
    """Unified interface around flash_attn MHA or standard MHA."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        try:
            from flash_attn.modules.mha import MHA
            self._attn = MHA(embed_dim, n_heads, dropout=dropout, use_flash_attn=True, causal=False)
            self._flash = True
        except ImportError:
            self._attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            self._flash = False

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self._flash:
            return self._attn(x)
        out, _ = self._attn(x, x, x, key_padding_mask=key_padding_mask)
        return out


# ---------------------------------------------------------------------------
# Shared embedding layer
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """
    Token embedding for multi-token-per-step sequences.
    Input shape: (B, T, step_width) of integer token IDs.
    Each of the step_width positions gets its own embedding; they are summed.
    """

    def __init__(self, vocab_size: int, embed_dim: int, step_width: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.step_width = step_width
        self.embed_dim  = embed_dim
        self.token_embs = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for _ in range(step_width)
        ])
        self.pos_emb  = nn.Embedding(max_seq_len, embed_dim)
        self.dropout  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, step_width) integer token IDs
        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = x.shape
        T = min(T, self.pos_emb.num_embeddings)
        x = x[:, :T, :]
        emb = sum(self.token_embs[i](x[:, :, i]) for i in range(self.step_width))
        pos = self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        return self.dropout(self.layer_norm(emb + pos))


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn    = _FlashAttnWrapper(embed_dim, n_heads, dropout)
        self.ff      = nn.Sequential(
            TECompatLinear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            TECompatLinear(ff_dim, embed_dim),
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch layer
# ---------------------------------------------------------------------------

class PatchLayer(nn.Module):
    """Groups T time-steps into non-overlapping patches of size patch_size."""

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = TECompatLinear(embed_dim * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (B, T, D)
        Returns: (B, T//patch_size, D)
        """
        B, T, D = x.shape
        n_patches = T // self.patch_size
        x = x[:, :n_patches * self.patch_size, :].reshape(B, n_patches, self.patch_size * D)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class PretrainHead(nn.Module):
    """Stage 1: predict masked token IDs."""
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.proj = TECompatLinear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # (B, T, vocab_size)


class JointHead(nn.Module):
    """Stage 2: probabilistic next-step prediction."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mu         = TECompatLinear(embed_dim, 1)
        self.log_sigma  = TECompatLinear(embed_dim, 1)
        self.point_pred = TECompatLinear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mu":         self.mu(x),
            "log_sigma":  self.log_sigma(x),
            "point_pred": self.point_pred(x),
        }


class MarketFinetuneHead(nn.Module):
    """Stage 3 (market): direction + volatility regime binary classifiers."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.direction = TECompatLinear(embed_dim, 1)
        self.vol       = TECompatLinear(embed_dim, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "direction_logit": self.direction(pooled),
            "vol_logit":       self.vol(pooled),
        }


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------

class MarketFoundationModel(nn.Module):
    """
    Base class for all market foundation model architectures.
    Subclasses implement _encode() and define architecture-specific layers.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(
            config.vocab_size, config.embed_dim, config.step_width,
            config.max_seq_len, config.dropout,
        )
        self.pretrain_head = PretrainHead(config.embed_dim, config.vocab_size)
        self.joint_head    = JointHead(config.embed_dim)
        self.finetune_head = MarketFinetuneHead(config.embed_dim)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_embeddings(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return CLS/mean-pool embeddings for embedding export."""
        encoded = self._encode(x, mask)
        # Mean pool over valid (non-padded) positions
        mask_f  = mask.float().unsqueeze(-1)  # (B, T, 1)
        pooled  = (encoded * mask_f[:, :encoded.shape[1], :]).sum(1) / mask_f[:, :encoded.shape[1], :].sum(1).clamp(min=1)
        return pooled

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, stage: str = "pretrain"
    ) -> dict[str, torch.Tensor]:
        encoded = self._encode(x, mask)  # (B, T', D)

        if stage == "pretrain":
            # Upsample back to T if patching was applied
            T = x.shape[1]
            if encoded.shape[1] != T:
                encoded = F.interpolate(
                    encoded.transpose(1, 2), size=T, mode="linear", align_corners=False
                ).transpose(1, 2)
            return {"logits": self.pretrain_head(encoded)}

        elif stage == "joint":
            T = x.shape[1]
            if encoded.shape[1] != T:
                encoded = F.interpolate(
                    encoded.transpose(1, 2), size=T, mode="linear", align_corners=False
                ).transpose(1, 2)
            return self.joint_head(encoded)

        else:  # finetune
            mask_f = mask.float().unsqueeze(-1)
            T_enc  = encoded.shape[1]
            mask_trunc = mask_f[:, :T_enc, :]
            pooled = (encoded * mask_trunc).sum(1) / mask_trunc.sum(1).clamp(min=1)
            return self.finetune_head(pooled)


# ---------------------------------------------------------------------------
# Architecture implementations
# ---------------------------------------------------------------------------

class PatchTSTModel(MarketFoundationModel):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.patch = PatchLayer(config.embed_dim, config.patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.ff_dim, config.dropout)
            for _ in range(config.n_layers)
        ])

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)            # (B, T, D)
        pat = self.patch(emb)          # (B, T//P, D)
        for blk in self.blocks:
            pat = blk(pat)
        return pat


class LightweightModel(MarketFoundationModel):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.block = TransformerBlock(config.embed_dim, config.n_heads, config.ff_dim, config.dropout)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        kpm = (mask == 0) if mask is not None else None
        return self.block(emb, kpm)


class LSTMBaseline(MarketFoundationModel):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.lstm = nn.LSTM(
            config.embed_dim, config.embed_dim // 2,
            num_layers=config.n_layers, batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=True,
        )
        self.proj = TECompatLinear(config.embed_dim, config.embed_dim)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.proj(out)


class HybridModel(MarketFoundationModel):
    """PatchTST encoder + gated residual for rich temporal modelling."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.patch = PatchLayer(config.embed_dim, config.patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.ff_dim, config.dropout)
            for _ in range(config.n_layers)
        ])
        # Gated residual mixing
        self.gate  = nn.Sequential(TECompatLinear(config.embed_dim * 2, config.embed_dim), nn.Sigmoid())
        self.mix   = TECompatLinear(config.embed_dim * 2, config.embed_dim)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)            # (B, T, D)
        pat = self.patch(emb)          # (B, T//P, D)
        skip = pat.clone()
        for blk in self.blocks:
            pat = blk(pat)
        # Gated skip connection
        cat  = torch.cat([pat, skip], dim=-1)
        gate = self.gate(cat)
        return gate * self.mix(cat) + (1 - gate) * skip


class TFTModel(MarketFoundationModel):
    """Simplified Temporal Fusion Transformer."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.lstm  = nn.LSTM(config.embed_dim, config.embed_dim, batch_first=True, bidirectional=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.ff_dim, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.grn_gate = nn.Sequential(TECompatLinear(config.embed_dim, config.embed_dim), nn.Sigmoid())

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        lstm_out, _ = self.lstm(emb)
        gated = self.grn_gate(lstm_out) * lstm_out
        for blk in self.blocks:
            gated = blk(gated)
        return gated


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: TrainingConfig) -> MarketFoundationModel:
    """Instantiate the requested architecture."""
    arch_map = {
        "patchtst":      PatchTSTModel,
        "hybrid":        HybridModel,
        "tft":           TFTModel,
        "lightweight":   LightweightModel,
        "lstm_baseline": LSTMBaseline,
    }
    cls = arch_map.get(config.architecture)
    if cls is None:
        raise ValueError(f"Unknown architecture: {config.architecture}")
    model = cls(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config.architecture}  ({n_params:,} parameters)")
    return model
