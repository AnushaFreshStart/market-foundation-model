"""
market_losses.py — Loss Functions for Market Foundation Model (Stage 3)
========================================================================
MarketFineTuneLoss: FocalLoss(direction) + vol_weight * BCE(vol_regime)

Stages 1 & 2 reuse the domain-agnostic losses from losses.py unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketFineTuneLoss(nn.Module):
    """
    Stage 3 fine-tune loss for stock market direction + volatility prediction.

    Loss = FocalLoss(direction) + vol_weight * BCE(vol_regime)

    Args:
        vol_weight:   Weight for the auxiliary volatility regime task (default: 0.4)
        focal_gamma:  Focal loss focusing parameter (default: 2.0)
        focal_alpha:  Focal loss class balancing weight (default: 0.6)
                      Slightly lower than credit (0.75) since direction labels
                      are less imbalanced than default events.
    """

    def __init__(
        self,
        vol_weight: float = 0.4,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.6,
    ):
        super().__init__()
        self.vol_weight   = vol_weight
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        direction_labels: torch.Tensor,
        vol_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions:      dict with 'direction_logit' (B,1) and 'vol_logit' (B,1)
            direction_labels: (B,) binary — 1 if price goes up > threshold
            vol_labels:       (B,) binary — 1 if high-volatility regime

        Returns:
            dict with 'loss', 'direction_loss', 'vol_loss'
        """
        dir_logit = predictions["direction_logit"].squeeze(-1)
        vol_logit = predictions["vol_logit"].squeeze(-1)

        direction_labels = direction_labels.float()
        vol_labels       = vol_labels.float()

        l_direction = self._focal_loss(dir_logit, direction_labels)
        l_vol       = F.binary_cross_entropy_with_logits(vol_logit, vol_labels)

        total = l_direction + self.vol_weight * l_vol

        return {
            "loss":           total,
            "direction_loss": l_direction,
            "vol_loss":       l_vol,
        }

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss: -α_t * (1 - p_t)^γ * log(p_t)."""
        probs     = torch.sigmoid(logits)
        p_t       = probs * targets + (1 - probs) * (1 - targets)
        alpha_t   = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_wt  = alpha_t * (1 - p_t) ** self.focal_gamma
        bce       = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_wt * bce).mean()
