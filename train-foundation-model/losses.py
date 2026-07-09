"""
losses.py — Loss Functions for Credit Foundation Model Training
================================================================
Stage 1: MaskedPredictionLoss  — Cross-entropy on masked positions
Stage 2: MultiObjectiveLoss    — α·MSE + β·NLL(Gaussian) + γ·ECE
Stage 3: MultiTaskFineTuneLoss — FocalLoss(default) + 0.5·BCE(cure)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPredictionLoss(nn.Module):
    """
    Stage 1: Cross-entropy loss on masked token positions only.
    Unmasked positions use ignore_index=-100.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, T, vocab_size)
            labels: (B, T) — event token IDs at masked positions, -100 elsewhere
        """
        B, T, V = logits.shape
        return self.ce(logits.reshape(B * T, V), labels.reshape(B * T))


class MultiObjectiveLoss(nn.Module):
    """
    Stage 2: Composite loss = α·MSE + β·NLL + γ·Calibration.

    Args:
        alpha: Weight for MSE (point accuracy)
        beta:  Weight for NLL (probabilistic quality)
        gamma: Weight for calibration (reliability)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with 'mu', 'log_sigma', 'point_pred'
            targets: (B, T, 1) or (B, T) actual values
            mask: (B, T) — 1 for valid positions
        """
        mu = predictions["mu"].squeeze(-1)
        log_sigma = predictions["log_sigma"].squeeze(-1)
        point = predictions["point_pred"].squeeze(-1)

        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        if mask is not None:
            valid = mask.bool()
            mu = mu[valid]
            log_sigma = log_sigma[valid]
            point = point[valid]
            targets = targets[valid]

        if targets.numel() == 0:
            zero = torch.tensor(0.0, device=mu.device, requires_grad=True)
            return {"loss": zero, "mse": zero, "nll": zero, "calibration": zero}

        # MSE
        l_mse = F.mse_loss(point, targets)

        # Gaussian NLL: 0.5 * (log_sigma + (target - mu)^2 / exp(log_sigma))
        log_sigma = torch.clamp(log_sigma, -6, 6)
        l_nll = 0.5 * (log_sigma + (targets - mu) ** 2 / (log_sigma.exp() + 1e-8)).mean()

        # Calibration (ECE): simplified differentiable approximation
        probs = torch.sigmoid(point)
        binary_targets = (targets > targets.median()).float()
        l_cal = self._ece(probs, binary_targets)

        total = self.alpha * l_mse + self.beta * l_nll + self.gamma * l_cal

        return {"loss": total, "mse": l_mse, "nll": l_nll, "calibration": l_cal}

    @staticmethod
    def _ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
        """Expected Calibration Error — differentiable approximation."""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        total = probs.numel()

        for i in range(n_bins):
            in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_conf = probs[in_bin].mean()
                bin_acc = targets[in_bin].mean()
                ece = ece + (in_bin.sum().float() / total) * (bin_conf - bin_acc).abs()

        return ece


class MultiTaskFineTuneLoss(nn.Module):
    """
    Stage 3: Multi-task loss for default + cure prediction.

    Loss = FocalLoss(default) + cure_weight * BCE(cure)

    Args:
        cure_weight: Weight for the auxiliary cure task (default 0.5)
        focal_gamma: Focal loss focusing parameter
        focal_alpha: Focal loss class balancing weight
    """

    def __init__(
        self,
        cure_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ):
        super().__init__()
        self.cure_weight = cure_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        default_labels: torch.Tensor,
        cure_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with 'default_logit' (B,1) and 'cure_logit' (B,1)
            default_labels: (B,) binary default labels
            cure_labels: (B,) binary cure labels
        """
        def_logit = predictions["default_logit"].squeeze(-1)
        cure_logit = predictions["cure_logit"].squeeze(-1)
        default_labels = default_labels.float()
        cure_labels = cure_labels.float()

        # Focal loss for default prediction
        l_default = self._focal_loss(def_logit, default_labels)

        # BCE for cure prediction
        l_cure = F.binary_cross_entropy_with_logits(cure_logit, cure_labels)

        total = l_default + self.cure_weight * l_cure

        return {
            "loss": total,
            "default_loss": l_default,
            "cure_loss": l_cure,
        }

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss: -α_t * (1 - p_t)^γ * log(p_t)."""
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_weight * bce).mean()
