"""
market_evaluator.py — Market-Specific Evaluation Metrics (Stage 3)
===================================================================
Replaces evaluate_finetune() for the market domain.
Stages 1 & 2 evaluations are domain-agnostic (reused from evaluator.py).

Metrics:
  - Directional Accuracy
  - AUC-ROC (direction)
  - AUC-ROC (volatility regime)
  - Top-Decile Precision
  - F1 Score (direction)
  - Sharpe-like Signal Quality Ratio
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class MarketEvaluator:
    """Evaluates market foundation model with finance-specific metrics."""

    @torch.no_grad()
    def evaluate_finetune(
        self, model, dataloader, device: torch.device
    ) -> dict:
        """Stage 3 metrics for market domain: direction + volatility."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            f1_score, brier_score_loss,
            roc_curve, precision_recall_curve,
        )

        model.eval()
        all_dir_logits:  list[torch.Tensor] = []
        all_vol_logits:  list[torch.Tensor] = []
        all_dir_labels:  list[torch.Tensor] = []
        all_vol_labels:  list[torch.Tensor] = []

        for batch in dataloader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            outputs = model(ids, mask, stage="finetune")

            all_dir_logits.append(outputs["direction_logit"].squeeze(-1).cpu())
            all_vol_logits.append(outputs["vol_logit"].squeeze(-1).cpu())
            all_dir_labels.append(batch["direction_label"].cpu())
            all_vol_labels.append(batch["vol_label"].cpu())

        dir_logits = torch.cat(all_dir_logits).numpy()
        vol_logits = torch.cat(all_vol_logits).numpy()
        dir_labels = torch.cat(all_dir_labels).numpy()
        vol_labels = torch.cat(all_vol_labels).numpy()

        dir_probs = 1 / (1 + np.exp(-dir_logits))
        vol_probs = 1 / (1 + np.exp(-vol_logits))

        results: dict = {}

        for name, probs, labels in [
            ("direction", dir_probs, dir_labels),
            ("vol",       vol_probs, vol_labels),
        ]:
            if len(np.unique(labels)) < 2:
                results[name] = {"auc_roc": 0.5, "note": "single class in labels"}
                continue

            auc    = float(roc_auc_score(labels, probs))
            ap     = float(average_precision_score(labels, probs))
            preds  = (probs > 0.5).astype(int)
            f1     = float(f1_score(labels, preds, zero_division=0))
            brier  = float(brier_score_loss(labels, probs))
            acc    = float((preds == labels).mean())

            # Top-decile precision: of top 10% confidence predictions, fraction correct
            top_decile_idx = np.argsort(probs)[::-1][:max(1, len(probs) // 10)]
            top_decile_prec = float(labels[top_decile_idx].mean())

            # Sharpe-like signal quality (only for direction)
            sharpe = 0.0
            if name == "direction":
                # Treat predictions as a long/flat signal: +1 if pred=UP, 0 otherwise
                # "returns" are proxied by (2*label - 1) as +1/-1
                signal   = (probs - 0.5)          # continuous signal strength
                outcomes = (2 * labels - 1).astype(float)  # +1 if UP, -1 if DOWN
                pnl      = signal * outcomes
                sharpe   = float(pnl.mean() / (pnl.std() + 1e-9)) * math.sqrt(252)

            # ROC curve (100 points)
            fpr, tpr, _ = roc_curve(labels, probs)
            idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)

            # PR curve
            prec, rec, _ = precision_recall_curve(labels, probs)
            idx_pr = np.linspace(0, len(prec) - 1, min(100, len(prec)), dtype=int)

            results[name] = {
                "auc_roc":          round(auc, 4),
                "avg_precision":    round(ap, 4),
                "f1_score":         round(f1, 4),
                "directional_accuracy": round(acc, 4),
                "brier_score":      round(brier, 6),
                "top_decile_precision": round(top_decile_prec, 4),
                "sharpe_signal":    round(sharpe, 4) if name == "direction" else None,
                "positive_rate":    round(float(labels.mean()), 4),
                "n_samples":        int(len(labels)),
                "roc_curve": {
                    "fpr": [round(float(v), 4) for v in fpr[idx]],
                    "tpr": [round(float(v), 4) for v in tpr[idx]],
                },
                "pr_curve": {
                    "precision": [round(float(v), 4) for v in prec[idx_pr]],
                    "recall":    [round(float(v), 4) for v in rec[idx_pr]],
                },
            }

        return results

    def compare_runs(
        self,
        run_results: list[dict],
        baseline_path: Optional[object] = None,
    ) -> dict:
        """Cross-run comparison sorted by directional AUC."""
        comparison = {"runs": [], "baseline": None}

        for run in run_results:
            metrics  = run.get("metrics", {})
            ft       = metrics.get("finetune", {})
            dir_m    = ft.get("direction", {}) if isinstance(ft, dict) else {}

            comparison["runs"].append({
                "run_id":               run.get("run_id", "unknown"),
                "architecture":         run.get("config", {}).get("architecture", "?"),
                "strategy":             run.get("config", {}).get("strategy", "?"),
                "auc_roc_direction":    dir_m.get("auc_roc", 0.5),
                "directional_accuracy": dir_m.get("directional_accuracy", 0.5),
                "top_decile_precision": dir_m.get("top_decile_precision", 0),
                "sharpe_signal":        dir_m.get("sharpe_signal", 0),
                "params":               run.get("params", 0),
            })

        comparison["runs"].sort(
            key=lambda x: x["auc_roc_direction"], reverse=True
        )
        return comparison
