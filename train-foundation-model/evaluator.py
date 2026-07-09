"""
evaluator.py — Comprehensive Accuracy Evaluation Engine
=========================================================
Evaluates model performance at each training stage with standardized
metrics. Supports cross-run comparison and XGBoost baseline overlay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class ModelEvaluator:
    """Evaluates trained models with full metrics suite."""

    @torch.no_grad()
    def evaluate_pretrain(
        self, model, dataloader, device: torch.device
    ) -> dict:
        """
        Stage 1 metrics: masked accuracy, top-3, per-class accuracy, perplexity.
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_top3 = 0
        total_masked = 0
        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}

        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(ids, mask, stage="pretrain")
            logits = outputs["logits"]  # (B, T, V)

            B, T, V = logits.shape
            flat_logits = logits.reshape(B * T, V)
            flat_labels = labels.reshape(B * T)

            # Loss
            total_loss += ce_loss(flat_logits, flat_labels).item()

            # Masked positions only
            valid = flat_labels != -100
            if valid.sum() == 0:
                continue

            valid_logits = flat_logits[valid]
            valid_labels = flat_labels[valid]
            n = valid.sum().item()
            total_masked += n

            # Top-1 accuracy
            preds = valid_logits.argmax(dim=-1)
            correct = (preds == valid_labels).sum().item()
            total_correct += correct

            # Top-3 accuracy
            top3 = valid_logits.topk(3, dim=-1).indices
            top3_correct = (top3 == valid_labels.unsqueeze(-1)).any(dim=-1).sum().item()
            total_top3 += top3_correct

            # Per-class accuracy
            for label_id in valid_labels.unique().tolist():
                mask_cls = valid_labels == label_id
                cls_correct = (preds[mask_cls] == label_id).sum().item()
                class_correct[label_id] = class_correct.get(label_id, 0) + cls_correct
                class_total[label_id] = class_total.get(label_id, 0) + mask_cls.sum().item()

        avg_loss = total_loss / max(total_masked, 1)
        perplexity = math.exp(min(avg_loss, 20))

        per_class = {}
        for cid in sorted(class_total.keys()):
            per_class[cid] = round(class_correct.get(cid, 0) / max(class_total[cid], 1), 4)

        return {
            "masked_accuracy": round(total_correct / max(total_masked, 1), 4),
            "top3_accuracy": round(total_top3 / max(total_masked, 1), 4),
            "per_class_accuracy": per_class,
            "perplexity": round(perplexity, 4),
            "avg_loss": round(avg_loss, 6),
            "total_masked": total_masked,
        }

    @torch.no_grad()
    def evaluate_joint(
        self, model, dataloader, device: torch.device
    ) -> dict:
        """Stage 2 metrics: MSE, NLL, ECE, R², Coverage@90."""
        model.eval()
        all_mu, all_sigma, all_point, all_targets = [], [], [], []

        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(ids, mask, stage="joint")
            mu = outputs["mu"].squeeze(-1)
            log_sigma = outputs["log_sigma"].squeeze(-1)
            point = outputs["point_pred"].squeeze(-1)

            valid = labels != -100
            if valid.sum() == 0:
                continue

            all_mu.append(mu[valid].cpu())
            all_sigma.append(log_sigma[valid].exp().cpu())
            all_point.append(point[valid].cpu())
            all_targets.append(labels[valid].float().cpu())

        if not all_mu:
            return {"mse": 0, "nll": 0, "ece": 0, "r2": 0, "coverage_at_90": 0}

        mu = torch.cat(all_mu)
        sigma = torch.cat(all_sigma)
        point = torch.cat(all_point)
        targets = torch.cat(all_targets)

        mse = F.mse_loss(point, targets).item()

        # Gaussian NLL
        nll = (0.5 * (sigma.log() + (targets - mu) ** 2 / (sigma + 1e-8))).mean().item()

        # R²
        ss_res = ((targets - point) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-8)

        # Coverage@90: fraction within 90% CI [mu - 1.645*sigma, mu + 1.645*sigma]
        lower = mu - 1.645 * sigma
        upper = mu + 1.645 * sigma
        in_interval = ((targets >= lower) & (targets <= upper)).float().mean().item()

        return {
            "mse": round(mse, 6),
            "nll": round(nll, 6),
            "r2": round(r2, 4),
            "coverage_at_90": round(in_interval, 4),
        }

    @torch.no_grad()
    def evaluate_finetune(
        self, model, dataloader, device: torch.device
    ) -> dict:
        """Stage 3 metrics for BOTH default and cure targets."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            brier_score_loss, log_loss, confusion_matrix,
            roc_curve, precision_recall_curve,
        )

        model.eval()
        all_default_logits, all_cure_logits = [], []
        all_default_labels, all_cure_labels = [], []

        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            outputs = model(ids, mask, stage="finetune")
            all_default_logits.append(outputs["default_logit"].squeeze(-1).cpu())
            all_cure_logits.append(outputs["cure_logit"].squeeze(-1).cpu())
            all_default_labels.append(batch["default_label"].cpu())
            all_cure_labels.append(batch["cure_label"].cpu())

        def_logits = torch.cat(all_default_logits).numpy()
        cure_logits = torch.cat(all_cure_logits).numpy()
        def_labels = torch.cat(all_default_labels).numpy()
        cure_labels = torch.cat(all_cure_labels).numpy()

        def_probs = 1 / (1 + np.exp(-def_logits))
        cure_probs = 1 / (1 + np.exp(-cure_logits))

        results = {}
        for name, probs, labels in [("default", def_probs, def_labels), ("cure", cure_probs, cure_labels)]:
            if len(np.unique(labels)) < 2:
                results[name] = {"auc_roc": 0, "note": "single class in labels"}
                continue

            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)
            preds_binary = (probs > 0.5).astype(int)
            f1 = f1_score(labels, preds_binary, zero_division=0)
            brier = brier_score_loss(labels, probs)
            ll = log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7))
            cm = confusion_matrix(labels, preds_binary).tolist()

            # KS statistic
            pos_probs = probs[labels == 1]
            neg_probs = probs[labels == 0]
            if len(pos_probs) > 0 and len(neg_probs) > 0:
                all_vals = np.sort(np.unique(probs))
                ks = 0.0
                for threshold in all_vals[::max(1, len(all_vals) // 100)]:
                    cdf_pos = (pos_probs <= threshold).mean()
                    cdf_neg = (neg_probs <= threshold).mean()
                    ks = max(ks, abs(cdf_pos - cdf_neg))
            else:
                ks = 0.0

            # ECE
            ece = self._compute_ece(probs, labels)

            # ROC curve (100 points)
            fpr, tpr, _ = roc_curve(labels, probs)
            idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)

            # PR curve
            prec, rec, _ = precision_recall_curve(labels, probs)
            idx_pr = np.linspace(0, len(prec) - 1, min(100, len(prec)), dtype=int)

            results[name] = {
                "auc_roc": round(float(auc), 4),
                "gini": round(float(2 * auc - 1), 4),
                "avg_precision": round(float(ap), 4),
                "f1_score": round(float(f1), 4),
                "brier_score": round(float(brier), 6),
                "ece": round(float(ece), 6),
                "ks_statistic": round(float(ks), 4),
                "log_loss": round(float(ll), 6),
                "confusion_matrix": cm,
                "roc_curve": {
                    "fpr": [round(float(v), 4) for v in fpr[idx]],
                    "tpr": [round(float(v), 4) for v in tpr[idx]],
                },
                "pr_curve": {
                    "precision": [round(float(v), 4) for v in prec[idx_pr]],
                    "recall": [round(float(v), 4) for v in rec[idx_pr]],
                },
                "positive_rate": round(float(labels.mean()), 4),
                "n_samples": int(len(labels)),
            }

        return results

    @staticmethod
    def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probs)
        for i in range(n_bins):
            in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_conf = probs[in_bin].mean()
                bin_acc = labels[in_bin].mean()
                ece += (in_bin.sum() / total) * abs(bin_conf - bin_acc)
        return ece

    def compare_runs(
        self,
        run_results: list[dict],
        baseline_path: Optional[Path] = None,
    ) -> dict:
        """
        Cross-strategy comparison table.

        Args:
            run_results: list of {run_id, config, metrics} dicts
            baseline_path: path to XGBoost model_results.json
        """
        comparison = {"runs": [], "baseline": None}

        for run in run_results:
            metrics = run.get("metrics", {})
            ft = metrics.get("finetune", {})
            default_m = ft.get("default", {}) if isinstance(ft, dict) else {}

            comparison["runs"].append({
                "run_id": run.get("run_id", "unknown"),
                "architecture": run.get("config", {}).get("architecture", "?"),
                "strategy": run.get("config", {}).get("strategy", "?"),
                "auc_roc_default": default_m.get("auc_roc", 0),
                "gini_default": default_m.get("gini", 0),
                "avg_precision": default_m.get("avg_precision", 0),
                "ks_statistic": default_m.get("ks_statistic", 0),
                "brier_score": default_m.get("brier_score", 0),
                "params": run.get("params", 0),
            })

        # Sort by AUC
        comparison["runs"].sort(key=lambda x: x["auc_roc_default"], reverse=True)

        # Load XGBoost baseline if available
        if baseline_path and baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                comparison["baseline"] = {
                    "auc_roc": baseline.get("auc_roc_test", 0),
                    "gini": baseline.get("gini_test", 0),
                    "avg_precision": baseline.get("avg_precision", 0),
                }
            except Exception:
                pass

        return comparison
