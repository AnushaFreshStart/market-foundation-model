"""
trainer.py — Training Orchestrator for Market Foundation Model
==============================================================
Runs 3-stage pipeline (pretrain → joint → finetune) with:
  - FP8 via NVIDIA Transformer Engine (H100) / FP16 AMP fallback
  - DDP-awareness (rank/world_size from env)
  - Gradient checkpointing support
  - Optimised DataLoader (4 workers, pin_memory, prefetch)
  - Market-domain label maps, loss, and evaluator
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from config import TrainingConfig
from models import build_model
from dataset import MarketSequenceDataset, collate_fn
from losses import MaskedPredictionLoss, MultiObjectiveLoss
from market_losses import MarketFineTuneLoss
from evaluator import ModelEvaluator
from market_evaluator import MarketEvaluator


def _make_fp8_context(device_type: str, enabled: bool):
    """Return Transformer Engine fp8_autocast if available, else AMP autocast."""
    if enabled and device_type == "cuda":
        try:
            import transformer_engine.pytorch as te
            return te.fp8_autocast(enabled=True)
        except ImportError:
            pass
    # Fallback to FP16 AMP
    import contextlib
    return torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)


class CreditModelTrainer:  # name kept for interface parity
    """
    Unified market training orchestrator.
    Supports all 5 architectures × 5 strategies with FP8/FP16,
    DDP, early stopping, and comprehensive evaluation.
    """

    def __init__(self, config: TrainingConfig, rank: int = 0, world_size: int = 1):
        self.config     = config
        self.rank       = rank
        self.world_size = world_size
        self.is_main    = (rank == 0)

        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        if self.is_main:
            print(f"  Device: {self.device}  |  world_size={world_size}  |  precision={config.precision}")

        self.model = build_model(config).to(self.device)

        # Compile model for H100 kernel optimization
        if config.profile == "h100_saturated":
            if self.is_main:
                print("  Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        # Gradient checkpointing
        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Wrap in DDP if multi-GPU
        if world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank], output_device=rank,
                find_unused_parameters=True,  # Allow multi-stage training (different heads per stage)
            )

        self.evaluator        = ModelEvaluator()
        self.market_evaluator = MarketEvaluator()
        config.resolve_paths()

    @property
    def _raw_model(self):
        """Unwrap DDP to access raw model methods."""
        return self.model.module if hasattr(self.model, "module") else self.model

    def train(self) -> dict:
        stages = self._resolve_stages()
        results = {
            "config":   self.config.to_dict(),
            "device":   str(self.device),
            "stages":   {},
            "domain":   self.config.domain,
        }
        total_t0 = time.perf_counter()

        for stage in stages:
            if self.is_main:
                print(f"\n{'=' * 60}\n  STAGE: {stage.upper()}\n{'=' * 60}")
            results["stages"][stage] = self._run_stage(stage)

        results["total_time_s"]   = round(time.perf_counter() - total_t0, 2)
        results["total_params"]   = sum(p.numel() for p in self._raw_model.parameters())
        results["trainable_params"] = sum(
            p.numel() for p in self._raw_model.parameters() if p.requires_grad
        )

        if self.config.save_embeddings and "finetune" in stages and self.is_main:
            if self.is_main:
                print(f"\n{'=' * 60}\n  EXPORTING EMBEDDINGS\n{'=' * 60}")
            results["embeddings"] = self._export_embeddings()

        return results

    def _resolve_stages(self) -> list[str]:
        strategies = {
            "full":              ["pretrain", "joint", "finetune"],
            "pretrain_only":     ["pretrain"],
            "pretrain_finetune": ["pretrain", "finetune"],
            "finetune_only":     ["finetune"],
            "joint_finetune":    ["joint", "finetune"],
        }
        return strategies[self.config.strategy]

    def _build_loader(self, ds, shuffle: bool, sampler=None) -> DataLoader:
        cfg = self.config
        use_workers = cfg.num_workers > 0
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            prefetch_factor=cfg.prefetch_factor if use_workers else None,
            persistent_workers=cfg.persistent_workers and use_workers,
        )

    def _run_stage(self, stage: str) -> dict:
        cfg = self.config

        # Load label maps for finetune
        label_maps = None
        if stage == "finetune":
            from market_labels import load_market_label_maps
            label_maps = load_market_label_maps(cfg.db_path)

        ds = MarketSequenceDataset(
            cfg.sequences_path, cfg.tokenizer_path,
            mode=stage, label_maps=label_maps,
            patch_size=cfg.patch_size,
        )
        train_ds, val_ds = ds.oot_split(train_year_max=2023)

        # DDP samplers
        train_sampler = DistributedSampler(train_ds, self.world_size, self.rank) if self.world_size > 1 else None
        train_loader  = self._build_loader(train_ds, shuffle=True, sampler=train_sampler)
        val_loader    = self._build_loader(val_ds, shuffle=False)

        # Loss function
        if stage == "pretrain":
            loss_fn  = MaskedPredictionLoss()
            n_epochs = cfg.pretrain_epochs
        elif stage == "joint":
            loss_fn  = MultiObjectiveLoss(cfg.alpha, cfg.beta, cfg.gamma)
            n_epochs = cfg.joint_epochs
        else:
            loss_fn  = MarketFineTuneLoss(cfg.vol_weight, cfg.focal_gamma, cfg.focal_alpha)
            n_epochs = cfg.finetune_epochs

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self._raw_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

        # Precision context
        use_amp     = cfg.use_amp and self.device.type == "cuda"
        use_fp8     = cfg.precision == "fp8" and use_amp
        scaler      = torch.amp.GradScaler("cuda") if (use_amp and not use_fp8) else None

        best_val_loss  = float("inf")
        patience_ctr   = 0
        patience       = 10
        best_state     = None
        train_losses: list[float] = []
        val_losses:   list[float] = []
        t0             = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for batch in train_loader:
                optimizer.zero_grad()
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)

                ctx = _make_fp8_context(self.device.type, use_fp8 or use_amp)
                with ctx:
                    outputs = self.model(ids, mask, stage=stage)
                    loss    = self._compute_loss(loss_fn, outputs, batch, stage)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self._raw_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._raw_model.parameters(), 1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            train_losses.append(round(avg_train, 6))

            # Validation
            self.model.eval()
            val_sum, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    ids  = batch["input_ids"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)
                    ctx  = _make_fp8_context(self.device.type, use_fp8 or use_amp)
                    with ctx:
                        outputs = self.model(ids, mask, stage=stage)
                        loss    = self._compute_loss(loss_fn, outputs, batch, stage)
                    val_sum += loss.item()
                    val_n   += 1

            avg_val = val_sum / max(val_n, 1)
            val_losses.append(round(avg_val, 6))

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_ctr  = 0
                best_state    = {k: v.cpu().clone() for k, v in self._raw_model.state_dict().items()}
            else:
                patience_ctr += 1

            if self.is_main:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:3d}/{n_epochs} | "
                    f"train={avg_train:.5f} | val={avg_val:.5f} | "
                    f"lr={lr:.2e} | patience={patience_ctr}/{patience} | "
                    f"{time.perf_counter()-t0:.1f}s"
                )

            if patience_ctr >= patience:
                if self.is_main:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if best_state:
            self._raw_model.load_state_dict(best_state)
            self._raw_model.to(self.device)

        # Evaluate
        if self.is_main:
            print(f"\n  Evaluating {stage}...")
        if stage == "pretrain":
            metrics = self.evaluator.evaluate_pretrain(self._raw_model, val_loader, self.device)
        elif stage == "joint":
            metrics = self.evaluator.evaluate_joint(self._raw_model, val_loader, self.device)
        else:
            metrics = self.market_evaluator.evaluate_finetune(self._raw_model, val_loader, self.device)

        return {
            "metrics":        metrics,
            "train_losses":   train_losses,
            "val_losses":     val_losses,
            "best_val_loss":  round(best_val_loss, 6),
            "epochs_trained": len(train_losses),
            "stage_time_s":   round(time.perf_counter() - t0, 2),
        }

    def _compute_loss(self, loss_fn, outputs, batch, stage):
        if stage == "pretrain":
            labels = batch["labels"].to(self.device)
            return loss_fn(outputs["logits"], labels)
        elif stage == "joint":
            labels = batch["labels"].to(self.device).float()
            mask   = batch["attention_mask"].to(self.device)
            return loss_fn(outputs, labels, mask)["loss"]
        else:  # finetune
            dir_labels = batch["direction_label"].to(self.device)
            vol_labels = batch["vol_label"].to(self.device)
            return loss_fn(outputs, dir_labels, vol_labels)["loss"]

    def _export_embeddings(self) -> dict:
        cfg = self.config
        from market_labels import load_market_label_maps
        label_maps = load_market_label_maps(cfg.db_path)

        ds = MarketSequenceDataset(
            cfg.sequences_path, cfg.tokenizer_path,
            mode="finetune", label_maps=label_maps,
        )
        loader = self._build_loader(ds, shuffle=False)

        self._raw_model.eval()
        all_embeddings, all_ids, all_dir, all_vol = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)

                emb = self._raw_model.get_embeddings(ids, mask)
                all_embeddings.append(emb.cpu().numpy())
                all_ids.extend(batch["loan_ids"])

                out = self._raw_model(ids, mask, stage="finetune")
                all_dir.append(torch.sigmoid(out["direction_logit"].squeeze(-1)).cpu().numpy())
                all_vol.append(torch.sigmoid(out["vol_logit"].squeeze(-1)).cpu().numpy())

        import pandas as pd
        embeddings = np.concatenate(all_embeddings, axis=0)
        dir_probs  = np.concatenate(all_dir)
        vol_probs  = np.concatenate(all_vol)

        embed_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        df = pd.DataFrame({
            "ticker":         all_ids,
            **embed_cols,
            "direction_prob": np.round(dir_probs, 6),
            "vol_prob":       np.round(vol_probs, 6),
        })

        out_path = Path(cfg.output_dir) / "embeddings.parquet"
        df.to_parquet(str(out_path), index=False)
        print(f"  OK Embeddings → {out_path.name} ({len(df):,} tickers × {embeddings.shape[1]} dims)")

        return {
            "n_tickers":       len(df),
            "embed_dim":       int(embeddings.shape[1]),
            "output_path":     str(out_path),
            "dir_prob_mean":   round(float(dir_probs.mean()), 4),
            "vol_prob_mean":   round(float(vol_probs.mean()), 4),
        }

    def get_checkpoint(self) -> dict:
        return {
            "model_state_dict": self._raw_model.state_dict(),
            "config":           self.config.to_dict(),
            "architecture":     self.config.architecture,
            "domain":           self.config.domain,
        }
