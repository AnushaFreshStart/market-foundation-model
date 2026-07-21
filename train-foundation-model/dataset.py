"""
dataset.py — Universal Dataset for Market Foundation Model Training
====================================================================
Generalised from the Credit Foundation Model dataset.py.
Supports all training modes: pretrain (masked patches), joint (next-step),
finetune (market: direction + vol labels).

Key differences from credit version:
  - entity_id replaces loan_id (works for tickers and loan IDs)
  - finetune mode returns 'direction_label' + 'vol_label' instead of
    'default_label' + 'cure_label'
  - Tokenizer is duck-typed: works with both LoanTokenizer and MarketTokenizer

Usage:
    ds = MarketSequenceDataset(seq_path, tok_path, mode='finetune', label_maps=maps)
    train_ds, test_ds = ds.oot_split(train_year_max=2023)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "market-tokenizer"))
from market_tokenizer import MarketTokenizer, MAX_SEQ_LEN, STEP_WIDTH


MASK_PROB = 0.15
EVENT_POS = 0   # position of regime/event token within each step


class MarketSequenceDataset(Dataset):
    """
    Universal dataset for all market training stages.

    Modes:
        pretrain:  Masked patch prediction (mask 15% of patches)
        joint:     Next-step regime prediction
        finetune:  Multi-task classification (direction + volatility labels)
    """

    def __init__(
        self,
        sequences_path: str | Path,
        tokenizer_path: str | Path,
        mode: Literal["pretrain", "joint", "finetune"] = "pretrain",
        label_maps: dict[str, dict[str, int]] | None = None,
        patch_size: int = 5,
        seed: int = 42,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        import pyarrow.parquet as pq

        self.path       = Path(sequences_path)
        self.tok        = MarketTokenizer.load(tokenizer_path)
        self.mode       = mode
        self.label_maps = label_maps or {"direction": {}, "vol": {}}
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.rng        = random.Random(seed)

        table   = pq.read_table(str(self.path))
        self.df = table.to_pandas()
        print(f"  MarketDataset: {len(self.df):,} tickers, mode={mode}, max_seq_len={max_seq_len}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row       = self.df.iloc[idx]
        entity_id = str(row["ticker"])
        seq_len   = int(row["seq_len"])
        flat      = list(row["seq_tokens"])

        flat = flat[: self.max_seq_len * STEP_WIDTH]
        tokens    = np.array(flat, dtype=np.int64).reshape(-1, STEP_WIDTH)
        attn_mask = np.zeros(self.max_seq_len, dtype=np.int64)
        attn_mask[: min(seq_len, self.max_seq_len)] = 1

        if self.mode == "pretrain":
            input_ids, labels = self._apply_patch_masking(tokens, min(seq_len, self.max_seq_len))
            return {
                "input_ids":      torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels":         torch.from_numpy(labels),
                "loan_id":        entity_id,   # keep key name for collate_fn compat
                "seq_len":        min(seq_len, self.max_seq_len),
            }

        elif self.mode == "joint":
            input_ids = tokens.copy()
            targets   = np.full(self.max_seq_len, -100, dtype=np.int64)
            eff_len   = min(seq_len - 1, self.max_seq_len - 1)
            for t in range(eff_len):
                targets[t] = tokens[t + 1, EVENT_POS]
            return {
                "input_ids":      torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels":         torch.from_numpy(targets),
                "loan_id":        entity_id,
                "seq_len":        min(seq_len, self.max_seq_len),
            }

        else:  # finetune
            input_ids       = tokens.copy()
            direction_label = self.label_maps.get("direction", {}).get(entity_id, 0)
            vol_label       = self.label_maps.get("vol", {}).get(entity_id, 0)
            return {
                "input_ids":        torch.from_numpy(input_ids),
                "attention_mask":   torch.from_numpy(attn_mask),
                "direction_label":  torch.tensor(direction_label, dtype=torch.long),
                "vol_label":        torch.tensor(vol_label, dtype=torch.long),
                "loan_id":          entity_id,
                "seq_len":          min(seq_len, self.max_seq_len),
            }

    def _apply_patch_masking(
        self, tokens: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mask 15% of patches (groups of patch_size steps)."""
        input_ids = tokens.copy()
        labels    = np.full(self.max_seq_len, -100, dtype=np.int64)

        n_patches = max(1, seq_len // self.patch_size)

        for p in range(n_patches):
            if self.rng.random() < MASK_PROB:
                start = p * self.patch_size
                end   = min(start + self.patch_size, seq_len)
                for t in range(start, end):
                    labels[t]            = tokens[t, EVENT_POS]
                    input_ids[t, EVENT_POS] = self.tok.mask_id

        return input_ids, labels

    def oot_split(
        self, train_year_max: int = 2023
    ) -> tuple["_SubsetDataset", "_SubsetDataset"]:
        """Out-of-Time split. Train ≤ train_year_max, Test > train_year_max."""
        if "obs_year_max" in self.df.columns:
            train_mask = self.df["obs_year_max"] <= train_year_max
            test_mask  = self.df["obs_year_min"] > train_year_max

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                train_df = self.df[train_mask].reset_index(drop=True)
                test_df  = self.df[test_mask].reset_index(drop=True)
                print(f"  OOT split: train={len(train_df):,}, test={len(test_df):,}")
                return _SubsetDataset(self, train_df), _SubsetDataset(self, test_df)

        # Fallback: random 80/20
        n_train  = int(len(self.df) * 0.8)
        shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = shuffled.iloc[:n_train].reset_index(drop=True)
        test_df  = shuffled.iloc[n_train:].reset_index(drop=True)
        print(f"  Random split (no OOT data): train={len(train_df):,}, test={len(test_df):,}")
        return _SubsetDataset(self, train_df), _SubsetDataset(self, test_df)

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size


class _SubsetDataset(Dataset):
    """Subset of MarketSequenceDataset with a filtered DataFrame."""

    def __init__(self, parent: MarketSequenceDataset, subset_df):
        self._parent = parent
        self.df      = subset_df
        self.mode    = parent.mode
        self.tok     = parent.tok
        self.max_seq_len = parent.max_seq_len
        self.patch_size = parent.patch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        entity_id = str(row["ticker"])
        seq_len   = int(row["seq_len"])
        flat      = list(row["seq_tokens"])

        flat = flat[: self.max_seq_len * STEP_WIDTH]
        tokens    = np.array(flat, dtype=np.int64).reshape(-1, STEP_WIDTH)
        attn_mask = np.zeros(self.max_seq_len, dtype=np.int64)
        attn_mask[: min(seq_len, self.max_seq_len)] = 1

        if self.mode == "pretrain":
            input_ids, labels = self._parent._apply_patch_masking(tokens, min(seq_len, self.max_seq_len))
            return {
                "input_ids":      torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels":         torch.from_numpy(labels),
                "loan_id":        entity_id,
                "seq_len":        min(seq_len, self.max_seq_len),
            }
        elif self.mode == "joint":
            input_ids = tokens.copy()
            targets   = np.full(self.max_seq_len, -100, dtype=np.int64)
            eff_len   = min(seq_len - 1, self.max_seq_len - 1)
            for t in range(eff_len):
                targets[t] = tokens[t + 1, EVENT_POS]
            return {
                "input_ids":      torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels":         torch.from_numpy(targets),
                "loan_id":        entity_id,
                "seq_len":        min(seq_len, self.max_seq_len),
            }
        else:
            input_ids       = tokens.copy()
            direction_label = self._parent.label_maps.get("direction", {}).get(entity_id, 0)
            vol_label       = self._parent.label_maps.get("vol", {}).get(entity_id, 0)
            return {
                "input_ids":        torch.from_numpy(input_ids),
                "attention_mask":   torch.from_numpy(attn_mask),
                "direction_label":  torch.tensor(direction_label, dtype=torch.long),
                "vol_label":        torch.tensor(vol_label, dtype=torch.long),
                "loan_id":          entity_id,
                "seq_len":          min(seq_len, self.max_seq_len),
            }

    @property
    def vocab_size(self):
        return self._parent.vocab_size


def collate_fn(batch: list[dict]) -> dict:
    """Universal collate for all modes."""
    result = {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "loan_ids":       [b["loan_id"] for b in batch],
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([b["labels"] for b in batch])
    if "direction_label" in batch[0]:
        result["direction_label"] = torch.stack([b["direction_label"] for b in batch])
        result["vol_label"]       = torch.stack([b["vol_label"]       for b in batch])
    return result
