"""
market_tokenizer.py — Market Event Tokenizer & Vocabulary Registry
===================================================================
Builds a three-tier vocabulary for stock market sequence tokenization:
  Tier 1: Special tokens  [PAD] [UNK] [BOS] [EOS] [MASK]
  Tier 2: Regime tokens   BULL CORR BEAR CRASH RECOV FLAT GAP
  Tier 3: Quantile-binned continuous features
          RSI_Qn / VOL_Qn / RET_Qn / MCAP_Qn / ATR_Qn  (10 bins each)

Total vocabulary: 5 + 7 + 50 = 62 tokens

Interface is identical to LoanTokenizer for drop-in compatibility.

Usage:
    tok = MarketTokenizer()
    tok.fit(duckdb_connection)
    ids = tok.encode_step(row_dict)  # -> list[int] of length STEP_WIDTH=5
    tok.save("market_tokenizer.json")
    tok2 = MarketTokenizer.load("market_tokenizer.json")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 60      # 60 trading days per sequence
N_BINS      = 10      # quantile bins for continuous features
STEP_WIDTH  = 5       # tokens per time-step: [regime, rsi, vol, ret, atr]

# Regime state tokens
REGIME_TOKENS = ["BULL", "CORR", "BEAR", "CRASH", "RECOV", "FLAT", "GAP"]

# Tier-1 special tokens
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]

# Tier-3 continuous feature names
CONTINUOUS_FEATURES = ["RSI", "VOL", "RET", "ATR"]
# Note: MCAP is intentionally excluded from CONTINUOUS_FEATURES step encoding
# as it is used as a metadata bin, not a per-step feature.
# The STEP_WIDTH=5 is: [regime, RSI, VOL, RET, ATR]


# ---------------------------------------------------------------------------
# VocabRegistry — identical to LoanTokenizer's implementation
# ---------------------------------------------------------------------------

class VocabRegistry:
    """Maps token names to integer IDs and back."""

    def __init__(self):
        self._tok2id: dict[str, int] = {}
        self._id2tok: dict[int, str] = {}

    def add(self, token: str) -> int:
        if token not in self._tok2id:
            idx = len(self._tok2id)
            self._tok2id[token] = idx
            self._id2tok[idx] = token
        return self._tok2id[token]

    def __getitem__(self, token: str) -> int:
        return self._tok2id.get(token, self._tok2id.get("[UNK]", 1))

    def decode(self, idx: int) -> str:
        return self._id2tok.get(idx, "[UNK]")

    def __len__(self) -> int:
        return len(self._tok2id)

    def to_dict(self) -> dict:
        return dict(self._tok2id)

    @classmethod
    def from_dict(cls, d: dict) -> "VocabRegistry":
        reg = cls()
        for tok, idx in sorted(d.items(), key=lambda x: x[1]):
            reg._tok2id[tok] = idx
            reg._id2tok[idx] = tok
        return reg


# ---------------------------------------------------------------------------
# MarketTokenizer
# ---------------------------------------------------------------------------

class MarketTokenizer:
    """
    Fits on the derived + regime_labels tables in market.db and encodes
    daily OHLCV snapshots into sequences of integer token IDs.
    """

    PAD_ID  = 0
    UNK_ID  = 1
    BOS_ID  = 2
    EOS_ID  = 3
    MASK_ID = 4

    def __init__(self):
        self.vocab     = VocabRegistry()
        self.bin_edges: dict[str, list[float]] = {}
        self._fitted   = False
        self._build_static_vocab()

    def _build_static_vocab(self):
        """Register special and regime tokens (Tier 1 + 2)."""
        for tok in SPECIAL_TOKENS:
            self.vocab.add(tok)
        for tok in REGIME_TOKENS:
            self.vocab.add(tok)

    def fit(self, con) -> "MarketTokenizer":
        """
        Compute quantile bin edges for continuous features by scanning
        the derived table in market.db.
        """
        print("  Fitting MarketTokenizer on derived table...")

        feature_queries = {
            "RSI": ("SELECT rsi_14 FROM derived WHERE rsi_14 IS NOT NULL AND rsi_14 BETWEEN 0 AND 100", "rsi_14"),
            "VOL": ("SELECT vol_5d FROM derived WHERE vol_5d IS NOT NULL AND vol_5d > 0", "vol_5d"),
            "RET": ("SELECT ret_5d FROM derived WHERE ret_5d IS NOT NULL AND ret_5d BETWEEN -0.5 AND 0.5", "ret_5d"),
            "ATR": ("SELECT atr_14 FROM derived WHERE atr_14 IS NOT NULL AND atr_14 > 0", "atr_14"),
        }

        for feat, (sql, col_name) in feature_queries.items():
            df = con.execute(sql).fetchdf()
            values = df[col_name].values
            values = values[~np.isnan(values)]

            quantiles = np.percentile(values, np.linspace(0, 100, N_BINS + 1)).tolist()
            edges = sorted(set(quantiles))
            if len(edges) < 2:
                edges = [float(values.min()), float(values.max()) + 1e-9]
            self.bin_edges[feat] = edges

            for i in range(N_BINS):
                self.vocab.add(f"{feat}_Q{i}")

            print(f"    {feat}: {len(edges)-1} bins, range [{values.min():.4f}, {values.max():.4f}]")

        self._fitted = True
        print(f"  OK Vocabulary size: {len(self.vocab)} tokens")
        return self

    def _quantize(self, feat: str, value: float | None) -> int:
        """Map a scalar value to its bin token ID."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return self.vocab["[UNK]"]
        edges = self.bin_edges.get(feat, [])
        if not edges:
            return self.vocab["[UNK]"]
        idx = int(np.searchsorted(edges[1:-1], value, side="right"))
        idx = min(idx, N_BINS - 1)
        return self.vocab[f"{feat}_Q{idx}"]

    def encode_step(self, row: dict[str, Any]) -> list[int]:
        """
        Encode one daily bar dict into STEP_WIDTH=5 token IDs.

        Returns: [regime_id, rsi_bin, vol_bin, ret_bin, atr_bin]
        """
        regime = str(row.get("regime", "FLAT"))
        regime_id = self.vocab[regime] if regime in REGIME_TOKENS else self.vocab["FLAT"]

        rsi_id = self._quantize("RSI", row.get("rsi_14"))
        vol_id = self._quantize("VOL", row.get("vol_5d"))
        ret_id = self._quantize("RET", row.get("ret_5d"))
        atr_id = self._quantize("ATR", row.get("atr_14"))

        return [regime_id, rsi_id, vol_id, ret_id, atr_id]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    @property
    def mask_id(self) -> int:
        return self.MASK_ID

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "vocab":       self.vocab.to_dict(),
            "bin_edges":   self.bin_edges,
            "n_bins":      N_BINS,
            "step_width":  STEP_WIDTH,
            "max_seq_len": MAX_SEQ_LEN,
            "version":     "1.0",
            "domain":      "market",
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  OK MarketTokenizer saved → {path.name}  (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str | Path) -> "MarketTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.vocab     = VocabRegistry.from_dict(data["vocab"])
        tok.bin_edges = data["bin_edges"]
        tok._fitted   = True
        return tok

    def summary(self) -> dict:
        tier1 = {t: self.vocab[t] for t in SPECIAL_TOKENS}
        tier2 = {t: self.vocab[t] for t in REGIME_TOKENS}
        tier3 = {}
        for feat in CONTINUOUS_FEATURES:
            tier3[feat] = {
                f"{feat}_Q{i}": self.vocab[f"{feat}_Q{i}"]
                for i in range(N_BINS)
                if f"{feat}_Q{i}" in self.vocab.to_dict()
            }
        return {
            "vocab_size":   self.vocab_size,
            "step_width":   STEP_WIDTH,
            "max_seq_len":  MAX_SEQ_LEN,
            "n_bins":       N_BINS,
            "domain":       "market",
            "tiers": {
                "special":  tier1,
                "regime":   tier2,
                "continuous": tier3,
            },
            "bin_edges": self.bin_edges,
        }
