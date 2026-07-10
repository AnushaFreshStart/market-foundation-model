"""
build_market_sequences.py — Build 60-Day Rolling Sequences from market.db (GPU-Accelerated)
============================================================================================
Reads derived + regime_labels from market.db, fits MarketTokenizer,
builds per-ticker 60-day rolling windows with GPU quantile binning,
writes market_sequences.parquet.

Uses PyTorch GPU acceleration for vectorized quantile encoding (1.78M+ encodes).

Usage:
    python build_market_sequences.py
    python build_market_sequences.py --db market-data/market.db --out market-tokenizer-result/ --gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from market_tokenizer import MarketTokenizer, MAX_SEQ_LEN, STEP_WIDTH


def _get_device(use_gpu: bool = True) -> torch.device:
    """Get GPU device if available, else CPU."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU enabled: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("  Using CPU (no GPU or --gpu not specified)")
        return torch.device("cpu")


def _encode_window_gpu(
    window: pd.DataFrame,
    tok: MarketTokenizer,
    device: torch.device,
) -> np.ndarray:
    """
    GPU-accelerated vectorized quantile encoding.
    Encodes all 60 days at once instead of per-row using vectorized searchsorted.
    
    Returns: tokens array (60, 5) of token IDs
    """
    n_rows = len(window)
    tokens = np.zeros((MAX_SEQ_LEN, STEP_WIDTH), dtype=np.int64)
    
    # Extract feature columns for quantile binning
    features = ["rsi_14", "vol_5d", "ret_5d", "atr_14"]
    feature_names = ["RSI", "VOL", "RET", "ATR"]
    
    # Get bin edges from tokenizer
    bin_edges_dict = tok.bin_edges  # Dict: {"RSI": [edges...], "VOL": [...], ...}
    
    for col_idx, (feat_col, feat_name) in enumerate(zip(features, feature_names)):
        if feat_col not in window.columns or feat_name not in bin_edges_dict:
            continue
        
        values = window[feat_col].values  # (n_rows,)
        edges = bin_edges_dict[feat_name]  # list of bin edge values
        
        # GPU-accelerated searchsorted
        values_tensor = torch.from_numpy(values.astype(np.float32)).to(device)
        edges_tensor = torch.from_numpy(np.array(edges[1:-1], dtype=np.float32)).to(device)
        
        # searchsorted finds which bin each value belongs to
        bin_indices = torch.searchsorted(edges_tensor, values_tensor, right=False)
        bin_indices = torch.clamp(bin_indices, 0, len(edges) - 2).cpu().numpy()
        
        # Convert bin index to token ID
        for t in range(min(n_rows, MAX_SEQ_LEN)):
            bin_idx = bin_indices[t]
            token_name = f"{feat_name}_Q{bin_idx}"
            tokens[t, col_idx + 1] = tok.vocab[token_name]
    
    # Encode regime tokens (position 0) using CPU (already fast)
    for t in range(min(n_rows, MAX_SEQ_LEN)):
        row = window.iloc[t]
        regime = str(row.get("regime", "FLAT"))
        regime_id = tok.vocab[regime] if regime in ["BULL", "CORR", "BEAR", "CRASH", "RECOV", "FLAT", "GAP"] else tok.vocab["FLAT"]
        tokens[t, 0] = regime_id
    
    # Mark BOS at position 0
    tokens[0, 0] = tok.bos_id
    
    return tokens


def build_sequences(
    db_path: str,
    out_dir: str,
    min_seq_len: int = 20,
    use_gpu: bool = False,
) -> None:
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(use_gpu)

    print(f"Connecting to {db_path}...")
    con = duckdb.connect(str(db_path), read_only=True)

    # Fit tokenizer
    tok = MarketTokenizer()
    tok.fit(con)
    tok_path = out_dir / "market_tokenizer.json"
    tok.save(tok_path)

    # Load all derived + regime data
    print("  Loading derived data...")
    df = con.execute("""
        SELECT
            d.ticker,
            d.date,
            d.rsi_14,
            d.vol_5d,
            d.ret_5d,
            d.ret_20d,
            d.atr_14,
            d.log_ret,
            d.mcap_tier,
            COALESCE(r.regime, 'FLAT') AS regime
        FROM derived d
        LEFT JOIN regime_labels r
            ON d.ticker = r.ticker AND d.date = r.date
        ORDER BY d.ticker, d.date
    """).fetchdf()
    con.close()

    df["date"] = pd.to_datetime(df["date"])
    df["obs_year"] = df["date"].dt.year

    tickers = df["ticker"].unique()
    print(f"  Building sequences for {len(tickers)} tickers (GPU={use_gpu})...")

    records = []
    step_size = 20  # 1-month sliding window
    total_windows = 0

    for ticker in tickers:
        tdf = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        n = len(tdf)

        if n < min_seq_len:
            continue

        # Extract sliding windows
        for start_idx in range(0, max(1, n - MAX_SEQ_LEN + 1), step_size):
            window = tdf.iloc[start_idx : start_idx + MAX_SEQ_LEN].reset_index(drop=True)
            actual_len = len(window)
            
            if actual_len < min_seq_len:
                continue

            end_date = window["date"].iloc[-1].strftime("%Y-%m-%d")
            seq_id = f"{ticker}_{end_date}"

            # GPU-accelerated encoding
            tokens = _encode_window_gpu(window, tok, device)

            had_crash  = (window["regime"] == "CRASH").any()
            had_gap    = (window["regime"] == "GAP").any()
            final_regime = window["regime"].iloc[-1] if actual_len > 0 else "FLAT"

            records.append({
                "ticker":       seq_id,
                "seq_tokens":   tokens.flatten().tolist(),
                "seq_len":      actual_len,
                "obs_year_max": int(window["obs_year"].max()),
                "obs_year_min": int(window["obs_year"].min()),
                "final_regime": final_regime,
                "had_crash":    bool(had_crash),
                "had_gap":      bool(had_gap),
            })
            total_windows += 1

    seq_df = pd.DataFrame(records)
    out_path = out_dir / "market_sequences.parquet"
    seq_df.to_parquet(str(out_path), index=False)
    print(f"  OK Sequences → {out_path.name}  ({len(seq_df):,} sequences)")

    # Write stats JSON
    stats = {
        "n_sequences":    int(len(seq_df)),
        "vocab_size":     tok.vocab_size,
        "max_seq_len":    MAX_SEQ_LEN,
        "step_width":     STEP_WIDTH,
        "avg_seq_len":    float(seq_df["seq_len"].mean()),
        "regime_dist":    seq_df["final_regime"].value_counts().to_dict(),
        "crash_rate":     float(seq_df["had_crash"].mean()),
        "gap_rate":       float(seq_df["had_gap"].mean()),
        "obs_year_range": [int(seq_df["obs_year_min"].min()), int(seq_df["obs_year_max"].max())],
        "gpu_accelerated": use_gpu,
    }
    stats_path = out_dir / "sequence_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  OK Stats → {stats_path.name}")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build market sequences parquet (GPU-accelerated)")
    parser.add_argument("--db",  default="market-data/market.db", help="Path to market.db")
    parser.add_argument("--out", default="market-tokenizer-result", help="Output directory")
    parser.add_argument("--min-seq-len", type=int, default=20, help="Minimum sequence length")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (if available)")
    args = parser.parse_args()
    build_sequences(args.db, args.out, args.min_seq_len, args.gpu)


if __name__ == "__main__":
    main()
