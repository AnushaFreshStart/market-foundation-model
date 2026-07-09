"""
build_market_sequences.py — Build 60-Day Rolling Sequences from market.db
==========================================================================
Reads derived + regime_labels from market.db, fits MarketTokenizer,
builds per-ticker 60-day rolling windows, writes market_sequences.parquet.

Usage:
    python build_market_sequences.py
    python build_market_sequences.py --db market-data/market.db --out market-tokenizer-result/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

import sys
sys.path.insert(0, str(Path(__file__).parent))
from market_tokenizer import MarketTokenizer, MAX_SEQ_LEN, STEP_WIDTH


def build_sequences(
    db_path: str,
    out_dir: str,
    min_seq_len: int = 20,
) -> None:
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  Building sequences for {len(tickers)} tickers...")

    records = []
    step_size = 20  # 1-month sliding window

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

            # Encode tokens: BOS + steps
            tokens = np.zeros((MAX_SEQ_LEN, STEP_WIDTH), dtype=np.int64)

            for t, row in window.iterrows():
                if t >= MAX_SEQ_LEN:
                    break
                step_ids = tok.encode_step(row.to_dict())
                tokens[t] = step_ids

            # Mark BOS at position 0 if space
            tokens[0, 0] = tok.bos_id

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

    seq_df = pd.DataFrame(records)
    out_path = out_dir / "market_sequences.parquet"
    seq_df.to_parquet(str(out_path), index=False)
    print(f"  OK Sequences → {out_path.name}  ({len(seq_df):,} tickers)")

    # Write stats JSON
    stats = {
        "n_tickers":      int(len(seq_df)),
        "vocab_size":     tok.vocab_size,
        "max_seq_len":    MAX_SEQ_LEN,
        "step_width":     STEP_WIDTH,
        "avg_seq_len":    float(seq_df["seq_len"].mean()),
        "regime_dist":    seq_df["final_regime"].value_counts().to_dict(),
        "crash_rate":     float(seq_df["had_crash"].mean()),
        "gap_rate":       float(seq_df["had_gap"].mean()),
        "obs_year_range": [int(seq_df["obs_year_min"].min()), int(seq_df["obs_year_max"].max())],
    }
    stats_path = out_dir / "sequence_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  OK Stats → {stats_path.name}")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build market sequences parquet")
    parser.add_argument("--db",  default="market-data/market.db")
    parser.add_argument("--out", default="market-tokenizer-result")
    parser.add_argument("--min-seq-len", type=int, default=20)
    args = parser.parse_args()
    build_sequences(args.db, args.out, args.min_seq_len)


if __name__ == "__main__":
    main()
