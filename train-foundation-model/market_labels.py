"""
market_labels.py — Direction & Volatility Label Maps from market.db
====================================================================
Replaces CreditSequenceDataset.load_label_maps() for the market domain.
Reads future price data from market.db to construct binary labels:

  direction: 1 if mean adj_close over next horizon_days > threshold, else 0
  vol:       1 if realised_vol_5d in top tercile across universe, else 0

Usage:
    from market_labels import load_market_label_maps
    label_maps = load_market_label_maps('market-data/market.db', horizon_days=5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import duckdb


def load_market_label_maps(
    db_path: str | Path,
    horizon_days: int = 5,
    direction_threshold: float = 0.005,
) -> dict[str, dict[str, int]]:
    """
    Build direction and volatility regime label maps per ticker.

    Args:
        db_path:             Path to market.db
        horizon_days:        Forward window for direction label (default: 5 trading days)
        direction_threshold: Min log return over horizon to label as UP (default: 0.5%)

    Returns:
        {
          'direction': {ticker: 0 or 1},
          'vol':       {ticker: 0 or 1}
        }
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"  [WARNING] market.db not found at {db_path}, using empty label maps")
        return {"direction": {}, "vol": {}}

    con = duckdb.connect(str(db_path), read_only=True)

    # Load derived table sorted by ticker, date
    try:
        df = con.execute(
            "SELECT ticker, date, log_ret, vol_5d FROM derived ORDER BY ticker, date"
        ).fetchdf()
    except Exception as e:
        print(f"  [WARNING] Could not load derived table: {e}")
        con.close()
        return {"direction": {}, "vol": {}}
    con.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    direction_map: dict[str, int] = {}
    vol_map: dict[str, int] = {}
    
    # Calculate global tercile boundary for volatility
    vol_values = df["vol_5d"].dropna().values
    tercile_boundary = float(np.percentile(vol_values, 66.67)) if len(vol_values) > 0 else 0.0

    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.reset_index(drop=True)
        n = len(tdf)
        if n < horizon_days + 1:
            continue
            
        # Calculate future 5-day return for every day
        tdf['future_ret'] = tdf['log_ret'].iloc[::-1].rolling(horizon_days, min_periods=horizon_days).sum().iloc[::-1].shift(-1)
        
        for _, row in tdf.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            seq_id = f"{ticker}_{date_str}"
            
            if pd.notna(row['future_ret']):
                direction_map[seq_id] = int(row['future_ret'] > direction_threshold)
            
            if pd.notna(row['vol_5d']):
                vol_map[seq_id] = int(row['vol_5d'] >= tercile_boundary)

    n_up = sum(direction_map.values())
    n_high_vol = sum(vol_map.values())
    print(
        f"  Market labels: {n_up}/{len(direction_map)} UP direction, "
        f"{n_high_vol}/{len(vol_map)} high-vol"
    )

    return {"direction": direction_map, "vol": vol_map}
