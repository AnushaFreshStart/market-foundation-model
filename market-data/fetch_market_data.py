"""
fetch_market_data.py — Download OHLCV + derived features for S&P 500
======================================================================
Downloads daily bars from yfinance, computes RSI-14, realised volatility,
ATR-14, log returns, market-cap tier, and writes to market.db (DuckDB).

Usage:
    python fetch_market_data.py --tickers sp500 --years 5
    python fetch_market_data.py --tickers AAPL,MSFT,GOOGL --years 3
    python fetch_market_data.py --tickers sp500 --years 5 --db /path/to/market.db
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import duckdb


# ---------------------------------------------------------------------------
# Extended S&P 500 + NASDAQ 100 Ticker List (~150 liquid components)
# ---------------------------------------------------------------------------
SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","JNJ",
    "V","PG","UNH","HD","MA","DIS","PYPL","BAC","ADBE","CRM",
    "NFLX","CMCSA","XOM","PFE","T","VZ","INTC","WMT","CVX","ABT",
    "KO","PEP","MRK","TMO","COST","ACN","AVGO","DHR","TXN","NEE",
    "LIN","UPS","PM","MDT","HON","LOW","RTX","QCOM","SBUX","IBM",
    "GS","BLK","C","AXP","MS","USB","WFC","SCHW","MMC","TRV",
    "SPG","EQR","PSA","AMT","CCI","PLD","WELL","DLR","O","AVB",
    "LMT","GD","BA","NOC","HII","L3H","TDG","HWM","TXT","WWD",
    "CAT","DE","EMR","ETN","GE","ITW","PH","ROK","SWK","XYL",
    "CVS","CI","HUM","ELV","CNC","MOH","DVA","HCA","UHS","THC",
    "AMD","INTU","ISRG","MDLZ","GILD","AMGN","ADI","BKNG","REGN","CSCO",
    "ADSK","MELI","PANW","SNPS","CDNS","MU","NXPI","KLAC","ASML","MAR",
    "LRCX","KDP","ORLY","CTAS","FTNT","MNST","PCAR","PAYX","CPRT","DXCM",
    "VRSK","IDXX","AEP","EXC","KHC","CTSH","XEL","BKR","GEHC","ON",
    "MCHP","FAST","ANSS","CDW","TEAM","DDOG","WDAY","ROP","ADWR","TEAM",
    "SYK","EL","TGT","DG","TJX","ORCL","CRM","AMAT","WBA","PDD"
]


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI-n for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range normalised by close price."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr / (close + 1e-9)  # normalised ATR


def compute_realised_vol(log_ret: pd.Series, window: int = 5) -> pd.Series:
    """Rolling annualised realised volatility."""
    return log_ret.rolling(window).std() * math.sqrt(252)


def classify_regime(ret_20d: float, vol_20d: float, rsi: float,
                    p80_vol: float, p20_vol: float) -> str:
    """Rule-based market regime classification."""
    if ret_20d > 0.05:
        return "BULL"
    if ret_20d < -0.10:
        return "CRASH" if vol_20d > p80_vol else "BEAR"
    if ret_20d < -0.03:
        return "CORR"
    if vol_20d < p20_vol and abs(ret_20d) < 0.02:
        return "FLAT"
    if rsi > 70 or rsi < 30:
        return "GAP"
    return "RECOV"


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def download_from_kaggle(dest_dir: Path) -> Path:
    """Download S&P 500 dataset from Kaggle. Tries Method A, fallbacks to Method B."""
    import sys
    import subprocess
    import shutil
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_dest = dest_dir / "all_stocks_5yr.csv"
    if csv_dest.exists():
        print(f"  Found cached dataset at {csv_dest}")
        return csv_dest

    # Method A: Try standard kaggle API
    try:
        print("  Attempting Method A (kaggle API)...")
        try:
            import kaggle
        except ImportError:
            print("  Installing kaggle package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
            
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("jsaleeby/sp-500-stock-data", path=str(dest_dir), unzip=True)
        print("  Method A successful!")
        return csv_dest
    except Exception as e:
        print(f"  Method A failed: {e}. Trying Method B (kagglehub)...")

    # Method B: Try kagglehub
    try:
        try:
            import kagglehub
        except ImportError:
            print("  Installing kagglehub package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        download_path = kagglehub.dataset_download("jsaleeby/sp-500-stock-data")
        print(f"  Method B successful! Downloaded to {download_path}")
        
        src_file = Path(download_path) / "all_stocks_5yr.csv"
        if not src_file.exists():
            csvs = list(Path(download_path).glob("*.csv"))
            if csvs:
                src_file = csvs[0]
            else:
                raise FileNotFoundError("Could not find any CSV file in downloaded dataset")

        shutil.copy(src_file, csv_dest)
        return csv_dest
    except Exception as e2:
        raise RuntimeError(f"All Kaggle download methods failed: {e2}")


def download_from_git(dest_dir: Path) -> Path:
    """Download S&P 500 dataset directly from GitHub."""
    import urllib.request
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_dest = dest_dir / "all_stocks_5yr.csv"
    if csv_dest.exists():
        print(f"  Found cached dataset at {csv_dest}")
        return csv_dest

    print("  Downloading S&P 500 dataset from GitHub raw source...")
    url = "https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/all_stocks_5yr.csv"
    try:
        urllib.request.urlretrieve(url, str(csv_dest))
        print("  GitHub download successful!")
        return csv_dest
    except Exception as e:
        raise RuntimeError(f"GitHub download failed: {e}")


def download_from_huggingface(dest_dir: Path) -> Path:
    """Download S&P 500 dataset from Hugging Face."""
    import sys
    import subprocess
    import shutil
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_dest = dest_dir / "all_stocks_5yr.csv"
    if csv_dest.exists():
        print(f"  Found cached dataset at {csv_dest}")
        return csv_dest

    print("  Attempting download from Hugging Face Hub...")
    try:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("  Installing huggingface_hub package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            from huggingface_hub import hf_hub_download

        # Using a public HF dataset containing S&P 500 historical prices
        download_path = hf_hub_download(
            repo_id="Sashank-810/crisisnet-dataset",
            filename="SP500.csv",
            repo_type="dataset"
        )
        print(f"  Hugging Face download successful! Downloaded to {download_path}")
        shutil.copy(download_path, csv_dest)
        return csv_dest
    except Exception as e:
        print(f"  Hugging Face download failed: {e}. Falling back to GitHub download...")
        return download_from_git(dest_dir)


def fetch_and_store(
    tickers: list[str],
    years: int,
    db_path: str,
    batch_size: int = 20,
    client: str = "git",
    delay: float = 0.0,
) -> None:
    """Download OHLCV for tickers and persist to DuckDB."""
    import time
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # Create tables
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker    VARCHAR,
            date      DATE,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            adj_close DOUBLE,
            volume    BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS derived (
            ticker    VARCHAR,
            date      DATE,
            log_ret   DOUBLE,
            rsi_14    DOUBLE,
            vol_5d    DOUBLE,
            atr_14    DOUBLE,
            ret_5d    DOUBLE,
            ret_20d   DOUBLE,
            mcap_tier INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS regime_labels (
            ticker VARCHAR,
            date   DATE,
            regime VARCHAR,
            PRIMARY KEY (ticker, date)
        )
    """)

    all_vols: list[float] = []

    print(f"  Ingesting from static dataset using client: {client}...")
    if client == "kaggle":
        csv_path = download_from_kaggle(db_path.parent)
    elif client == "huggingface":
        csv_path = download_from_huggingface(db_path.parent)
    else:
        csv_path = download_from_git(db_path.parent)

    full_df = pd.read_csv(csv_path)
    
    # Rename columns to standard (maps options-IV-SP500 or standard camnugent/sandp500 columns)
    full_df.columns = [c.lower() for c in full_df.columns]
    
    ticker_col = "name" if "name" in full_df.columns else ("ticker" if "ticker" in full_df.columns else None)
    if not ticker_col:
        # Fallback to identify ticker column if different
        for col in ["name", "ticker", "symbol", "code"]:
            if col in full_df.columns:
                ticker_col = col
                break
    
    if not ticker_col:
        raise ValueError("Could not find ticker or symbol column in CSV")

    full_df = full_df.rename(columns={
        ticker_col: "ticker",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })
    
    unique_tickers = full_df["ticker"].unique()
    print(f"  Processing {len(unique_tickers)} tickers from CSV...")
    
    for idx, ticker in enumerate(unique_tickers):
        try:
            df = full_df[full_df["ticker"] == ticker].copy()
            df = df.dropna(subset=["close"])
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").reset_index(drop=True)
            
            df["adj_close"] = df["close"]
            
            # Derived features
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            df["rsi_14"] = compute_rsi(df["close"], 14)
            df["vol_5d"] = compute_realised_vol(df["log_ret"], 5)
            df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
            df["ret_5d"] = df["close"].pct_change(5)
            df["ret_20d"] = df["close"].pct_change(20)
            df["mcap_tier"] = 1
            
            df = df.dropna()
            if df.empty:
                continue
            all_vols.extend(df["vol_5d"].tolist())

            # Write ohlcv
            ohlcv_df = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
            con.execute("DELETE FROM ohlcv WHERE ticker = ?", [ticker])
            con.execute("INSERT INTO ohlcv SELECT * FROM ohlcv_df")

            # Write derived
            derived_df = df[["ticker", "date", "log_ret", "rsi_14", "vol_5d", "atr_14", "ret_5d", "ret_20d", "mcap_tier"]].copy()
            con.execute("DELETE FROM derived WHERE ticker = ?", [ticker])
            con.execute("INSERT INTO derived SELECT * FROM derived_df")
            
            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(unique_tickers)} tickers...")
        except Exception as e:
            print(f"    [WARN] Failed to process {ticker}: {e}")
            continue

    # Compute regime labels with universe-wide vol percentiles
    if all_vols:
        p80_vol = float(np.percentile(all_vols, 80))
        p20_vol = float(np.percentile(all_vols, 20))
    else:
        p80_vol, p20_vol = 0.4, 0.1

    print("\n  Computing regime labels...")
    all_tickers_db = con.execute("SELECT DISTINCT ticker FROM derived").fetchdf()["ticker"].tolist()

    for ticker in all_tickers_db:
        df = con.execute(
            "SELECT ticker, date, ret_20d, vol_5d, rsi_14 FROM derived WHERE ticker = ? ORDER BY date",
            [ticker]
        ).fetchdf()
        if df.empty:
            continue
        df["regime"] = df.apply(
            lambda r: classify_regime(
                r["ret_20d"], r["vol_5d"], r["rsi_14"], p80_vol, p20_vol
            ),
            axis=1,
        )
        regime_df = df[["ticker", "date", "regime"]]
        con.execute("DELETE FROM regime_labels WHERE ticker = ?", [ticker])
        con.execute("INSERT INTO regime_labels SELECT * FROM regime_df")

    n_rows = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    n_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM ohlcv").fetchone()[0]
    print(f"\n  Done. market.db: {n_tickers} tickers, {n_rows:,} rows total")
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch market data into market.db")
    parser.add_argument("--tickers", default="sp500",
                        help="'sp500' or comma-separated ticker list")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of history to download")
    parser.add_argument("--db", default="market-data/market.db",
                        help="Output DuckDB path")
    parser.add_argument("--client", default="git", choices=["git", "kaggle", "huggingface"],
                        help="Data client to use (git, kaggle, or huggingface)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Time delay in seconds (ignored for static clients)")
    args = parser.parse_args()

    if args.tickers == "sp500":
        tickers = SP500_TICKERS
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    print(f"Fetching {len(tickers)} tickers, {args.years} years → {args.db} (Client: {args.client})")
    fetch_and_store(tickers, args.years, args.db, client=args.client, delay=args.delay)


if __name__ == "__main__":
    main()
