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

def fetch_and_store(
    tickers: list[str],
    years: int,
    db_path: str,
    batch_size: int = 20,
) -> None:
    """Download OHLCV for tickers and persist to DuckDB."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + 30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

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

    # Batch download
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        print(f"  Downloading batch {i // batch_size + 1}: {batch[:3]}...")
        try:
            raw = yf.download(
                batch, start=start_str, end=end_str,
                auto_adjust=True, progress=False, threads=True,
            )
        except Exception as e:
            print(f"  [WARN] Download failed for batch: {e}")
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                else:
                    if ticker not in raw.columns.get_level_values(1):
                        continue
                    df = raw.xs(ticker, axis=1, level=1).copy()

                df = df.dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index).normalize()

                # Rename columns
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                df["adj_close"] = df["close"]
                df["ticker"] = ticker
                df["date"] = df.index

                # Derived features
                df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
                df["rsi_14"] = compute_rsi(df["close"], 14)
                df["vol_5d"] = compute_realised_vol(df["log_ret"], 5)
                df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
                df["ret_5d"] = df["close"].pct_change(5)
                df["ret_20d"] = df["close"].pct_change(20)
                df["mcap_tier"] = 1  # placeholder; set by market cap tier logic below

                df = df.dropna()
                all_vols.extend(df["vol_5d"].tolist())

                # Write ohlcv
                ohlcv_df = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
                con.execute("DELETE FROM ohlcv WHERE ticker = ?", [ticker])
                con.execute("INSERT INTO ohlcv SELECT * FROM ohlcv_df")

                # Write derived
                derived_df = df[["ticker", "date", "log_ret", "rsi_14", "vol_5d", "atr_14", "ret_5d", "ret_20d", "mcap_tier"]].copy()
                con.execute("DELETE FROM derived WHERE ticker = ?", [ticker])
                con.execute("INSERT INTO derived SELECT * FROM derived_df")

                print(f"    {ticker}: {len(df)} rows")

            except Exception as e:
                print(f"    [WARN] {ticker}: {e}")
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
    parser.add_argument("--years", type=int, default=20,
                        help="Years of history to download")
    parser.add_argument("--db", default="market-data/market.db",
                        help="Output DuckDB path")
    args = parser.parse_args()

    if args.tickers == "sp500":
        tickers = SP500_TICKERS
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    print(f"Fetching {len(tickers)} tickers, {args.years} years → {args.db}")
    fetch_and_store(tickers, args.years, args.db)


if __name__ == "__main__":
    main()
