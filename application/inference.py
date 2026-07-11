"""
inference.py — Live Inference Pipeline for Market Foundation Model
==================================================================
Connects to Alpha Vantage or Polygon.io to fetch the last 100 days of data
for a given ticker, preprocesses it, tokenizes the sequence, and runs
it through the trained model checkpoint for next-day predictions.

Usage:
  export ALPHAVANTAGE_API_KEY="your_api_key"
  python application/inference.py --ticker AAPL --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import requests

# Add the parent directory to sys.path to import training/tokenization modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from train-foundation-model.config import TrainingConfig
    from train-foundation-model.models import build_model
    from market-tokenizer.market-tokenizer import MarketTokenizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running from the market-foundation-model-workspace root.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Technical indicator helpers (from fetch_market_data.py)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr / (close + 1e-9)

def compute_realised_vol(log_ret: pd.Series, window: int = 5) -> pd.Series:
    return log_ret.rolling(window).std() * math.sqrt(252)

def classify_regime(ret_20d: float, vol_20d: float, rsi: float, p80_vol: float, p20_vol: float) -> str:
    if ret_20d > 0.05: return "BULL"
    if ret_20d < -0.10: return "CRASH" if vol_20d > p80_vol else "BEAR"
    if ret_20d < -0.03: return "CORR"
    if vol_20d < p20_vol and abs(ret_20d) < 0.02: return "FLAT"
    if rsi > 70 or rsi < 30: return "GAP"
    return "RECOV"


# ---------------------------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------------------------

def fetch_alphavantage(ticker: str) -> pd.DataFrame:
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable not set. Please export it.")
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={api_key}"
    print(f"  Fetching from Alpha Vantage: {ticker}...")
    resp = requests.get(url)
    data = resp.json()
    
    if "Time Series (Daily)" not in data:
        if "Information" in data:
            raise RuntimeError(f"API Rate limit reached: {data['Information']}")
        raise RuntimeError(f"Unexpected API response: {data}")
        
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", 
        "3. low": "low", "4. close": "close", "5. volume": "volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    df = df.astype(float)
    return df

def fetch_polygon(ticker: str) -> pd.DataFrame:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set. Please export it.")
        
    # Polygon free tier requires setting the dates.
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=150)).strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=120&apiKey={api_key}"
    
    print(f"  Fetching from Polygon.io: {ticker}...")
    resp = requests.get(url)
    data = resp.json()
    
    if "results" not in data:
        if "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")
        raise RuntimeError(f"Unexpected API response: {data}")
        
    df = pd.DataFrame(data["results"])
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "date"})
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df = df.set_index("date")
    df = df.sort_index(ascending=True)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and regimes."""
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["vol_5d"] = compute_realised_vol(df["log_ret"], 5)
    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)
    
    df = df.dropna().copy()
    
    # Use standard percentiles for regime fallback if we don't have universe distribution
    p80_vol, p20_vol = 0.4, 0.1  
    
    df["regime"] = df.apply(
        lambda r: classify_regime(r["ret_20d"], r["vol_5d"], r["rsi_14"], p80_vol, p20_vol), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# Main Inference
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live Inference Pipeline")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--provider", default="alphavantage", choices=["alphavantage", "polygon"], help="API provider")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to trained model .pt file")
    parser.add_argument("--tokenizer", default="market-tokenizer-result/market_tokenizer.json", help="Path to tokenizer .json")
    parser.add_argument("--config", default="market", help="Config profile name")
    args = parser.parse_args()

    print(f"=== Live Inference Pipeline ===")
    print(f"Ticker:    {args.ticker}")
    print(f"Provider:  {args.provider}")
    
    # 1. Fetch Data
    try:
        if args.provider == "alphavantage":
            raw_df = fetch_alphavantage(args.ticker)
        else:
            raw_df = fetch_polygon(args.ticker)
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch data: {e}")
        sys.exit(1)
        
    print(f"  Received {len(raw_df)} daily bars ending on {raw_df.index[-1].strftime('%Y-%m-%d')}.")
    
    # 2. Preprocess
    print("  Computing indicators and regimes...")
    df = preprocess_features(raw_df)
    
    if len(df) < 60:
        print(f"\n[ERROR] Not enough valid data points after feature generation. Need >= 60, got {len(df)}.")
        sys.exit(1)
        
    # Take latest 60 days
    df_seq = df.tail(60).copy()
    print(f"  Extracting latest sequence of {len(df_seq)} trading days.")

    # 3. Tokenize
    if not os.path.exists(args.tokenizer):
        print(f"\n[ERROR] Tokenizer not found at {args.tokenizer}. Have you built it?")
        sys.exit(1)
        
    print("  Loading MarketTokenizer...")
    tok = MarketTokenizer.load(args.tokenizer)
    
    sequence_ids = []
    for _, row in df_seq.iterrows():
        # encode_step expects dictionary
        row_dict = row.to_dict()
        ids = tok.encode_step(row_dict)
        sequence_ids.append(ids)
        
    # Shape: (1, 60, 5)
    seq_tensor = torch.tensor([sequence_ids], dtype=torch.long)
    mask_tensor = torch.ones((1, 60), dtype=torch.long)
    
    # 4. Model Inference
    print(f"  Loading Model Config Profile: {args.config}...")
    config = TrainingConfig.load_profile(args.config)
    # override vocab_size to match tokenizer
    config.vocab_size = tok.vocab_size
    
    model = build_model(config)
    
    if os.path.exists(args.checkpoint):
        print(f"  Loading Weights from {args.checkpoint}...")
        # Checkpoint might be a bare state_dict or a dict with "model_state_dict"
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"  [WARN] Checkpoint {args.checkpoint} not found. Running inference with UNTRAINED model for demonstration.")
        
    model.eval()
    
    print("\n=== Prediction for Next Trading Day ===")
    with torch.no_grad():
        # Run stage="finetune"
        out = model(seq_tensor, mask_tensor, stage="finetune")
        
        dir_logit = out["direction_logit"].item()
        vol_logit = out["vol_logit"].item()
        
        # Apply sigmoid to convert logit -> probability
        p_up = 1.0 / (1.0 + math.exp(-dir_logit))
        p_vol = 1.0 / (1.0 + math.exp(-vol_logit))
        
        print(f"  Ticker:               {args.ticker}")
        print(f"  Model Direction (UP): {p_up:.2%} probability")
        print(f"  Model Volatility:     {p_vol:.2%} probability (High Volatility Regime)")
        
        # Human readable conclusion
        if p_up > 0.6:
            trend = "STRONGLY BULLISH 🟢"
        elif p_up > 0.5:
            trend = "BULLISH ↗️"
        elif p_up > 0.4:
            trend = "BEARISH ↘️"
        else:
            trend = "STRONGLY BEARISH 🔴"
            
        print(f"  Overall Trend Signal: {trend}")

if __name__ == "__main__":
    main()
