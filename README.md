# Market Foundation Model

**Stock price direction & volatility prediction using a Transformer Foundation Model**
Adapted from the Credit Foundation Model (PRAGMA) architecture.

---

## Architecture

Same 3-tier tokenization + PatchTST/Hybrid transformer, retargeted at equity markets:

| Layer | Detail |
|:---|:---|
| Vocabulary | 62 tokens: 5 special + 7 regime (BULL/BEAR/CRASH…) + 50 continuous bins |
| Sequence | 60 trading days per ticker, step = `[regime, RSI, vol, ret, ATR]` |
| Pre-training | Masked Patch Modeling (15%, same as credit CFM) |
| Fine-tuning | Direction (5-day UP/DOWN) + Volatility regime classification |
| Hardware | 2× NVIDIA H100 80GB, FP8 via `transformer_engine`, FlashAttention-3 |

---

## Quickstart (H100 server)

```bash
# 1. Clone / copy this workspace to the H100
cd /workspace

# 2. Single command: validates GPU, installs deps, fetches data, trains
chmod +x train-foundation-model/launch_h100.sh
./train-foundation-model/launch_h100.sh --arch hybrid --profile market --strategy full

# Smoke test (2 epochs, no H100 required)
./train-foundation-model/launch_h100.sh --arch lightweight --profile fast --pretrain-epochs 2
```

---

## Manual Step-by-Step

```bash
# Step 1: Fetch data
python market-data/fetch_market_data.py --tickers sp500 --years 5

# Step 2: Build sequences
python market-tokenizer/build_market_sequences.py

# Step 3: Single-GPU train
python train-foundation-model/train_foundation.py \
    --arch hybrid --profile market --strategy full

# Step 4: Multi-GPU (2× H100) train
torchrun --nproc_per_node=2 \
    train-foundation-model/train_foundation_ddp.py \
    --arch hybrid --profile market --strategy full
```

---

## Directory Structure

```
market-foundation-model-workspace/
│
├── market-data/
│   └── fetch_market_data.py        ← yfinance → market.db
│
├── market-tokenizer/
│   ├── market_tokenizer.py         ← MarketTokenizer (vocab=62)
│   └── build_market_sequences.py  ← 60-day windows → parquet
│
├── market-tokenizer-result/        ← auto-created after Step 2
│   ├── market_tokenizer.json
│   ├── market_sequences.parquet
│   └── sequence_stats.json
│
└── train-foundation-model/
    ├── config.py                   ← market profile + domain field
    ├── models.py                   ← PatchTST / Hybrid / TFT / LSTM
    ├── dataset.py                  ← MarketSequenceDataset (all modes)
    ├── losses.py                   ← Stage 1+2 (domain-agnostic)
    ├── market_losses.py            ← Stage 3 MarketFineTuneLoss
    ├── evaluator.py                ← Stage 1+2 evaluation
    ├── market_evaluator.py         ← Stage 3 market metrics
    ├── market_labels.py            ← direction + vol label maps
    ├── trainer.py                  ← Full training orchestrator
    ├── run_manager.py              ← Run tracking
    ├── train_foundation.py         ← Single-GPU CLI
    ├── train_foundation_ddp.py     ← torchrun DDP CLI
    └── launch_h100.sh              ← Bootstrap + launch
```

---

## Key Metrics

| Metric | Threshold for "working" |
|:---|:---|
| Directional Accuracy | > 52% (random = 50%) |
| AUC-ROC (direction) | > 0.55 |
| Top-Decile Precision | > 60% |
| Sharpe Signal Quality | > 0.3 (annualised) |

---

## Profiles

| Profile | embed_dim | layers | epochs | Use case |
|:---|:---|:---|:---|:---|
| `fast` | 32 | 1 | 5/3/3 | Smoke test, no GPU needed |
| `default` | 64 | 3 | 50/30/20 | Development |
| `market` | 128 | 4 | 80/50/30 | H100 production |
| `market_large` | 256 | 6 | 100/60/40 | Research ablations |
