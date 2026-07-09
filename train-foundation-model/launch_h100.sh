#!/usr/bin/env bash
# =============================================================================
# launch_h100.sh — Bootstrap & Launch Script for 2× H100 Pre-Training
# =============================================================================
# Validates 2× H100 GPUs, installs required packages, and fires torchrun.
#
# Usage (on the H100 server):
#   chmod +x launch_h100.sh
#   ./launch_h100.sh --arch hybrid --profile market --strategy full
#   ./launch_h100.sh --arch lightweight --profile fast --pretrain-epochs 3   # smoke test
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments (forward all unknown flags to train_foundation_ddp.py)
# ---------------------------------------------------------------------------
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    TRAIN_ARGS+=("$1")
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  Market Foundation Model — H100 Launch Script"
echo "  Workspace: $WORKSPACE_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. GPU Validation
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Validating GPU setup..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "      Detected: $N_GPUS GPU(s)"

if [ "$N_GPUS" -lt 2 ]; then
    echo "  [WARN] Only $N_GPUS GPU detected. Running single-GPU mode."
    NPROC=1
else
    NPROC=2
    echo "      Running 2-GPU DDP mode."
fi

# ---------------------------------------------------------------------------
# 2. Python / pip check
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Checking Python..."
python3 --version || { echo "ERROR: python3 not found"; exit 1; }
PYTHON=python3

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installing/verifying dependencies..."

pip install --quiet --upgrade \
    "torch>=2.4" \
    torchvision \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "pyarrow>=14.0" \
    duckdb \
    yfinance \
    scikit-learn \
    || echo "  [WARN] Some packages may already be installed"

# Transformer Engine (FP8, H100-native)
if python3 -c "import transformer_engine" 2>/dev/null; then
    echo "      transformer_engine: OK"
else
    echo "      Installing transformer_engine..."
    pip install --quiet "transformer-engine[pytorch]>=1.9" \
        || echo "  [WARN] transformer_engine install failed — falling back to FP16"
fi

# FlashAttention-3
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "      flash_attn: OK"
else
    echo "      Installing flash_attn (may take a few minutes for CUDA compilation)..."
    pip install --quiet flash-attn --no-build-isolation \
        || echo "  [WARN] flash_attn install failed — falling back to nn.MultiheadAttention"
fi

# ---------------------------------------------------------------------------
# 4. Data pipeline (if market.db doesn't exist)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Checking data pipeline..."

DB_PATH="$WORKSPACE_DIR/market-data/market.db"
SEQ_PATH="$WORKSPACE_DIR/market-tokenizer-result/market_sequences.parquet"

if [ ! -f "$DB_PATH" ]; then
    echo "      market.db not found. Fetching market data (S&P 500, 5 years)..."
    cd "$WORKSPACE_DIR"
    $PYTHON market-data/fetch_market_data.py --tickers sp500 --years 5 --db "$DB_PATH"
else
    echo "      market.db: found ($DB_PATH)"
fi

if [ ! -f "$SEQ_PATH" ]; then
    echo "      Sequences not found. Building market_sequences.parquet..."
    cd "$WORKSPACE_DIR"
    $PYTHON market-tokenizer/build_market_sequences.py \
        --db "$DB_PATH" \
        --out "$WORKSPACE_DIR/market-tokenizer-result"
else
    echo "      market_sequences.parquet: found"
fi

# ---------------------------------------------------------------------------
# 5. Launch training
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Launching torchrun with $NPROC GPU(s)..."
echo "      Args: ${TRAIN_ARGS[*]:-'(none — using defaults)'}"
echo ""

cd "$SCRIPT_DIR"

if [ "$NPROC" -eq 2 ]; then
    torchrun \
        --nproc_per_node=2 \
        --master_addr=localhost \
        --master_port=29500 \
        train_foundation_ddp.py \
        "${TRAIN_ARGS[@]}"
else
    $PYTHON train_foundation.py "${TRAIN_ARGS[@]}"
fi

echo ""
echo "============================================================"
echo "  Training complete. Check train-foundation-model/runs/"
echo "============================================================"
