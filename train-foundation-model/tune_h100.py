"""
tune_h100.py — H100 Throughput & Memory Profiler / Auto-Tuner
==============================================================
Benchmarks various model profiles, batch sizes, and precisions
to find the optimal training configuration for maximum H100 throughput.

Usage:
    python tune_h100.py --gpus 2
"""

from __future__ import annotations

import argparse
import time
import torch
import numpy as np
import pandas as pd

from config import TrainingConfig
from models import build_model
from losses import MaskedPredictionLoss

def benchmark_config(
    arch: str,
    embed_dim: int,
    n_heads: int,
    n_layers: int,
    batch_size: int,
    precision: str,
    max_seq_len: int,
    steps: int = 15,
    warmup_steps: int = 5,
) -> dict:
    """Run forward and backward passes on dummy data to measure speed and VRAM."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Force clean start without triggering CUDA asserts on some driver/runtime combinations
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    
    cfg = TrainingConfig(
        architecture=arch,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        batch_size=batch_size,
        precision=precision,
        max_seq_len=max_seq_len,
        vocab_size=62,
        step_width=5,
    )
    
    try:
        model = build_model(cfg).to(device)
        loss_fn = MaskedPredictionLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # FP8 Autocast Setup
        use_amp = True
        use_fp8 = (precision == "fp8")
        scaler = torch.amp.GradScaler("cuda") if (use_amp and not use_fp8) else None
        
        if use_fp8:
            try:
                import transformer_engine.pytorch as te
                ctx = te.fp8_autocast(enabled=True)
            except Exception:
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        else:
            ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)

        # Generate dummy batch
        input_ids = torch.randint(0, 62, (batch_size, max_seq_len, 5), device=device)
        attention_mask = torch.ones((batch_size, max_seq_len), dtype=torch.long, device=device)
        labels = torch.randint(-100, 62, (batch_size, max_seq_len), device=device)
        labels[:, ::2] = -100 # Mask alternate tokens

        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            with ctx:
                outputs = model(input_ids, attention_mask, stage="pretrain")
                loss = loss_fn(outputs["logits"], labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        
        # Benchmark loop with a smaller, safer number of iterations
        start_time = time.perf_counter()
        for _ in range(min(steps, 8)):
            optimizer.zero_grad()
            with ctx:
                outputs = model(input_ids, attention_mask, stage="pretrain")
                loss = loss_fn(outputs["logits"], labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        bench_steps = min(steps, 8)
        avg_step_time = total_time / bench_steps
        throughput = (batch_size * bench_steps) / total_time
        
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        return {
            "status": "SUCCESS",
            "avg_step_ms": round(avg_step_time * 1000, 2),
            "throughput_seq_sec": round(throughput, 2),
            "throughput_tokens_sec": round(throughput * max_seq_len, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "vram_util_pct": round((peak_vram_gb / 80.0) * 100, 1),
        }
        
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            return {"status": "OOM", "avg_step_ms": 0, "throughput_seq_sec": 0, "throughput_tokens_sec": 0, "peak_vram_gb": 0, "vram_util_pct": 0}
        return {"status": f"ERROR: {str(e)[:80]}", "avg_step_ms": 0, "throughput_seq_sec": 0, "throughput_tokens_sec": 0, "peak_vram_gb": 0, "vram_util_pct": 0}

def main():
    parser = argparse.ArgumentParser(description="H100 Optimization Auto-Tuner")
    parser.add_argument("--gpus", type=int, default=2, help="Number of H100 GPUs available")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available. Auto-tuning requires at least one H100 GPU.")
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    print(f"============================================================")
    print(f"  H100 AUTO-TUNING BENCHMARK SUITE")
    print(f"  Primary Device: {gpu_name}")
    print(f"  GPUs configurations scaled for {args.gpus}x H100 DDP setup")
    print(f"============================================================\n")

    # Grid of candidate configurations
    search_space = [
        # Small / Debug models
        {"arch": "hybrid", "embed_dim": 64, "n_heads": 4, "n_layers": 3, "batch_size": 1024, "precision": "fp16", "max_seq_len": 60},
        {"arch": "hybrid", "embed_dim": 64, "n_heads": 4, "n_layers": 3, "batch_size": 2048, "precision": "fp8",  "max_seq_len": 60},
        
        # Standard Production configurations
        {"arch": "hybrid", "embed_dim": 128, "n_heads": 8, "n_layers": 4, "batch_size": 1024, "precision": "fp16", "max_seq_len": 60},
        {"arch": "hybrid", "embed_dim": 128, "n_heads": 8, "n_layers": 4, "batch_size": 2048, "precision": "fp8",  "max_seq_len": 60},
        {"arch": "hybrid", "embed_dim": 128, "n_heads": 8, "n_layers": 4, "batch_size": 4096, "precision": "fp8",  "max_seq_len": 60},
        
        # Large scaling candidates
        {"arch": "hybrid", "embed_dim": 256, "n_heads": 8, "n_layers": 6, "batch_size": 1024, "precision": "fp16", "max_seq_len": 60},
        {"arch": "hybrid", "embed_dim": 256, "n_heads": 8, "n_layers": 6, "batch_size": 2048, "precision": "fp8",  "max_seq_len": 60},
        {"arch": "hybrid", "embed_dim": 256, "n_heads": 8, "n_layers": 6, "batch_size": 4096, "precision": "fp8",  "max_seq_len": 60},
        
        # Extreme sequences length scaling (120 trading days ~ 6 months window)
        {"arch": "hybrid", "embed_dim": 128, "n_heads": 8, "n_layers": 4, "batch_size": 2048, "precision": "fp8",  "max_seq_len": 120},
    ]

    results = []
    
    for i, cfg in enumerate(search_space):
        print(f"Testing {i+1}/{len(search_space)}: Arch={cfg['arch']} | Embed={cfg['embed_dim']} | BS={cfg['batch_size']} | Prec={cfg['precision']} | SeqLen={cfg['max_seq_len']}...")
        metrics = benchmark_config(**cfg)
        print(f"  -> Status: {metrics['status']} | Throughput: {metrics['throughput_tokens_sec']} tokens/s | Peak VRAM: {metrics['peak_vram_gb']} GB")
        results.append({**cfg, **metrics})
        
    df = pd.DataFrame(results)
    
    # Compute multi-GPU DDP projected performance
    df["projected_ddp_seq_sec"] = df["throughput_seq_sec"] * args.gpus * 0.92  # 92% scaling efficiency factor
    df["projected_ddp_tokens_sec"] = df["projected_ddp_seq_sec"] * df["max_seq_len"]

    # Filter out failures
    success_df = df[df["status"] == "SUCCESS"].copy()
    
    print(f"\n\n============================================================")
    print(f"  BENCHMARK RESULTS (Ranked by Projected DDP Throughput)")
    print(f"============================================================")
    
    if success_df.empty:
        print("No configurations succeeded. Check CUDA availability and installations.")
        return
        
    sorted_df = success_df.sort_values(by="projected_ddp_tokens_sec", ascending=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(sorted_df[["embed_dim", "n_layers", "batch_size", "precision", "max_seq_len", "throughput_tokens_sec", "projected_ddp_tokens_sec", "peak_vram_gb", "vram_util_pct"]])
    
    best_config = sorted_df.iloc[0]
    print(f"\n============================================================")
    print(f"  🏆 RECOMMENDED CONFIGURATION FOR MAXIMUM H100 UTILIZATION")
    print(f"============================================================")
    print(f"  Model Size:  Embed Dim={best_config['embed_dim']} | Layers={best_config['n_layers']} | Heads={best_config['n_heads']}")
    print(f"  Sequence:    Max Sequence Length={best_config['max_seq_len']}")
    print(f"  Batch Size:  {best_config['batch_size']} (per-GPU) -> Total DDP batch size of {best_config['batch_size'] * args.gpus}")
    print(f"  Precision:   {best_config['precision'].upper()} (Transformer Engine Autocast)")
    print(f"  Peak VRAM:   {best_config['peak_vram_gb']} GB ({best_config['vram_util_pct']}% of H100 80GB)")
    print(f"  Throughput:  {best_config['projected_ddp_tokens_sec']:.0f} tokens/second across both GPUs")
    print(f"============================================================")
    
    print("\nTo launch DDP training with this optimal configuration, run:")
    print(f"torchrun --nproc_per_node={args.gpus} train_foundation_ddp.py \\")
    print(f"    --arch {best_config['arch']} \\")
    print(f"    --embed-dim {best_config['embed_dim']} \\")
    print(f"    --n-heads {best_config['n-heads'] if 'n-heads' in best_config else best_config['n_heads']} \\")
    print(f"    --n-layers {best_config['n_layers']} \\")
    print(f"    --batch-size {best_config['batch_size']} \\")
    print(f"    --profile custom")

if __name__ == "__main__":
    main()
