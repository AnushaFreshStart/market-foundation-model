#!/usr/bin/env python
"""
train_foundation_ddp.py — DDP Entry Point for 2× H100 Pre-Training
======================================================================
Launched by torchrun. Initialises NCCL process group, sets rank/world_size,
wraps model in DistributedDataParallel, runs training, and saves results
only from rank-0.

Launch:
    torchrun --nproc_per_node=2 train_foundation_ddp.py \
        --arch hybrid --profile market --strategy full
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.distributed as dist

from config import TrainingConfig, VALID_ARCHITECTURES


def main():
    parser = argparse.ArgumentParser(description="Market Foundation Model — DDP Training")
    parser.add_argument("--arch",            default="hybrid",
                        choices=list(VALID_ARCHITECTURES) + ["all"])
    parser.add_argument("--strategy",        default="full",
                        choices=["full", "pretrain_only", "pretrain_finetune",
                                 "finetune_only", "joint_finetune"])
    parser.add_argument("--profile",         default="market",
                        choices=["default", "small", "fast", "market", "market_large", "custom"])
    parser.add_argument("--domain",          default="market", choices=["market", "credit"])
    parser.add_argument("--embed-dim",       type=int,   default=None)
    parser.add_argument("--n-heads",         type=int,   default=None)
    parser.add_argument("--n-layers",        type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--pretrain-epochs", type=int,   default=None)
    parser.add_argument("--joint-epochs",    type=int,   default=None)
    parser.add_argument("--finetune-epochs", type=int,   default=None)
    parser.add_argument("--db",              type=str,   default=None)
    parser.add_argument("--sequences",       type=str,   default=None)
    parser.add_argument("--output",          type=str,   default=None)
    args = parser.parse_args()

    # ---- Distributed setup ----
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"  DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    from trainer import CreditModelTrainer
    from run_manager import RunManager

    base_dir = Path(__file__).parent
    run_mgr  = RunManager(base_dir)

    architectures = list(VALID_ARCHITECTURES) if args.arch == "all" else [args.arch]
    all_results   = []

    for arch in architectures:
        if rank == 0:
            print(f"\n{'=' * 60}\n  TRAINING: {arch.upper()} / {args.strategy} / {args.profile}\n{'=' * 60}")

        overrides = {
            "architecture": arch,
            "strategy":     args.strategy,
            "domain":       args.domain,
            "world_size":   world_size,
            "rank":         rank,
        }
        for flag, key in [
            (args.embed_dim,       "embed_dim"),
            (args.n_heads,         "n_heads"),
            (args.n_layers,        "n_layers"),
            (args.lr,              "learning_rate"),
            (args.batch_size,      "batch_size"),
            (args.pretrain_epochs, "pretrain_epochs"),
            (args.joint_epochs,    "joint_epochs"),
            (args.finetune_epochs, "finetune_epochs"),
        ]:
            if flag is not None:
                overrides[key] = flag
        if args.db:        overrides["db_path"]       = args.db
        if args.sequences: overrides["sequences_path"] = args.sequences
        if args.output:    overrides["output_dir"]     = args.output

        config = TrainingConfig.load_profile(args.profile, **overrides)
        config.resolve_paths()

        if rank == 0:
            warnings = config.validate()
            for w in warnings:
                print(f"  \u26a0 {w}")
            print(config.summary())

        # Only rank-0 creates the run directory
        run_id = run_mgr.start_run(config) if rank == 0 else "ddp_run"

        # Synchronise run_id across ranks
        if world_size > 1:
            run_id_list = [run_id]
            dist.broadcast_object_list(run_id_list, src=0)
            run_id = run_id_list[0]

        trainer = CreditModelTrainer(config, rank=rank, world_size=world_size)
        results = trainer.train()
        results["run_id"] = run_id

        # Only rank-0 saves
        if rank == 0:
            run_mgr.save_run(run_id, results, checkpoint=trainer.get_checkpoint())
            all_results.append({"run_id": run_id, "architecture": arch, "results": results})

    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print("\n---JSON_RESULTS_START---")
        print(json.dumps(all_results, indent=2, default=str))
        print("---JSON_RESULTS_END---")


if __name__ == "__main__":
    main()
