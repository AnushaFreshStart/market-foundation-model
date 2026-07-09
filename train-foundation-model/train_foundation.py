#!/usr/bin/env python
"""
train_foundation.py — CLI Entry Point (single-GPU) for Market Foundation Model
================================================================================
Called via subprocess or directly from command line.

Usage:
    python train_foundation.py --arch hybrid --strategy full --profile market
    python train_foundation.py --arch all --strategy pretrain_only
    python train_foundation.py --list-runs
    python train_foundation.py --compare-all
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from config import TrainingConfig, VALID_ARCHITECTURES


def main():
    parser = argparse.ArgumentParser(
        description="Market Foundation Model Training (single-GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default="hybrid",
                        choices=list(VALID_ARCHITECTURES) + ["all"])
    parser.add_argument("--strategy", default="full",
                        choices=["full", "pretrain_only", "pretrain_finetune",
                                 "finetune_only", "joint_finetune"])
    parser.add_argument("--profile", default="market",
                        choices=["default", "small", "fast", "market", "market_large", "custom"])
    parser.add_argument("--domain", default="market", choices=["market", "credit"])
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
    parser.add_argument("--list-runs",   action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--compare",     nargs="+",      default=None)
    args = parser.parse_args()

    from trainer import CreditModelTrainer
    from run_manager import RunManager

    base_dir = Path(__file__).parent
    run_mgr  = RunManager(base_dir)

    if args.list_runs:
        runs = run_mgr.list_runs()
        if not runs:
            print("No completed runs found.")
        else:
            print(f"\n{'\u2500' * 80}")
            print(f"  {'Run ID':<45} {'Arch':<12} {'Dir AUC':>8}  {'DirAcc':>7}  {'Time':>6}")
            print(f"{'\u2500' * 80}")
            for r in runs:
                print(
                    f"  {r['run_id']:<45} "
                    f"{r['architecture']:<12} "
                    f"{r.get('auc_roc_direction', 0):>8.4f}  "
                    f"{r.get('directional_accuracy', 0):>7.4f}  "
                    f"{r.get('total_time_s', 0):>5.0f}s"
                )
            print(f"{'\u2500' * 80}")
        print(json.dumps(runs, indent=2))
        return

    if args.compare_all or args.compare:
        comparison = run_mgr.compare_runs(args.compare)
        print(json.dumps(comparison, indent=2))
        return

    architectures = list(VALID_ARCHITECTURES) if args.arch == "all" else [args.arch]
    all_results   = []

    for arch in architectures:
        print(f"\n{'=' * 60}")
        print(f"  TRAINING: {arch.upper()} / {args.strategy} / {args.profile}")
        print(f"{'=' * 60}")

        overrides = {"architecture": arch, "strategy": args.strategy, "domain": args.domain}
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

        warnings = config.validate()
        for w in warnings:
            print(f"  \u26a0 {w}")
        print(config.summary())

        run_id  = run_mgr.start_run(config)
        trainer = CreditModelTrainer(config, rank=0, world_size=1)
        results = trainer.train()
        results["run_id"] = run_id
        run_mgr.save_run(run_id, results, checkpoint=trainer.get_checkpoint())

        all_results.append({"run_id": run_id, "architecture": arch, "results": results})

    if len(architectures) > 1:
        print(f"\n{'=' * 60}\n  COMPARISON ACROSS {len(architectures)} ARCHITECTURES\n{'=' * 60}")
        comparison = run_mgr.compare_runs([r["run_id"] for r in all_results])
        print(json.dumps(comparison, indent=2))

    print("\n---JSON_RESULTS_START---")
    print(json.dumps(all_results, indent=2, default=str))
    print("---JSON_RESULTS_END---")


if __name__ == "__main__":
    main()
