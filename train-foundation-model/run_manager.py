"""
run_manager.py â€” Multi-Run Save/Load/Compare for Training Experiments
=======================================================================
Manages training runs under runs/<run_id>/ with config, results,
checkpoints, and embeddings. Supports A/B comparison.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


class RunManager:
    """Manages multiple training runs for comparison."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, config) -> str:
        """Create a new run directory and save config. Returns run_id."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name or f"{config.architecture}_{config.strategy}_{config.profile}"
        run_id = f"{run_name}_{timestamp}"

        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(run_dir / "config.json")
        print(f"  Run: {run_id}")
        return run_id

    def save_run(
        self,
        run_id: str,
        results: dict,
        checkpoint: dict | None = None,
    ) -> None:
        """Save training results and optional checkpoint."""
        run_dir = self.runs_dir / run_id

        # Save results
        results_path = run_dir / "training_results.json"
        results_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"  OK Results â†’ {results_path.name}")

        # Save checkpoint
        if checkpoint:
            ckpt_path = run_dir / "checkpoint.pt"
            torch.save(checkpoint, str(ckpt_path))
            size_mb = ckpt_path.stat().st_size / 1_048_576
            print(f"  OK Checkpoint â†’ {ckpt_path.name} ({size_mb:.1f} MB)")

        # Copy embeddings if they exist
        embed_src = self.base_dir / "embeddings.parquet"
        if embed_src.exists():
            import shutil
            shutil.copy2(embed_src, run_dir / "embeddings.parquet")
            print(f"  OK Embeddings â†’ {run_id}/embeddings.parquet")

    def list_runs(self) -> list[dict]:
        """List all completed runs with summary metrics."""
        runs = []
        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            results_path = run_dir / "training_results.json"
            config_path = run_dir / "config.json"

            if not results_path.exists():
                continue

            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
                config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

                # Extract key metrics
                ft_metrics = results.get("stages", {}).get("finetune", {}).get("metrics", {})
                default_m = ft_metrics.get("default", {})

                runs.append({
                    "run_id": run_dir.name,
                    "architecture": config.get("architecture", "?"),
                    "strategy": config.get("strategy", "?"),
                    "profile": config.get("profile", "?"),
                    "auc_roc_direction: default_m.get("auc_roc_direction", 0.5),,
                    "gini_default": default_m.get("gini", 0),
                    "total_params": results.get("total_params", 0),
                    "total_time_s": results.get("total_time_s", 0),
                    "has_checkpoint": (run_dir / "checkpoint.pt").exists(),
                    "has_embeddings": (run_dir / "embeddings.parquet").exists(),
                })
            except Exception:
                continue

        # Sort by AUC descending
        runs.sort(key=lambda x: x.get("auc_roc_default", 0), reverse=True)
        return runs

    def load_run(self, run_id: str) -> dict:
        """Load full training results for a run."""
        results_path = self.runs_dir / run_id / "training_results.json"
        if not results_path.exists():
            return {"error": f"Run '{run_id}' not found"}
        return json.loads(results_path.read_text(encoding="utf-8"))

    def compare_runs(self, run_ids: list[str] | None = None) -> dict:
        """Compare multiple runs side by side."""
        from market_evaluator import MarketEvaluator as ModelEvaluator

        if run_ids is None:
            all_runs = self.list_runs()
            run_ids = [r["run_id"] for r in all_runs]

        run_results = []
        for run_id in run_ids:
            results = self.load_run(run_id)
            if "error" not in results:
                run_results.append({
                    "run_id": run_id,
                    "config": results.get("config", {}),
                    "metrics": {
                        stage: data.get("metrics", {})
                        for stage, data in results.get("stages", {}).items()
                    },
                    "params": results.get("total_params", 0),
                })

        evaluator = ModelEvaluator()
        baseline_path = self.base_dir.parent / "pipelines" / "model_results.json"
        return evaluator.compare_runs(run_results, baseline_path)

    def get_best_run(self, metric: str = "auc_roc_default") -> dict | None:
        """Find the run with the best value for a given metric."""
        runs = self.list_runs()
        if not runs:
            return None
        return max(runs, key=lambda x: x.get(metric, 0))

