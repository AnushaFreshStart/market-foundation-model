"""
config.py — Training Configuration for Market Foundation Model
================================================================
Drops the credit-specific terminology; adds 'market' profile and
a 'domain' field to switch between credit and market data pipelines.

Usage:
    cfg = TrainingConfig.load_profile('market')     # market profile on H100
    cfg = TrainingConfig.load_profile('fast')        # quick smoke-test
    cfg = TrainingConfig.from_dict({...})            # from JSON
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hyperparameter Profiles
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "embed_dim": 64,  "n_heads": 4,  "n_layers": 3,
        "patch_size": 5,  "ff_dim": 256, "dropout": 0.1,
        "batch_size": 64, "learning_rate": 1e-4,
        "pretrain_epochs": 50, "joint_epochs": 30, "finetune_epochs": 20,
        "vocab_size": 62, "max_seq_len": 60,
    },
    "small": {
        "embed_dim": 32,  "n_heads": 2,  "n_layers": 2,
        "patch_size": 5,  "ff_dim": 128, "dropout": 0.1,
        "batch_size": 128, "learning_rate": 2e-4,
        "pretrain_epochs": 30, "joint_epochs": 20, "finetune_epochs": 15,
        "vocab_size": 62, "max_seq_len": 60,
    },
    "fast": {
        "embed_dim": 32,  "n_heads": 2,  "n_layers": 1,
        "patch_size": 5,  "ff_dim": 64,  "dropout": 0.05,
        "batch_size": 128, "learning_rate": 3e-4,
        "pretrain_epochs": 5,  "joint_epochs": 3, "finetune_epochs": 3,
        "vocab_size": 62, "max_seq_len": 60,
    },
    "market": {
        # H100 profile — large model, FP8, 60-day sequences
        "embed_dim": 128, "n_heads": 8,  "n_layers": 4,
        "patch_size": 5,  "ff_dim": 512, "dropout": 0.1,
        "batch_size": 2048, "learning_rate": 8e-4,
        "pretrain_epochs": 80,  "joint_epochs": 50, "finetune_epochs": 30,
        "vocab_size": 62, "max_seq_len": 60,
        "precision": "fp8",
        "num_workers": 4, "pin_memory": True, "prefetch_factor": 2,
        "persistent_workers": True, "gradient_checkpointing": False,
    },
    "market_large": {
        "embed_dim": 256, "n_heads": 8,  "n_layers": 6,
        "patch_size": 5,  "ff_dim": 1024, "dropout": 0.1,
        "batch_size": 4096, "learning_rate": 1e-3,
        "pretrain_epochs": 100, "joint_epochs": 60, "finetune_epochs": 40,
        "vocab_size": 62, "max_seq_len": 60,
        "precision": "fp8",
        "num_workers": 4, "pin_memory": True, "prefetch_factor": 2,
        "persistent_workers": True, "gradient_checkpointing": False,
    },
    "h100_saturated": {
        "embed_dim": 768, "n_heads": 12, "n_layers": 12,
        "patch_size": 8, "ff_dim": 3072, "dropout": 0.1,
        "batch_size": 2048, "learning_rate": 8e-4,
        "pretrain_epochs": 100, "joint_epochs": 60, "finetune_epochs": 40,
        "vocab_size": 62, "max_seq_len": 512,
        "precision": "fp8",
        "num_workers": 8, "pin_memory": True, "prefetch_factor": 4,
        "persistent_workers": True, "gradient_checkpointing": False,
    },
}

VALID_ARCHITECTURES = ("patchtst", "tft", "hybrid", "lightweight", "lstm_baseline")
VALID_STRATEGIES    = ("full", "pretrain_only", "pretrain_finetune", "finetune_only", "joint_finetune")
VALID_DOMAINS       = ("market", "credit")


@dataclass
class TrainingConfig:
    """Full training configuration for the Market Foundation Model."""

    # -- Domain --
    domain: str = "market"      # 'market' or 'credit'

    # -- Architecture & strategy --
    architecture: str = "hybrid"
    strategy:     str = "full"
    profile:      str = "market"

    # -- Model hyperparameters --
    embed_dim:   int   = 128
    n_heads:     int   = 8
    n_layers:    int   = 4
    patch_size:  int   = 5
    ff_dim:      int   = 512
    dropout:     float = 0.1
    vocab_size:  int   = 62        # 5 special + 7 regime + 50 continuous bins
    max_seq_len: int   = 60        # 60 trading days
    step_width:  int   = 5

    # -- Training hyperparameters --
    batch_size:      int   = 256
    learning_rate:   float = 2e-4
    weight_decay:    float = 0.01
    pretrain_epochs: int   = 80
    joint_epochs:    int   = 50
    finetune_epochs: int   = 30

    # -- Loss weights (Stage 2: multi-objective) --
    alpha: float = 1.0
    beta:  float = 0.5
    gamma: float = 0.3

    # -- Loss weights (Stage 3: market multi-task) --
    vol_weight:   float = 0.4
    focal_gamma:  float = 2.0
    focal_alpha:  float = 0.6

    # -- Self-supervised --
    mask_prob: float = 0.15

    # -- Acceleration (H100 defaults) --
    use_amp:    bool = True
    precision:  str  = "fp8"     # 'fp8' on H100, 'fp16' on older GPUs

    # -- DataLoader (Linux/H100 optimised) --
    num_workers:        int  = 4
    pin_memory:         bool = True
    prefetch_factor:    int  = 2
    persistent_workers: bool = True

    # -- Memory --
    gradient_checkpointing: bool = False

    # -- Multi-GPU (set by torchrun env) --
    world_size: int = 1
    rank:       int = 0

    # -- Run management --
    save_embeddings: bool = True
    run_name:        str  = ""

    # -- Paths (auto-resolved via resolve_paths) --
    workspace_dir:    str = ""
    db_path:          str = ""
    sequences_path:   str = ""
    tokenizer_path:   str = ""
    output_dir:       str = ""

    # ---------------------------------------------------------------
    # Class methods
    # ---------------------------------------------------------------

    @classmethod
    def load_profile(cls, name: str, **overrides) -> "TrainingConfig":
        """Create a config from a named profile with optional overrides."""
        if name not in PROFILES and name != "custom":
            raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
        params = copy.deepcopy(PROFILES.get(name, {}))
        params["profile"] = name
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    # ---------------------------------------------------------------
    # Instance methods
    # ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def resolve_paths(self) -> None:
        """Auto-resolve paths relative to the workspace directory."""
        if not self.workspace_dir:
            self.workspace_dir = str(Path(__file__).parent.parent)

        ws = Path(self.workspace_dir)

        if not self.db_path:
            self.db_path = str(ws / "market-data" / "market.db")
        if not self.sequences_path:
            self.sequences_path = str(ws / "market-tokenizer-result" / "market_sequences.parquet")
        if not self.tokenizer_path:
            self.tokenizer_path = str(ws / "market-tokenizer-result" / "market_tokenizer.json")
        if not self.output_dir:
            self.output_dir = str(Path(__file__).parent)

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of warnings."""
        warnings_list: list[str] = []

        if self.architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"Invalid architecture '{self.architecture}'. Choose from: {VALID_ARCHITECTURES}")
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{self.strategy}'. Choose from: {VALID_STRATEGIES}")
        if self.domain not in VALID_DOMAINS:
            raise ValueError(f"Invalid domain '{self.domain}'. Choose from: {VALID_DOMAINS}")
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})")
        if self.max_seq_len % self.patch_size != 0:
            warnings_list.append(
                f"max_seq_len ({self.max_seq_len}) not divisible by patch_size ({self.patch_size}). "
                f"Last partial patch will be dropped."
            )
        if self.learning_rate > 1e-2:
            warnings_list.append(f"Learning rate {self.learning_rate} is very high for transformers.")

        return warnings_list

    def summary(self) -> str:
        return (
            f"Config: domain={self.domain}, arch={self.architecture}, strategy={self.strategy}, "
            f"profile={self.profile}\n"
            f"  Model: embed={self.embed_dim}, heads={self.n_heads}, "
            f"layers={self.n_layers}, patch={self.patch_size}, vocab={self.vocab_size}\n"
            f"  Training: bs={self.batch_size}, lr={self.learning_rate}, "
            f"epochs={self.pretrain_epochs}/{self.joint_epochs}/{self.finetune_epochs}\n"
            f"  Loss: α={self.alpha}, β={self.beta}, γ={self.gamma}, "
            f"vol_w={self.vol_weight}\n"
            f"  AMP: {self.precision if self.use_amp else 'disabled'}, "
            f"workers={self.num_workers}, pin_mem={self.pin_memory}\n"
            f"  GPU: world_size={self.world_size}, rank={self.rank}"
        )
