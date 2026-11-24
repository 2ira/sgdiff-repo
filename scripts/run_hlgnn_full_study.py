"""Run validation-selected HL-GNN training (2 profiles x 3 datasets pilot, then 27 formal fits)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hlgnn.model import PROFILES, VARIANTS

DATASETS = ("cora", "citeseer", "pubmed")


def run_one(a, dataset: str, profile: str, variant: str, seed: int) -> dict:
    label = f"{dataset}_{profile}_{variant}_s{seed}"
    log = a.output / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_hlgnn_full.py"),
        "--peer",
        str(a.peer),
        "--data-root",
        str(a.data_root),
        "--output",
        str(a.output),
        "--dataset",
        dataset,
        "--profile",
        profile,
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--epochs",
        str(a.epochs),
        "--eval-every",
        str(a.eval_every),
        "--patience",
        str(a.patience),
        "--device",
        a.device,
        "--threads",
        str(a.threads),
    ]
    with log.open("a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    result = json.loads((a.output / "models" / label / "complete.json").read_text())
    print(json.dumps(dict(run=result["run"], best_epoch=result["best_epoch"], epochs_run=result["epochs_run"], val_auc=result["best_validation"]["auc"])), flush=True)
    return result


def main(a) -> None:
    a.output.mkdir(parents=True, exist_ok=True)
    choices = {}
    for dataset in DATASETS:
        candidates = [run_one(a, dataset, profile, "full", 0) for profile in PROFILES]
        winner = max(zip(PROFILES, candidates), key=lambda x: x[1]["best_validation"]["auc"])[0]
        choices[dataset] = dict(
            profile=winner,
            selection_metric="seed0 clean validation AUROC",
            candidates=[
                {
                    "run": c["run"],
                    "profile": c["profile"],
                    "best_epoch": c["best_epoch"],
                    "epochs_run": c["epochs_run"],
                    "updates": c["updates"],
                    "best_validation": c["best_validation"],
                    "checkpoint_sha256": c["checkpoint_sha256"],
                }
                for c in candidates
            ],
        )
        (a.output / f"selection_{dataset}.json").write_text(json.dumps(choices[dataset], indent=2) + "\n")

    (a.output / "selection.json").write_text(json.dumps(choices, indent=2) + "\n")
    print("SELECTION_COMPLETE:", json.dumps({d: v["profile"] for d, v in choices.items()}), flush=True)

    for dataset in DATASETS:
        profile = choices[dataset]["profile"]
        for seed in (0, 1, 2):
            for variant in VARIANTS:
                run_one(a, dataset, profile, variant, seed)

    (a.output / "TRAINING_COMPLETE").write_text("27 formal fits completed.\n")
    print("ALL_27_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    main(p.parse_args())
