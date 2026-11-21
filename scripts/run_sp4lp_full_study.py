"""Run validation-selected full-data SP4LP and two independently trained ablations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from pathlib import Path

DATASETS = ("cora", "citeseer", "pubmed")
PROFILES = ("full64", "paper128", "release256", "full64_std", "paper128_std", "release256_std")


def run(a, dataset, profile, variant, seed):
    label = f"{dataset}_{profile}_{variant}_s{seed}"
    log = a.output / "logs" / f"{label}.log"
    log.parent.mkdir(exist_ok=True, parents=True)
    command = [
        sys.executable,
        "-u",
        "scripts/train_sp4lp_full.py",
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
    ]
    with log.open("a") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    result = json.loads((a.output / "models" / label / "complete.json").read_text())
    print(json.dumps(result), flush=True)
    return result


def select_dataset(a, dataset):
    candidates = [run(a, dataset, profile, "full", 0) for profile in PROFILES]
    winner = max(zip(PROFILES, candidates), key=lambda x: x[1]["best_validation"]["auc"])[0]
    choice = dict(profile=winner, selection_metric="seed0 validation AUROC", candidates=candidates)
    (a.output / f"selection_{dataset}.json").write_text(json.dumps(choice, indent=2) + "\n")
    return dataset, choice


def run_dataset(a, dataset):
    _, choice = select_dataset(a, dataset)
    for seed in (0, 1, 2):
        for variant in ("full", "endpoints", "featureless"):
            run(a, dataset, choice["profile"], variant, seed)
    return dataset, choice


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(select_dataset, a, dataset) for dataset in DATASETS]
        choices = dict(future.result() for future in futures)
        futures = [
            pool.submit(run, a, dataset, choices[dataset]["profile"], variant, seed)
            for dataset in DATASETS
            for seed in (0, 1, 2)
            for variant in ("full", "endpoints", "featureless")
        ]
        for future in futures:
            future.result()
    (a.output / "selection.json").write_text(json.dumps(choices, indent=2) + "\n")
    (a.output / "TRAINING_COMPLETE").write_text(
        "27 selected fits plus 15 unselected pilot fits completed.\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("peer", "data-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
