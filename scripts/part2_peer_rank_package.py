"""Package replay inputs and the complete per-seed node-rank curves."""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    manifests, results = a.manifests.resolve(), a.results.resolve()
    with zipfile.ZipFile(results / "replay_inputs.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(manifests.glob("*.json")):
            data = json.loads(path.read_text())
            dataset, seed = data["dataset"], data["seed"]
            split_arc = f"inputs/splits/{dataset}_seed{seed}.npz"
            z.write(data["split_path"], split_arc)
            data["split_path"] = f"../splits/{dataset}_seed{seed}.npz"
            for row in data["records"]:
                model = row["variant"]
                condition = row["condition"]
                arc = f"inputs/scores/{dataset}_{model}_s{seed}/predictions_{condition}.npz"
                z.write(row["scores_path"], arc)
                row["scores_path"] = f"../scores/{dataset}_{model}_s{seed}/predictions_{condition}.npz"
            z.writestr(f"inputs/manifests/{path.name}", json.dumps(data, indent=2) + "\n")
    with zipfile.ZipFile(results / "per_seed_curves.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder in sorted(p for p in results.glob("*_seed*") if p.is_dir()):
            for name in ("per_seed_curves.csv", "paired_deltas.csv", "node_ranks.csv", "audit.json"):
                z.write(folder / name, f"{folder.name}/{name}")
    for name in ("replay_inputs.zip", "per_seed_curves.zip"):
        print(name, (results / name).stat().st_size, flush=True)


if __name__ == "__main__":
    main()
