"""Bundle verified results and inference weights without optimizer or dataset caches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_reproduction import historical, read_seed
from sgdiff.engine.reproduction import HEURISTICS, OWNED, PRIMARY, checkpoint_path, digest, expanded


def package(dataset, seed, names, root, weights, out):
    read_seed(root, dataset, seed, names, historical(dataset))
    base = root / dataset / f"seed_{seed}"
    selected = list(expanded(names))
    inference = {}
    for name in selected:
        if name in HEURISTICS:
            continue
        path = checkpoint_path(weights, dataset, seed, name)
        recipe = json.loads((base / name / "recipe.json").read_text())
        if digest(path) != recipe["checkpoint"]:
            raise ValueError(f"changed checkpoint: {path}")
        inference[path] = Path("runs/pair") / path.relative_to(weights)
        if name == "sgdiff":
            for extra in ("metrics.jsonl", "resolved_config.json", "result.json"):
                inference[path.parent / extra] = inference[path].parent / extra
    if not (base / "graphs/training_counts.json").exists():
        raise FileNotFoundError("run audit_reproduction.py to record the training cohorts first")
    results = {f: Path("results/paired-reproduction") / f.relative_to(root)
               for section in ["graphs", *selected] for f in (base / section).iterdir()
               if f.is_file() and not f.name.startswith(".")}
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temp:
        path = Path(temp) / f"{dataset}_seed{seed}.zip"
        with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=3) as z:
            for f, name in sorted({**inference, **results}.items()):
                z.write(f, str(name))
        with ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError(f"archive verification failed: {path}")
        path.replace(out / path.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=PRIMARY, default=list(OWNED))
    p.add_argument("--datasets", nargs="+", default=["cora","citeseer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--root", type=Path, default=Path("results/paired-reproduction"))
    p.add_argument("--weights", type=Path, default=Path("runs/pair"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    for d in a.datasets:
        for s in a.seeds:
            package(d, s, a.models, a.root, a.weights, a.out)
    with (a.out / "SHA256SUMS").open("w") as stream:
        for archive in sorted(a.out.glob("*.zip")):
            stream.write(f"{digest(archive)}  {archive.name}\n")
