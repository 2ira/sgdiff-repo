"""Package trained weights, histories and predictions independently of remote paths."""
import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile, ZIP_DEFLATED


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    for filename in ("per_seed.csv", "provenance.json"):
        copyfile(a.evaluation / filename, a.output / filename)
    copyfile(a.training / "selection.json", a.output / "selection.json")
    copyfile(a.training / "runtime.json", a.output / "runtime.json")
    checksums = []
    for dataset in ("cora", "citeseer", "pubmed"):
        for seed in (0, 1, 2):
            dest = a.output / f"{dataset}_seed{seed}.zip"
            with ZipFile(dest, "w", ZIP_DEFLATED, compresslevel=3) as z:
                for run in sorted((a.training / "models").glob(f"{dataset}_*_s{seed}")):
                    for name in ("best.pt", "receipt.json", "history.json", "complete.json"):
                        z.write(run / name, f"training/models/{run.name}/{name}")
                for run in sorted((a.evaluation / "runs").glob(f"{dataset}_*_s{seed}")):
                    for file in sorted(run.glob("*")):
                        if file.suffix in (".npz", ".json"):
                            z.write(file, f"evaluation/runs/{run.name}/{file.name}")
            with ZipFile(dest) as z:
                assert z.testzip() is None
            checksums.append(f"{hashlib.sha256(dest.read_bytes()).hexdigest()}  {dest.name}")
    dest = a.output / "evaluation-inputs.zip"
    with ZipFile(dest, "w", ZIP_DEFLATED, compresslevel=3) as z:
        for folder in ("attacks", "splits"):
            for file in sorted((a.evaluation / folder).glob("*.npz")):
                z.write(file, f"evaluation/{folder}/{file.name}")
        for dataset in ("cora", "citeseer", "pubmed"):
            file = a.data_root / "processed" / f"{dataset}.npz"
            z.write(file, f"data/processed/{file.name}")
    with ZipFile(dest) as z:
        assert z.testzip() is None
    checksums.append(f"{hashlib.sha256(dest.read_bytes()).hexdigest()}  {dest.name}")
    for name in ("per_seed.csv", "provenance.json", "selection.json", "runtime.json"):
        checksums.append(f"{hashlib.sha256((a.output / name).read_bytes()).hexdigest()}  {name}")
    (a.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("training", "evaluation", "data-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
