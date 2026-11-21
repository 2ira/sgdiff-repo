"""Evaluate the two frozen course families with Lin Junxiong's pinned native evaluator."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments
from sgdiff.engine.course_baselines import score_course_model
from sgdiff.engine.factory import load_split
from sgdiff.engine.reproduction import checkpoint_path, digest as file_digest, expanded, restore
from sgdiff.engine.team_study import settings

REVISION = "95083da808e500c02da458e205f4b861ceb80871"


def bind_peer(peer):
    """Load only the original attack and evaluation modules at the recorded revision."""
    rev = subprocess.check_output(["git", "-C", str(peer), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError(f"Expected peer checkout {REVISION}, found {rev}")
    experiments.__path__.append(str((peer / "experiments").resolve()))
    from experiments.new_link_models import protocol, run

    for module in (protocol, run):
        relative = f"experiments/new_link_models/{Path(module.__file__).name}"
        source = (peer / relative).resolve()
        original = subprocess.check_output(["git", "-C", str(peer), "show", f"{REVISION}:{relative}"])
        if Path(module.__file__).resolve() != source or source.read_bytes() != original:
            raise ValueError(f"Evaluator does not match pinned peer source: {source}")
    return protocol, run


class FrozenPredictor:
    device = torch.device("cpu")

    def __init__(self, model):
        self.model = model.eval()

    def eval(self):
        self.model.eval()

    def context(self, graph):
        return graph

    def score(self, graph, pairs):
        return torch.from_numpy(score_course_model(self.model, graph, pairs.cpu().numpy()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--peer-data", type=Path, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--weights", type=Path, default=Path("runs/pair"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--pubmed-ea-cache", type=Path,
                   help="Also run all 14 PubMed conditions using the published H200 graphs")
    a = p.parse_args()
    protocol, native = bind_peer(a.peer)
    digest = protocol.digest
    evaluator_hash = file_digest(Path(native.__file__))

    torch.set_num_threads(2)
    peer_results = a.peer / "experiments/new_link_models/benchmarks"
    published = {x["file"]: x["graph_sha256"] for x in json.loads((peer_results / "protocol_audit.json").read_text())["attacks"]}
    a.tables.mkdir(exist_ok=True, parents=True)
    rows = []
    provenance = []
    for dataset in ("cora", "citeseer"):
        graph = protocol.load_data(a.peer_data, dataset)
        for seed in (0, 1, 2):
            split, split_sha = protocol.split_data(graph, seed, a.output)
            cfg = settings(dataset, seed, "reference", str(a.data_root), "cpu")
            old, original = load_split(cfg)
            np.testing.assert_array_equal(old.edges, graph.edges)
            np.testing.assert_array_equal(old.x, graph.x)
            for k in ("train", "val", "test"):
                for label in ("pos", "neg"):
                    key = f"{k}_{label}"
                    np.testing.assert_array_equal(getattr(split, key), getattr(original, key))
            for name in expanded(("katz", "sp4lp")):
                ckpt = checkpoint_path(a.weights, dataset, seed, name)
                model = restore(name, cfg, graph, ckpt, "cpu", None)
                run = a.output / "runs" / f"{dataset}_{name}_s{seed}"
                run.mkdir(parents=True, exist_ok=True)
                source = dict(dataset=dataset, seed=seed, model=name, peer_commit=REVISION, split=split_sha,
                              weights=file_digest(ckpt) if name.startswith("sp4lp") else "parameter-free",
                              positive=digest(split.test_pos), negative=digest(split.test_neg),
                              features=digest(graph.x), classes=digest(graph.y))
                receipt = run / "receipt.json"
                if receipt.exists() and json.loads(receipt.read_text()) != source:
                    raise ValueError(f"inputs changed: {run}")
                receipt.write_text(json.dumps(source, indent=2)+"\n")
                provenance.append(dict(**source, evaluator="experiments.new_link_models.run.evaluate",
                                       evaluator_sha256=evaluator_hash))
                family = "sp4lp" if name.startswith("sp4lp") else "katz"
                args = SimpleNamespace(dataset=dataset, seed=seed, model=family, variant=name,
                                       output=a.output, eval_batch_size=512)
                records = native.evaluate(FrozenPredictor(model), graph, split, args, run)
                assert len(records) == len(native.variants()) == 14
                for r in records:
                    attack, ratio = r["attack"], r["ratio"]
                    tag = attack if attack in ("clean", "feature_shuffle") else f"{attack}_{ratio:g}"
                    with np.load(run / f"predictions_{tag}.npz") as scores:
                        pos, neg = scores["positive"], scores["negative"]
                        assert len(pos)==len(split.test_pos) and len(neg)==len(split.test_neg)
                        assert np.isfinite(pos).all() and np.isfinite(neg).all()
                        replay = protocol.metrics(pos,neg)
                    for key in replay:
                        assert replay[key] == r["metrics"][key]
                    actual = r["statistics"]["graph_sha256"]
                    saved_tag = "clean" if attack == "feature_shuffle" else tag
                    expected = published[f"{dataset}_s{seed}_{saved_tag}.npz"]
                    rows.append(dict(dataset=dataset, seed=seed, family=family, model=name, condition=tag,
                                     positives=len(pos), negatives=len(neg), graph_hash_matches=int(actual==expected),
                                     graph_sha256=actual, peer_graph_sha256=expected,
                                     **{k:replay[k] for k in ("auc","ap","hits100","precision100")}))
    with (a.tables / "per_seed.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (a.tables / "provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    assert len(rows)==504
    print(f"Replayed all {len(rows)} cells; {sum(r['graph_hash_matches'] for r in rows)} match the published graph hashes.")

    from summarize_peer_protocol import summarize
    summarize(a.tables)
    if a.pubmed_ea_cache:
        from evaluate_pubmed_extension import evaluate_pubmed
        pubmed_args = SimpleNamespace(peer=a.peer, data_root=a.data_root, weights=a.weights,
                                      tables=a.tables / "pubmed", output=a.output / "pubmed",
                                      ea_cache=a.pubmed_ea_cache)
        evaluate_pubmed(pubmed_args)
        summarize(pubmed_args.tables)


if __name__ == "__main__":
    main()
