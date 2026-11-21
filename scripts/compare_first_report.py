"""Join five model families after checking Junxiong's first-report graphs, splits and predictions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd

from evaluate_peer_protocol import REVISION, bind_peer

GRAPH_RELEASE = "5d8242701e2325b2dec8461a16d3b4963876b892"
BENCHMARKS = "experiments/new_link_models/benchmarks"
METRICS = ["auc", "ap", "hits100", "precision100"]


def compare(peer, tables):
    protocol, native = bind_peer(peer)

    def blob(commit, path):
        return subprocess.check_output(
            ["git", "-C", str(peer), "show", f"{commit}:{BENCHMARKS}/{path}"]
        )

    manifest = json.loads(blob(GRAPH_RELEASE, "ea_graphs/manifest.json"))
    original_audit = json.loads(blob(REVISION, "protocol_audit.json"))
    expected = {x["file"]: x["graph_sha256"] for x in original_audit["attacks"]}
    own = pd.read_csv(tables / "per_seed.csv", float_precision="round_trip")
    previous = pd.read_csv(
        io.BytesIO(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(peer),
                    "show",
                    "d1457c549c05dae91330d445c5efb6ce682b551f:results/peer-protocol/per_seed.csv",
                ]
            )
        ),
        float_precision="round_trip",
    )
    keys = ["dataset", "seed", "model", "condition"]
    keep = ~own.condition.str.startswith("ea_")
    pd.testing.assert_frame_equal(
        own[keep].set_index(keys).sort_index(),
        previous[~previous.condition.str.startswith("ea_")].set_index(keys).sort_index(),
    )
    assert len(own) == 504 and keep.sum() == 396 and own.graph_hash_matches.eq(1).all()
    published = pd.read_csv(
        io.BytesIO(blob(REVISION, "per_seed.csv")), float_precision="round_trip"
    )
    published["condition"] = [
        attack if not ratio else f"{attack}_{ratio:g}"
        for attack, ratio in zip(published.attack, published.ratio)
    ]
    conditions = {a if not r else f"{a}_{r:g}" for a, r in native.variants()}
    assert set(own.condition) == set(published.condition) == conditions
    checks = []
    with ZipFile(tables / "predictions-and-graphs.zip") as bundle:
        for ctx in manifest["contexts"]:
            d, s = ctx["dataset"], ctx["seed"]
            split_name = ctx["split_file"]
            raw = blob(GRAPH_RELEASE, "ea_graphs/" + split_name)
            assert hashlib.sha256(raw).hexdigest() == manifest["file_sha256"][split_name]
            with np.load(io.BytesIO(raw)) as ref, np.load(
                io.BytesIO(bundle.read(split_name))
            ) as actual:
                assert set(ref.files) == set(actual.files)
                for k in ref.files:
                    np.testing.assert_array_equal(ref[k], actual[k])
                clean = set(map(tuple, np.vstack((ref["train_pos"], ref["val_pos"]))))
                protected = set(
                    map(tuple, np.vstack((ref["test_pos"], ref["test_neg"], ref["val_neg"])))
                )
                npos, nneg = len(ref["test_pos"]), len(ref["test_neg"])
            clean_digest = protocol.digest(np.array(sorted(clean), dtype=np.int64))
            assert clean_digest == ctx["base_sha256"]
            for g in [g for g in manifest["graphs"] if (g["dataset"], g["seed"]) == (d, s)]:
                tag = f"ea_{g['ratio']:g}"
                name = Path(g["file"]).name
                raw = bundle.read("attacks/" + name)
                assert raw == blob(GRAPH_RELEASE, "ea_graphs/" + g["file"])
                sha = hashlib.sha256(raw).hexdigest()
                assert sha == manifest["file_sha256"][g["file"]]
                with np.load(io.BytesIO(raw)) as obj:
                    fingerprint = protocol.digest(obj["edges"])
                    edges = set(map(tuple, obj["edges"]))
                    assert str(obj["base_sha256"]) == clean_digest
                assert fingerprint == expected[name] == g["graph_sha256"]
                assert not edges & protected and len(edges ^ clean) == round(
                    g["ratio"] * len(clean)
                )
                subset = own[(own.dataset == d) & (own.seed == s) & (own.condition == tag)]
                assert len(subset) == 6 and subset.graph_sha256.eq(fingerprint).all()
                for row in subset.itertuples():
                    path = f"runs/{d}_{row.model}_s{s}/predictions_{tag}.npz"
                    with np.load(io.BytesIO(bundle.read(path))) as pred:
                        pos, neg = pred["positive"], pred["negative"]
                        assert (len(pos), len(neg)) == (npos, nneg)
                        vals = protocol.metrics(pos, neg)
                    for metric in METRICS:
                        assert vals[metric] == getattr(row, metric)
                peer_rows = published[
                    (published.dataset == d) & (published.seed == s) & (published.condition == tag)
                ]
                assert len(peer_rows) == 9
                assert {(x["model"], x["component"], x["seed"]) for x in g["used_by"]} == set(
                    peer_rows[["model", "component", "seed"]].itertuples(index=False, name=None)
                )
                checks.append(
                    dict(
                        dataset=d,
                        seed=s,
                        condition=tag,
                        file_sha256=sha,
                        edge_sha256=fingerprint,
                        clean_edges=len(clean),
                        inserted=len(edges - clean),
                        deleted=len(clean - edges),
                        test_positives=npos,
                        test_negatives=nneg,
                    )
                )
    assert len(checks) == 18
    own["component"] = own.model.where(own.model != own.family, "full")
    own["model"] = own.family
    columns = ["dataset", "seed", "model", "component", "condition", *METRICS]
    df = pd.concat([own[columns], published[columns]], ignore_index=True)
    assert len(df) == 1260 and not df.duplicated(columns[:5]).any()

    def graph_hash(r):
        tag = "clean" if r.condition == "feature_shuffle" else r.condition
        return expected[f"{r.dataset}_s{r.seed}_{tag}.npz"]

    df["graph_sha256"] = [graph_hash(r) for r in df.itertuples()]
    output = tables / "five-models"
    output.mkdir(exist_ok=True)
    df.sort_values(columns[:5]).to_csv(output / "per_seed.csv", index=False)
    clean = df[df.condition == "clean"].drop(columns="condition")
    full = df[df.component == "full"].drop(columns="component")
    paired = df.merge(
        clean,
        on=["dataset", "seed", "model", "component"],
        suffixes=("", "_clean"),
        validate="many_to_one",
    )
    paired = paired.merge(
        full,
        on=["dataset", "seed", "model", "condition"],
        suffixes=("", "_full"),
        validate="many_to_one",
    )
    fields = []
    for metric in METRICS:
        paired[metric + "_drop"] = paired[metric + "_clean"] - paired[metric]
        paired[metric + "_full_minus_variant"] = paired[metric + "_full"] - paired[metric]
        fields.extend((metric, metric + "_drop", metric + "_full_minus_variant"))
    grouped = paired.groupby(["dataset", "model", "component", "condition"])
    assert grouped.size().eq(3).all()
    out = grouped[fields].agg(["mean", "std"])
    out.columns = ["_".join(pair) for pair in out.columns]
    out.to_csv(output / "summary.csv")
    pd.DataFrame(checks).to_csv(output / "ea_graph_checks.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=Path("results/peer-protocol"))
    args = parser.parse_args()
    compare(args.peer, args.tables)
