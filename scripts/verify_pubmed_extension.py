"""Verify all PubMed predictions, protected queries and graph edit budgets, then pack each seed."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from evaluate_peer_protocol import bind_peer
from sgdiff.engine.reproduction import checkpoint_path, digest as file_digest, expanded


def verify(a):
    protocol, native = bind_peer(a.peer)
    digest, metrics = protocol.digest, protocol.metrics

    rows = pd.read_csv(a.tables / "per_seed.csv").set_index(
        ["seed", "condition", "model"], verify_integrity=True
    )
    provenance = {
        (r["seed"], r["model"]): r for r in json.loads((a.tables / "provenance.json").read_text())
    }
    assert len(rows) == 252
    assert set(rows.index.get_level_values("model")) == set(expanded(("katz", "sp4lp")))
    native_sha = file_digest(Path(native.__file__))
    assert all(r["evaluator_sha256"] == native_sha for r in provenance.values())
    manifest = json.loads((a.ea_cache / "manifest.json").read_text())
    entries = {entry["file"]: entry for entry in manifest["graphs"]}
    assert len(entries) == 9
    artifact_files = [(a.ea_cache / "manifest.json", "attacks/manifest.json")]
    checksums = []
    for seed in range(3):
        base = a.output / f"pubmed_s{seed}"
        splitfile = a.output / "splits" / f"pubmed_seed{seed}.npz"
        with np.load(splitfile) as split:
            clean = set(map(tuple, np.vstack((split["train_pos"], split["val_pos"]))))
            forbidden = set(
                map(tuple, np.vstack((split["test_pos"], split["test_neg"], split["val_neg"])))
            )
            pairs = np.vstack((split["test_pos"], split["test_neg"]))
            pos = len(split["test_pos"])
        artifact_files.extend(
            [
                (splitfile, f"splits/{splitfile.name}"),
                (base / "queries.npz", f"queries/pubmed_s{seed}.npz"),
            ]
        )
        with np.load(base / "queries.npz") as q:
            np.testing.assert_array_equal(pairs, q["pairs"])
            assert int(q["positive_count"]) == pos
        labels = np.r_[np.ones(pos), np.zeros(len(pairs) - pos)]
        for cond in rows.loc[seed].index.get_level_values("condition").unique():
            with np.load(base / f"graph_{cond}.npz") as g, np.load(
                base / f"scores_{cond}.npz"
            ) as scores:
                edges = set(map(tuple, g["edges"]))
                assert not (edges & forbidden)
                if cond in ("clean", "feature_shuffle"):
                    assert edges == clean
                else:
                    assert len(clean ^ edges) == round(float(cond.rsplit("_", 1)[1]) * len(clean))
                if cond.startswith("ea_"):
                    path = a.ea_cache / f"pubmed_s{seed}_{cond}.npz"
                    record = entries[path.name]
                    assert file_digest(path) == record["sha256"]
                    assert digest(g["edges"]) == record["graph_sha256"]
                    with np.load(path) as c:
                        np.testing.assert_array_equal(c["edges"], g["edges"])
                        np.testing.assert_array_equal(
                            c["added"],
                            np.asarray(sorted(edges - clean), dtype=np.int64).reshape(-1, 2),
                        )
                        np.testing.assert_array_equal(
                            c["removed"],
                            np.asarray(sorted(clean - edges), dtype=np.int64).reshape(-1, 2),
                        )
                        assert str(c["base_sha256"]) == digest(
                            np.asarray(sorted(clean), dtype=np.int64)
                        )
                        assert str(c["split_file_sha256"]) == file_digest(splitfile)
                    assert (len(edges - clean), len(clean - edges)) == (
                        record["inserted"],
                        record["deleted"],
                    )
                    artifact_files.append((path, f"attacks/{path.name}"))
                if cond == "feature_shuffle":
                    np.testing.assert_array_equal(
                        np.random.default_rng(seed + 7183).permutation(19717), g["permutation"]
                    )
                    with np.load(base / "scores_clean.npz") as original:
                        for m in ("katz", "katz_h2", "katz_h3", "sp4lp_featureless"):
                            np.testing.assert_array_equal(original[m], scores[m])
                for m in expanded(("katz", "sp4lp")):
                    r = rows.loc[(seed, cond, m)]
                    v = scores[m]
                    assert v.shape == labels.shape and np.isfinite(v).all()
                    assert r.graph_sha256 == digest(g["edges"])
                    values = metrics(v[:pos], v[pos:])
                    np.testing.assert_allclose(
                        [r[k] for k in ("auc", "ap", "hits100", "precision100")],
                        [
                            roc_auc_score(labels, v),
                            average_precision_score(labels, v),
                            values["hits100"],
                            values["precision100"],
                        ],
                        rtol=0,
                        atol=1e-12,
                    )
        archive = a.tables / f"pubmed_seed{seed}.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=3) as z:
            for p in sorted(base.glob("*.npz")):
                z.write(p, f"pubmed_s{seed}/{p.name}")
            z.write(splitfile, f"splits/{splitfile.name}")
            for m in expanded(("sp4lp",)):
                w = checkpoint_path(a.weights, "pubmed", seed, m)
                assert file_digest(w) == provenance[seed, m]["weights"]
                z.write(w, str(Path("runs/pair") / w.relative_to(a.weights)))
        with ZipFile(archive) as z:
            assert z.testzip() is None
        checksums.append(f"{file_digest(archive)}  {archive.name}")
    bundle = a.tables / "pubmed_ea_graphs.zip"
    assert len(artifact_files) == 16
    with ZipFile(bundle, "w", ZIP_DEFLATED, compresslevel=3) as z:
        for path, member in sorted(artifact_files):
            z.write(path, member)
    with ZipFile(bundle) as z:
        assert z.testzip() is None
    checksums.append(f"{file_digest(bundle)}  {bundle.name}")
    (a.tables / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print("Verified 252 cells, nine EA caches, three seed archives and the reusable EA bundle.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--ea-cache", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=Path("runs/pair"))
    verify(p.parse_args())
