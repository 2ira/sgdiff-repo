"""Corrected experiments must not reuse or merge historical-method results."""

import json

import pytest

from scripts import collect_results, run_experiments
from sgdiff.version import METHOD_VERSION


def test_grid_outputs_are_separate_from_legacy_results():
    assert run_experiments.RESULTS.parts[-1] == METHOD_VERSION
    assert collect_results.RESULTS == run_experiments.RESULTS
    paper = run_experiments.PROFILES["paper"]
    assert paper["subgraph.max_nodes_total"] is None
    assert paper["train.freeze_diffusion"] is True
    assert paper["train.finetune_mc_samples"] is None


def test_merge_rejects_unversioned_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_results, "RESULTS", tmp_path)
    (tmp_path / "cora_seed0_paper.json").write_text(json.dumps({"auc": 0.8, "ap": 0.8, "hit@100": 0.7}))
    with pytest.raises(ValueError, match="legacy/mixed"):
        collect_results.merge_indomain("paper")
    assert not (tmp_path / "summary_indomain_paper.json").exists()


def test_merge_preserves_method_version(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_results, "RESULTS", tmp_path)
    for seed, auc in enumerate([0.6, 0.8]):
        payload = {"method_version": METHOD_VERSION, "auc": auc, "ap": auc, "hit@100": 0.7}
        (tmp_path / f"cora_seed{seed}_paper.json").write_text(json.dumps(payload))
    summary = collect_results.merge_indomain("paper")
    assert summary["cora"]["method_version"] == METHOD_VERSION
    assert summary["cora"]["auc"]["mean"] == pytest.approx(0.7)
    assert summary["cora"]["num_seeds"] == 2
