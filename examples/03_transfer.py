"""Zero-shot cross-dataset transfer (RQ1).

    python examples/03_transfer.py --source cora --target citeseer
"""

import argparse

from sgdiff import api
from sgdiff.data.splits import make_link_split
from sgdiff.engine.evaluator import evaluate_link_prediction
from sgdiff.engine.factory import build_extractor
from sgdiff.engine.trainer import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="cora")
    parser.add_argument("--target", default="citeseer")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        _, summary, _ = api.train(
            overrides=[f"data.name={args.source}", f"train.epochs={args.epochs}",
                       "data.max_train_pairs=1500", "data.max_eval_pairs=400"]
        )
        checkpoint = f"{summary['run_dir']}/best.pt"

    model, cfg = load_checkpoint(checkpoint, device="cpu")
    graph = api.load_graph(args.target)
    split = make_link_split(graph, seed=cfg.data.seed)
    extractor = build_extractor(
        graph.drop_edges(split.test_pos.tolist()), cfg,
        target_feature_dim=int(model.feature_dim),
    )
    metrics = evaluate_link_prediction(
        model, extractor, split.test_pos[:500], split.test_neg[:500], device="cpu"
    )
    print(f"{args.source} -> {args.target}:", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
