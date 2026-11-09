"""Train SGDIFF and evaluate it on the test split.

    python examples/02_train_sgdiff.py --dataset cora --epochs 10
"""

import argparse

from sgdiff import api
from sgdiff.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-pairs", type=int, default=2000)
    args = parser.parse_args()

    model, summary, cfg = api.train(
        overrides=[
            f"data.name={args.dataset}",
            f"train.epochs={args.epochs}",
            f"data.max_train_pairs={args.train_pairs}",
            "data.max_eval_pairs=500",
        ]
    )
    print("test metrics:", summary["test"])
    save_json(summary["test"], f"results/example_{args.dataset}.json")

    graph, split = api.load_split(cfg)
    extractor = api.build_extractor(graph.drop_edges(split.test_pos.tolist()), cfg)
    scores = api.predict(model, graph, split.test_pos[:5], cfg, eval_graph=extractor.graph)
    print("first five positive scores:", scores.round(4).tolist())


if __name__ == "__main__":
    main()
