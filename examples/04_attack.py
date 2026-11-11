"""Robustness against structural attacks (RQ3): RF, DICE and EA.

    python examples/04_attack.py --checkpoint runs/cora_main/best.pt
    python examples/04_attack.py --checkpoint runs/cora_main/best.pt --attacks rf ea
"""

import argparse

from sgdiff import api
from sgdiff.data.attack import apply_attack
from sgdiff.data.splits import make_link_split
from sgdiff.engine.evaluator import evaluate_link_prediction
from sgdiff.engine.factory import build_extractor
from sgdiff.engine.trainer import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--attacks", nargs="*", default=["rf", "dice", "ea"],
                        choices=["rf", "dice", "ea"])
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint, device="cpu")
    graph = api.load_graph(args.dataset)
    split = make_link_split(graph, seed=cfg.data.seed)
    pos, neg = split.test_pos[: args.pairs // 2], split.test_neg[: args.pairs // 2]

    for ratio in (0.0, 0.05, 0.1, 0.2):
        variants = {"clean": graph} if ratio == 0 else {
            method: apply_attack(graph, method, ratio=ratio, seed=0) for method in args.attacks
        }
        for name, attacked in variants.items():
            extractor = build_extractor(
                attacked.drop_edges(split.test_pos.tolist()), cfg,
                target_feature_dim=int(model.feature_dim),
            )
            metrics = evaluate_link_prediction(model, extractor, pos, neg, device="cpu")
            extra = attacked.meta.get("attack", {})
            print(
                f"ratio={ratio:<5} {name:<7} AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f}"
                + (f" deleted={extra['deleted']} inserted={extra['inserted']}" if extra else "")
            )


if __name__ == "__main__":
    main()
