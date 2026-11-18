"""Command line: train the full pipeline and print a results summary."""

from __future__ import annotations

import argparse
import logging

from .config import PipelineConfig
from .pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the startup-match pipeline on synthetic data.")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--startups", type=int, default=6000)
    ap.add_argument("--investors", type=int, default=400)
    ap.add_argument("--embeddings", choices=["auto", "sentence-transformers", "lsa"], default="auto")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    m = run(PipelineConfig(n_startups=a.startups, n_investors=a.investors,
                           embedding_backend=a.embeddings, seed=a.seed), a.out)
    ev = m["evaluator"]
    print(f"\nEvaluator (time-split test, n={ev['calibrated']['n']})")
    print(f"  AUC {ev['calibrated']['auc']:.3f} | Brier {ev['calibrated']['brier']:.3f} | "
          f"ECE {ev['uncalibrated']['ece']:.3f} -> {ev['calibrated']['ece']:.3f} after calibration")
    print(f"  recency-only baseline AUC {ev['baseline_last_round_recency_auc']:.3f} | "
          f"random-split AUC (leaky, for comparison) {ev['random_split_auc_for_comparison']:.3f}")
    print(f"\nEmbeddings: {m['embedding_backend']} | sector-match AUC {m['embedding_sector_match_auc']:.3f}")
    print(f"\nRanking on {m['data']['test_investors']} held-out investors (NDCG@10, 95% CI)")
    for name, r in sorted(m["ranking"].items(), key=lambda kv: -kv[1]["ndcg@10"]):
        lo, hi = r["ndcg@10_ci95"]
        print(f"  {r['ndcg@10']:.3f} [{lo:.3f}, {hi:.3f}]  {name}")
    c = m["ranker_vs_best_baseline"]
    print(f"\nRanker - best baseline ({c['baseline']}): {c['mean_diff']:+.3f} "
          f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")
    print(f"\nArtifacts written to {a.out}/")


if __name__ == "__main__":
    main()
