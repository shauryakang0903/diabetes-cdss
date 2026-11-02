"""Run post-hoc subgroup transparency audit without fitting any model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nhanes_subgroup_audit import BASE_SEED, BOOTSTRAP_REPLICATES, run_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POST-HOC / EXPLORATORY frozen-model subgroup audit; not fairness certification")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    development, temporal, dev_ranges, temporal_ranges, summary = run_audit(replicates=args.bootstrap_replicates, seed=args.seed)
    print("Frozen artifact unchanged:", summary["sha256_before"]["artifact"] == summary["sha256_after"]["artifact"])
    print("Development instability flags:", summary["unstable_groups"]["development"])
    print("Temporal POST-HOC / EXPLORATORY instability flags:", summary["unstable_groups"]["temporal_post_hoc"])
    print(temporal[["subgroup_family", "group_label", "n", "positive_count", "roc_auc", "sensitivity", "specificity", "estimates_unstable"]].to_string(index=False))
