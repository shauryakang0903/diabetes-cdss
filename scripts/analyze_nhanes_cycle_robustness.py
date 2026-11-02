"""Run: python -m scripts.analyze_nhanes_cycle_robustness."""
from src.evaluation.nhanes_cycle_robustness import run_cycle_robustness

if __name__ == "__main__":
    metrics, thresholds, _ = run_cycle_robustness()
    print(metrics.to_string(index=False))
    print(thresholds[["model", "held_out_cycle", "operating_point", "threshold", "sensitivity",
                      "specificity", "absolute_sensitivity_target_deviation"]].to_string(index=False))
