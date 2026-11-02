"""First frozen temporal evaluation: python -m scripts.evaluate_nhanes_temporal."""
from src.evaluation.nhanes_temporal_validation import run_temporal_validation

if __name__ == "__main__":
    metrics, thresholds, _, completeness = run_temporal_validation()
    print(metrics[["model", "roc_auc", "average_precision", "brier_score", "log_loss",
                   "calibration_intercept", "calibration_slope", "ece"]].to_string(index=False))
    print(thresholds[["model", "operating_point", "threshold", "sensitivity", "specificity",
                      "absolute_sensitivity_target_deviation"]].to_string(index=False))
    print(completeness.to_string(index=False))
