"""Run with python -m scripts.compare_nhanes_baseline_models."""

from src.evaluation.nhanes_baseline_comparison import run_baseline_comparison


if __name__ == "__main__":
    _, summary = run_baseline_comparison()
    print(summary.drop(columns=["model_parameters", "feature_list"]).to_string(index=False))
