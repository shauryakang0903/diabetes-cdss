"""Run: python -m scripts.analyze_nhanes_thresholds."""
from src.evaluation.nhanes_threshold_analysis import run_threshold_analysis

if __name__ == "__main__":
    _, candidates, _, _ = run_threshold_analysis()
    print(candidates.to_string(index=False))
