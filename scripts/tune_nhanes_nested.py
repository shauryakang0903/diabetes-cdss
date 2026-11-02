"""Run: python -m scripts.tune_nhanes_nested."""
from src.evaluation.nhanes_nested_tuning import run_nested_tuning

if __name__ == "__main__":
    _, summary, _ = run_nested_tuning()
    print(summary.to_string(index=False))
