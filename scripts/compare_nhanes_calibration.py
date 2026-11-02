"""Run: python -m scripts.compare_nhanes_calibration."""
from src.evaluation.nhanes_calibration_comparison import run_calibration_comparison

if __name__ == "__main__":
    _, summary, _ = run_calibration_comparison()
    print(summary.to_string(index=False))
