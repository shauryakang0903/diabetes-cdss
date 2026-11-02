"""Run with: python -m scripts.compare_nhanes_preprocessing."""

from src.evaluation.nhanes_preprocessing_comparison import (
    compare_preprocessing, load_development_data,
)


def main():
    _, summary = compare_preprocessing(load_development_data())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
