"""Reproducible DCA of existing predictions; no training or threshold selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nhanes_decision_curve import run_dca


if __name__ == "__main__":
    run_dca()
