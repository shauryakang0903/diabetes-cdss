"""Synthetic software demonstrations, never clinical patient archetypes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import CORE_FEATURES
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS


def synthetic_profiles():
    return {name: dict(zip(CORE_FEATURES, values)) for name, values in (
        ("relatively_lower_scoring_example", [25, 2, 3, 22, 78, 112, 72, 65, 190, 0]),
        ("intermediate_scoring_example", [48, 1, 3, 28, 98, 128, 80, 48, 185, 1]),
        ("higher_scoring_example", [68, 1, 3, 35, 118, 145, 78, 35, 150, 2]),
    )}


def main():
    service = NhanesDiabetesCDSS()
    print("Synthetic software demonstrations only; labels are not clinical risk bands.")
    for name, patient in synthetic_profiles().items():
        print(json.dumps({"example": name, **service.predict(patient)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
