"""Generate deterministic development-only input reference JSON; no model fitting."""
from src.decision_support.nhanes_input_quality import REFERENCE_PATH, save_reference_metadata

if __name__ == '__main__':
    save_reference_metadata()
    print(REFERENCE_PATH)
