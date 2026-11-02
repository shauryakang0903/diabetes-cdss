"""Run: python -m scripts.explain_nhanes_shap_global. Never trains a model."""
from src.explainability.nhanes_shap_global import run_global_shap

if __name__ == "__main__":
    transformed, original, _, provenance = run_global_shap()
    print(transformed.to_string(index=False))
    print(original.to_string(index=False))
    print(provenance["additivity"])
