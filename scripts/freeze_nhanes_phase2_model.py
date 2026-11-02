"""Run once: python -m scripts.freeze_nhanes_phase2_model."""
from src.models.nhanes_phase2_model import build_frozen_artifact

if __name__ == "__main__":
    artifact, metadata_path, metadata = build_frozen_artifact()
    print(f"Artifact: {artifact}")
    print(f"Metadata: {metadata_path}")
    print(f"Artifact SHA-256: {metadata['artifact_sha256']}")
    print(f"Fitted N: {metadata['development_n']}")
    print(f"Features: {metadata['input_feature_count']} -> {metadata['transformed_feature_count']}")
    print(f"Reload maximum probability difference: {metadata['serialization']['maximum_absolute_prediction_difference']}")
