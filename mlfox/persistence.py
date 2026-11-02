"""Trusted-artifact loading for fitted MLFOX models."""
from pathlib import Path
import warnings
import joblib

from .mlfox import MLFOX, ARTIFACT_FORMAT_VERSION


def load_model(path: str | Path) -> MLFOX:
    """Load a Joblib artifact.

    Joblib uses pickle deserialization, which may execute arbitrary code. Load
    artifacts only from a source you trust.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"MLFOX model artifact not found: {source}")
    warnings.warn(
        "Joblib model loading can execute code embedded in the artifact; only load trusted files.",
        UserWarning,
        stacklevel=2,
    )
    try:
        artifact = joblib.load(source)
    except Exception as exc:
        raise ValueError(f"Unable to load MLFOX model artifact: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("format") != "vprojectx.mlfox":
        raise ValueError("Unsupported or invalid MLFOX model artifact")
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"Unsupported MLFOX artifact version: {artifact.get('format_version')}")
    required = {"pipeline", "task_type", "target_column", "feature_schema", "report"}
    if not required.issubset(artifact):
        raise ValueError("MLFOX artifact is missing required model metadata")
    model = MLFOX.__new__(MLFOX)
    config = artifact.get("configuration", {})
    for key, default in (("task", "auto"), ("metric", None), ("cv", 5), ("test_size", .2),
                         ("random_state", 42), ("tune", False), ("max_tuning_trials", 10)):
        setattr(model, key, config.get(key, default))
    model.progress = False
    model.best_pipeline_ = artifact["pipeline"]
    model.task_type_ = artifact["task_type"]
    model.target_column_ = artifact["target_column"]
    model.feature_schema_ = list(artifact["feature_schema"])
    model.holdout_metrics_ = artifact.get("holdout_metrics")
    model.loaded_report_ = artifact["report"]
    model.report_ = model.loaded_report_
    model._explainability_ = model.loaded_report_.get("explainability", {
        "global_feature_importance": None, "individual_explanations": []
    })
    model.fitted_ = True
    return model
