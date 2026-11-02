"""Public M11 orchestration over the existing M1–M10 components."""
from time import perf_counter
from pathlib import Path
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from typing import Any
import pandas as pd

from .ingestion import load_dataset
from .profiler import profile_dataset, detect_task, is_discrete_numeric_multiclass
from .splitter import split_dataset
from .column_intelligence import analyze_columns
from .model_selection import ModelSelectionEngine
from .tuning import HyperparameterTuner, default_search_space

ARTIFACT_FORMAT_VERSION = 1


def _running_in_notebook() -> bool:
    """Return whether an IPython notebook would auto-display cell results."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _print_summary(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    """Print only the concise candidate comparison and final metrics."""
    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title="MLFOX benchmark")
        table.add_column("Model")
        table.add_column("CV mean")
        table.add_column("Status")
        for row in candidates:
            table.add_row(str(row.get("model", "")), str(row.get("cv_mean")), str(row.get("status", "")))
        console = Console()
        console.print(table)
        console.print(f"Best: {payload.get('best_estimator')} | Holdout: {payload.get('holdout_metrics')}")
        console.print()
        console.print("[dim italic]~ Product by vprojectx[/dim italic]")
    except ImportError:
        print(f"MLFOX best model: {payload.get('best_estimator')}; holdout metrics: {payload.get('holdout_metrics')}")
        print("\n~ Product by vprojectx")


class MLFOX:
    """Fit and use an MLFOX model-selection, tuning, and inference workflow."""

    def __init__(self, task="auto", metric=None, cv=5, test_size=0.2,
                 random_state=42, tune=False, max_tuning_trials=10, progress=True):
        if task not in {"auto", "binary_classification", "multiclass_classification", "regression"}:
            raise ValueError("task must be auto, binary_classification, multiclass_classification, or regression")
        if cv < 2: raise ValueError("cv must be at least 2")
        if not 0 < test_size < 1: raise ValueError("test_size must be between 0 and 1")
        if max_tuning_trials < 1: raise ValueError("max_tuning_trials must be positive")
        self.task, self.metric, self.cv = task, metric, cv
        self.test_size, self.random_state = test_size, random_state
        self.tune, self.max_tuning_trials, self.progress = tune, max_tuning_trials, progress

    def fit(self, data: Any, target: str) -> "MLFOX":
        """Fit from a CSV/Excel path or DataFrame; holdout is evaluated last."""
        self.fitted_ = False
        started = perf_counter()
        if isinstance(data, (str, bytes)) and str(data).lower().endswith((".csv", ".xlsx", ".xls")) is False:
            raise ValueError("Unsupported formats. Supported input formats: CSV, XLSX, XLS")
        df = load_dataset(data)
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        valid_target = df[target].dropna()
        if len(valid_target) < 2 or valid_target.nunique() < 2:
            raise ValueError("Target must contain at least two valid observations and two distinct values")
        self.profile_ = profile_dataset(df, target=target)
        detected = detect_task(df, target)
        task_type = detected["task_type"] if self.task == "auto" else self.task
        if self.task != "auto" and "classification" in self.task:
            class_count = int(valid_target.nunique())
            numeric = pd.api.types.is_numeric_dtype(valid_target)
            discrete_numeric_multiclass = numeric and is_discrete_numeric_multiclass(valid_target)
            required_count = 2 if self.task == "binary_classification" else 3
            if class_count < required_count or (self.task == "binary_classification" and class_count != 2):
                if self.task == "binary_classification" and numeric and not discrete_numeric_multiclass:
                    raise ValueError("Configured classification task is incompatible with a continuous numeric target")
                raise ValueError(f"{self.task} requires exactly {required_count if self.task == 'binary_classification' else 'at least 3'} distinct target labels")
            if "classification" not in detected["task_type"]:
                if not (self.task == "multiclass_classification" and discrete_numeric_multiclass):
                    raise ValueError("Configured classification task requires categorical labels or discrete integer multiclass labels")
        if self.task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
            raise ValueError("Regression task requires a numeric target")
        self.task_report_ = dict(detected, task_type=task_type)
        self.task_type_ = task_type
        self.target_column_ = target
        self.split_ = split_dataset(df, target, self.test_size, self.random_state, task_type)
        self.feature_schema_ = list(self.split_.X_train.columns)
        self.column_report_ = analyze_columns(self.split_.X_train, self.split_.y_train)
        if len(self.split_.X_train) < self.cv:
            raise ValueError("Insufficient training rows for the configured CV count")
        selection_started = perf_counter()
        self.engine_ = ModelSelectionEngine(cv=self.cv, random_state=self.random_state,
            progress=self.progress, primary_metric=self.metric)
        self.engine_.fit(self.split_.X_train, self.split_.y_train, task_type)
        self.tuning_report_ = None
        self.tuning_runtime_seconds_ = 0.0
        self.evaluation_runtime_seconds_ = 0.0
        if self.tune:
            estimator = self.engine_.best_model_.named_steps["model"]
            tuner = HyperparameterTuner(estimator, default_search_space(estimator),
                max_trials=self.max_tuning_trials, random_state=self.random_state,
                cv=self.cv, task_type=task_type, primary_metric=self.metric,
                progress=self.progress)
            tuning_started = perf_counter()
            tuner.fit(self.split_.X_train, self.split_.y_train,
                      self.split_.X_test, self.split_.y_test)
            self.tuning_runtime_seconds_ = getattr(tuner, "tuning_runtime_seconds_", perf_counter() - tuning_started)
            self.evaluation_runtime_seconds_ = getattr(tuner, "holdout_evaluation_runtime_seconds_", 0.0)
            self.tuner_ = tuner
            self.best_pipeline_ = tuner.best_estimator_
            self.holdout_metrics_ = tuner.holdout_metrics_
            self.tuning_report_ = tuner.report_
        else:
            # Selection/refit used training data only; this is the single holdout evaluation.
            self.engine_.holdout_metrics_ = None
            from .evaluation import metric_values
            evaluation_started = perf_counter()
            self.holdout_metrics_ = metric_values(self.engine_.best_model_, self.split_.X_test, self.split_.y_test, task_type)
            self.evaluation_runtime_seconds_ = perf_counter() - evaluation_started
            self.best_pipeline_ = self.engine_.best_model_
        self.selection_runtime_seconds_ = perf_counter() - selection_started
        self.training_runtime_seconds_ = perf_counter() - started
        self._explainability_ = {"global_feature_importance": None, "individual_explanations": []}
        self.fitted_ = True
        return self

    def report(self) -> None:
        """Print the concise benchmark summary; detailed content is in ``html()``."""
        payload = self._build_report()
        selection = payload.get("model_selection", {})
        candidates = selection.get("benchmark_table", [])
        _print_summary(payload, candidates)
        return None

    def _build_report(self) -> dict[str, Any]:
        """Build the structured report for internal exports and artifacts."""
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() before report().")
        if getattr(self, "loaded_report_", None) is not None:
            self.loaded_report_.setdefault("explainability", {"global_feature_importance": None, "individual_explanations": []})
            self.loaded_report_["explainability"].update(getattr(self, "_explainability_", {}))
            self.report_ = self.loaded_report_
            return self.loaded_report_
        profile = self.profile_
        payload = {"dataset": {"rows": self.profile_.rows, "columns": self.profile_.columns,
                    "warnings": self.profile_.warnings,
                    "feature_types": {"numerical": list(profile.numerical_columns), "categorical": list(profile.categorical_columns),
                                      "datetime": list(profile.datetime_candidates), "other": list(profile.other_columns)},
                    "missing_values": profile.missing_analysis}, "task": self.task_report_,
            "column_intelligence": self.column_report_.to_dict(),
            "preprocessing": {"pipeline": "fold-aware M4–M8", "excluded_features": self.best_pipeline_.named_steps["preprocessing"].excluded_features_},
            "feature_selection": self.best_pipeline_.named_steps["preprocessing"].selector_.to_dict(),
            "model_selection": self.engine_.report(), "best_estimator": type(self.best_pipeline_.named_steps["model"]).__name__,
            "best_hyperparameters": self.tuner_.best_params_ if self.tune else None,
            "cv_metrics": self.engine_.cv_results_.get(self.engine_.best_model_name_, {}).get("metrics"),
            "holdout_metrics": self.holdout_metrics_, "tuning": self.tuning_report_,
            "runtime_seconds": {"training_total": self.training_runtime_seconds_, "selection_and_fit": self.selection_runtime_seconds_,
                                "tuning": getattr(self, "tuning_runtime_seconds_", 0.0), "evaluation": getattr(self, "evaluation_runtime_seconds_", 0.0)},
            "explainability": getattr(self, "_explainability_", {"global_feature_importance": None, "individual_explanations": []})}
        self.report_ = payload
        return payload

    def _load_explanation_frame(self, data: Any, target: str) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() or load a fitted model before explainability analysis.")
        frame = load_dataset(data)
        if target not in frame.columns:
            raise ValueError(f"Target column {target!r} not found in evaluation data")
        return frame

    def feature_importance(self, data: Any, target: str, n_repeats: int = 10,
                           random_state: int = 42, metric: str | None = None) -> pd.DataFrame:
        """Calculate permutation importance on explicit labeled evaluation data."""
        from .explainability import calculate_importance
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() or load a fitted model before calculating feature importance.")
        if not hasattr(self, "_explainability_"):
            self._explainability_ = {"global_feature_importance": None, "individual_explanations": []}
        return calculate_importance(self, data, target, n_repeats, random_state, metric)

    def explain_prediction(self, data: Any, background_data: Any,
                           target_class: Any = None) -> pd.DataFrame:
        """Explain one prediction using feature perturbation, not causal effects."""
        from .explainability import explain_observation
        if not hasattr(self, "_explainability_"):
            self._explainability_ = {"global_feature_importance": None, "individual_explanations": []}
        return explain_observation(self, data, background_data, target_class)

    def export_report(self, path: str | Path) -> Path:
        """Write a self-contained HTML or strict JSON report."""
        from .report_export import export_report
        return export_report(self, path)

    def html(self, path: str | Path = "mlfox_report.html") -> Path:
        """Write the detailed, styled HTML report and return its file path."""
        return self.export_report(path)

    def _inference_frame(self, data: Any) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() or load a fitted model before prediction.")
        frame = load_dataset(data)
        if self.target_column_ in frame.columns:
            frame = frame.drop(columns=[self.target_column_])
        missing = [c for c in self.feature_schema_ if c not in frame.columns]
        if missing:
            raise ValueError(f"Inference data is missing required features: {missing}")
        extras = [c for c in frame.columns if c not in self.feature_schema_]
        if extras:
            self.last_inference_warnings_ = [f"Ignored additional columns: {extras}"]
        else:
            self.last_inference_warnings_ = []
        return frame.loc[:, self.feature_schema_]

    def predict(self, data: Any) -> pd.Series:
        """Predict target values while preserving input row indices."""
        frame = self._inference_frame(data)
        values = self.best_pipeline_.predict(frame)
        return pd.Series(values, index=frame.index, name="prediction")

    def predict_proba(self, data: Any) -> pd.DataFrame:
        """Return class probabilities for classification models."""
        if self.task_type_ == "regression":
            raise ValueError("Probability prediction is only available for classification tasks")
        frame = self._inference_frame(data)
        if not hasattr(self.best_pipeline_, "predict_proba"):
            raise ValueError("The selected estimator does not support probability prediction")
        values = self.best_pipeline_.predict_proba(frame)
        labels = self.best_pipeline_.named_steps["model"].classes_
        return pd.DataFrame(values, index=frame.index, columns=labels)

    def save(self, path: str | Path) -> Path:
        """Atomically save the fitted pipeline and report metadata as Joblib."""
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Cannot save an unfitted model.")
        import joblib
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        versions = {}
        for package in ("mlfox", "numpy", "pandas", "scikit-learn", "joblib"):
            try: versions[package] = version(package)
            except PackageNotFoundError: versions[package] = "unknown"
        report = self._build_report()
        artifact = {"format": "vprojectx.mlfox", "format_version": ARTIFACT_FORMAT_VERSION,
            "pipeline": self.best_pipeline_, "task_type": self.task_type_,
            "target_column": self.target_column_, "feature_schema": list(self.feature_schema_),
            "estimator_name": type(self.best_pipeline_.named_steps["model"]).__name__,
            "hyperparameters": self.tuner_.best_params_ if getattr(self, "tune", False) and hasattr(self, "tuner_") else None,
            "configuration": {"task": self.task, "metric": self.metric, "cv": self.cv,
                "test_size": self.test_size, "random_state": self.random_state,
                "tune": self.tune, "max_tuning_trials": self.max_tuning_trials},
            "cv_metrics": report.get("cv_metrics"),
            "holdout_metrics": self.holdout_metrics_, "report": report, "versions": versions}
        handle = tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent), delete=False)
        temp_path = Path(handle.name)
        handle.close()
        try:
            joblib.dump(artifact, temp_path)
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists(): temp_path.unlink()
        return destination


def mlfox(task="auto", metric=None, cv=5, test_size=0.2, random_state=42,
          tune=False, max_tuning_trials=10, progress=True) -> MLFOX:
    """Create an MLFOX estimator with deterministic, sequential defaults."""
    return MLFOX(task, metric, cv, test_size, random_state, tune, max_tuning_trials, progress)
