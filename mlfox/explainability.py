"""Model-agnostic explainability helpers for fitted MLFOX pipelines."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


_CLASSIFICATION_METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"}
_REGRESSION_METRICS = {"rmse", "mae", "r2"}


def metric_scoring(task_type, metric):
    allowed = _REGRESSION_METRICS if task_type == "regression" else _CLASSIFICATION_METRICS
    if metric not in allowed:
        raise ValueError(f"Unsupported or incompatible metric {metric!r} for {task_type}; choose from {sorted(allowed)}")
    if metric == "precision":
        return "precision_weighted"
    if metric == "recall":
        return "recall_weighted"
    if metric == "f1":
        return "f1_weighted"
    if metric == "roc_auc" and task_type == "multiclass_classification":
        return "roc_auc_ovr_weighted"
    return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}.get(metric, metric)


def calculate_importance(model, data, target, n_repeats=10, random_state=42, metric=None):
    if not getattr(model, "fitted_", False):
        raise RuntimeError("Call fit() or load a fitted model before calculating feature importance.")
    if not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    frame = model._load_explanation_frame(data, target)
    y = frame.pop(target)
    if y.isna().any():
        raise ValueError("Evaluation target contains missing values")
    if y.nunique() < 2:
        raise ValueError("Evaluation target must contain at least two distinct values")
    if model.task_type_ == "regression":
        if not pd.api.types.is_numeric_dtype(y) or not np.isfinite(pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)).all():
            raise ValueError("Regression evaluation target must contain only finite numeric values")
    else:
        fitted_classes = list(getattr(model.best_pipeline_.named_steps["model"], "classes_", []))
        unknown = [label for label in pd.unique(y) if fitted_classes and label not in fitted_classes]
        if unknown:
            raise ValueError(f"Evaluation target contains labels not present during model fitting: {unknown}")
    X = model._inference_frame(frame)
    metric = metric or getattr(model, "metric", None) or ("rmse" if model.task_type_ == "regression" else "balanced_accuracy")
    scoring = metric_scoring(model.task_type_, metric)
    if metric == "roc_auc" and not hasattr(model.best_pipeline_, "predict_proba"):
        raise ValueError("roc_auc requires a selected estimator with probability estimates")
    if model.task_type_ == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression feature importance requires a numeric target")
    result = permutation_importance(model.best_pipeline_, X, y, scoring=scoring,
                                    n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    importance = pd.DataFrame({
        "feature": list(model.feature_schema_),
        "importance_mean": result.importances_mean.astype(float),
        "importance_std": result.importances_std.astype(float),
    }).sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance = importance[["feature", "importance_mean", "importance_std", "rank"]]
    # sklearn scorers are oriented so larger is always better. For loss metrics,
    # importance remains the loss reduction caused by shuffling the feature.
    model._explainability_["global_feature_importance"] = {
        "method": "permutation importance on the complete fitted pipeline",
        "metric": metric,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "features": importance.to_dict(orient="records"),
        "limitations": ["Correlated features can share or mask permutation importance.",
                        "Importance measures predictive sensitivity, not causation.",
                        "Features removed during preprocessing remain in the report and may have zero importance."],
    }
    return importance


def _baseline_value(series, name):
    nonmissing = series.dropna()
    if nonmissing.empty:
        raise ValueError(f"Background feature {name!r} has no non-missing values")
    if pd.api.types.is_numeric_dtype(series):
        return nonmissing.median()
    if pd.api.types.is_datetime64_any_dtype(series):
        return nonmissing.sort_values().iloc[len(nonmissing) // 2]
    modes = nonmissing.mode(dropna=True)
    if modes.empty:
        raise ValueError(f"Unable to determine a background baseline for {name!r}")
    return modes.iloc[0]


def explain_observation(model, data, background_data, target_class=None):
    if not getattr(model, "fitted_", False):
        raise RuntimeError("Call fit() or load a fitted model before explaining predictions.")
    if isinstance(data, pd.DataFrame) and len(data) != 1:
        raise ValueError("explain_prediction requires exactly one observation")
    observation = model._inference_frame(data)
    if len(observation) != 1:
        raise ValueError("explain_prediction requires exactly one observation")
    background = model._inference_frame(background_data)
    if background.empty:
        raise ValueError("background_data must contain at least one row")
    baseline = {name: _baseline_value(background[name], name) for name in model.feature_schema_}
    original_prediction = model.best_pipeline_.predict(observation)[0]
    is_classification = model.task_type_ != "regression"
    original_probabilities = None
    selected_class = target_class
    if is_classification:
        classes = list(getattr(model.best_pipeline_.named_steps["model"], "classes_", []))
        if selected_class is None:
            selected_class = original_prediction
        if classes and selected_class not in classes:
            raise ValueError(f"target_class {selected_class!r} is not among fitted classes {classes!r}")
        if hasattr(model.best_pipeline_, "predict_proba"):
            probability = model.best_pipeline_.predict_proba(observation)[0]
            original_probabilities = {str(label): float(value) for label, value in zip(classes, probability)}
    rows = []
    for feature in model.feature_schema_:
        perturbed = observation.copy(deep=True)
        perturbed.loc[perturbed.index[0], feature] = baseline[feature]
        perturbed_prediction = model.best_pipeline_.predict(perturbed)[0]
        row = {"feature": feature, "original_value": observation.iloc[0][feature],
               "background_baseline": baseline[feature], "original_prediction": original_prediction,
               "perturbed_prediction": perturbed_prediction, "method": "single-feature perturbation / sensitivity analysis"}
        if is_classification:
            row["original_class_probabilities"] = original_probabilities
            row["target_class"] = selected_class
            row["class_changed"] = bool(perturbed_prediction != original_prediction)
            if hasattr(model.best_pipeline_, "predict_proba"):
                perturbed_prob = model.best_pipeline_.predict_proba(perturbed)[0]
                classes = list(model.best_pipeline_.named_steps["model"].classes_)
                idx = classes.index(selected_class)
                row["perturbed_target_probability"] = float(perturbed_prob[idx])
                row["probability_change"] = float(perturbed_prob[idx] - model.best_pipeline_.predict_proba(observation)[0][idx])
            else:
                row["probability_change"] = None
        else:
            row["prediction_change"] = float(perturbed_prediction - original_prediction)
        rows.append(row)
    result = pd.DataFrame(rows)
    report_rows = result.drop(columns=["original_value", "background_baseline"], errors="ignore")
    model._explainability_["individual_explanations"].append({
        "method": "single-feature perturbation / sensitivity analysis",
        "target_class": selected_class,
        # Keep prediction deltas in report metadata without retaining input-row values.
        "results": report_rows.to_dict(orient="records"),
        "limitations": ["Correlated features and feature interactions are not modeled by this one-feature-at-a-time analysis.",
                        "Changes describe model sensitivity and are not causal effects.",
                        "This method is not SHAP."],
    })
    return result
