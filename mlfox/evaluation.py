"""Cross-validation and one-time holdout metrics."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error)

def metric_values(model, X, y, task_type):
    pred = model.predict(X)
    if task_type == "regression":
        return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred)))}
    result = {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "precision": float(precision_score(y, pred, average="weighted", zero_division=0)), "recall": float(recall_score(y, pred, average="weighted", zero_division=0)), "f1": float(f1_score(y, pred, average="weighted", zero_division=0))}
    try:
        prob = model.predict_proba(X)
        result["roc_auc"] = float(roc_auc_score(y, prob[:, 1] if prob.shape[1] == 2 else prob, multi_class="ovr" if prob.shape[1] > 2 else "raise", average="weighted"))
    except (AttributeError, ValueError):
        result["roc_auc"] = None
    return result

def cross_validate_pipeline(pipeline, X, y, task_type, cv=5, random_state=42,
                            event_callback=None, candidate_name=None, diagnostics=None):
    if diagnostics is None:
        diagnostics = event_callback is not None
    counts = y.value_counts() if task_type != "regression" else None
    folds = min(cv, int(counts.min())) if counts is not None else min(cv, len(y))
    if folds < 2: raise ValueError("At least two valid CV folds are required.")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state) if counts is not None else KFold(folds, shuffle=True, random_state=random_state)
    collected = {}
    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(X, y), 1):
        candidate = clone(pipeline)
        pre_id = id(candidate.named_steps["preprocessing"])
        def emit(event_type, details=None):
            if event_callback:
                event_callback({"event_type": event_type, "candidate": candidate_name,
                    "fold": fold_number, "preprocessor_id": pre_id,
                    "details": details or {}})
        emit("fold_started", {"train_indices": X.index[train_idx].tolist(), "validation_indices": X.index[valid_idx].tolist()} if diagnostics else {})
        try:
            candidate.fit(X.iloc[train_idx], y.iloc[train_idx])
            if diagnostics:
                details = {"train_indices": X.index[train_idx].tolist(),
                           "imputer_statistics": candidate.named_steps["preprocessing"].imputer_.fill_values_.copy(),
                           "selector_input_indices": X.index[train_idx].tolist(),
                           "selector_labels": y.iloc[train_idx].tolist(),
                           "preprocessor": candidate.named_steps["preprocessing"],
                           "fitted_pipeline": candidate}
            else:
                details = {"training_rows": len(train_idx)}
            emit("preprocessing_fitted", details)
            emit("model_fitted")
        except Exception as exc:
            emit("candidate_failed", {"exception": repr(exc)})
            raise
        scores = metric_values(candidate, X.iloc[valid_idx], y.iloc[valid_idx], task_type)
        emit("fold_completed", {"metrics": scores})
        for key, value in scores.items():
            if value is not None: collected.setdefault(key, []).append(value)
    summary = {key: {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0} for key, vals in collected.items()}
    return summary, folds
