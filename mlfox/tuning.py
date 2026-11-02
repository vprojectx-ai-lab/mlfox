"""Bounded randomized parameter search over fold-aware MLFOX pipelines."""
from time import perf_counter
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
from .pipeline import FoldAwarePreprocessor
from .evaluation import cross_validate_pipeline, metric_values
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor

def default_search_space(estimator):
    """Return a small bounded grid for supported M9 estimator families."""
    spaces = {
        LogisticRegression: {"C": [0.1, 1.0, 10.0]},
        LinearRegression: {"fit_intercept": [True, False]},
        DecisionTreeClassifier: {"max_depth": [None, 4, 8], "min_samples_split": [2, 5]},
        RandomForestClassifier: {"n_estimators": [50, 100], "max_depth": [None, 8]},
        ExtraTreesClassifier: {"n_estimators": [50, 100], "max_depth": [None, 8]},
        HistGradientBoostingClassifier: {"learning_rate": [0.05, 0.1], "max_iter": [50, 100]},
        SVC: {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        Ridge: {"alpha": [0.1, 1.0, 10.0]},
        ElasticNet: {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        DecisionTreeRegressor: {"max_depth": [None, 4, 8], "min_samples_split": [2, 5]},
        RandomForestRegressor: {"n_estimators": [50, 100], "max_depth": [None, 8]},
        ExtraTreesRegressor: {"n_estimators": [50, 100], "max_depth": [None, 8]},
        HistGradientBoostingRegressor: {"learning_rate": [0.05, 0.1], "max_iter": [50, 100]},
        XGBClassifier: {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        XGBRegressor: {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    }
    for cls, space in spaces.items():
        if isinstance(estimator, cls): return space
    if isinstance(estimator, CalibratedClassifierCV) and isinstance(getattr(estimator, "estimator", None), SVC):
        return {"estimator__C": [0.1, 1.0, 10.0],
                "estimator__kernel": ["rbf", "linear"]}
    # Support _LabelEncodingClassifier wrapper (e.g. around XGBClassifier)
    from .model_selection import _LabelEncodingClassifier
    if isinstance(estimator, _LabelEncodingClassifier):
        inner_space = default_search_space(estimator.estimator)
        return {f"estimator__{k}": v for k, v in inner_space.items()}
    raise ValueError(f"No default search space for estimator type {type(estimator).__name__}")

class HyperparameterTuner:
    def __init__(self, estimator, param_space, max_trials=20, random_state=42,
                 cv=5, task_type="binary_classification", primary_metric=None,
                 progress=False, n_jobs=1, event_callback=None):
        if max_trials < 1: raise ValueError("max_trials must be positive")
        if cv < 2: raise ValueError("cv must be at least 2")
        if n_jobs != 1: raise ValueError("Only sequential n_jobs=1 is supported")
        self.estimator, self.param_space, self.max_trials = estimator, param_space, max_trials
        self.random_state, self.cv, self.task_type = random_state, cv, task_type
        self.primary_metric = primary_metric
        self.progress, self.event_callback = progress, event_callback

    def _emit(self, event_type, details=None):
        if self.event_callback:
            self.event_callback({"event_type": event_type, "candidate": "tuning", "fold": None,
                                 "preprocessor_id": None, "details": details or {}})

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        metric = self.primary_metric or ("rmse" if self.task_type == "regression" else "balanced_accuracy")
        valid = {"rmse", "mae", "r2"} if self.task_type == "regression" else {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"}
        if metric not in valid:
            raise ValueError(f"Unsupported primary metric {metric!r} for {self.task_type}; choose from {sorted(valid)}")
        maximize = metric not in {"rmse", "mae"}
        trials = list(ParameterSampler(self.param_space, n_iter=self.max_trials, random_state=self.random_state))
        if not trials: raise ValueError("Search space produced no trials")
        self.trial_results_ = []
        started_all = perf_counter()
        for number, params in enumerate(trials, 1):
            started = perf_counter()
            record = {"trial": number, "params": params, "status": "failed", "failure": None}
            try:
                pipe = Pipeline([("preprocessing", FoldAwarePreprocessor(task_type=self.task_type)), ("model", clone(self.estimator))])
                pipe.set_params(**{f"model__{key}": value for key, value in params.items()})
                scores, folds = cross_validate_pipeline(pipe, X_train, y_train, self.task_type, self.cv, self.random_state)
                if metric not in scores: raise ValueError(f"Primary metric {metric!r} unavailable for estimator")
                record.update(status="success", metrics=scores, folds=folds)
            except Exception as exc:
                record["failure"] = repr(exc)
            record["runtime_seconds"] = perf_counter() - started
            self.trial_results_.append(record)
            if self.progress: print(f"[MLFOX tuning] trial {number}/{len(trials)} {record['status']}", flush=True)
        eligible = [r for r in self.trial_results_ if r["status"] == "success"]
        if not eligible: raise RuntimeError("All tuning trials failed")
        self.best_trial_ = sorted(eligible, key=lambda r: r["metrics"][metric]["mean"], reverse=maximize)[0]
        self.best_params_ = self.best_trial_["params"]
        self.best_score_ = self.best_trial_["metrics"][metric]["mean"]
        self.best_estimator_ = Pipeline([("preprocessing", FoldAwarePreprocessor(task_type=self.task_type)), ("model", clone(self.estimator))])
        self.best_estimator_.set_params(**{f"model__{k}": v for k, v in self.best_params_.items()})
        self._emit("best_model_selected", {"params": self.best_params_, "cv_score": self.best_score_})
        self.best_estimator_.fit(X_train, y_train)
        holdout_started = perf_counter()
        self.holdout_metrics_ = metric_values(self.best_estimator_, X_test, y_test, self.task_type) if X_test is not None else None
        self.holdout_evaluation_runtime_seconds_ = perf_counter() - holdout_started if X_test is not None else 0.0
        self.total_runtime_seconds_ = perf_counter() - started_all
        self.tuning_runtime_seconds_ = max(0.0, self.total_runtime_seconds_ - self.holdout_evaluation_runtime_seconds_)
        self.report_ = {"trials_attempted": len(self.trial_results_), "successful": sum(r["status"] == "success" for r in self.trial_results_), "failed": sum(r["status"] == "failed" for r in self.trial_results_), "best_params": self.best_params_, "best_cv_mean": self.best_score_, "best_cv_std": self.best_trial_["metrics"][metric]["std"], "primary_metric": metric, "trials": self.trial_results_, "total_runtime_seconds": self.total_runtime_seconds_, "holdout_metrics": self.holdout_metrics_}
        return self
