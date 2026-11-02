"""CPU model eligibility, fold benchmarking and final refit."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from xgboost import XGBClassifier, XGBRegressor
import time
from .pipeline import FoldAwarePreprocessor
from .evaluation import cross_validate_pipeline, metric_values

class ModelSelectionEngine:
    def __init__(self, cv=5, random_state=42, progress=True, estimators=None, primary_metric=None, n_jobs=1, event_callback=None):
        if cv < 2: raise ValueError("cv must be at least 2.")
        if n_jobs != 1: raise ValueError("M9 currently supports sequential n_jobs=1 only.")
        self.cv, self.random_state, self.progress = cv, random_state, progress
        self.estimators, self.primary_metric = estimators, primary_metric
        self.event_callback = event_callback

    def _candidates(self, task, n, p):
        if self.estimators is not None: return dict(self.estimators), {}
        if task == "regression":
            models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(), "ElasticNet": ElasticNet(random_state=self.random_state), "Decision Tree": DecisionTreeRegressor(random_state=self.random_state), "Random Forest": RandomForestRegressor(n_estimators=80, n_jobs=1, random_state=self.random_state), "Extra Trees": ExtraTreesRegressor(n_estimators=80, n_jobs=1, random_state=self.random_state), "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, tree_method="hist", device="cpu", n_jobs=1, random_state=self.random_state, verbosity=0)}
            if n >= 200: models["HistGradientBoosting"] = HistGradientBoostingRegressor(random_state=self.random_state)
        else:
            models = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state), "Decision Tree": DecisionTreeClassifier(random_state=self.random_state), "Random Forest": RandomForestClassifier(n_estimators=80, n_jobs=1, random_state=self.random_state), "Extra Trees": ExtraTreesClassifier(n_estimators=80, n_jobs=1, random_state=self.random_state), "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, tree_method="hist", device="cpu", n_jobs=1, random_state=self.random_state, eval_metric="logloss", verbosity=0)}
            if n >= 200: models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=self.random_state)
            if n <= 5000: models["SVM"] = CalibratedClassifierCV(SVC(random_state=self.random_state), cv=2)
        skipped = {"SVM": "Skipped: more than 5000 samples."} if task != "regression" and n > 5000 else {}
        if p > 1000: models.pop("SVM", None); skipped["SVM"] = "Skipped: more than 1000 input features."
        return models, skipped

    def fit(self, X_train, y_train, task_type, X_test=None, y_test=None):
        if task_type not in {"binary_classification", "multiclass_classification", "regression"}: raise ValueError(f"Unsupported task_type: {task_type}")
        if X_test is not None and y_test is None: raise ValueError("y_test is required with X_test.")
        default_metric = "rmse" if task_type == "regression" else "balanced_accuracy"
        self.primary_metric_ = self.primary_metric or default_metric
        allowed = ({"rmse", "mae", "r2"} if task_type == "regression" else
                   {"balanced_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc"})
        if self.primary_metric_ not in allowed:
            raise ValueError(f"Unsupported primary metric '{self.primary_metric_}' for {task_type}; choose from {sorted(allowed)}.")
        models, self.skipped_models_ = self._candidates(task_type, len(X_train), X_train.shape[1])
        if task_type != "regression" and len(X_train) < 200:
            self.skipped_models_["HistGradientBoosting"] = "Skipped: eligible from 200 training rows to avoid unstable/small-sample boosting."
        if task_type == "regression" and len(X_train) < 200:
            self.skipped_models_["HistGradientBoosting"] = "Skipped: eligible from 200 training rows to avoid unstable/small-sample boosting."
        self.task_type_ = task_type
        direction = -1 if self.primary_metric_ in {"rmse", "mae"} else 1
        self.cv_results_, self.failed_models_ = {}, {}
        self.events_ = []
        self.benchmark_table_ = []
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            rich_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), disable=not self.progress or not getattr(__import__('sys').stdout, 'isatty', lambda: False)())
        except ImportError:
            rich_progress = None
        if rich_progress:
            rich_progress.start()
            progress_task = rich_progress.add_task("MLFOX candidates", total=len(models) + len(self.skipped_models_))
        def emit(event_type, candidate=None, fold=None, pre_id=None, details=None):
            event = {"event_type": event_type, "candidate": candidate, "fold": fold,
                     "preprocessor_id": pre_id, "details": details or {}}
            if callable(getattr(self, "event_callback", None)):
                self.event_callback(event)
            # Fold diagnostics can contain a complete fitted CV pipeline and
            # copied labels/indices. Deliver those only to an explicit observer;
            # keep the engine's lifecycle history small and serializable.
            retained_details = {}
            if event_type == "candidate_failed":
                retained_details = {key: value for key, value in event["details"].items() if key == "exception"}
            elif event_type == "candidate_skipped":
                retained_details = {"reason": event["details"].get("reason")}
            elif event_type == "candidate_completed":
                retained_details = {"elapsed_seconds": event["details"].get("elapsed_seconds")}
            self.events_.append({"event_type": event_type, "candidate": candidate,
                                 "fold": fold, "preprocessor_id": pre_id,
                                 "details": retained_details})
        for name, estimator in models.items():
            started = time.perf_counter()
            if self.progress and rich_progress:
                rich_progress.update(progress_task, description=f"Current: {name}")
            elif self.progress: print(f"[MLFOX] Benchmarking {name}", flush=True)
            pipe = Pipeline([("preprocessing", FoldAwarePreprocessor(task_type=task_type, random_state=self.random_state)), ("model", clone(estimator))])
            try:
                def record(event):
                    emit(event["event_type"], name, event["fold"], event["preprocessor_id"], event["details"])
                    if self.progress and event["event_type"] == "fold_started":
                        if rich_progress:
                            rich_progress.update(progress_task, description=f"{name}: fold {event['fold']} (completed {len(self.cv_results_)}, failed {len(self.failed_models_)})")
                        else: print(f"[MLFOX] {name}: fold {event['fold']}", flush=True)
                scores, folds = cross_validate_pipeline(
                    pipe, X_train, y_train, task_type, self.cv, self.random_state,
                    record, name, diagnostics=self.event_callback is not None)
                self.cv_results_[name] = {"metrics": scores, "folds": folds, "status": "success"}
            except Exception as exc:
                failure = {"exception": repr(exc), "fold": None}
                if self.events_ and self.events_[-1]["candidate"] == name:
                    failure["fold"] = self.events_[-1]["fold"]
                self.failed_models_[name] = failure
                emit("candidate_failed", name, failure["fold"], details=failure)
                self.cv_results_[name] = {"status": "failed", "reason": failure}
            elapsed = time.perf_counter() - started
            if self.cv_results_[name]["status"] == "success":
                emit("candidate_completed", name, details={"elapsed_seconds": elapsed})
            self.benchmark_table_.append({"model": name, "primary_metric": self.primary_metric_,
                "direction": "minimize" if direction < 0 else "maximize",
                "cv_mean": self.cv_results_[name].get("metrics", {}).get(self.primary_metric_, {}).get("mean"),
                "cv_std": self.cv_results_[name].get("metrics", {}).get(self.primary_metric_, {}).get("std"),
                "execution_seconds": elapsed, "status": self.cv_results_[name]["status"],
                "failure_reason": self.cv_results_[name].get("reason")})
            if rich_progress: rich_progress.advance(progress_task)
        for skipped_name, reason in self.skipped_models_.items():
            emit("candidate_skipped", skipped_name, details={"reason": reason})
            self.benchmark_table_.append({"model": skipped_name, "primary_metric": self.primary_metric_,
                "direction": "minimize" if direction < 0 else "maximize", "cv_mean": None,
                "cv_std": None, "execution_seconds": 0.0, "status": "skipped", "failure_reason": reason})
            if rich_progress: rich_progress.advance(progress_task)
        if rich_progress: rich_progress.stop()
        successful = [(name, row) for name, row in self.cv_results_.items() if row["status"] == "success" and self.primary_metric_ in row["metrics"]]
        if not successful: raise RuntimeError(f"All candidates failed: {self.failed_models_}")
        best_name, _ = sorted(successful, key=lambda item: direction * item[1]["metrics"][self.primary_metric_]["mean"], reverse=True)[0]
        self.best_model_name_ = best_name
        emit("best_model_selected", best_name)
        self.best_model_ = Pipeline([("preprocessing", FoldAwarePreprocessor(task_type=task_type, random_state=self.random_state)), ("model", clone(models[best_name]))])
        emit("final_refit_started", best_name)
        self.best_model_.fit(X_train, y_train)
        emit("final_refit_completed", best_name)
        self.holdout_metrics_ = None
        if X_test is not None:
            emit("holdout_evaluation_started", best_name)
            self.holdout_metrics_ = metric_values(self.best_model_, X_test, y_test, task_type)
            emit("holdout_evaluation_completed", best_name)
        self.results_ = self.cv_results_
        return self

    def report(self):
        return {"primary_metric": self.primary_metric_, "optimization_direction": "minimize" if self.primary_metric_ in {"rmse", "mae"} else "maximize", "best_model": self.best_model_name_, "benchmark_table": self.benchmark_table_, "cv_results": self.cv_results_, "failed_models": self.failed_models_, "skipped_models": self.skipped_models_, "holdout_metrics": self.holdout_metrics_}
