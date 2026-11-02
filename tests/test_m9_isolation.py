import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from mlfox.model_selection import ModelSelectionEngine


class DeliberatelyBroken(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        raise RuntimeError("deliberate fit failure")

class GuardedSeries(pd.Series):
    _metadata = ["blocked"]
    @property
    def _constructor(self): return GuardedSeries
    def __getitem__(self, key):
        if getattr(self, "blocked", False): raise AssertionError("holdout labels accessed before final evaluation")
        return super().__getitem__(key)


def _data():
    rng = np.random.default_rng(19)
    X = pd.DataFrame({"signal": np.r_[rng.normal(-20, 1, 20), rng.normal(30, 1, 20), rng.normal(0, 1, 20)],
                      "missing": np.r_[np.full(20, np.nan), np.arange(20.), np.arange(20.) + 50],
                      "category": np.where(np.arange(60) % 2, "a", "b")}, index=np.arange(100, 160))
    y = pd.Series(np.tile([0, 1], 30), index=X.index)
    return X, y


def test_fold_preprocessors_imputers_and_selector_labels_are_training_local():
    X, y = _data()
    events = []
    engine = ModelSelectionEngine(cv=3, progress=False, estimators={"tree": DecisionTreeClassifier(random_state=1)}, event_callback=events.append)
    engine.fit(X, y, "binary_classification")
    fitted = [e for e in events if e["event_type"] == "preprocessing_fitted"]
    assert len(fitted) == 3
    assert len({e["preprocessor_id"] for e in fitted}) == 3
    for event in fitted:
        train_indices = event["details"]["train_indices"]
        assert set(train_indices).isdisjoint(set(next(x["details"]["validation_indices"] for x in events if x["event_type"] == "fold_started" and x["fold"] == event["fold"])))
        expected = X.loc[train_indices, "missing"].median()
        assert event["details"]["imputer_statistics"]["missing"] == expected
        assert event["details"]["selector_labels"] == y.loc[train_indices].tolist()
        assert event["details"]["selector_input_indices"] == train_indices


def test_candidates_have_independent_fitted_preprocessors_and_validation_cannot_change_fit():
    from sklearn.tree import DecisionTreeClassifier
    from mlfox.evaluation import cross_validate_pipeline
    from sklearn.pipeline import Pipeline
    from mlfox.pipeline import FoldAwarePreprocessor
    X, y = _data()
    events = []
    models = {"one": DecisionTreeClassifier(random_state=3), "two": DecisionTreeClassifier(max_depth=2, random_state=4)}
    engine = ModelSelectionEngine(cv=3, progress=False, estimators=models, event_callback=events.append).fit(X, y, "binary_classification")
    fitted = [e["details"]["preprocessor"] for e in events if e["event_type"] == "preprocessing_fitted"]
    assert len(fitted) == 6
    assert len({id(p) for p in fitted}) == 6
    pipe = Pipeline([("preprocessing", FoldAwarePreprocessor()), ("model", DecisionTreeClassifier())])
    base_events = []
    altered = X.copy()
    # Alter rows that are validation data in a deterministic single split.
    from sklearn.model_selection import StratifiedKFold
    train_idx, valid_idx = next(StratifiedKFold(3, shuffle=True, random_state=11).split(X, y))
    cross_validate_pipeline(pipe, X.iloc[train_idx], y.iloc[train_idx], "binary_classification", cv=2, event_callback=base_events.append)
    # For a fixed training partition, alter only held-out rows in the outer split;
    # fitted imputer statistics in those CV folds remain determined by their train index sets.
    altered.iloc[valid_idx, altered.columns.get_loc("missing")] = 999999
    changed_events = []
    cross_validate_pipeline(pipe, altered.iloc[train_idx], y.iloc[train_idx], "binary_classification", cv=2, event_callback=changed_events.append)
    stats = lambda es: [e["details"]["imputer_statistics"] for e in es if e["event_type"] == "preprocessing_fitted"]
    assert stats(base_events) == stats(changed_events)


def test_holdout_order_and_failure_continuation():
    X, y = _data()
    events = []
    engine = ModelSelectionEngine(cv=3, progress=False,
        estimators={"broken": DeliberatelyBroken(), "tree": DecisionTreeClassifier(random_state=2)}, event_callback=events.append)
    engine.fit(X.iloc[:45], y.iloc[:45], "binary_classification", X.iloc[45:], y.iloc[45:])
    assert engine.cv_results_["broken"]["status"] == "failed"
    assert "deliberate fit failure" in engine.failed_models_["broken"]["exception"]
    assert engine.cv_results_["tree"]["status"] == "success"
    types = [e["event_type"] for e in events]
    assert types.index("best_model_selected") < types.index("final_refit_started") < types.index("final_refit_completed") < types.index("holdout_evaluation_started") < types.index("holdout_evaluation_completed")
    assert sum(t == "holdout_evaluation_completed" for t in types) == 1
    assert events[types.index("holdout_evaluation_started")]["candidate"] == engine.best_model_name_
    assert engine.failed_models_["broken"]["fold"] == 1
    assert len([e for e in events if e["event_type"] == "holdout_evaluation_completed"]) == 1

def test_holdout_labels_are_not_accessed_until_after_selection_and_refit(monkeypatch):
    from mlfox import model_selection
    X, y = _data()
    train_x, test_x = X.iloc[:45], X.iloc[45:]
    train_y = y.iloc[:45]
    holdout_y = GuardedSeries(y.iloc[45:].copy())
    holdout_y.blocked = True
    actual_metric = model_selection.metric_values
    calls = []
    def guarded_metric(model, features, labels, task):
        if labels is holdout_y:
            calls.append(model)
            holdout_y.blocked = False
        return actual_metric(model, features, labels, task)
    monkeypatch.setattr(model_selection, "metric_values", guarded_metric)
    engine = ModelSelectionEngine(cv=3, progress=False, estimators={"a": DecisionTreeClassifier(random_state=1), "b": DecisionTreeClassifier(max_depth=2, random_state=2)})
    engine.fit(train_x, train_y, "binary_classification", test_x, holdout_y)
    assert len(calls) == 1
    assert calls[0] is engine.best_model_
    kinds = [event["event_type"] for event in engine.events_]
    assert kinds.index("best_model_selected") < kinds.index("final_refit_completed") < kinds.index("holdout_evaluation_started") < kinds.index("holdout_evaluation_completed")

def test_skipped_histogram_candidate_has_reason_and_cannot_win():
    X, y = _data()
    engine = ModelSelectionEngine(cv=2, progress=False, estimators={"tree": DecisionTreeClassifier(random_state=1)}).fit(X.iloc[:40], y.iloc[:40], "binary_classification")
    assert engine.skipped_models_["HistGradientBoosting"]
    assert engine.best_model_name_ != "HistGradientBoosting"
