import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from mlfox.tuning import HyperparameterTuner
from mlfox.tuning import default_search_space
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVC

def data():
    x, y = make_classification(n_samples=48, n_features=4, n_informative=3, n_redundant=0, random_state=5)
    return pd.DataFrame(x, columns=list("abcd")), pd.Series(y)

def test_bounded_reproducible_trials_and_holdout_separate():
    X, y = data(); space = {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]}
    first = HyperparameterTuner(LogisticRegression(max_iter=300), space, max_trials=2, random_state=7, cv=2).fit(X[:40], y[:40], X[40:], y[40:])
    second = HyperparameterTuner(LogisticRegression(max_iter=300), space, max_trials=2, random_state=7, cv=2).fit(X[:40], y[:40])
    assert [r["params"] for r in first.trial_results_] == [r["params"] for r in second.trial_results_]
    assert len(first.trial_results_) == 2 and first.holdout_metrics_ is not None

def test_failed_trial_does_not_prevent_later_trials():
    X,y=data()
    tuner=HyperparameterTuner(LogisticRegression(max_iter=300), {"C":[1.0],"bogus":[1]}, max_trials=1, cv=2)
    with pytest.raises(RuntimeError, match="All tuning trials failed"):
        tuner.fit(X,y)
    assert tuner.trial_results_[0]["status"] == "failed"

def test_trial_count_seed_variation_and_shared_fold_indices(monkeypatch):
    from mlfox import tuning
    X,y=data(); seen=[]
    real=tuning.cross_validate_pipeline
    def wrapped(pipe,x,y,task,cv,seed):
        from sklearn.model_selection import StratifiedKFold
        seen.append([tuple(v.tolist()) for _,v in StratifiedKFold(cv,shuffle=True,random_state=seed).split(x,y)])
        return real(pipe,x,y,task,cv,seed)
    monkeypatch.setattr(tuning,"cross_validate_pipeline",wrapped)
    space={"C":[0.01,0.1,1,10,100],"solver":["liblinear"]}
    a=HyperparameterTuner(LogisticRegression(max_iter=300),space,max_trials=3,random_state=2,cv=2).fit(X,y)
    b=HyperparameterTuner(LogisticRegression(max_iter=300),space,max_trials=3,random_state=2,cv=2).fit(X,y)
    c=HyperparameterTuner(LogisticRegression(max_iter=300),space,max_trials=3,random_state=3,cv=2).fit(X,y)
    assert len(a.trial_results_) == 3
    assert [r["params"] for r in a.trial_results_] == [r["params"] for r in b.trial_results_]
    assert [r["params"] for r in a.trial_results_] != [r["params"] for r in c.trial_results_]
    assert all(x == seen[0] for x in seen[:3])
    assert all(x == seen[3] for x in seen[3:6])
    assert all(x == seen[6] for x in seen[6:9])

def test_holdout_labels_do_not_affect_cv_ranking(monkeypatch):
    X,y=data(); space={"C":[0.1,1,10],"solver":["liblinear"]}
    first=HyperparameterTuner(LogisticRegression(max_iter=300),space,max_trials=3,random_state=8,cv=2).fit(X[:40],y[:40],X[40:],y[40:])
    altered=HyperparameterTuner(LogisticRegression(max_iter=300),space,max_trials=3,random_state=8,cv=2).fit(X[:40],y[:40],X[40:],1-y[40:])
    assert first.best_params_ == altered.best_params_
    assert first.report_["holdout_metrics"] != altered.report_["holdout_metrics"]

def test_metric_ranking_uses_only_mean_direction(monkeypatch):
    from mlfox import tuning
    X,y=data()
    # Deliberately conflicting std values prove ranking uses the configured primary mean only.
    scores={0.1:(0.70,0.0), 1.0:(0.90,99.0)}
    def fake_cv(pipe,x,y,task,cv,seed):
        c=pipe.named_steps["model"].C
        mean,std=scores[c]
        return {"accuracy":{"mean":mean,"std":std},"rmse":{"mean":1-mean,"std":std}},2
    monkeypatch.setattr(tuning,"cross_validate_pipeline",fake_cv)
    maximize=HyperparameterTuner(LogisticRegression(),{"C":[0.1,1.0]},2,1,2,"binary_classification","accuracy")
    maximize.fit(X,y)
    assert maximize.best_params_["C"] == 1.0
    scores[0.1]=(0.1,0.0); scores[1.0]=(0.9,99.0)
    scores[0.1]=(0.9,0.0); scores[1.0]=(0.1,99.0)
    from sklearn.linear_model import Ridge
    minimize=HyperparameterTuner(Ridge(),{"alpha":[0.1,1.0]},2,1,2,"regression","rmse")
    def fake_reg(pipe,x,y,task,cv,seed):
        mean,std=scores[pipe.named_steps["model"].alpha]
        return {"rmse":{"mean":mean,"std":std}},2
    monkeypatch.setattr(tuning,"cross_validate_pipeline",fake_reg)
    minimize.fit(X,pd.Series(np.arange(len(y),dtype=float)))
    assert minimize.best_params_["alpha"] == 1.0

def test_failed_trials_excluded_and_holdout_called_once_after_refit(monkeypatch):
    from mlfox import tuning
    X,y=data(); events=[]; calls=[]
    real=tuning.cross_validate_pipeline
    def cv(pipe,x,y,task,folds,seed):
        if pipe.named_steps["model"].C == 0.1: raise RuntimeError("broken config")
        return real(pipe,x,y,task,folds,seed)
    actual_metric=tuning.metric_values
    def metric(model,x,y,task):
        calls.append((model, len(x)))
        return actual_metric(model,x,y,task)
    monkeypatch.setattr(tuning,"cross_validate_pipeline",cv)
    monkeypatch.setattr(tuning,"metric_values",metric)
    tuner=HyperparameterTuner(LogisticRegression(max_iter=200),{"C":[0.1,1.]},2,2,2,event_callback=events.append)
    tuner.fit(X[:40],y[:40],X[40:],y[40:])
    assert tuner.trial_results_[0]["status"] == "failed"
    assert tuner.best_params_["C"] == 1.0
    assert len(calls) == 1 and calls[0][0] is tuner.best_estimator_
    assert calls[0][1] == 8
    assert tuner.report_["holdout_metrics"] is not None and "holdout_metrics" not in tuner.report_["trials"][0]
    assert events[-1]["event_type"] == "best_model_selected"

def test_validation_rows_cannot_change_fold_imputer_stats(monkeypatch):
    from mlfox import tuning
    X,y=data(); saved=[]; real=tuning.cross_validate_pipeline
    def capture(pipe,x,y,task,cv,seed):
        def event(e):
            if e["event_type"] == "preprocessing_fitted": saved.append((e["details"]["train_indices"],e["details"]["imputer_statistics"]))
        return real(pipe,x,y,task,cv,seed,event_callback=event)
    monkeypatch.setattr(tuning,"cross_validate_pipeline",capture)
    params={"C":[1.0]}
    HyperparameterTuner(LogisticRegression(max_iter=200),params,1,cv=2).fit(X,y)
    baseline=saved.copy(); saved.clear()
    changed=X.copy(); changed.iloc[0,:]=99999
    HyperparameterTuner(LogisticRegression(max_iter=200),params,1,cv=2).fit(changed,y)
    # Compare fold whose validation set contains row zero: it must retain identical train stats.
    b=dict((tuple(i),s) for i,s in baseline); c=dict((tuple(i),s) for i,s in saved)
    for train_indices in b:
        if 0 not in train_indices: assert b[train_indices] == c[train_indices]

def test_guarded_holdout_labels_only_accessed_by_final_metric(monkeypatch):
    from mlfox import tuning
    class Guarded(pd.Series):
        _metadata=["allow"]
        @property
        def _constructor(self): return Guarded
        def __getitem__(self,key):
            if not getattr(self,"allow",False): raise AssertionError("holdout accessed before final metric")
            return super().__getitem__(key)
    X,y=data(); hold=Guarded(y.iloc[40:].copy()); hold.allow=False; events=[]; calls=[]
    metric=tuning.metric_values
    def guarded(model,x,labels,task):
        if labels is hold:
            calls.append(model); assert events[-1]["event_type"] == "best_model_selected"; hold.allow=True
        return metric(model,x,labels,task)
    monkeypatch.setattr(tuning,"metric_values",guarded)
    tuner=HyperparameterTuner(LogisticRegression(max_iter=200),{"C":[0.1,1]},2,cv=2,event_callback=events.append)
    tuner.fit(X.iloc[:40],y.iloc[:40],X.iloc[40:],hold)
    assert len(calls)==1 and calls[0] is tuner.best_estimator_
    assert tuner.holdout_metrics_ is not None and "holdout_metrics" not in tuner.trial_results_[0]

def test_default_search_space_for_supported_families():
    estimators=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),ExtraTreesClassifier(),HistGradientBoostingClassifier(),SVC(),Ridge(),ElasticNet(),DecisionTreeRegressor(),RandomForestRegressor(),ExtraTreesRegressor(),HistGradientBoostingRegressor()]
    assert all(default_search_space(est) for est in estimators)
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_svm = CalibratedClassifierCV(SVC(), cv=2)
    assert "estimator__C" in default_search_space(calibrated_svm)

def test_regression_tuner_bounded_trials():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    X,y=make_regression(n_samples=36,n_features=3,random_state=11)
    tuner=HyperparameterTuner(Ridge(),{"alpha":[0.1,1,10]},max_trials=2,cv=2,task_type="regression",primary_metric="rmse")
    tuner.fit(pd.DataFrame(X,columns=["a","b","c"]),pd.Series(y))
    assert len(tuner.trial_results_)==2 and tuner.best_params_["alpha"] in {0.1,1,10}
