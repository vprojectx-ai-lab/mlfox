import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_regression
from mlfox.model_selection import ModelSelectionEngine

def test_primary_metric_validation_and_direction():
    x = pd.DataFrame({"x": np.arange(40.), "z": np.tile([0., 1.], 20)})
    y = pd.Series(np.tile([0, 1], 20))
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        ModelSelectionEngine(estimators={"tree": DecisionTreeClassifier()}, primary_metric="rmse", progress=False).fit(x, y, "binary_classification")
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        ModelSelectionEngine(estimators={"tree": DecisionTreeRegressor()}, primary_metric="accuracy", progress=False).fit(x, pd.Series(np.arange(40.)), "regression")

def test_m9_binary_holdout_and_custom_estimators():
    rng = np.random.default_rng(22)
    x = pd.DataFrame({"x": rng.normal(size=80), "cat": np.where(np.arange(80)%2, "a", "b")})
    y = pd.Series((x.x > 0).astype(int))
    engine = ModelSelectionEngine(cv=3, progress=False, estimators={"weak": DecisionTreeClassifier(max_depth=1, random_state=1), "strong": DecisionTreeClassifier(random_state=1)})
    engine.fit(x.iloc[:65], y.iloc[:65], "binary_classification", x.iloc[65:], y.iloc[65:])
    assert engine.best_model_name_ in engine.cv_results_
    assert engine.holdout_metrics_ is not None

def test_m9_multiclass_and_regression_runs():
    iris = load_iris(as_frame=True)
    x = iris.data.iloc[:90].copy()
    y = iris.target.iloc[:90].astype(str)
    cls = ModelSelectionEngine(cv=2, progress=False, estimators={"tree": DecisionTreeClassifier(random_state=2)}).fit(x, y, "multiclass_classification")
    assert cls.best_model_name_ == "tree"
    xx, yy = make_regression(n_samples=60, n_features=4, random_state=3)
    reg = ModelSelectionEngine(cv=2, progress=False, estimators={"tree": DecisionTreeRegressor(random_state=2)}).fit(pd.DataFrame(xx, columns=[f"x{i}" for i in range(xx.shape[1])]), pd.Series(yy), "regression")
    assert reg.best_model_name_ == "tree"
