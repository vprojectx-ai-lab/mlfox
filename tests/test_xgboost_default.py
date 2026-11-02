import pandas as pd
from sklearn.datasets import make_classification, make_regression
from xgboost import XGBClassifier, XGBRegressor

from mlfox.model_selection import ModelSelectionEngine
from mlfox.tuning import default_search_space


def test_xgboost_is_in_default_candidate_sets_and_cpu_sequential():
    engine = ModelSelectionEngine(progress=False)
    classification, _ = engine._candidates("binary_classification", 300, 8)
    regression, _ = engine._candidates("regression", 300, 8)

    classifier = classification["XGBoost"]
    regressor = regression["XGBoost Regressor"]
    assert isinstance(classifier, XGBClassifier)
    assert isinstance(regressor, XGBRegressor)
    for estimator in (classifier, regressor):
        assert estimator.get_params()["tree_method"] == "hist"
        assert estimator.get_params()["n_jobs"] == 1
        assert estimator.get_params()["device"] == "cpu"


def test_builtin_xgboost_estimators_have_bounded_tuning_spaces():
    assert default_search_space(XGBClassifier())
    assert default_search_space(XGBRegressor())


def test_xgboost_classification_and_regression_run_inside_fold_pipeline():
    Xc, yc = make_classification(n_samples=48, n_features=5, n_informative=3,
                                 n_redundant=0, random_state=18)
    classification = pd.DataFrame(Xc, columns=list("abcde"))
    engine = ModelSelectionEngine(cv=2, random_state=18, progress=False,
                                  estimators={"XGBoost": XGBClassifier(
                                      n_estimators=8, max_depth=2, tree_method="hist",
                                      n_jobs=1, random_state=18, eval_metric="logloss",
                                      verbosity=0)})
    engine.fit(classification, pd.Series(yc), "binary_classification")
    assert engine.cv_results_["XGBoost"]["status"] == "success"
    assert engine.best_model_name_ == "XGBoost"

    Xr, yr = make_regression(n_samples=48, n_features=5, noise=1, random_state=19)
    regression = pd.DataFrame(Xr, columns=list("abcde"))
    engine = ModelSelectionEngine(cv=2, random_state=19, progress=False,
                                  estimators={"XGBoost Regressor": XGBRegressor(
                                      n_estimators=8, max_depth=2, tree_method="hist",
                                      n_jobs=1, random_state=19, verbosity=0)})
    engine.fit(regression, pd.Series(yr), "regression")
    assert engine.cv_results_["XGBoost Regressor"]["status"] == "success"
    assert engine.best_model_name_ == "XGBoost Regressor"
