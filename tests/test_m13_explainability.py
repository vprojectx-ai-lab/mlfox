import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from mlfox import load_model, mlfox


def classifier_frame(n=72):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3,
                               n_redundant=0, random_state=73)
    frame = pd.DataFrame(X, columns=["signal", "amount", "measure", "noise"])
    frame["segment"] = np.where(np.arange(n) % 2, "retail", "business")
    frame.loc[::13, "amount"] = np.nan
    frame["target"] = y
    return frame


def regression_frame(n=72):
    X, y = make_regression(n_samples=n, n_features=4, noise=3, random_state=74)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["target"] = y
    return frame


def fitted_classifier():
    frame = classifier_frame()
    return mlfox(cv=2, progress=False).fit(frame, "target"), frame


def test_classification_permutation_importance_ranking_and_determinism(tmp_path):
    model, frame = fitted_classifier()
    first = model.feature_importance(frame, "target", n_repeats=3, random_state=19)
    second = model.feature_importance(frame, "target", n_repeats=3, random_state=19)
    assert list(first.columns) == ["feature", "importance_mean", "importance_std", "rank"]
    assert set(first.feature) == set(model.feature_schema_)
    assert first["rank"].tolist() == list(range(1, len(first) + 1))
    pd.testing.assert_frame_equal(first, second)
    assert model._build_report()["explainability"]["global_feature_importance"]["metric"] == "balanced_accuracy"


def test_regression_permutation_importance_and_metric(tmp_path):
    frame = regression_frame()
    model = mlfox(task="regression", cv=2, progress=False).fit(frame, "target")
    importance = model.feature_importance(frame, "target", n_repeats=2, random_state=5, metric="rmse")
    assert set(importance.feature) == set(frame.columns) - {"target"}
    assert np.isfinite(importance.importance_mean).all()
    assert model._build_report()["explainability"]["global_feature_importance"]["metric"] == "rmse"


def test_invalid_importance_metrics_and_missing_target():
    model, frame = fitted_classifier()
    with pytest.raises(ValueError, match="Unsupported or incompatible metric"):
        model.feature_importance(frame, "target", metric="rmse")
    with pytest.raises(ValueError, match="not found"):
        model.feature_importance(frame.drop(columns="target"), "target")
    with pytest.raises(ValueError, match="positive integer"):
        model.feature_importance(frame, "target", n_repeats=0)


def test_classification_perturbation_explanation_and_no_refit(monkeypatch):
    model, frame = fitted_classifier()
    observation = frame.drop(columns="target").iloc[[2]]
    background = frame.drop(columns="target").iloc[10:30]
    before_predictions = model.predict(observation)
    original_fit = model.best_pipeline_.fit

    def fail_fit(*args, **kwargs):
        raise AssertionError("explanation must not fit the model")

    monkeypatch.setattr(model.best_pipeline_, "fit", fail_fit)
    result = model.explain_prediction(observation, background)
    assert len(result) == len(model.feature_schema_)
    assert result.method.str.contains("perturbation").all()
    assert "original_class_probabilities" in result
    assert model.predict(observation).equals(before_predictions)
    assert "original_value" not in model._build_report()["explainability"]["individual_explanations"][0]["results"][0]
    monkeypatch.setattr(model.best_pipeline_, "fit", original_fit)


def test_regression_perturbation_and_background_validation():
    frame = regression_frame()
    model = mlfox(task="regression", cv=2, progress=False).fit(frame, "target")
    obs = frame.drop(columns="target").iloc[[1]]
    result = model.explain_prediction(obs, frame.drop(columns="target").iloc[20:35])
    assert "prediction_change" in result
    assert np.isfinite(result.prediction_change).all()
    with pytest.raises(ValueError, match="exactly one observation"):
        model.explain_prediction(obs.iloc[0:0], frame.drop(columns="target").iloc[20:35])
    with pytest.raises(ValueError, match="no non-missing values"):
        bad_background = frame.drop(columns="target").iloc[20:35].copy()
        bad_background["a"] = np.nan
        model.explain_prediction(obs, bad_background)


def test_csv_inputs_and_html_json_export_escaping_and_report_integration(tmp_path):
    frame = classifier_frame()
    model = mlfox(cv=2, progress=False).fit(frame, "target")
    evaluation_path = tmp_path / "evaluation.csv"
    frame.to_csv(evaluation_path, index=False)
    model.feature_importance(evaluation_path, "target", n_repeats=2)
    obs_path = tmp_path / "customer.csv"
    frame.drop(columns="target").iloc[[0]].to_csv(obs_path, index=False)
    background_path = tmp_path / "background.csv"
    frame.drop(columns="target").iloc[20:32].to_csv(background_path, index=False)
    model.explain_prediction(obs_path, background_path)
    model.profile_.warnings.append('<script>alert("x")</script>')
    html_path = model.export_report(tmp_path / "report.html")
    json_path = model.export_report(tmp_path / "report.json")
    html = html_path.read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    assert "<script>alert(\"x\")</script>" not in html
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["explainability"]["global_feature_importance"]["features"]
    assert payload["explainability"]["individual_explanations"]
    assert payload["holdout_metrics"] == model.holdout_metrics_
    with pytest.raises(ValueError, match="only .html and .json"):
        model.export_report(tmp_path / "report.csv")


def test_m12_artifact_compatibility_and_m13_metadata_persistence(tmp_path):
    model, frame = fitted_classifier()
    model.feature_importance(frame, "target", n_repeats=2)
    model.explain_prediction(frame.drop(columns="target").iloc[[0]], frame.drop(columns="target").iloc[10:25])
    path = model.save(tmp_path / "m13.joblib")
    restored = load_model(path)
    assert restored._build_report()["explainability"]["global_feature_importance"] is not None
    assert len(restored._build_report()["explainability"]["individual_explanations"]) == 1

    # Emulate an M12 artifact whose structured report predates optional M13 metadata.
    artifact = joblib.load(path)
    artifact["report"].pop("explainability", None)
    old_path = tmp_path / "m12.joblib"
    joblib.dump(artifact, old_path)
    old_loaded = load_model(old_path)
    assert old_loaded.predict(frame.drop(columns="target").iloc[:2]).shape == (2,)
    assert old_loaded._build_report()["explainability"]["global_feature_importance"] is None
