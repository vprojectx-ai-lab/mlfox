import importlib.metadata
import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


def _classification_frame(n=80, classes=2, categorical=False):
    X, y = make_classification(n_samples=n, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=classes,
                               n_clusters_per_class=1, random_state=141)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    if categorical:
        y = y.astype(str)
    frame["target"] = y
    frame.loc[::17, "x0"] = np.nan
    return frame


def test_public_imports_distribution_and_runtime_configuration():
    import mlfox as mlfox_package
    from mlfox import MLFOX, load_model, mlfox
    from mlfox.mlfox import mlfox as factory

    assert callable(mlfox) and callable(load_model)
    assert callable(mlfox_package)
    assert isinstance(mlfox_package(), MLFOX)
    assert MLFOX.__name__ == "MLFOX"
    assert callable(factory)
    assert importlib.metadata.version("mlfox") == "1.0.1"
    model = mlfox(cv=3, test_size=0.25, random_state=17, tune=True,
                  max_tuning_trials=2, progress=False)
    assert (model.cv, model.test_size, model.random_state) == (3, .25, 17)
    assert model.tune and model.max_tuning_trials == 2 and not model.progress


def test_default_lifecycle_history_does_not_retain_fold_pipelines():
    from mlfox import mlfox
    frame = _classification_frame()
    model = mlfox(cv=2, progress=False).fit(frame, "target")
    assert model.engine_.events_
    assert all("fitted_pipeline" not in event["details"] and "preprocessor" not in event["details"]
               for event in model.engine_.events_)


def test_date_transformer_report_width_matches_actual_output():
    from mlfox.feature_engineering import FeatureEngineeringIntelligence
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=12),
                          "amount": np.arange(12, dtype=float) ** 2})
    transformer = FeatureEngineeringIntelligence(skew_threshold=0.5, min_unique_values=5).fit(frame)
    transformed = transformer.transform(frame)
    assert transformer.to_dict()["output_features"] == transformed.shape[1]


def test_classification_prediction_alignment_missing_values_and_probability_consistency():
    from mlfox import mlfox
    frame = _classification_frame()
    model = mlfox(cv=2, progress=False).fit(frame, "target")
    expected = model.predict(frame.drop(columns="target").iloc[:8])
    reordered = frame.drop(columns="target").iloc[:8].loc[:, ["x4", "x2", "x0", "x3", "x1"]]
    pd.testing.assert_series_equal(expected, model.predict(reordered))
    repeated = model.predict(reordered)
    pd.testing.assert_series_equal(expected, repeated)
    probabilities = model.predict_proba(reordered)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(probabilities.idxmax(axis=1).to_numpy(), expected.to_numpy())
    with_extra = reordered.assign(extra=1)
    pd.testing.assert_series_equal(expected, model.predict(with_extra))


def test_regression_workflow_and_round_trip(tmp_path):
    from mlfox import load_model, mlfox
    X, y = make_regression(n_samples=70, n_features=4, noise=2, random_state=42)
    frame = pd.DataFrame(X, columns=list("abcd")); frame["target"] = y
    model = mlfox(task="regression", cv=2, progress=False).fit(frame, "target")
    before = model.predict(frame.drop(columns="target").iloc[:9])
    path = model.save(tmp_path / "regression.joblib")
    with pytest.warns(UserWarning, match="only load trusted files"):
        restored = load_model(path)
    pd.testing.assert_series_equal(before, restored.predict(frame.drop(columns="target").iloc[:9]))
    assert restored._build_report()["holdout_metrics"] == model.holdout_metrics_


def test_explicit_numeric_multiclass_and_invalid_continuous_target():
    from mlfox import mlfox
    frame = _classification_frame(n=120, classes=3)
    model = mlfox(task="multiclass_classification", cv=2, progress=False).fit(frame, "target")
    assert model.task_type_ == "multiclass_classification"
    assert model._build_report()["task"]["task_type"] == "multiclass_classification"
    X, y = make_regression(n_samples=80, n_features=4, random_state=5)
    continuous = pd.DataFrame(X, columns=list("abcd")); continuous["target"] = y
    with pytest.raises(ValueError, match="discrete integer multiclass"):
        mlfox(task="multiclass_classification", cv=2, progress=False).fit(continuous, "target")
    with pytest.raises(ValueError, match="exactly 2"):
        mlfox(task="binary_classification", cv=2, progress=False).fit(frame, "target")


def test_auto_detects_integer_coded_multiclass_and_keeps_continuous_regression():
    from mlfox.profiler import detect_task
    from mlfox import mlfox

    rng = np.random.RandomState(122)
    numeric_classes = pd.DataFrame(rng.normal(size=(140, 5)), columns=list("abcde"))
    numeric_classes["target"] = np.tile([1, 2, 3, 4, 5, 6, 7], 20)
    assert detect_task(numeric_classes, "target")["task_type"] == "multiclass_classification"
    fitted = mlfox(cv=2, progress=False, random_state=122).fit(numeric_classes, "target")
    assert fitted.task_type_ == "multiclass_classification"

    _, continuous = make_regression(n_samples=120, n_features=4, random_state=122)
    regression = pd.DataFrame({"target": continuous})
    assert detect_task(regression, "target")["task_type"] == "regression"


def test_persistence_artifact_errors_and_atomic_roundtrip(tmp_path):
    import joblib
    from mlfox import load_model, mlfox
    frame = _classification_frame()
    fitted = mlfox(cv=2, progress=False).fit(frame, "target")
    destination = tmp_path / "nested" / "model.joblib"
    fitted.save(destination)
    assert destination.is_file()
    with pytest.warns(UserWarning, match="only load trusted files"):
        loaded = load_model(destination)
    assert loaded._build_report()["best_estimator"] == fitted._build_report()["best_estimator"]
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
    invalid = tmp_path / "invalid.joblib"
    joblib.dump({"format": "wrong"}, invalid)
    with pytest.warns(UserWarning, match="only load trusted files"):
        with pytest.raises(ValueError, match="Unsupported or invalid"):
            load_model(invalid)


def test_calibrated_svm_has_compatible_bounded_tuning_space():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    from mlfox.tuning import HyperparameterTuner, default_search_space
    frame = _classification_frame(n=48)
    X = frame.drop(columns="target")
    y = frame["target"]
    estimator = CalibratedClassifierCV(SVC(random_state=3), cv=2)
    tuner = HyperparameterTuner(estimator, default_search_space(estimator),
                                max_trials=1, cv=2, random_state=3,
                                task_type="binary_classification")
    tuner.fit(X, y)
    assert tuner.trial_results_[0]["status"] == "success"
    assert "estimator__C" in tuner.best_params_


def test_explainability_and_reports_remain_safe_and_serializable(tmp_path, monkeypatch):
    from mlfox import mlfox
    frame = _classification_frame()
    model = mlfox(cv=2, progress=False).fit(frame, "target")
    query = frame.drop(columns="target").iloc[[0]]
    reference = frame.drop(columns="target").iloc[20:35]
    prediction = model.predict(query)
    model.feature_importance(frame, "target", n_repeats=2, random_state=42)
    model.explain_prediction(query, reference)
    pd.testing.assert_series_equal(prediction, model.predict(query))
    model.holdout_metrics_["optional_metric"] = None
    html_path = model.export_report(tmp_path / "report.html")
    styled_html_path = model.html(tmp_path / "styled_report.html")
    json_path = model.export_report(tmp_path / "report.json")
    assert "https://" not in html_path.read_text(encoding="utf-8")
    styled_html = styled_html_path.read_text(encoding="utf-8")
    assert "<style>" in styled_html and "Dataset overview" in styled_html
    payload = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert payload["explainability"]["global_feature_importance"]
    assert payload["explainability"]["individual_explanations"]
    assert payload["holdout_metrics"]["optional_metric"] is None
    assert "original_value" not in payload["explainability"]["individual_explanations"][0]["results"][0]
    monkeypatch.setitem(sys.modules, "rich", None)
    assert model.report() is None
    assert model._build_report()["best_estimator"]


def test_invalid_configuration_and_report_before_fit():
    from mlfox import mlfox
    with pytest.raises(ValueError, match="cv must be at least 2"):
        mlfox(cv=1)
    with pytest.raises(RuntimeError, match="Call fit"):
        mlfox(progress=False).report()


def test_report_prints_vprojectx_brand_footer(capsys, monkeypatch):
    from mlfox import mlfox
    monkeypatch.setitem(sys.modules, "rich", None)
    model = mlfox(cv=2, progress=False).fit(_classification_frame(), "target")
    model.report()
    assert "\n~ Product by vprojectx" in capsys.readouterr().out


def test_notebook_report_suppresses_automatic_dict_output(monkeypatch):
    import importlib
    from mlfox import mlfox
    implementation = importlib.import_module("mlfox.mlfox")
    monkeypatch.setattr(implementation, "_running_in_notebook", lambda: True)
    model = mlfox(cv=2, progress=False).fit(_classification_frame(), "target")
    assert model.report() is None
    assert model._build_report()["best_estimator"]
