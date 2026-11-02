"""Generic M1–M13 integration smoke run; detailed checks live in tests/."""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from . import (
    EncodingScalingIntelligence,
    FeatureEngineeringIntelligence,
    FeatureSelectionIntelligence,
    MissingValueIntelligence,
    analyze_columns,
    detect_task,
    split_dataset,
    load_dataset,
    profile_dataset,
    ModelSelectionEngine,
    HyperparameterTuner,
)


def main():
    rng = np.random.default_rng(42)
    rows = 120
    target = pd.Series(np.tile([0, 1], rows // 2), name="target")
    frame = pd.DataFrame({
        "amount": rng.lognormal(2, 1, rows),
        "category": np.where(np.arange(rows) % 3, "standard", "premium"),
        "event_date": pd.date_range("2024-01-01", periods=rows),
        "sometimes_missing": rng.normal(size=rows),
        "identifier": np.arange(rows),
        "target_copy": target,
        "target": target,
    })
    frame.loc[::11, "sometimes_missing"] = np.nan

    frame = load_dataset(frame)
    profile = profile_dataset(frame, target="target")
    assert profile.rows == rows

    task = detect_task(frame, "target")
    split = split_dataset(frame, "target", task_type=task["task_type"])
    column_report = analyze_columns(split.X_train, split.y_train)
    excluded = set(column_report.constant_columns)
    excluded.update(x["column"] for x in column_report.duplicate_columns)
    excluded.update(x["column"] for x in column_report.possible_identifiers)
    excluded.update(x["column"] for x in column_report.possible_target_leakage)
    train = split.X_train.drop(columns=list(excluded), errors="ignore")
    test = split.X_test.drop(columns=list(excluded), errors="ignore")

    imputer = MissingValueIntelligence()
    train, test = imputer.fit_transform(train), imputer.transform(test)
    engineering = FeatureEngineeringIntelligence()
    train, test = engineering.fit_transform(train), engineering.transform(test)
    encoder = EncodingScalingIntelligence()
    train, test = encoder.fit_transform(train), encoder.transform(test)
    selector = FeatureSelectionIntelligence(min_features=2)
    train = selector.fit_transform(train, split.y_train, split.task_type)
    test = selector.transform(test)

    assert train.shape[0] == len(split.y_train)
    assert test.shape[0] == len(split.y_test)
    assert list(train.columns) == list(test.columns)
    assert np.isfinite(train.to_numpy()).all()
    assert np.isfinite(test.to_numpy()).all()
    engine = ModelSelectionEngine(cv=2, progress=False, estimators={
        "logistic": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=500)
    })
    engine.fit(split.X_train, split.y_train, task["task_type"], split.X_test, split.y_test)
    assert engine.best_model_ is not None and engine.holdout_metrics_ is not None
    tuner = HyperparameterTuner(
        __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=300),
        {"C": [0.1, 1.0]}, max_trials=2, cv=2, random_state=42)
    tuner.fit(split.X_train, split.y_train)
    assert tuner.best_estimator_ is not None and tuner.report_["trials_attempted"] <= 2
    from .mlfox import mlfox
    from .persistence import load_model
    with tempfile.TemporaryDirectory(prefix="mlfox-m13-") as directory:
        directory = Path(directory)
        train_path = directory / "dataset.csv"
        evaluation_path = directory / "evaluation.csv"
        customer_path = directory / "customer.csv"
        background_path = directory / "reference.csv"
        frame.to_csv(train_path, index=False)
        frame.to_csv(evaluation_path, index=False)
        split.X_test.iloc[[0]].to_csv(customer_path, index=False)
        split.X_train.iloc[:12].to_csv(background_path, index=False)
        api_model = mlfox(cv=2, progress=False).fit(train_path, "target")
        assert api_model._build_report()["holdout_metrics"]
        predictions = api_model.predict(split.X_test)
        probabilities = api_model.predict_proba(split.X_test)
        assert len(predictions) == len(split.X_test)
        assert probabilities.shape[0] == len(split.X_test)
        api_model.feature_importance(evaluation_path, "target", n_repeats=2, random_state=42)
        api_model.explain_prediction(customer_path, background_path)
        html_report = api_model.export_report(Path(directory) / "integration.html")
        json_report = api_model.export_report(Path(directory) / "integration.json")
        assert "MLFOX Model Report" in html_report.read_text(encoding="utf-8")
        import json
        assert json.loads(json_report.read_text(encoding="utf-8"))["explainability"]["global_feature_importance"]
        artifact = api_model.save(Path(directory) / "integration.joblib")
        reloaded = load_model(artifact)
        pd.testing.assert_series_equal(predictions, reloaded.predict(split.X_test))
        pd.testing.assert_frame_equal(probabilities, reloaded.predict_proba(split.X_test))
        assert reloaded._build_report()["explainability"]["global_feature_importance"]
        assert reloaded._build_report()["best_estimator"] == api_model._build_report()["best_estimator"]
    print(f"M1–M13 integration passed: train={train.shape}, test={test.shape}, model={engine.best_model_name_}, tuning_trials={tuner.report_['trials_attempted']}, api_model={api_model.report_['best_estimator']}, prediction_rows={len(predictions)}")


if __name__ == "__main__":
    main()
