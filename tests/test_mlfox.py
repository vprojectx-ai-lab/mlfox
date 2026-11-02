import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression

from vprojectx1.mlfox import (
    EncodingScalingIntelligence,
    FeatureEngineeringIntelligence,
    FeatureSelectionIntelligence,
    MissingValueIntelligence,
    analyze_columns,
    detect_task,
    load_dataset,
    profile_dataset,
    split_dataset,
)


def run_preprocessing(split):
    columns = analyze_columns(split.X_train, split.y_train)
    drop = set(columns.constant_columns)
    drop.update(item["column"] for item in columns.duplicate_columns)
    drop.update(item["column"] for item in columns.possible_identifiers)
    drop.update(item["column"] for item in columns.possible_target_leakage)
    x_train = split.X_train.drop(columns=list(drop), errors="ignore")
    x_test = split.X_test.drop(columns=list(drop), errors="ignore")

    imputer = MissingValueIntelligence(add_indicators=True)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    engineer = FeatureEngineeringIntelligence()
    x_train = engineer.fit_transform(x_train)
    x_test = engineer.transform(x_test)
    encoder = EncodingScalingIntelligence()
    x_train = encoder.fit_transform(x_train)
    x_test = encoder.transform(x_test)
    selector = FeatureSelectionIntelligence(min_features=2)
    x_train = selector.fit_transform(x_train, split.y_train, split.task_type)
    x_test = selector.transform(x_test)
    return x_train, x_test, columns, selector


def test_recovery_exports_and_task_detection():
    from vprojectx1.mlfox.feature_engineering import FeatureEngineeringReport, FeatureEngineeringIntelligence
    from vprojectx1.mlfox.feature_selection import FeatureSelectionReport, FeatureSelectionIntelligence

    assert FeatureEngineeringReport is not FeatureSelectionReport
    assert FeatureEngineeringIntelligence is not FeatureSelectionIntelligence
    iris = load_iris(as_frame=True).frame
    assert detect_task(iris, "target")["task_type"] == "multiclass_classification"
    categorical = iris.assign(target=iris.target.astype(str))
    assert detect_task(categorical, "target")["task_type"] == "multiclass_classification"


def test_ingestion_profile_and_split_preserve_indices(tmp_path):
    df = pd.DataFrame({"x": range(30), "target": [0, 1] * 15}, index=np.arange(100, 130))
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.shape == df.shape
    assert profile_dataset(loaded, target="target").rows == 30
    split = split_dataset(df, "target", random_state=3)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert split.X_train.index.equals(split.y_train.index)
    assert len(split.X_train) + len(split.X_test) == len(df)


def test_m4_flags_constants_duplicates_ids_leakage_and_correlation():
    target = pd.Series([0, 1] * 25, name="target")
    x = pd.DataFrame({
        "constant": 1,
        "sequence_id": range(50),
        "signal": target,
        "signal_copy": target,
        "leak": target,
    })
    report = analyze_columns(x, target)
    assert "constant" in report.constant_columns
    assert {item["column"] for item in report.duplicate_columns} >= {"signal_copy", "leak"}
    assert "sequence_id" in {item["column"] for item in report.possible_identifiers}
    assert "leak" in {item["column"] for item in report.possible_target_leakage}
    assert report.high_correlation_pairs


def test_m5_and_m6_fit_on_train_and_transform_unseen_rows():
    train = pd.DataFrame({
        "amount": [0, 1, 2, 3, 4, 100, 200, 400, 800, 1600, 3200, 6400],
        "event_date": pd.date_range("2020-01-01", periods=12),
        "category": ["a", "b"] * 6,
        "missing": [np.nan] + list(range(11)),
    })
    test = pd.DataFrame({
        "amount": [12_800], "event_date": [pd.Timestamp("2021-12-25")],
        "category": ["new"], "missing": [np.nan],
    }, index=[99])
    imputer = MissingValueIntelligence()
    train_i, test_i = imputer.fit_transform(train), imputer.transform(test)
    engineer = FeatureEngineeringIntelligence(skew_threshold=0.5, min_unique_values=5)
    train_f, test_f = engineer.fit_transform(train_i), engineer.transform(test_i)
    assert engineer.to_dict()["output_features"] == train_f.shape[1]
    assert "amount__log1p" in train_f
    assert "event_date" not in train_f
    assert "event_date__is_weekend" in train_f
    assert list(train_f.columns) == list(test_f.columns)
    assert test_f.index.tolist() == [99]
    assert test_f["event_date__is_weekend"].iloc[0] == 1


def test_full_binary_pipeline_avoids_mi_warnings_and_leakage():
    rng = np.random.default_rng(42)
    n = 240
    target = pd.Series(np.tile([0, 1], n // 2), name="churn")
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n),
        "spend": rng.lognormal(3, 1, n),
        "segment": np.where(np.arange(n) % 3, "standard", "premium"),
        "missing_col": rng.normal(size=n),
        "constant": 9,
        "duplicate": np.arange(n),
        "id": np.arange(n),
        "exact_leak": target,
        "target": target,
    })
    df.loc[::9, "missing_col"] = np.nan
    split = split_dataset(df, "target", random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train, test, column_report, selector = run_preprocessing(split)
    assert not caught, [str(w.message) for w in caught]
    assert "exact_leak" in {x["column"] for x in column_report.possible_target_leakage}
    assert "num__exact_leak" not in train
    assert list(train.columns) == list(test.columns)
    assert np.isfinite(train.to_numpy()).all() and np.isfinite(test.to_numpy()).all()
    assert selector.selected_features_


def test_iris_multiclass_and_synthetic_regression_pipeline():
    iris = load_iris(as_frame=True).frame
    iris["target"] = iris["target"].astype(str)
    iris_split = split_dataset(iris, "target", random_state=7)
    train, test, _, _ = run_preprocessing(iris_split)
    assert train.shape[1] > 0 and list(train.columns) == list(test.columns)

    x, y = make_regression(n_samples=180, n_features=6, noise=5, random_state=8)
    regression = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    regression["target"] = y
    regression_split = split_dataset(regression, "target", random_state=8)
    train, test, _, _ = run_preprocessing(regression_split)
    assert train.shape[1] > 0 and list(train.columns) == list(test.columns)


def test_invalid_inputs_and_unseen_categories():
    with pytest.raises(ValueError, match="empty"):
        load_dataset(pd.DataFrame())
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="test_size"):
        split_dataset(pd.DataFrame({"x": [1, 2], "y": [0, 1]}), "y", test_size=1)
    with pytest.raises(ValueError, match="All features are constant"):
        FeatureSelectionIntelligence().fit(
            pd.DataFrame({"fixed": [1, 1]}), pd.Series([0, 1]), "binary_classification"
        )
    encoder = EncodingScalingIntelligence().fit(pd.DataFrame({"kind": ["a", "b"]}))
    result = encoder.transform(pd.DataFrame({"kind": ["unseen"]}))
    assert result.shape == (1, 2)
    assert np.isfinite(result.to_numpy()).all()
