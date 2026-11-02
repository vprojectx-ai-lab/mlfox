
from dataclasses import dataclass, asdict
import re

import pandas as pd


@dataclass
class ColumnIntelligenceReport:
    numerical_columns: list
    categorical_columns: list
    datetime_columns: list
    other_columns: list
    constant_columns: list
    duplicate_columns: list
    possible_identifiers: list
    high_missing_columns: list
    high_cardinality_columns: list
    high_correlation_pairs: list
    possible_target_leakage: list
    column_details: dict
    warnings: list

    def to_dict(self):
        return asdict(self)


def _is_identifier_name(column):
    name = str(column).strip().lower()

    # Pandas-generated CSV index columns
    if re.fullmatch(r"unnamed(?::\s*\d+)?", name):
        return True

    compact = re.sub(r"[^a-z0-9]", "", name)
    tokens = re.split(r"[^a-z0-9]+", name)

    identifier_tokens = {
        "id", "uuid", "identifier", "index",
        "record", "row", "serial"
    }

    return (
        any(token in identifier_tokens for token in tokens)
        or compact.endswith("id")
        or compact in {
            "index", "rownumber", "serialnumber"
        }
    )


def _is_one_to_one_mapping(feature, target, max_levels=20):
    paired = pd.DataFrame({
        "feature": feature,
        "target": target
    }).dropna()

    if paired.empty:
        return False

    feature_count = paired["feature"].nunique()
    target_count = paired["target"].nunique()

    if (
        feature_count < 2
        or feature_count > max_levels
        or feature_count != target_count
    ):
        return False

    feature_mapping = (
        paired.groupby("feature")["target"].nunique()
    )

    target_mapping = (
        paired.groupby("target")["feature"].nunique()
    )

    return (
        feature_mapping.max() == 1
        and target_mapping.max() == 1
    )


def analyze_columns(
    X_train,
    y_train=None,
    target=None,
    missing_threshold=0.50,
    max_categories=50,
    correlation_threshold=0.98
):
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if X_train.empty:
        raise ValueError("Training features are empty.")

    if not 0 < missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1.")

    if max_categories < 2:
        raise ValueError("max_categories must be at least 2.")

    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be between 0 and 1.")

    if not X_train.columns.is_unique:
        raise ValueError("Duplicate feature column names detected.")

    if y_train is not None:
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        if not X_train.index.equals(y_train.index):
            raise ValueError("X_train and y_train indices must match.")

    numerical = []
    categorical = []
    datetime = []
    other = []

    constant = []
    duplicates = []
    identifiers = []
    high_missing = []
    high_cardinality = []
    leakage = []
    correlation_pairs = []
    details = {}
    warnings = []

    # Analyze individual columns
    for column in X_train.columns:
        series = X_train[column]

        missing_count = int(series.isna().sum())
        missing_ratio = missing_count / len(X_train)

        unique_count = int(series.nunique(dropna=True))
        non_missing_count = int(series.notna().sum())

        unique_ratio = (
            unique_count / non_missing_count
            if non_missing_count else 0.0
        )

        if pd.api.types.is_datetime64_any_dtype(series):
            role = "datetime"
            datetime.append(column)

        elif pd.api.types.is_bool_dtype(series):
            role = "categorical"
            categorical.append(column)

        elif pd.api.types.is_numeric_dtype(series):
            role = "numerical"
            numerical.append(column)

        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            role = "categorical"
            categorical.append(column)

        else:
            role = "other"
            other.append(column)

        details[str(column)] = {
            "dtype": str(series.dtype),
            "role": role,
            "missing_count": missing_count,
            "missing_percentage": round(missing_ratio * 100, 2),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 4)
        }

        if unique_count <= 1:
            constant.append(column)

        if missing_ratio >= missing_threshold:
            high_missing.append({
                "column": column,
                "missing_percentage": round(missing_ratio * 100, 2)
            })

        if (
            role == "categorical"
            and unique_count > max_categories
        ):
            high_cardinality.append({
                "column": column,
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4)
            })

        # Possible identifiers
        name_hint = _is_identifier_name(column)

        sequential = (
            non_missing_count > 1
            and unique_count == non_missing_count
            and (
                series.is_monotonic_increasing
                or series.is_monotonic_decreasing
            )
        )

        if name_hint or (
            unique_ratio >= 0.98 and sequential
        ):
            identifiers.append({
                "column": column,
                "unique_ratio": round(unique_ratio, 4),
                "reasons": [
                    reason for condition, reason in [
                        (name_hint, "identifier-like column name"),
                        (sequential, "unique sequential values")
                    ] if condition
                ]
            })

        # Potential target leakage
        if y_train is not None:
            paired = pd.concat(
                [series.rename("feature"), y_train.rename("target")],
                axis=1
            ).dropna()

            exact_match = (
                not paired.empty
                and paired["feature"].reset_index(drop=True).equals(
                    paired["target"].reset_index(drop=True)
                )
            )

            one_to_one = _is_one_to_one_mapping(
                series, y_train
            )

            if exact_match or one_to_one:
                leakage.append({
                    "column": column,
                    "reason": (
                        "exact target match"
                        if exact_match
                        else "one-to-one mapping with target"
                    )
                })

    # Detect duplicate columns
    columns = list(X_train.columns)

    for i, column in enumerate(columns):
        for other_column in columns[i + 1:]:
            left = X_train[column].reset_index(drop=True)
            right = X_train[other_column].reset_index(drop=True)

            if left.equals(right):
                duplicates.append({
                    "column": other_column,
                    "duplicate_of": column
                })

    # Detect highly correlated numerical features
    if len(numerical) > 1:
        corr = X_train[numerical].corr()

        for i, column_a in enumerate(numerical):
            for column_b in numerical[i + 1:]:
                value = corr.loc[column_a, column_b]

                if (
                    pd.notna(value)
                    and abs(value) >= correlation_threshold
                ):
                    correlation_pairs.append({
                        "feature_1": column_a,
                        "feature_2": column_b,
                        "correlation": round(float(value), 4)
                    })

    if identifiers:
        warnings.append("Possible identifier columns detected.")

    if high_missing:
        warnings.append("Columns with high missingness detected.")

    if high_cardinality:
        warnings.append("High-cardinality categorical columns detected.")

    if leakage:
        warnings.append("Possible target leakage detected.")

    if correlation_pairs:
        warnings.append("Highly correlated numerical features detected.")

    return ColumnIntelligenceReport(
        numerical_columns=numerical,
        categorical_columns=categorical,
        datetime_columns=datetime,
        other_columns=other,
        constant_columns=constant,
        duplicate_columns=duplicates,
        possible_identifiers=identifiers,
        high_missing_columns=high_missing,
        high_cardinality_columns=high_cardinality,
        high_correlation_pairs=correlation_pairs,
        possible_target_leakage=leakage,
        column_details=details,
        warnings=warnings
    )
