
import re
from dataclasses import dataclass, asdict

import pandas as pd


# ==========================================
# DATA PROFILE STRUCTURE
# ==========================================

@dataclass
class DatasetProfile:
    rows: int
    columns: int
    target: object

    numerical_columns: list
    categorical_columns: list
    datetime_candidates: list
    other_columns: list

    missing_analysis: dict
    unique_analysis: dict
    numerical_summary: dict
    categorical_summary: dict

    duplicate_rows: int
    duplicate_columns: list
    constant_columns: list
    possible_identifiers: list

    warnings: list

    def to_dict(self):
        return asdict(self)


# ==========================================
# DATETIME DETECTION
# ==========================================

DATE_TOKENS = {
    "date", "datetime", "timestamp", "time",
    "dob", "birthdate", "created_at", "updated_at"
}


def is_datetime_candidate(column, series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return False

    sample = series.dropna().astype(str).head(100)

    if sample.empty:
        return False

    name_hint = any(
        token in str(column).lower()
        for token in ["date", "time", "timestamp", "dob", "birth"]
    )

    pattern = r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}"

    value_ratio = sample.str.contains(
        pattern, regex=True
    ).mean()

    if not name_hint and value_ratio < 0.8:
        return False

    parsed = pd.to_datetime(
        sample,
        format="mixed",
        errors="coerce"
    )

    return parsed.notna().mean() >= 0.8
# ==========================================
# IDENTIFIER DETECTION
# ==========================================

def detect_possible_identifier(column, series):

    name = str(column).lower()

    unique_count = series.nunique(dropna=True)
    total = series.notna().sum()

    ratio = unique_count / total if total else 0

    reasons = []

    tokens = set(
        re.findall(r"[a-z0-9]+", name)
    )

    if tokens.intersection(
        {"id", "uuid", "identifier", "ticket", "reference"}
    ):
        reasons.append("identifier-like column name")

    if ratio >= 0.95 and unique_count > 1:
        reasons.append("very high uniqueness")

    if reasons:
        return {
            "column": str(column),
            "reasons": reasons,
            "unique_ratio": round(ratio, 4)
        }

    return None


# ==========================================
# MAIN PROFILER
# ==========================================

def profile_dataset(df, target=None, source=None):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    numerical = []
    categorical = []
    datetime_columns = []
    other = []

    missing_analysis = {}
    unique_analysis = {}

    numerical_summary = {}
    categorical_summary = {}

    possible_identifiers = []
    constant_columns = []

    warnings = []

    # ======================================
    # ANALYSE EACH COLUMN
    # ======================================

    for column in df.columns:

        series = df[column]
        name = str(column)

        missing_count = int(series.isna().sum())
        total_count = len(series)

        missing_pct = (
            missing_count / total_count * 100
        )

        unique_count = int(series.nunique(dropna=True))

        non_missing = int(series.notna().sum())

        unique_ratio = (
            unique_count / non_missing
            if non_missing else 0
        )

        # Missing-value analysis
        missing_analysis[name] = {
            "count": missing_count,
            "percentage": round(missing_pct, 2)
        }

        # Unique-value analysis
        unique_analysis[name] = {
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 4)
        }

        # Constant columns
        if series.nunique(dropna=False) <= 1:
            constant_columns.append(name)

        # Datetime detection comes before
        # ordinary numerical/categorical detection.
        if is_datetime_candidate(column, series):

            datetime_columns.append(name)

        elif pd.api.types.is_bool_dtype(series):

            categorical.append(name)

            categorical_summary[name] = {
                "top_values": [
                    {
                        "value": str(value),
                        "count": int(count)
                    }
                    for value, count in
                    series.value_counts().head(10).items()
                ]
            }

        elif pd.api.types.is_numeric_dtype(series):

            numerical.append(name)

            stats = series.describe()

            numerical_summary[name] = {
                key: (
                    float(value)
                    if pd.notna(value)
                    else None
                )
                for key, value in stats.items()
            }

        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):

            categorical.append(name)

            categorical_summary[name] = {
                "top_values": [
                    {
                        "value": str(value),
                        "count": int(count)
                    }
                    for value, count in
                    series.value_counts().head(10).items()
                ]
            }

        else:
            other.append(name)

        # Possible identifier detection
        identifier = detect_possible_identifier(
            column, series
        )

        if identifier:
            possible_identifiers.append(identifier)

        # Data quality warnings
        if missing_count > 0:
            warnings.append(
                f"{name}: {missing_pct:.2f}% missing values"
            )

    # ======================================
    # DUPLICATE ROWS
    # ======================================

    duplicate_rows = int(df.duplicated().sum())

    # ======================================
    # DUPLICATE COLUMNS
    # ======================================

    duplicate_columns = []

    for i in range(len(df.columns)):

        for j in range(i + 1, len(df.columns)):

            if df.iloc[:, i].equals(df.iloc[:, j]):

                duplicate_columns.append({
                    "first": str(df.columns[i]),
                    "duplicate": str(df.columns[j])
                })

    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate rows detected"
        )

    if duplicate_columns:
        warnings.append(
            f"{len(duplicate_columns)} duplicate columns detected"
        )

    # ======================================
    # BUILD PROFILE
    # ======================================

    profile = DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        target=target,

        numerical_columns=numerical,
        categorical_columns=categorical,
        datetime_candidates=datetime_columns,
        other_columns=other,

        missing_analysis=missing_analysis,
        unique_analysis=unique_analysis,

        numerical_summary=numerical_summary,
        categorical_summary=categorical_summary,

        duplicate_rows=duplicate_rows,
        duplicate_columns=duplicate_columns,
        constant_columns=constant_columns,
        possible_identifiers=possible_identifiers,

        warnings=warnings
    )

    return profile
def is_discrete_numeric_multiclass(values):
    """Recognize repeated, integer-coded multiclass labels by cardinality.

    Continuous and high-cardinality numeric targets remain regression. If a
    numeric target is ambiguous, callers can explicitly set the task.
    """
    import numpy as np
    import pandas as pd

    valid = pd.Series(values).dropna()
    if not pd.api.types.is_numeric_dtype(valid.dtype) or len(valid) == 0:
        return False
    unique_count = int(valid.nunique())
    cardinality_limit = min(100, max(20, int(2 * np.sqrt(len(valid)))))
    numeric = pd.to_numeric(valid, errors="coerce")
    return bool(
        unique_count >= 3
        and unique_count <= cardinality_limit
        and numeric.notna().all()
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and (numeric % 1 == 0).all()
        and valid.value_counts().min() >= 2
    )


def detect_task(df, target):
    import pandas as pd

    if target not in df.columns:
        matches = [
            col for col in df.columns
            if str(col).strip().lower() == str(target).strip().lower()
        ]

        if len(matches) == 1:
            target = matches[0]
        else:
            raise ValueError(f"Target column '{target}' not found.")

    y = df[target]
    valid_y = y.dropna()

    if valid_y.empty:
        raise ValueError("Target column contains no valid values.")

    unique_count = valid_y.nunique()

    if unique_count < 2:
        raise ValueError("Target must contain at least 2 unique values.")

    dtype = y.dtype

    is_categorical = (
        pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )

    if is_categorical:
        task_type = (
            "binary_classification"
            if unique_count == 2
            else "multiclass_classification"
        )

    elif pd.api.types.is_numeric_dtype(dtype):
        if unique_count == 2:
            task_type = "binary_classification"
        elif is_discrete_numeric_multiclass(valid_y):
            task_type = "multiclass_classification"
        else:
            task_type = "regression"

    else:
        raise ValueError(f"Unsupported target datatype: {dtype}")

    result = {
        "target": target,
        "task_type": task_type,
        "target_dtype": str(dtype),
        "unique_values": int(unique_count),
        "valid_samples": int(len(valid_y)),
        "missing_samples": int(y.isna().sum()),
        "class_distribution": None,
        "warnings": []
    }

    if "classification" in task_type:
        distribution = valid_y.value_counts()
        result["class_distribution"] = {
            str(label): int(count)
            for label, count in distribution.items()
        }

    if y.isna().any():
        result["warnings"].append(
            f"{int(y.isna().sum())} missing target values."
        )

    return result
