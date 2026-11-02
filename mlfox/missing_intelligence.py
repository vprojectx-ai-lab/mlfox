
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class MissingValueReport:
    strategies: dict
    dropped_columns: list
    missing_indicators: dict
    warnings: list
    training_rows: int

    def to_dict(self):
        return asdict(self)


class MissingValueIntelligence:

    def __init__(
        self,
        high_missing_threshold=0.20,
        add_indicators=True
    ):
        if not 0 < high_missing_threshold <= 1:
            raise ValueError(
                "high_missing_threshold must be between 0 and 1."
            )

        self.high_missing_threshold = high_missing_threshold
        self.add_indicators = add_indicators
        self.fitted_ = False

    @staticmethod
    def _missing_token(series):
        existing = set(series.dropna().astype(str))

        token = "__MLFOX_MISSING__"
        suffix = 1

        while token in existing:
            token = f"__MLFOX_MISSING_{suffix}__"
            suffix += 1

        return token

    @staticmethod
    def _safe_value(value):
        if value is None or value is pd.NaT:
            return None

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, np.generic):
            value = value.item()

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        return str(value)

    def fit(self, X_train):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if X_train.empty:
            raise ValueError("Training dataset is empty.")

        if not X_train.columns.is_unique:
            raise ValueError("Duplicate column names detected.")

        self.feature_columns_ = list(X_train.columns)
        self.fill_values_ = {}
        self.fill_methods_ = {}
        self.dropped_columns_ = []
        self.indicator_columns_ = {}
        self.strategies_ = {}
        self.warnings_ = []

        used_names = set(X_train.columns)

        for column in X_train.columns:
            series = X_train[column]

            missing_count = int(series.isna().sum())
            missing_ratio = missing_count / len(X_train)

            # Entirely missing columns have no learnable information.
            if missing_count == len(X_train):
                self.dropped_columns_.append(column)

                self.strategies_[str(column)] = {
                    "strategy": "drop_all_missing",
                    "missing_count": missing_count,
                    "missing_percentage": 100.0,
                    "fill_value": None
                }

                self.warnings_.append(
                    f"{column}: entirely missing, will be dropped."
                )
                continue

            # Datetime columns
            if pd.api.types.is_datetime64_any_dtype(series):
                fill_value = series.dropna().median()
                method = "datetime_median"

            # Boolean columns
            elif pd.api.types.is_bool_dtype(series):
                fill_value = series.value_counts().index[0]
                method = "most_frequent"

            # Numerical columns
            elif pd.api.types.is_numeric_dtype(series):
                fill_value = series.median()
                method = "median"

            # Categorical and other columns
            else:
                if (
                    missing_count > 0
                    and missing_ratio >= self.high_missing_threshold
                ):
                    fill_value = self._missing_token(series)
                    method = "missing_category"
                else:
                    fill_value = series.value_counts().index[0]
                    method = "most_frequent"

            self.fill_values_[column] = fill_value
            self.fill_methods_[column] = method

            self.strategies_[str(column)] = {
                "strategy": method,
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_ratio * 100, 2
                ),
                "fill_value": self._safe_value(fill_value)
            }

            if missing_count > 0:
                self.warnings_.append(
                    f"{column}: {missing_count} missing values "
                    f"({missing_ratio * 100:.2f}%)."
                )

                if self.add_indicators:
                    indicator = f"{column}__missing"
                    suffix = 1

                    while indicator in used_names:
                        indicator = (
                            f"{column}__missing_{suffix}"
                        )
                        suffix += 1

                    used_names.add(indicator)
                    self.indicator_columns_[column] = indicator

        self.report_ = MissingValueReport(
            strategies=self.strategies_,
            dropped_columns=self.dropped_columns_,
            missing_indicators=self.indicator_columns_,
            warnings=self.warnings_,
            training_rows=len(X_train)
        )

        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError(
                "Call fit() before transform()."
            )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        missing_columns = [
            col for col in self.feature_columns_
            if col not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing expected columns: {missing_columns}"
            )

        result = X.loc[:, self.feature_columns_].copy()

        # Generate indicators before imputation.
        indicators = {}

        for column, indicator in self.indicator_columns_.items():
            indicators[indicator] = (
                result[column].isna().astype("int8")
            )

        # Apply strategies learned from training data.
        for column, fill_value in self.fill_values_.items():
            if not result[column].isna().any():
                continue

            method = self.fill_methods_[column]

            if method == "missing_category":
                result[column] = (
                    result[column]
                    .astype("object")
                    .fillna(fill_value)
                )

            elif method == "median":
                try:
                    result[column] = result[column].fillna(
                        fill_value
                    )
                except (TypeError, ValueError):
                    result[column] = (
                        pd.to_numeric(
                            result[column], errors="coerce"
                        )
                        .astype("float64")
                        .fillna(fill_value)
                    )

            else:
                try:
                    result[column] = result[column].fillna(
                        fill_value
                    )
                except (TypeError, ValueError):
                    result[column] = (
                        result[column]
                        .astype("object")
                        .fillna(fill_value)
                    )

        # Drop columns that were entirely missing in training.
        if self.dropped_columns_:
            result = result.drop(
                columns=self.dropped_columns_
            )

        # Append missingness indicators.
        for indicator, values in indicators.items():
            result[indicator] = values

        return result

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def to_dict(self):
        if not self.fitted_:
            raise RuntimeError("The imputer has not been fitted.")

        return self.report_.to_dict()
