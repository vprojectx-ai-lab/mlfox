"""Training-fitted datetime and skew transformations (MLFOX M6)."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringReport:
    input_features: int
    output_features: int
    datetime_features: dict
    log_transformed_features: list
    warnings: list

    def to_dict(self):
        return asdict(self)


class FeatureEngineeringIntelligence:
    """Add date parts/cycles and log1p features using training-only choices.

    Detected datetime source columns are replaced by numeric calendar features.
    Numeric source columns are retained when a log1p feature is added.
    """

    def __init__(self, skew_threshold=1.0, min_unique_values=10):
        if not isinstance(skew_threshold, (int, float)) or skew_threshold < 0:
            raise ValueError("skew_threshold must be a non-negative number.")
        if not isinstance(min_unique_values, int) or min_unique_values < 2:
            raise ValueError("min_unique_values must be an integer of at least 2.")
        self.skew_threshold = float(skew_threshold)
        self.min_unique_values = min_unique_values
        self.fitted_ = False

    @staticmethod
    def _parse_datetime(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors="coerce", utc=True)
        if (pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)):
            sample = series.dropna().astype(str).head(100)
            if sample.empty:
                return None
            ratio = sample.str.contains(
                r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}", regex=True
            ).mean()
            if ratio < 0.8:
                return None
            try:
                parsed_sample = pd.to_datetime(sample, format="mixed", errors="coerce", utc=True)
            except TypeError:  # pandas versions before format="mixed"
                parsed_sample = pd.to_datetime(sample, errors="coerce", utc=True)
            if parsed_sample.notna().mean() < 0.8:
                return None
            try:
                return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
            except TypeError:
                return pd.to_datetime(series, errors="coerce", utc=True)
        return None

    @staticmethod
    def _unique_name(base, used):
        name, suffix = base, 1
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        return name

    def fit(self, X_train):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if X_train.empty:
            raise ValueError("Training dataset is empty.")
        if not X_train.columns.is_unique:
            raise ValueError("Duplicate feature names detected.")
        if not all(isinstance(col, str) for col in X_train.columns):
            raise TypeError("All feature column names must be strings.")

        self.feature_columns_ = list(X_train.columns)
        self.datetime_columns_ = {}
        self.log_columns_ = []
        self.warnings_ = []

        for col in self.feature_columns_:
            parsed = self._parse_datetime(X_train[col])
            if parsed is not None:
                self.datetime_columns_[col] = [
                    f"{col}__year", f"{col}__month", f"{col}__day",
                    f"{col}__weekday", f"{col}__is_weekend",
                    f"{col}__month_sin", f"{col}__month_cos",
                    f"{col}__weekday_sin", f"{col}__weekday_cos",
                ]

        datetime_set = set(self.datetime_columns_)
        for col in self.feature_columns_:
            series = X_train[col]
            if col in datetime_set or not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if (len(values) < self.min_unique_values
                    or np.unique(values).size < self.min_unique_values
                    or not np.isfinite(values).all()
                    or np.any(values < 0)):
                continue
            skew = pd.Series(values).skew()
            if pd.notna(skew) and skew > self.skew_threshold:
                self.log_columns_.append(col)

        self.report_ = FeatureEngineeringReport(
            input_features=len(self.feature_columns_),
            output_features=0,
            datetime_features={col: list(names) for col, names in self.datetime_columns_.items()},
            log_transformed_features=list(self.log_columns_),
            warnings=self.warnings_,
        )
        self.fitted_ = True
        # The fitted choices determine the output width; running transform on
        # the full training frame here and again in fit_transform duplicated a
        # full pass and transient DataFrame allocation for every CV fold.
        self.report_.output_features = (
            len(self.feature_columns_)
            + 8 * len(self.datetime_columns_)
            + len(self.log_columns_)
        )
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        missing = [col for col in self.feature_columns_ if col not in X.columns]
        if missing:
            raise ValueError(f"Missing expected features: {missing}")

        result = X.loc[:, self.feature_columns_].copy()
        used = set(result.columns)
        for col, names in self.datetime_columns_.items():
            parsed = self._parse_datetime(result[col])
            if parsed is None:
                parsed = pd.to_datetime(result[col], errors="coerce", utc=True)
            month = parsed.dt.month
            weekday = parsed.dt.day_of_week
            parts = {
                names[0]: parsed.dt.year,
                names[1]: month,
                names[2]: parsed.dt.day,
                names[3]: weekday,
                names[4]: weekday.isin([5, 6]).astype("int8"),
                names[5]: np.sin(2 * np.pi * (month - 1) / 12),
                names[6]: np.cos(2 * np.pi * (month - 1) / 12),
                names[7]: np.sin(2 * np.pi * weekday / 7),
                names[8]: np.cos(2 * np.pi * weekday / 7),
            }
            result = result.drop(columns=[col])
            used.discard(col)
            for name, values in parts.items():
                actual = self._unique_name(name, used)
                result[actual] = values.to_numpy()

        for col in self.log_columns_:
            values = pd.to_numeric(result[col], errors="coerce")
            if (values.dropna() < 0).any():
                raise ValueError(f"Feature '{col}' has negative values; log1p is undefined.")
            name = self._unique_name(f"{col}__log1p", used)
            result[name] = np.log1p(values)

        return result

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def get_feature_names_out(self):
        if not self.fitted_:
            raise RuntimeError("Feature engineering has not been fitted.")
        return list(self.transform(pd.DataFrame(columns=self.feature_columns_)).columns)

    def to_dict(self):
        if not self.fitted_:
            raise RuntimeError("Feature engineering has not been fitted.")
        return self.report_.to_dict()
