
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class EncodingReport:
    numerical_columns: list
    categorical_columns: list
    scaled_columns: list
    encoded_columns: list
    input_features: int
    output_features: int
    max_categories: int
    warnings: list

    def to_dict(self):
        return asdict(self)


class EncodingScalingIntelligence:

    def __init__(self, max_categories=100):
        if max_categories < 2:
            raise ValueError("max_categories must be at least 2.")

        self.max_categories = max_categories
        self.fitted_ = False

    @staticmethod
    def _create_encoder(max_categories):
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=max_categories
            )
        except TypeError:
            try:
                return OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                    max_categories=max_categories
                )
            except TypeError:
                return OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False
                )

    def fit(self, X_train):

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if X_train.empty:
            raise ValueError("Training dataset is empty.")

        if not X_train.columns.is_unique:
            raise ValueError("Duplicate feature names detected.")

        if not all(isinstance(c, str) for c in X_train.columns):
            raise TypeError("All feature column names must be strings.")

        if X_train.isna().any().any():
            raise ValueError(
                "Missing values detected. Run M5 before M7."
            )

        self.feature_columns_ = list(X_train.columns)

        self.numeric_columns_ = [
            col for col in self.feature_columns_
            if (
                pd.api.types.is_numeric_dtype(X_train[col])
                and not pd.api.types.is_bool_dtype(X_train[col])
            )
        ]

        self.categorical_columns_ = [
            col for col in self.feature_columns_
            if col not in self.numeric_columns_
        ]

        # Datetime features should already be handled by M6.
        datetime_columns = [
            col for col in self.feature_columns_
            if pd.api.types.is_datetime64_any_dtype(X_train[col])
        ]

        if datetime_columns:
            raise ValueError(
                "Datetime columns remain. Run M6 first: "
                f"{datetime_columns}"
            )

        if not self.numeric_columns_ and not self.categorical_columns_:
            raise ValueError("No usable features found.")

        self.scaler_ = None
        self.encoder_ = None

        output_names = []
        self.warnings_ = []

        # Numerical scaling
        if self.numeric_columns_:
            numeric_data = X_train[
                self.numeric_columns_
            ].astype("float64")

            if not np.isfinite(numeric_data.to_numpy()).all():
                raise ValueError(
                    "Numerical features contain infinite values."
                )

            self.scaler_ = StandardScaler()
            self.scaler_.fit(numeric_data)

            output_names.extend([
                f"num__{col}"
                for col in self.numeric_columns_
            ])

        # Categorical encoding
        if self.categorical_columns_:
            categorical_data = (
                X_train[self.categorical_columns_]
                .astype(str)
            )

            self.encoder_ = self._create_encoder(
                self.max_categories
            )

            self.encoder_.fit(categorical_data)

            encoded_names = (
                self.encoder_.get_feature_names_out(
                    self.categorical_columns_
                )
            )

            output_names.extend([
                f"cat__{name}"
                for name in encoded_names
            ])

        self.feature_names_out_ = output_names

        if not self.categorical_columns_:
            self.warnings_.append(
                "No categorical columns required encoding."
            )

        if not self.numeric_columns_:
            self.warnings_.append(
                "No numerical columns required scaling."
            )

        self.report_ = EncodingReport(
            numerical_columns=self.numeric_columns_,
            categorical_columns=self.categorical_columns_,
            scaled_columns=self.numeric_columns_,
            encoded_columns=self.categorical_columns_,
            input_features=len(self.feature_columns_),
            output_features=len(self.feature_names_out_),
            max_categories=self.max_categories,
            warnings=self.warnings_
        )

        self.fitted_ = True
        return self

    def transform(self, X):

        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        missing = [
            col for col in self.feature_columns_
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing expected columns: {missing}"
            )

        data = X.loc[:, self.feature_columns_]

        if data.isna().any().any():
            raise ValueError(
                "Missing values detected. Run M5 before M7."
            )

        output_blocks = []

        # Transform numerical features
        if self.numeric_columns_:
            numeric_data = data[
                self.numeric_columns_
            ].astype("float64")

            numeric_array = numeric_data.to_numpy()

            if not np.isfinite(numeric_array).all():
                raise ValueError(
                    "Numerical features contain infinite values."
                )

            scaled = self.scaler_.transform(numeric_data)
            output_blocks.append(scaled)

        # Transform categorical features
        if self.categorical_columns_:
            categorical_data = (
                data[self.categorical_columns_]
                .astype(str)
            )

            encoded = self.encoder_.transform(
                categorical_data
            )

            output_blocks.append(encoded)

        output = np.hstack(output_blocks)

        return pd.DataFrame(
            output,
            index=X.index,
            columns=self.feature_names_out_
        )

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def get_feature_names_out(self):
        if not self.fitted_:
            raise RuntimeError("The encoder has not been fitted.")

        return list(self.feature_names_out_)

    def to_dict(self):
        if not self.fitted_:
            raise RuntimeError("The encoder has not been fitted.")

        return self.report_.to_dict()
