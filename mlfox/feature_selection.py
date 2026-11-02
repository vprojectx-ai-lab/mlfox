
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)


@dataclass
class FeatureSelectionReport:
    input_features: int
    selected_feature_count: int
    selected_features: list
    constant_features: list
    correlated_features: list
    weak_features: list
    feature_scores: dict
    selection_percentile: float
    correlation_threshold: float
    warnings: list

    def to_dict(self):
        return asdict(self)


class FeatureSelectionIntelligence:

    def __init__(
        self,
        correlation_threshold=0.98,
        selection_percentile=25,
        min_features=5,
        random_state=42
    ):
        if not 0 < correlation_threshold <= 1:
            raise ValueError(
                "correlation_threshold must be between 0 and 1."
            )

        if not 0 <= selection_percentile < 100:
            raise ValueError(
                "selection_percentile must be in [0, 100)."
            )

        if min_features < 1:
            raise ValueError(
                "min_features must be at least 1."
            )

        self.correlation_threshold = correlation_threshold
        self.selection_percentile = selection_percentile
        self.min_features = min_features
        self.random_state = random_state
        self.fitted_ = False

    def fit(self, X_train, y_train, task_type):

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a DataFrame.")

        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a Series.")

        if X_train.empty:
            raise ValueError("Training features are empty.")

        if not X_train.columns.is_unique:
            raise ValueError("Duplicate feature names detected.")

        if not X_train.index.equals(y_train.index):
            raise ValueError(
                "X_train and y_train indices must match."
            )

        if y_train.isna().any():
            raise ValueError(
                "Training target contains missing values."
            )

        if y_train.nunique() < 2:
            raise ValueError(
                "Target must contain at least 2 values."
            )

        valid_tasks = {
            "binary_classification",
            "multiclass_classification",
            "regression"
        }

        if task_type not in valid_tasks:
            raise ValueError(
                f"Unsupported task type: {task_type}"
            )

        if task_type == "regression":
            if not pd.api.types.is_numeric_dtype(y_train):
                raise TypeError(
                    "Regression target must be numeric."
                )

        if X_train.isna().any().any():
            raise ValueError("Run M5 before feature selection.")

        non_numeric = [
            col for col in X_train.columns
            if not pd.api.types.is_numeric_dtype(X_train[col])
        ]

        if non_numeric:
            raise TypeError(
                f"Numerical features expected. Found: {non_numeric}"
            )

        X = X_train.astype("float64")

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "Features contain infinite values."
            )

        self.input_features_ = list(X.columns)
        self.task_type_ = task_type

        # 1. Remove constant features
        self.constant_features_ = [
            col for col in X.columns
            if X[col].nunique() <= 1
        ]

        candidates = [
            col for col in X.columns
            if col not in self.constant_features_
        ]

        if not candidates:
            raise ValueError(
                "All features are constant."
            )

        # 2. Detect discrete features
        discrete_limit = min(
            20,
            max(2, int(np.sqrt(len(X))))
        )

        discrete_mask = np.array([
            X[col].nunique() <= discrete_limit
            for col in candidates
        ], dtype=bool)

        # IMPORTANT:
        # Convert discrete features to integer category codes.
        # This prevents sklearn from treating scaled binary
        # and categorical features as continuous labels.
        X_mi = X[candidates].copy()

        for index, column in enumerate(candidates):

            if discrete_mask[index]:
                codes, _ = pd.factorize(
                    X_mi[column],
                    sort=True
                )

                X_mi[column] = codes.astype("int64")

        # 3. Mutual Information scoring
        if task_type == "regression":

            scores = mutual_info_regression(
                X_mi,
                y_train.to_numpy(dtype=float),
                discrete_features=discrete_mask,
                random_state=self.random_state
            )

        else:

            scores = mutual_info_classif(
                X_mi,
                y_train.to_numpy(),
                discrete_features=discrete_mask,
                random_state=self.random_state
            )

        self.feature_scores_ = {
            col: float(score)
            for col, score in zip(candidates, scores)
        }

        # 4. Rank features by Mutual Information
        ranked_features = sorted(
            candidates,
            key=lambda col: (
                -self.feature_scores_[col],
                self.input_features_.index(col)
            )
        )

        # 5. Remove highly correlated features
        corr_matrix = X[candidates].corr().abs()

        retained = []
        self.correlated_features_ = []

        for candidate in ranked_features:

            redundant_with = None
            max_correlation = 0.0

            for existing in retained:

                corr_value = corr_matrix.loc[
                    candidate, existing
                ]

                if (
                    pd.notna(corr_value)
                    and corr_value >= self.correlation_threshold
                    and corr_value > max_correlation
                ):
                    max_correlation = float(corr_value)
                    redundant_with = existing

            if redundant_with is not None:

                self.correlated_features_.append({
                    "feature": candidate,
                    "redundant_with": redundant_with,
                    "correlation": round(
                        max_correlation, 4
                    ),
                    "mutual_information": round(
                        self.feature_scores_[candidate], 6
                    )
                })

            else:
                retained.append(candidate)

        # 6. Remove weak features
        retained_scores = np.array([
            self.feature_scores_[col]
            for col in retained
        ])

        self.warnings_ = []

        if (
            len(retained) <= self.min_features
            or np.ptp(retained_scores) <= 1e-12
        ):
            selected = retained.copy()

            if np.ptp(retained_scores) <= 1e-12:
                self.warnings_.append(
                    "Feature scores have little variation. "
                    "Weak-feature filtering skipped."
                )

        else:

            threshold = float(
                np.percentile(
                    retained_scores,
                    self.selection_percentile
                )
            )

            selected = [
                col for col in retained
                if self.feature_scores_[col] > threshold
            ]

            if len(selected) < self.min_features:
                selected = retained[:min(
                    self.min_features,
                    len(retained)
                )]

        selected_set = set(selected)

        # Preserve original feature order
        self.selected_features_ = [
            col for col in self.input_features_
            if col in selected_set
        ]

        self.weak_features_ = [
            col for col in retained
            if col not in selected_set
        ]

        if not self.correlated_features_:
            self.warnings_.append(
                "No highly correlated features detected."
            )

        if not self.weak_features_:
            self.warnings_.append(
                "No weak features removed."
            )

        # 7. Generate report
        self.report_ = FeatureSelectionReport(
            input_features=len(self.input_features_),
            selected_feature_count=len(
                self.selected_features_
            ),
            selected_features=self.selected_features_,
            constant_features=self.constant_features_,
            correlated_features=self.correlated_features_,
            weak_features=self.weak_features_,
            feature_scores={
                col: round(score, 6)
                for col, score in sorted(
                    self.feature_scores_.items(),
                    key=lambda item: -item[1]
                )
            },
            selection_percentile=self.selection_percentile,
            correlation_threshold=self.correlation_threshold,
            warnings=self.warnings_
        )

        self.fitted_ = True
        return self

    def transform(self, X):

        if not self.fitted_:
            raise RuntimeError(
                "Call fit() before transform()."
            )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")

        missing = [
            col for col in self.input_features_
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing expected features: {missing}"
            )

        result = X.loc[
            :, self.selected_features_
        ].copy()

        if result.isna().any().any():
            raise ValueError(
                "Missing values found during transform."
            )

        if not np.isfinite(
            result.astype("float64").to_numpy()
        ).all():
            raise ValueError(
                "Non-finite values found during transform."
            )

        return result

    def fit_transform(self, X_train, y_train, task_type):

        return self.fit(
            X_train,
            y_train,
            task_type
        ).transform(X_train)

    def get_feature_names_out(self):

        if not self.fitted_:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return list(self.selected_features_)

    def to_dict(self):

        if not self.fitted_:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return self.report_.to_dict()
