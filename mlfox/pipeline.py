"""Fold-local preprocessing that composes the existing M4–M8 components."""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from .column_intelligence import analyze_columns
from .missing_intelligence import MissingValueIntelligence
from .feature_engineering import FeatureEngineeringIntelligence
from .encoding import EncodingScalingIntelligence
from .feature_selection import FeatureSelectionIntelligence


class FoldAwarePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, task_type="binary_classification", exclude_identifiers=True,
                 high_missing_threshold=0.20, max_categories=100,
                 selection_percentile=25, min_features=5, random_state=42):
        self.task_type = task_type
        self.exclude_identifiers = exclude_identifiers
        self.high_missing_threshold = high_missing_threshold
        self.max_categories = max_categories
        self.selection_percentile = selection_percentile
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be a DataFrame and y a Series.")
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match.")
        self.feature_names_in_ = list(X.columns)
        self.column_report_ = analyze_columns(X, y)
        remove = set(self.column_report_.constant_columns)
        remove.update(item["column"] for item in self.column_report_.duplicate_columns)
        # M4's one-to-one heuristic is useful for categorical lookup columns,
        # but continuous numeric predictors can appear one-to-one on small
        # folds. Exclude exact target copies and categorical one-to-one flags.
        for item in self.column_report_.possible_target_leakage:
            col = item["column"]
            exact = item.get("reason") == "exact target match"
            categorical = col in self.column_report_.categorical_columns
            if exact or categorical:
                remove.add(col)
        if self.exclude_identifiers:
            remove.update(item["column"] for item in self.column_report_.possible_identifiers)
        self.excluded_features_ = sorted(remove, key=self.feature_names_in_.index)
        retained = [c for c in self.feature_names_in_ if c not in remove]
        if not retained:
            raise ValueError("M4 exclusions removed every feature.")
        self.imputer_ = MissingValueIntelligence(self.high_missing_threshold)
        imputed = self.imputer_.fit_transform(X[retained])
        self.engineer_ = FeatureEngineeringIntelligence()
        engineered = self.engineer_.fit_transform(imputed)
        self.encoder_ = EncodingScalingIntelligence(self.max_categories)
        encoded = self.encoder_.fit_transform(engineered)
        self.selector_ = FeatureSelectionIntelligence(
            selection_percentile=self.selection_percentile,
            min_features=min(self.min_features, max(1, encoded.shape[1])),
            random_state=self.random_state)
        self.selector_.fit(encoded, y, self.task_type)
        self.n_features_out_ = len(self.selector_.selected_features_)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        missing = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected features: {missing}")
        data = X.loc[:, self.feature_names_in_].drop(columns=self.excluded_features_)
        data = self.imputer_.transform(data)
        data = self.engineer_.transform(data)
        data = self.encoder_.transform(data)
        return self.selector_.transform(data)

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("y is required because M4/M8 use target information.")
        return self.fit(X, y).transform(X)
