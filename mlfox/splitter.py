
from dataclasses import dataclass
from math import ceil
from numbers import Real

import pandas as pd
from sklearn.model_selection import train_test_split

from .profiler import detect_task


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target: str
    task_type: str
    test_size: float
    random_state: int
    dropped_target_rows: int

    def summary(self):
        result = {
            "target": self.target,
            "task_type": self.task_type,
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "features": self.X_train.shape[1],
            "dropped_target_rows": self.dropped_target_rows,
            "train_shape": self.X_train.shape,
            "test_shape": self.X_test.shape,
        }

        if "classification" in self.task_type:
            result["train_class_distribution"] = {
                str(k): int(v)
                for k, v in self.y_train.value_counts().items()
            }
            result["test_class_distribution"] = {
                str(k): int(v)
                for k, v in self.y_test.value_counts().items()
            }

        return result


def split_dataset(
    df,
    target,
    test_size=0.2,
    random_state=42,
    task_type=None
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not isinstance(test_size, Real) or isinstance(test_size, bool):
        raise TypeError("test_size must be a numeric proportion.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Resolve target name case-insensitively
    matches = [
        col for col in df.columns
        if str(col).strip().lower() == str(target).strip().lower()
    ]

    if len(matches) != 1:
        raise ValueError(f"Target column '{target}' not found or ambiguous.")

    target = matches[0]

    # Exclude rows with missing target values
    original_rows = len(df)
    missing_target = df[target].isna()
    dropped_rows = int(missing_target.sum())
    # Avoid a full frame copy when no rows need to be removed. Downstream
    # split/drop operations do not mutate the caller's frame.
    clean_df = df.loc[~missing_target] if dropped_rows else df

    if len(clean_df) < 2:
        raise ValueError("Not enough valid target samples to split.")

    X = clean_df.drop(columns=[target])
    y = clean_df[target]

    if X.shape[1] == 0:
        raise ValueError("Dataset has no features besides the target.")

    # Detect task if not supplied
    if task_type is None:
        task_type = detect_task(clean_df, target)["task_type"]

    valid_tasks = {
        "binary_classification",
        "multiclass_classification",
        "regression"
    }

    if task_type not in valid_tasks:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Stratify classification datasets
    stratify = None

    if "classification" in task_type:
        class_counts = y.value_counts()
        n_classes = len(class_counts)

        if class_counts.min() < 2:
            raise ValueError(
                "Each class needs at least 2 samples for stratified splitting."
            )

        n_test = ceil(len(y) * test_size)
        n_train = len(y) - n_test

        if n_test < n_classes or n_train < n_classes:
            raise ValueError(
                "Not enough samples in train/test sets to represent every class."
            )

        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target=target,
        task_type=task_type,
        test_size=float(test_size),
        random_state=random_state,
        dropped_target_rows=dropped_rows
    )
