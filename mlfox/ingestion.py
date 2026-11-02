
from pathlib import Path

import pandas as pd


def load_dataset(data):
    """
    Load a CSV, Excel file, or Pandas DataFrame.
    Returns a copy so the original data is never modified.
    """

    if isinstance(data, pd.DataFrame):
        df = data.copy(deep=True)

    elif isinstance(data, (str, Path)):
        path = Path(data)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)

        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        else:
            raise ValueError(
                "Supported formats: CSV, XLSX, XLS"
            )

    else:
        raise TypeError(
            "Input must be a file path or Pandas DataFrame."
        )

    # Basic validation
    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")

    # Detect duplicate column names
    duplicate_names = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicate_names:
        raise ValueError(
            f"Duplicate column names: {duplicate_names}"
        )

    return df
