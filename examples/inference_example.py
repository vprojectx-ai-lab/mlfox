"""Persist a fitted model and load it for inference."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.datasets import make_classification
import mlfox

X, y = make_classification(n_samples=160, n_features=6, n_informative=4,
                           n_redundant=0, random_state=13)
data = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
data["target"] = y
model = mlfox(cv=3, random_state=13, progress=False).fit(data, target="target")
with TemporaryDirectory(prefix="mlfox-inference-", dir=Path.cwd()) as directory:
    artifact = model.save(Path(directory) / "model.joblib")
    loaded = mlfox.load_model(artifact)
    print(loaded.predict(data.drop(columns="target").head()))
