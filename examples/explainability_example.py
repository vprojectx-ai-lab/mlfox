"""Permutation importance, perturbation explanation, and report export."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
import mlfox

X, y = make_classification(n_samples=180, n_features=6, n_informative=4,
                           n_redundant=0, random_state=14)
features = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
data = features.copy(); data["target"] = y
model = mlfox(cv=3, random_state=14, progress=False).fit(data, target="target")
print(model.feature_importance(data, target="target", n_repeats=3).head())
print(model.explain_prediction(features.iloc[[0]], features.iloc[20:40]))
model.export_report(Path("mlfox_explainability_report.html"))
model.export_report(Path("mlfox_explainability_report.json"))
