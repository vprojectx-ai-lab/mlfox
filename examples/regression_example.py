"""Regression with the public MLFOX API."""
import pandas as pd
from sklearn.datasets import make_regression
import mlfox

X, y = make_regression(n_samples=180, n_features=6, noise=5, random_state=11)
data = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
data["target"] = y
model = mlfox(task="regression", metric="rmse", cv=3, random_state=11, progress=False)
model.fit(data, target="target")
model.report()
