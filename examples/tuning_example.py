"""Bounded CPU-only hyperparameter tuning."""
import pandas as pd
from sklearn.datasets import make_classification
import mlfox

X, y = make_classification(n_samples=180, n_features=6, n_informative=4,
                           n_redundant=0, random_state=12)
data = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
data["target"] = y
model = mlfox(task="binary_classification", tune=True, max_tuning_trials=3,
              cv=3, random_state=12, progress=False)
model.fit(data, target="target")
model.report()
