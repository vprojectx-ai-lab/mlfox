"""Binary classification with the public MLFOX API."""
import pandas as pd
from sklearn.datasets import make_classification
import mlfox

X, y = make_classification(n_samples=180, n_features=7, n_informative=5,
                           n_redundant=0, random_state=10)
data = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
data["target"] = y
model = mlfox(task="binary_classification", cv=3, random_state=10, progress=False)
model.fit(data, target="target")
model.report()
print(model.predict_proba(data.drop(columns="target").head()))
