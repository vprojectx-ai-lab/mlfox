# MLFOX

[![PyPI version](https://img.shields.io/pypi/v/mlfox)](https://pypi.org/project/mlfox/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlfox)](https://pypi.org/project/mlfox/)
[![CI](https://github.com/vprojectx-ai-lab/mlfox/actions/workflows/ci.yml/badge.svg)](https://github.com/vprojectx-ai-lab/mlfox/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Automated tabular machine learning on CPU. Give it a CSV, get a tested model and a report.**

MLFOX takes a CSV, Excel file, or pandas DataFrame and handles the rest: it profiles the data, builds leakage-safe preprocessing, compares candidate models with cross-validation, evaluates the winner once on an untouched holdout set, and writes a report.

Built by [vprojectx](https://github.com/vprojectx-ai-lab) · Source: [github.com/vprojectx-ai-lab/mlfox](https://github.com/vprojectx-ai-lab/mlfox)

## Highlights

- **One-line workflow:** `mlfox().fit("data.csv", target="y")`
- **Classification and regression**, with automatic task detection
- **Fold-aware preprocessing:** imputation, encoding, scaling, feature engineering, and feature selection are learned only from each fold's training rows, so no information leaks from validation data
- **Honest evaluation:** the holdout set is never used to pick the model
- **Reports:** a short console summary, a full self-contained HTML report, and JSON export
- **Explainability:** permutation importance and single-row perturbation explanations
- **Runs on CPU only:** no GPU setup needed

## Installation

```bash
python -m pip install mlfox
```

Requires **Python 3.11 or later** (CI covers 3.11–3.13).

Optional extras:

```bash
python -m pip install "mlfox[rich]"   # Rich progress bars and report formatting
python -m pip install "mlfox[excel]"  # .xlsx / .xls input support
python -m pip install "mlfox[test]"   # pytest
```

To install from a local checkout:

```bash
git clone https://github.com/vprojectx-ai-lab/mlfox.git
cd mlfox
python -m pip install -e .
```

> XGBoost is a required dependency. Windows and Linux install `xgboost-cpu`; macOS installs `xgboost` configured for CPU execution.

## Quick start

The package itself is callable, so use `import mlfox` and then `mlfox(...)`.

```python
import mlfox

model = mlfox()
model.fit("customers.csv", target="Churn")
model.report()
```

`target` is the column you want to predict. To control the run, pass options:

```python
model = mlfox(task="auto", cv=5, random_state=42)
model.fit("customers.csv", target="Churn")
model.report()
```

## Examples

### Classification

```python
import pandas as pd
import mlfox

data = pd.read_csv("customers.csv")
X = data.drop(columns="Churn")

model = mlfox(task="binary_classification", cv=5, random_state=42)
model.fit(data, target="Churn")

predictions = model.predict(X)
print(predictions.head())

# Available when the selected classifier supports probabilities
probabilities = model.predict_proba(X)
print(probabilities.head())
```

For three or more classes, use `task="multiclass_classification"` when you know the target type.

### Regression

```python
import pandas as pd
import mlfox

data = pd.read_csv("sales.csv")

model = mlfox(task="regression", metric="rmse", cv=5, random_state=42)
model.fit(data, target="Sales")

predictions = model.predict(data.drop(columns="Sales"))
print(predictions.head())
```

More runnable scripts are in [`examples/`](examples/).

## How `fit()` works

1. **Load and profile:** validates the input, profiles columns, and detects or checks the task.
2. **Split:** divides the data into training and holdout partitions.
3. **Benchmark:** cross-validates every eligible candidate model on the training partition.
4. **Preprocess per fold:** column decisions, imputation, feature engineering, encoding/scaling, and feature selection are fit only on each fold's training rows.
5. **Select and evaluate:** picks the best candidate by the primary CV metric, refits it on the full training partition, and evaluates it once on the holdout.

Classification uses stratified CV and defaults to balanced accuracy. Regression uses K-fold CV and defaults to RMSE. Configure the run with `cv`, `metric`, `test_size`, and `random_state`.

## Models

| Classification | Regression |
| --- | --- |
| Logistic Regression | Linear Regression |
| Decision Tree | Ridge |
| Random Forest | ElasticNet |
| Extra Trees | Decision Tree Regressor |
| XGBoost | Random Forest Regressor |
| HistGradientBoosting (when eligible) | Extra Trees Regressor |
| Calibrated SVM (within size limits) | XGBoost Regressor |
| | HistGradientBoosting Regressor (when eligible) |

Models run sequentially on CPU. Some candidates are skipped when a dataset exceeds their size or dimensionality limits. Any candidate failure (with the exception and fold) is recorded in the HTML report.

## Metrics

| Task | Reported | Primary metric options |
| --- | --- | --- |
| Classification | accuracy, balanced accuracy, weighted precision, weighted recall, weighted F1, ROC-AUC (when supported) | `balanced_accuracy`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc` |
| Regression | R², MAE, RMSE | `rmse`, `mae`, `r2` |

## Benchmarks

Produced by the repository's [benchmark runner](https://github.com/vprojectx-ai-lab/mlfox/blob/main/test2.py) using an external holdout. Fit time is the time spent in `model.fit()`. Runs used default five-fold CV and three seeds, except full Covertype, which used one.

| Dataset | Rows | Accuracy | Macro F1 | R² | RMSE | Mean fit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | 48,842 | 0.8732 | 0.8149 | — | — | 132.6 s |
| Credit-g | 1,000 | 0.7667 | 0.6995 | — | — | 8.5 s |
| Covertype (full) | 581,012 | 0.9582 | 0.9309 | — | — | 1,079.2 s |
| California Housing | 20,640 | — | — | 0.8291 | 0.4751 | 62.0 s |
| Diamonds | 53,940 | — | — | 0.9762 | 612.62 | 195.8 s |

These are example runs, not guarantees. Hardware, dependency versions, random splits, and dataset characteristics all affect results, and Credit-g varied more across seeds than the others. Because every candidate is cross-validated, expect training to take from seconds on small data to several minutes (or longer) on large data.

## Bounded tuning

Tuning is optional and capped. It is not an open-ended search.

```python
import mlfox

model = mlfox(
    task="binary_classification",
    tune=True,
    max_tuning_trials=8,
    cv=3,
    random_state=42,
)
model.fit("customers.csv", target="Churn")
model.report()
```

## Reports

| Method | Output |
| --- | --- |
| `model.report()` | Concise console summary: benchmark table, best model, holdout metrics |
| `model.html()` | Writes the full self-contained report to `mlfox_report.html` |
| `model.export_report("training_report.html")` | Full report to a path you choose |
| `model.export_report("training_report.json")` | Machine-readable JSON report |

The full HTML report covers the dataset, preprocessing, model comparison (including failures), holdout results, explainability, and runtime.

## Saving and loading

```python
import mlfox

model = mlfox().fit("training.csv", target="target")
model.save("model.joblib")

loaded = mlfox.load_model("model.joblib")
predictions = loaded.predict("new_rows.csv")
```

`fit()` returns the same model object, so calls can be chained.

- `predict()` returns a one-dimensional pandas Series named `prediction`, with the input index preserved:

  ```text
  0    yes
  1     no
  Name: prediction, dtype: object
  ```

- `predict_proba()` returns a DataFrame of shape `(n_rows, n_classes)` when the selected classifier supports probabilities.

> **Security:** Joblib uses pickle-based deserialization, which can execute code embedded in a file. Only load model files from sources you trust.

## Explainability

```python
importance = model.feature_importance(
    data="labeled_evaluation.csv", target="target", n_repeats=10
)

explanation = model.explain_prediction(
    data="one_customer.csv", background_data="reference_rows.csv"
)
```

- **Permutation importance:** global feature importance.
- **Perturbation explanation:** explains a single row against background data.

These describe model sensitivity, not causal effects. SHAP is not currently supported.

## Task detection

With `task="auto"`, MLFOX treats an integer-valued numeric target as multiclass when it has at least 3 distinct labels, at least 2 rows per label, and no more labels than `min(100, max(20, int(2 * sqrt(n_valid_targets))))`. Anything else numeric is treated as regression.

This is a heuristic. If integer values actually represent a continuous or count-like target, set `task` explicitly to `binary_classification`, `multiclass_classification`, or `regression`.

## Limitations

- Tabular data only; free-form text is not vectorized automatically.
- Random/stratified splits only. Time-aware and group-aware holdout or CV are not supported.
- Duplicate rows are reported but not removed. Outliers are not clipped or removed.
- High-cardinality categorical features can use a lot of memory during one-hot encoding.
- No GPU training and no parallel model benchmarking.

## Development

```bash
git clone https://github.com/vprojectx-ai-lab/mlfox.git
cd mlfox
python -m pip install -e ".[test,rich,excel]"
python -m compileall -q mlfox tests
python -m pytest -vv
python -m mlfox.test
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

Released under the [MIT License](LICENSE).

---

Made by **vprojectx**