# Changelog

Release notes for MLFOX / the `mlfox` Python distribution.

## 1.0.1 — 2026-11-02

- Made the `mlfox` distribution and `from mlfox import mlfox` the recommended installation and public import.
- Added bounded tuning, model evaluation, prediction, persistence, explainability, and report export to the release workflow.
- Added CPU XGBoost dependencies selected by platform: `xgboost-cpu` on Windows/Linux and `xgboost` on macOS.
- Added CI coverage for tests, integration, package building, and clean-wheel smoke tests.

## 1.0.0

- Initial MLFOX v1.0 release with tabular classification and regression workflows, fold-aware preprocessing, model selection, reporting, and holdout evaluation.
