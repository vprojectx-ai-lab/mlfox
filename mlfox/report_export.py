"""JSON and self-contained HTML serialization for MLFOX reports."""
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _native(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_native(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _native(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _native(value.to_dict())
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _native(value.to_dict())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def json_text(payload):
    return json.dumps(_native(payload), indent=2, ensure_ascii=False, allow_nan=False)


def _table(headers, rows):
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def html_document(payload):
    safe = _native(payload)
    dataset = safe.get("dataset", {})
    task = safe.get("task", {})
    selection = safe.get("model_selection", {})
    importance = safe.get("explainability", {}).get("global_feature_importance") or {}
    metric_pairs = list((safe.get("holdout_metrics") or {}).items())
    metric_cards = "".join(f"<div class='metric'><span>{html.escape(str(k))}</span><strong>{html.escape(str(v))}</strong></div>" for k, v in metric_pairs)
    candidates = selection.get("benchmark_table", [])
    successful_candidates = sum(row.get("status") == "success" for row in candidates)
    rows_count = dataset.get("rows", "Unknown")
    columns_count = dataset.get("columns", "Unknown")
    task_name = task.get("task_type", "ML task")
    target_name = task.get("target") or "the selected target"
    best_estimator = safe.get("best_estimator", "Not selected")
    overview_text = (
        f"This {task_name} report summarizes {rows_count} rows and {columns_count} columns "
        f"for predicting {target_name}. MLFOX compared {len(candidates)} candidate models; "
        f"{successful_candidates} completed successfully."
    )
    dataset_cards = "".join(
        f"<div class='stat'><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in (("Rows", rows_count), ("Columns", columns_count), ("Target", target_name))
    )
    candidate_rows = [[r.get("model", ""), r.get("primary_metric", ""), r.get("cv_mean", ""), r.get("cv_std", ""), r.get("execution_seconds", r.get("runtime_seconds", "")), r.get("status", ""), r.get("failure_reason", "") or (r.get("failure_details") or r.get("reason", ""))] for r in candidates]
    importance_rows = [[r.get("rank", ""), r.get("feature", ""), r.get("importance_mean", ""), r.get("importance_std", "")] for r in importance.get("features", [])]
    importance_values = [max(0.0, float(r.get("importance_mean", 0) or 0)) for r in importance.get("features", [])]
    importance_max = max(importance_values, default=0.0)
    importance_chart = "".join(
        f"<p>{html.escape(str(row.get('feature','')))}<div class='bar'><i style='width:{(max(0.0, float(row.get('importance_mean',0) or 0))/importance_max*100 if importance_max else 0):.2f}%'></i></div></p>"
        for row in importance.get("features", [])[:12]
    )
    feature_types = dataset.get("feature_types", {})
    warnings = dataset.get("warnings", [])
    report_json = html.escape(json.dumps(safe, ensure_ascii=False, indent=2))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLFOX Model Report</title><style>
:root{{--ink:#172033;--muted:#5b6475;--line:#dce2ec;--accent:#2457c5;--bg:#f5f7fb;--brand:#102b62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:32px auto;padding:0 18px}}.brand{{display:flex;align-items:center;gap:11px;margin:0 0 16px;color:var(--brand)}}.brand-mark{{width:38px;height:38px;display:grid;place-items:center;border-radius:11px;background:var(--brand);color:#fff;font-weight:800;font-size:21px}}.brand-name{{font-weight:750;letter-spacing:.02em}}.brand-caption{{display:block;color:var(--muted);font-size:11px;letter-spacing:.14em}}.brand-tag{{margin-left:auto;border:1px solid var(--line);background:#fff;border-radius:99px;padding:5px 11px;color:var(--muted);font-size:12px}}
header{{background:linear-gradient(120deg,#142d63,#3478d4);color:#fff;padding:30px;border-radius:16px;box-shadow:0 12px 30px #183a741f}}
h1{{margin:0 0 8px;font-size:30px}}header p{{margin:0;opacity:.9;max-width:780px}}section{{background:#fff;padding:22px;margin:18px 0;border:1px solid var(--line);border-radius:12px;box-shadow:0 4px 16px #16254a0a}}
h2{{font-size:20px;margin:0 0 14px}}h3{{font-size:16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:18px}}
.metric{{background:#fff;color:var(--ink);padding:14px;border-radius:10px;display:flex;flex-direction:column;gap:5px}}.metric span{{color:var(--muted);text-transform:capitalize}}.metric strong{{font-size:20px;overflow-wrap:anywhere}}
.dataset-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}.stat{{background:#f2f6fc;padding:14px;border:1px solid var(--line);border-radius:10px;min-width:0}}.stat span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.stat strong{{display:block;margin-top:4px;overflow-wrap:anywhere}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#f0f4fa;white-space:nowrap}}
code,pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.muted{{color:var(--muted)}}.bar{{height:9px;background:#e8edf6;border-radius:9px;min-width:100px}}.bar i{{display:block;height:100%;background:var(--accent);border-radius:9px}}
@media(max-width:600px){{main{{margin:14px auto}}section{{padding:16px}}h1{{font-size:24px}}}}
</style></head><body><main><div class="brand"><span class="brand-mark">v</span><div><span class="brand-name">vprojectx</span><small class="brand-caption">MACHINE LEARNING</small></div><span class="brand-tag">MLFOX REPORT</span></div>
<header><h1>Model performance</h1><p>{html.escape(overview_text)}</p><div class="cards">{metric_cards}</div></header>
<section><h2>Dataset overview</h2><p><b>Task:</b> {html.escape(str(task_name))} · <b>Best model:</b> {html.escape(str(best_estimator))}</p><div class="dataset-cards">{dataset_cards}</div><h3>Feature types</h3><pre>{html.escape(json.dumps(feature_types, ensure_ascii=False, indent=2))}</pre><h3>Missing values</h3><pre>{html.escape(json.dumps(dataset.get('missing_values',{}), ensure_ascii=False, indent=2))}</pre><h3>Warnings</h3><ul>{''.join('<li>'+html.escape(str(item))+'</li>' for item in warnings) or '<li>None</li>'}</ul></section>
<section><h2>Model selection</h2><p><b>Best estimator:</b> {html.escape(str(safe.get('best_estimator','')))}<br><b>CV metrics:</b> {html.escape(json.dumps(safe.get('cv_metrics'), ensure_ascii=False))}<br><b>Tuning:</b> {html.escape(json.dumps(safe.get('tuning'), ensure_ascii=False))}<br><b>Best hyperparameters:</b> {html.escape(json.dumps(safe.get('best_hyperparameters'), ensure_ascii=False))}</p>{_table(['Model','Metric','CV mean','CV std','Runtime (s)','Status','Details'], candidate_rows) if candidate_rows else '<p class="muted">No candidate comparison recorded.</p>'}</section>
<section><h2>Final holdout evaluation</h2><p>Evaluated separately after candidate selection.</p>{_table(['Metric','Value'], metric_pairs)}</section>
<section><h2>Explainability</h2><p><b>Global method:</b> {html.escape(str(importance.get('method','Not computed')))}; metric: {html.escape(str(importance.get('metric','')))}</p>{_table(['Rank','Feature','Importance mean','Std'], importance_rows) if importance_rows else '<p class="muted">Global feature importance has not been computed.</p>'}{importance_chart}<p class="muted">Permutation importance reflects predictive sensitivity. Correlated features may share or mask importance; this is not causal attribution.</p><h3>Individual perturbation explanations</h3><pre>{html.escape(json.dumps(safe.get('explainability',{}).get('individual_explanations',[]), ensure_ascii=False, indent=2))}</pre><p class="muted">Single-feature perturbation does not model feature interactions and is not SHAP.</p></section>
<section><h2>Runtime</h2><pre>{html.escape(json.dumps(safe.get('runtime_seconds',{}), ensure_ascii=False, indent=2))}</pre></section>
<details><summary>Structured report data</summary><pre>{report_json}</pre></details><footer class="muted">MLFOX · A product by vprojectx · Self-contained report</footer></main></body></html>'''


def export_report(model, path):
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".html", ".json"}:
        raise ValueError("Report export supports only .html and .json files")
    payload = model._build_report()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = html_document(payload) if suffix == ".html" else json_text(payload)
    destination.write_text(content, encoding="utf-8")
    return destination
