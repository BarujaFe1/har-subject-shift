"""Pre-registered experiment: nested grouped CV, diagnostics, final one-time holdout."""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from .data import load_split, require_audit, atomic_json, sha256, LABELS, IDS
from .metrics import score, subject_scores, family_errors, cluster_bootstrap
from .models import candidates, fixed_logistic, selection_pipeline

SEED = 42
B = 2000


def fold_invariant(groups: np.ndarray, fit: np.ndarray, val: np.ndarray) -> bool:
    if len(np.intersect1d(fit, val)) > 0 or len(set(groups[fit]) & set(groups[val])) > 0:
        raise AssertionError("Vazamento: mesmo sujeito entre ajuste e validação")
    return True


def scorer(y, p):
    return f1_score(y, p, labels=list(IDS), average="macro", zero_division=0)


def outer_group_predictions(estimator, x, y, groups, *, grid=None, outer_splits=5, inner_splits=3):
    """Fit transforms inside each fold, optionally select hyperparameters via grouped inner CV."""
    if len(np.unique(groups)) < outer_splits or len(np.unique(groups)) < inner_splits + 1:
        raise ValueError("Número insuficiente de participantes")
    y_pred = np.full(len(y), -1, dtype=np.int16)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    fits = []
    folds_manifest = []
    outer = GroupKFold(n_splits=outer_splits)
    for fold, (tr, val) in enumerate(outer.split(x, y, groups), 1):
        fold_invariant(groups, tr, val)
        start = time.perf_counter()
        if grid:
            inner = GroupKFold(n_splits=inner_splits)
            # Prevalidate all inner splits in the coordinates of this outer training set.
            for i_tr, i_val in inner.split(x[tr], y[tr], groups[tr]):
                fold_invariant(groups[tr], i_tr, i_val)
            gs = GridSearchCV(clone(estimator), grid, cv=inner, scoring=make_scorer(scorer), refit=True, n_jobs=1, error_score="raise")
            gs.fit(x[tr], y[tr], groups=groups[tr])
            fit = gs.best_estimator_
            params = gs.best_params_
        else:
            fit = clone(estimator)
            fit.fit(x[tr], y[tr])
            params = {}
        y_pred[val] = fit.predict(x[val])
        fold_ids[val] = fold
        secs = time.perf_counter() - start
        m = score(y[val], y_pred[val])
        fits.append({"fold": fold, "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"], "n_subjects": int(len(np.unique(groups[val]))), "n_windows": len(val), "fit_and_predict_seconds": secs, "params": params})
        folds_manifest.append({"fold": fold, "train_subjects": sorted(set(map(int, groups[tr]))), "validation_subjects": sorted(set(map(int, groups[val])))})
    if np.any(y_pred == -1) or np.any(fold_ids == -1):
        raise AssertionError("OOF incompleto")
    return y_pred, fold_ids, fits, folds_manifest


def illustrate_random_windows(x, y, groups):
    """H1: fixed untuned logistic. Both CV arms use identical official TRAIN windows."""
    fixed = fixed_logistic()
    group_p, group_folds, group_scores, group_list = outer_group_predictions(fixed, x, y, groups)
    naive_p = np.full(len(y), -1, dtype=np.int16)
    for tr, val in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y):
        fit = clone(fixed).fit(x[tr], y[tr])
        naive_p[val] = fit.predict(x[val])
    if np.any(naive_p == -1):
        raise AssertionError("Predições ausentes em split por janela")
    correct, incorrect = score(y, group_p), score(y, naive_p)
    return {"n_windows": len(y), "n_subjects": len(np.unique(groups)), "fixed_model": "logistic C=1 (no tuning)", "correct_grouped_macro_f1": correct["macro_f1"], "incorrect_random_window_macro_f1": incorrect["macro_f1"], "delta_f1_percentage_points": 100 * (incorrect["macro_f1"] - correct["macro_f1"]), "correct_grouped_ba": correct["balanced_accuracy"], "incorrect_random_window_ba": incorrect["balanced_accuracy"], "delta_ba_percentage_points": 100*(incorrect["balanced_accuracy"]-correct["balanced_accuracy"]), "interpretation": "Protocol contrast within official training set, not a causal leakage estimate"}, pd.DataFrame({"row_id": np.arange(len(y)), "subject_id": groups, "y_true": y, "group_y_pred": group_p, "naive_y_pred": naive_p})


def feature_ablation(x, y, groups, folds=5):
    """Exploratory H2, never a selector of the final model."""
    records, selections = [], {}
    for k in (64, 128, 256, "all"):
        pred = np.full(len(y), -1, dtype=np.int16)
        selected_sets = []
        for train, valid in GroupKFold(folds).split(x, y, groups):
            fold_invariant(groups, train, valid)
            pipe = selection_pipeline(k)
            start = time.perf_counter()
            pipe.fit(x[train], y[train])
            secs = time.perf_counter() - start
            pred[valid] = pipe.predict(x[valid])
            # Mapping to original feature positions through fitted VarianceThreshold.
            retained = pipe.named_steps["variance"].get_support(indices=True)
            local = pipe.named_steps["select"].get_support(indices=True)
            selected_sets.append(set(map(int, retained[local])))
            records.append({"k": str(k), "fold": int(len(selected_sets)), "fit_seconds": secs, "n_selected": int(len(selected_sets[-1]))})
        js = [len(a & b) / len(a | b) if a | b else 1 for a, b in itertools.combinations(selected_sets, 2)]
        m = score(y, pred)
        fold_time = [r["fit_seconds"] for r in records if r["k"] == str(k) and isinstance(r["fold"], int)]
        records.append({"k": str(k), "fold": "OOF", "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"], "jaccard_mean": float(np.mean(js)), "mean_fit_seconds": float(np.mean(fold_time)), "n_selected": len(selected_sets[0])})
        selections[str(k)] = [sorted(s) for s in selected_sets]
    return pd.DataFrame(records), selections


def train(root: Path, mode="full", run_hypotheses=True) -> dict:
    require_audit(root)
    out = root / "artifacts"
    for path in ("test_started.json", "test_metrics.json", "protocol_locked.json"):
        if (out / path).exists():
            raise RuntimeError(f"Protocolo já congelado/testado ({path}); evite retuning pós-holdout")
    x, y, groups = load_split(root, "train")  # NO test split loaded
    start = time.perf_counter()
    all_preds, fold_rows, metric_rows, subject_rows, runtime_rows = [], [], [], [], []
    configs = candidates(mode)
    for name, (est, grid) in configs.items():
        t0 = time.perf_counter()
        p, folds, fold_metrics, fold_manifest = outer_group_predictions(est, x, y, groups, grid=grid)
        for row in fold_manifest:
            fold_rows.append({"model": name, **row})
        current = score(y, p)
        sub = subject_scores(y, p, groups)
        sub["model"] = name
        subject_rows.append(sub)
        elapsed = time.perf_counter() - t0
        metric_rows.append({"model": name, "macro_f1": current["macro_f1"], "balanced_accuracy": current["balanced_accuracy"], "mean_subject_f1": float(sub.macro_f1.mean()), "sd_subject_f1": float(sub.macro_f1.std()), "nested_outer_folds": len(fold_metrics), "mode": mode, "train_only": True})
        runtime_rows.append({"model": name, "total_nested_cv_seconds": elapsed, "fold_fit_seconds_total": sum(f["fit_and_predict_seconds"] for f in fold_metrics)})
        all_preds.append(pd.DataFrame({"model": name, "row_id": [f"train_{i+1:06d}" for i in range(len(x))], "subject_id": groups, "fold_id": folds, "y_true": y, "y_pred": p}))
        atomic_json(out / f"{name}_fold_metrics.json", fold_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True)
    # Validate exactly one OOF prediction per training row *per model*.
    if preds_df.duplicated(["model", "row_id"]).any() or preds_df.groupby("model").size().ne(len(x)).any():
        raise AssertionError("OOF duplicated or missing")
    metrics_df = pd.DataFrame(metric_rows).sort_values(["macro_f1", "sd_subject_f1", "model"], ascending=[False, True, True]).reset_index(drop=True)
    chosen = str(metrics_df.iloc[0].model)
    winner_predictions = preds_df.loc[preds_df.model == chosen].sort_values("row_id")
    if not np.array_equal(winner_predictions.y_true.to_numpy(), y):
        raise AssertionError("Previsões OOF do vencedor desalinhadas")
    oof_ci = cluster_bootstrap(y, winner_predictions.y_pred.to_numpy(), groups, b=B, seed=SEED)
    atomic_json(out / "cv_bootstrap_by_subject.json", oof_ci)
    base_est, grid = configs[chosen]
    if grid:
        fit = GridSearchCV(clone(base_est), grid, cv=GroupKFold(3), scoring=make_scorer(scorer), n_jobs=1, refit=True, error_score="raise")
        fit.fit(x, y, groups=groups)
        final_params = fit.best_params_
    else:
        final_params = {}
    # Lock the selection choice. The held-out test remains completely unopened.
    if run_hypotheses:
        h1, h1_preds = illustrate_random_windows(x, y, groups)
        atomic_json(out / "h1_protocol_contrast.json", h1)
        h1_preds.to_csv(out / "h1_predictions.csv", index=False)
        h2, selected = feature_ablation(x, y, groups)
        h2.to_csv(out / "h2_feature_ablation.csv", index=False)
        atomic_json(out / "h2_selected_indices.json", selected)
    preds_df.to_csv(out / "oof_predictions.csv", index=False)
    metrics_df.to_csv(out / "cv_metrics.csv", index=False)
    pd.concat(subject_rows).to_csv(out / "subject_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out / "runtime.csv", index=False)
    atomic_json(out / "fold_subjects.json", fold_rows)
    selection = {"model": chosen, "params": final_params, "mode": mode, "selection_rule": "max nested-CV OOF macro-F1; tie lower per-subject SD, then alphabetical", "selection_is_posthoc_across_families": True, "test_touched": False, "scores": metrics_df.to_dict(orient="records")}
    atomic_json(out / "model_selection.json", selection)
    manifest = json.loads((out / "data_manifest.json").read_text())
    locked = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "primary_dataset_sha256": manifest["sha256"], "train_y_subjects_only_until_now": True, "primary_metric": "six-class macro-F1", "validation": "nested GroupKFold outer 5 inner 3", "illustrative_only": "StratifiedKFold 5 for H1, never model selection", "subject_aware_bootstrap_B": B, "seed": SEED, "selected_model": chosen, "params": final_params, "mode": mode, "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "elapsed_train_seconds": time.perf_counter()-start, "code_sha256": {f.name: sha256(f) for f in sorted(Path(__file__).parent.glob("*.py"))}, "test_status": "NEVER_OPENED_FOR_MODELING"}
    atomic_json(out / "protocol_locked.json", locked)
    return locked


def final_test(root: Path) -> dict:
    """Irreversible one-shot evaluation: lock must exist; never use test for selection."""
    require_audit(root)
    out = root / "artifacts"
    lock_path = out / "protocol_locked.json"
    if not lock_path.exists():
        raise RuntimeError("Falta protocol_locked.json: execute train antes do teste")
    if (out / "test_started.json").exists() or (out / "test_metrics.json").exists():
        raise RuntimeError("Teste oficial já iniciou ou terminou; repetição bloqueada")
    locked = json.loads(lock_path.read_text(encoding="utf8"))
    manifest = json.loads((out / "data_manifest.json").read_text(encoding="utf8"))
    if locked["primary_dataset_sha256"] != manifest["sha256"]:
        raise RuntimeError("Hash mudou após lock")
    for name, digest in locked["code_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise RuntimeError(f"Código alterado após lock: {name}")
    selection = json.loads((out / "model_selection.json").read_text(encoding="utf8"))
    model, _ = candidates(selection["mode"])[selection["model"]]
    model = clone(model).set_params(**selection["params"])
    x_train, y_train, s_train = load_split(root, "train")
    # Dedicated marker before opening test. A crash requires manual audit, NOT silent rerun.
    atomic_json(out / "test_started.json", {"utc": datetime.now(timezone.utc).isoformat(), "model": selection["model"], "final_test_executions_started": 1})
    x_test, y_test, s_test = load_split(root, "test")
    if set(s_train) & set(s_test):
        raise AssertionError("Holdout subject leakage")
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    secs = time.perf_counter() - t0
    pred = model.predict(x_test)
    # Inference latency: 30 repetitions after warm-up, on identical CPU/hardware.
    timings = {}
    for size in (1, 100, 1000):
        batch = x_test[:size]
        model.predict(batch)
        measurements = []
        for _ in range(30):
            t1 = time.perf_counter()
            model.predict(batch)
            measurements.append((time.perf_counter() - t1) * 1000)
        timings[str(size)] = {"median_ms_batch": float(np.median(measurements)), "p95_ms_batch": float(np.percentile(measurements, 95)), "repetitions": 30}
    metric = score(y_test, pred)
    subject_df = subject_scores(y_test, pred, s_test)
    ci = cluster_bootstrap(y_test, pred, s_test, b=B)
    metric.update({"model": selection["model"], "params": selection["params"], "n_subjects": len(np.unique(s_test)), "mean_subject_macro_f1": float(subject_df.macro_f1.mean()), "family_errors": family_errors(y_test, pred), "bootstrap_subjects": ci, "fit_seconds": secs, "predict_latency": timings, "process_rss_mib_at_end": psutil.Process().memory_info().rss / 2**20, "note": "ONE final held-out test after protocol lock; no retuning"})
    pd.DataFrame({"row_id": [f"test_{i+1:06d}" for i in range(len(x_test))], "subject_id": s_test, "y_true": y_test, "y_pred": pred}).to_csv(out / "test_predictions.csv", index=False)
    subject_df.to_csv(out / "test_subject_metrics.csv", index=False)
    joblib.dump(model, out / "best_pipeline.joblib", compress=3)
    metric["model_size_mib"] = (out / "best_pipeline.joblib").stat().st_size / 2**20
    atomic_json(out / "test_metrics.json", metric)
    return metric
