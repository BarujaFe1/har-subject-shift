"""Subject-aware evaluation: windows are predictions, people are clusters."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from .data import IDS, LABELS

STATIC = {4, 5, 6}
DYNAMIC = {1, 2, 3}


def score(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y e p devem estar alinhados e não vazios")
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=list(IDS), zero_division=0)
    # Balanced accuracy on fixed six labels: average of six recalls.
    # scikit-learn's BA ignores absent true classes, while we explicitly fix six.
    ba = float(np.mean(recall))
    return {"macro_f1": float(np.mean(f1)), "balanced_accuracy": ba, "n_windows": int(len(y)), "per_class": {LABELS[k]: {"precision": float(precision[k-1]), "recall": float(recall[k-1]), "f1": float(f1[k-1]), "support": int(support[k-1])} for k in IDS}, "confusion": confusion_matrix(y, p, labels=list(IDS)).tolist(), "absent_true_classes": [LABELS[k] for k in IDS if not np.any(y == k)]}


def subject_scores(y, p, subjects) -> pd.DataFrame:
    records = []
    for s in sorted(np.unique(subjects)):
        mask = np.asarray(subjects) == s
        m = score(np.asarray(y)[mask], np.asarray(p)[mask])
        records.append({"subject_id": int(s), "n_windows": m["n_windows"], "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"], "n_present_classes": 6 - len(m["absent_true_classes"])})
    return pd.DataFrame.from_records(records)


def family_errors(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    result = {}
    for group_name, cls in (("static", STATIC), ("dynamic", DYNAMIC)):
        mask = np.isin(y, list(cls))
        n = int(mask.sum())
        wrong = mask & (y != p)
        crossing = wrong & ~np.isin(p, list(cls))
        result[group_name] = {"n_windows": n, "error_rate": float(wrong.sum()/n) if n else None, "within_family_errors": int(wrong.sum()-crossing.sum()), "cross_family_errors": int(crossing.sum())}
    return result


def cluster_bootstrap(y, p, subjects, b: int = 2000, seed: int = 42) -> dict:
    """Bootstrap unique subject IDs, all windows per sampled ID, multiplicities retained."""
    y, p, subjects = map(np.asarray, (y, p, subjects))
    ids = np.unique(subjects)
    if len(ids) < 2:
        raise ValueError("Bootstrap exige ao menos dois participantes")
    subject_indices = {s: np.where(subjects == s)[0] for s in ids}
    individual = subject_scores(y, p, subjects).set_index("subject_id")["macro_f1"].to_dict()
    rng = np.random.default_rng(seed)
    pool, mean_pool, missing = [], [], 0
    for _ in range(b):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([subject_indices[s] for s in draw])
        m = score(y[idx], p[idx])
        if m["absent_true_classes"]:
            missing += 1
        pool.append(m["macro_f1"])
        mean_pool.append(float(np.mean([individual[int(s)] for s in draw])))
    lo, hi = np.percentile(pool, [2.5, 97.5])
    ml, mh = np.percentile(mean_pool, [2.5, 97.5])
    return {"n_subjects": int(len(ids)), "B": b, "seed": seed, "global_macro_f1_ci_95": [float(lo), float(hi)], "subject_mean_macro_f1_ci_95": [float(ml), float(mh)], "replicates_with_absent_classes": missing, "global_macro_f1": float(score(y, p)["macro_f1"]), "subject_mean_macro_f1": float(np.mean(list(individual.values()))), "interpretation": "descriptive conditional on fitted model; sampled subjects are few"}


def paired_delta_bootstrap(y, p1, p2, subjects, b=2000, seed=42):
    y, p1, p2, subjects = map(np.asarray, (y, p1, p2, subjects))
    ids = np.unique(subjects)
    idx_by_subject = {s: np.where(subjects == s)[0] for s in ids}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(b):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_by_subject[s] for s in draw])
        values.append(score(y[idx], p1[idx])["macro_f1"] - score(y[idx], p2[idx])["macro_f1"])
    return {"n_subjects": int(len(ids)), "delta_macro_f1": float(score(y,p1)["macro_f1"]-score(y,p2)["macro_f1"]), "ci_95": [float(v) for v in np.percentile(values, [2.5,97.5])], "bootstrap_B": b}
