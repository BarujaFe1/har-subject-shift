"""Predeclared, CPU-only scikit-learn model families and grids."""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier


def candidates(mode: str = "full"):
    if mode not in ("quick", "full"):
        raise ValueError("mode deve ser 'quick' ou 'full'")
    v = ("variance", VarianceThreshold(threshold=0))
    scale = ("scale", StandardScaler())
    all_models = {
        "dummy": (Pipeline([("model", DummyClassifier(strategy="most_frequent"))]), {}),
        "logistic": (Pipeline([v, scale, ("model", LogisticRegression(max_iter=1500, solver="lbfgs", random_state=42))]), {"model__C": [0.1, 1, 10]}),
        "knn": (Pipeline([v, scale, ("model", KNeighborsClassifier())]), {"model__n_neighbors": [3, 7, 15]}),
        "linear_svc": (Pipeline([v, scale, ("model", LinearSVC(max_iter=8000, random_state=42))]), {"model__C": [0.1, 1, 10]}),
        "rbf_svc": (Pipeline([v, scale, ("model", SVC(kernel="rbf"))]), {"model__C": [1, 10], "model__gamma": ["scale", 0.01]}),
        "random_forest": (Pipeline([v, ("model", RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=2))]), {"model__min_samples_leaf": [1, 3]}),
    }
    if mode == "quick":
        # One deterministic config per model; still evaluates ALL six families.
        return {name: (est, {k: [vals[0]] for k, vals in grid.items()}) for name, (est, grid) in all_models.items()}
    return all_models


def fixed_logistic():
    """Fixed (untuned) diagnostic model for H1, chosen before seeing scores."""
    return Pipeline([("variance", VarianceThreshold(0)), ("scale", StandardScaler()), ("model", LogisticRegression(C=1, max_iter=1500, solver="lbfgs", random_state=42))])


def selection_pipeline(k: int | str):
    """SelectKBest is inside the Pipeline and fitted per training fold."""
    return Pipeline([("variance", VarianceThreshold(0)), ("scale", StandardScaler()), ("select", SelectKBest(score_func=f_classif, k=k)), ("model", LogisticRegression(C=1, solver="lbfgs", max_iter=1500, random_state=42))])
