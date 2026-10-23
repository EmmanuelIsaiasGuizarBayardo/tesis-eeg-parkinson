"""Modelos prerregistrados: uno principal y dos de sensibilidad."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Rejilla de la regularización (C = 1/λ), elegida con validación anidada por sujeto.
C_GRID = np.logspace(-4, 1, 6)
INNER_SPLITS = 3
MODELS = ("lr", "slda", "rf")


def build_model(name: str, seed: int) -> GridSearchCV | Pipeline:
    """Construye el modelo; la imputación y la escala se ajustan solo con entrenamiento.

    - ``lr``: regresión logística L2 con C elegido por validación anidada (principal).
    - ``slda``: LDA con contracción de Ledoit-Wolf; analítica, sin hiperparámetros.
    - ``rf``: bosque aleatorio con valores por defecto; sondea señal no lineal.
    """
    impute = ("impute", SimpleImputer(strategy="median"))
    scale = ("scale", StandardScaler())
    if name == "lr":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        pipe = Pipeline([impute, scale, ("clf", clf)])
        # Interna estratificada por sujeto: sin estratificar, un pliegue con una sola
        # clase deja el AUC indefinido y la búsqueda elegiría C a ciegas.
        inner = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=seed)
        return GridSearchCV(pipe, {"clf__C": C_GRID}, cv=inner, scoring="roc_auc", n_jobs=1)
    if name == "slda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        return Pipeline([impute, scale, ("clf", clf)])
    if name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", random_state=seed, n_jobs=1
        )
        return Pipeline([impute, ("clf", clf)])
    raise ValueError(f"Modelo desconocido: {name}")


def fit_model(model: GridSearchCV | Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Ajusta; la búsqueda anidada recibe los grupos para no partir sujetos."""
    if isinstance(model, GridSearchCV):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)
