"""Validación por sujeto, inferencia por permutación, IC por bootstrap y control de fuga.

Unidad de decisión: el registro (sujeto y sesión). El clasificador ve épocas; la
probabilidad del registro es el promedio de las de sus épocas, y se clasifica como
positivo si es >= 0.5. Toda partición, permutación y remuestreo respeta al sujeto.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from tesis_eeg_parkinson.classification.data import Dataset
from tesis_eeg_parkinson.classification.models import INNER_SPLITS, build_model, fit_model

OUTER_SPLITS = 5


def permute_labels(ds: Dataset, rng: np.random.Generator) -> np.ndarray:
    """Etiquetas permutadas respetando la unidad intercambiable.

    Contrastes entre grupos: se permutan las etiquetas entre sujetos (todas las
    épocas de un sujeto conservan una sola etiqueta). Contraste pareado: dentro de
    cada paciente se intercambian on y off con probabilidad 1/2.
    """
    subjects = np.unique(ds.subject)
    if ds.paired:
        flip = dict(zip(subjects, rng.random(subjects.size) < 0.5, strict=True))
        mask = np.array([flip[s] for s in ds.subject])
        return np.where(mask, 1 - ds.y, ds.y)
    labels = np.array([ds.y[ds.subject == s][0] for s in subjects])
    mapping = dict(zip(subjects, rng.permutation(labels), strict=True))
    return np.array([mapping[s] for s in ds.subject])


def oof_probabilities(
    ds: Dataset, y: np.ndarray, model_name: str, repeats: int, seed: int, scheme: str
) -> np.ndarray:
    """Probabilidades fuera de pliegue por época, shape (repeats, n_epochs).

    ``scheme="subject"``: pliegues estratificados por sujeto (la evaluación de la
    tesis). ``scheme="epoch"``: pliegues por época, que mezclan épocas de un mismo
    sujeto entre entrenamiento y prueba; es el control de fuga y no mide desempeño.
    """
    out = np.full((repeats, y.size), np.nan)
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        if scheme == "subject":
            cv = StratifiedGroupKFold(OUTER_SPLITS, shuffle=True, random_state=seed + r)
            splits = cv.split(ds.X, y, ds.subject)
        else:
            splits = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed + r).split(
                ds.X, y
            )
        for train, test in splits:
            model = build_model(model_name, seed + r)
            if scheme == "subject":
                groups = ds.subject[train]
            else:  # búsqueda interna también por épocas, como en un análisis ingenuo
                groups = rng.integers(0, INNER_SPLITS, train.size)
            fit_model(model, ds.X[train], y[train], groups)
            out[r, test] = model.predict_proba(ds.X[test])[:, 1]
    return out


def recording_table(ds: Dataset, y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Probabilidad por registro y repetición (promedio de sus épocas)."""
    frame = pd.DataFrame(proba.T, columns=[f"rep{r}" for r in range(proba.shape[0])])
    frame["recording"], frame["subject"], frame["y"] = ds.recording, ds.subject, y
    reps = [c for c in frame.columns if c.startswith("rep")]
    agg = frame.groupby("recording").agg(
        {"subject": "first", "y": "first", **dict.fromkeys(reps, "mean")}
    )
    agg["proba"] = agg[reps].mean(axis=1)
    return agg


def balanced_accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    """Exactitud balanceada con umbral 0.5."""
    return float(balanced_accuracy_score(y, (proba >= 0.5).astype(int)))


def bootstrap_ci(
    table: pd.DataFrame, n_boot: int, seed: int, level: float = 0.95
) -> dict[str, tuple[float, float]]:
    """IC por percentiles, remuestreando sujetos (con sus dos sesiones si es pareado)."""
    rng = np.random.default_rng(seed)
    by_subject = {s: g for s, g in table.groupby("subject")}
    subjects = np.array(list(by_subject))
    stats: dict[str, list[float]] = {"bacc": [], "auc": []}
    for _ in range(n_boot):
        sample = pd.concat([by_subject[s] for s in rng.choice(subjects, subjects.size)])
        if sample["y"].nunique() < 2:
            continue
        stats["bacc"].append(balanced_accuracy(sample["y"].to_numpy(), sample["proba"].to_numpy()))
        stats["auc"].append(float(roc_auc_score(sample["y"], sample["proba"])))
    tail = (1 - level) / 2 * 100
    return {k: tuple(np.percentile(v, [tail, 100 - tail])) for k, v in stats.items()}


def haufe_pattern(ds: Dataset, model_name: str, seed: int) -> pd.Series | None:
    """Patrón de activación A = Cov(Z)·w del modelo lineal ajustado con todos los datos.

    Los pesos de un clasificador lineal no se interpretan como actividad: el patrón
    sí (Haufe et al., 2014). Z son las features ya imputadas y estandarizadas.
    """
    if model_name == "rf":
        return None
    model = fit_model(build_model(model_name, seed), ds.X, ds.y, ds.subject)
    pipe = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    z = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(ds.X))
    w = pipe.named_steps["clf"].coef_.ravel()
    return pd.Series(np.cov(z, rowvar=False) @ w, index=ds.feature_names)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Corrección de Holm-Bonferroni."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, adjusted = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def _null_statistic(ds: Dataset, model_name: str, repeats: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    y_perm = permute_labels(ds, rng)
    proba = oof_probabilities(ds, y_perm, model_name, repeats, seed, "subject")
    table = recording_table(ds, y_perm, proba)
    return balanced_accuracy(table["y"].to_numpy(), table["proba"].to_numpy())


def run_contrast(
    ds: Dataset,
    model_name: str,
    repeats: int,
    n_perm: int,
    null_repeats: int,
    seed: int,
    n_jobs: int = 1,
    n_boot: int = 2000,
    leakage_control: bool = True,
    on_perm: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evalúa un contraste completo según el prerregistro.

    Returns
    -------
    dict
        ``summary`` (métricas), ``recordings`` (probabilidades por registro),
        ``null`` (distribución nula) y ``pattern`` (patrón de Haufe o None).
    """
    proba = oof_probabilities(ds, ds.y, model_name, repeats, seed, "subject")
    table = recording_table(ds, ds.y, proba)
    y_rec, p_rec = table["y"].to_numpy(), table["proba"].to_numpy()
    reps = [c for c in table.columns if c.startswith("rep")]
    per_repeat = [balanced_accuracy(y_rec, table[c].to_numpy()) for c in reps]
    epoch_bacc = np.mean([balanced_accuracy(ds.y, p) for p in proba])
    ci = bootstrap_ci(table, n_boot, seed)
    summary: dict[str, Any] = {
        "model": model_name,
        "n_subjects": len(np.unique(ds.subject)),
        "n_recordings": len(table),
        "n_positive": int(y_rec.sum()),
        "n_epochs": ds.y.size,
        "n_features": ds.X.shape[1],
        "bacc": balanced_accuracy(y_rec, p_rec),
        "bacc_ci_low": ci["bacc"][0],
        "bacc_ci_high": ci["bacc"][1],
        "auc": float(roc_auc_score(y_rec, p_rec)),
        "auc_ci_low": ci["auc"][0],
        "auc_ci_high": ci["auc"][1],
        "bacc_repeat_mean": float(np.mean(per_repeat)),
        "bacc_repeat_sd": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0,
        "epoch_bacc_subject_cv": float(epoch_bacc),
    }
    if leakage_control:
        leak = oof_probabilities(ds, ds.y, model_name, repeats, seed, "epoch")
        leak_table = recording_table(ds, ds.y, leak)
        summary["epoch_bacc_epoch_cv"] = float(np.mean([balanced_accuracy(ds.y, p) for p in leak]))
        summary["bacc_epoch_cv"] = balanced_accuracy(
            leak_table["y"].to_numpy(), leak_table["proba"].to_numpy()
        )
    null = np.array([])
    if n_perm > 0:
        tasks = (delayed(_null_statistic)(ds, model_name, null_repeats, seed + 10_000 + k)
                 for k in range(n_perm))  # fmt: skip
        values = []
        for done, value in enumerate(
            Parallel(n_jobs=n_jobs, return_as="generator_unordered")(tasks)
        ):
            values.append(value)
            if on_perm:
                on_perm(done + 1, n_perm)
        null = np.array(values)
        summary["p_value"] = float((1 + np.sum(null >= summary["bacc"])) / (1 + n_perm))
        summary["chance_mean"] = float(null.mean())
        summary["chance_p95"] = float(np.percentile(null, 95))
    return {
        "summary": summary,
        "recordings": table[["subject", "y", "proba"]],
        "null": null,
        "pattern": haufe_pattern(ds, model_name, seed),
    }
