"""Patrones de Haufe del análisis principal y estabilidad de su signo por bootstrap.

La estabilidad es exploratoria: el prerregistro declaró los patrones descriptivos.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tesis_eeg_parkinson.classification.data import CONTRASTS, Dataset, load_table, make_dataset
from tesis_eeg_parkinson.classification.evaluate import haufe_pattern
from tesis_eeg_parkinson.reproducibility import get_run_context

STABLE = 0.95  # fracción mínima de remuestreos con el mismo signo


def resample_subjects(ds: Dataset, rng: np.random.Generator) -> Dataset:
    """Remuestreo de sujetos con reemplazo, con todas sus épocas (y sesiones).

    Los duplicados conservan su identificador, así que en la búsqueda interna caen
    en el mismo pliegue. Se repite el sorteo si queda una sola clase.
    """
    subjects = np.unique(ds.subject)
    while True:
        chosen = rng.choice(subjects, subjects.size, replace=True)
        idx = np.concatenate([np.flatnonzero(ds.subject == s) for s in chosen])
        if np.unique(ds.y[idx]).size == 2:
            return replace(ds, X=ds.X[idx], y=ds.y[idx], subject=ds.subject[idx],
                           recording=ds.recording[idx])  # fmt: skip


def _one(ds: Dataset, seed: int) -> np.ndarray:
    pattern = haufe_pattern(resample_subjects(ds, np.random.default_rng(seed)), "lr", seed)
    return pattern.to_numpy()


def bootstrap_patterns(
    ds: Dataset,
    n_boot: int,
    seed: int,
    n_jobs: int = 1,
    on_boot: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Patrón con todos los datos y fracción de remuestreos con su mismo signo.

    Returns
    -------
    DataFrame
        Índice: feature. Columnas ``pattern`` y ``stability`` (0 a 1).
    """
    full = haufe_pattern(ds, "lr", seed)
    tasks = (delayed(_one)(ds, seed + 20_000 + b) for b in range(n_boot))
    boots = []
    for done, value in enumerate(Parallel(n_jobs=n_jobs, return_as="generator_unordered")(tasks)):
        boots.append(value)
        if on_boot:
            on_boot(done + 1, n_boot)
    same = np.sign(np.vstack(boots)) == np.sign(full.to_numpy())
    return pd.DataFrame({"pattern": full, "stability": same.mean(axis=0)})


def run_interpretation(
    features_path: Path,
    out_dir: Path,
    n_boot: int = 200,
    seed: int = 28,
    n_jobs: int = 1,
    on_event: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Patrones y estabilidad de los tres contrastes del análisis principal.

    Escribe ``patterns.tsv`` (formato largo) y ``run.json``; devuelve la tabla.
    """
    table = load_table(features_path)
    frames = []
    for contrast in CONTRASTS:
        ds = make_dataset(table, contrast)
        notify = (lambda d, n, c=contrast: on_event(c, d, n)) if on_event else None
        result = bootstrap_patterns(ds, n_boot, seed, n_jobs, notify)
        parts = result.index.to_series().str.split("__", expand=True)
        frames.append(result.assign(contrast=contrast, metric=parts[0], signal=parts[1],
                                    channel=parts[2]).reset_index(names="feature"))  # fmt: skip
    long = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "patterns.tsv", sep="\t", index=False)
    meta = {"features": str(features_path), "n_boot": n_boot, "stable": STABLE,
            "exploratorio": True, "run_context": get_run_context(seed=seed).as_dict()}  # fmt: skip
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return long
