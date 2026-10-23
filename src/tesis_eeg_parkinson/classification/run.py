"""Orquesta un análisis prerregistrado y escribe sus tablas."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from tesis_eeg_parkinson.classification.data import CONTRASTS, load_table, make_dataset
from tesis_eeg_parkinson.classification.evaluate import holm, run_contrast
from tesis_eeg_parkinson.reproducibility import get_run_context


def run_analysis(
    features_path: Path,
    out_dir: Path,
    model_name: str = "lr",
    contrasts: tuple[str, ...] = tuple(CONTRASTS),
    repeats: int = 20,
    n_perm: int = 1000,
    null_repeats: int = 2,
    seed: int = 28,
    n_jobs: int = 1,
    include_excluded: bool = False,
    intersect_with: Path | None = None,
    channels: tuple[str, ...] | None = None,
    leakage_control: bool = True,
    on_event: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Corre los contrastes pedidos y escribe ``summary.tsv``, ``recordings.tsv``,
    ``null.tsv``, ``patterns.tsv`` y ``run.json`` en ``out_dir``.

    Parameters
    ----------
    intersect_with : Path, optional
        Otra tabla de features: se conservan solo los sujetos presentes en ambas
        para cada contraste (comparación entre equipos sobre las mismas personas).
    on_event : callable, optional
        Recibe (contraste, permutaciones hechas, total) para la barra de progreso.
    """
    table = load_table(features_path)
    other = load_table(intersect_with) if intersect_with else None
    rows, recordings, nulls, patterns = [], [], {}, {}
    for contrast in contrasts:
        subjects = None
        if other is not None:
            mine = set(make_dataset(table, contrast, include_excluded, channels=channels).subject)
            theirs = set(make_dataset(other, contrast, include_excluded).subject)
            subjects = mine & theirs
        ds = make_dataset(table, contrast, include_excluded, subjects, channels)
        notify = (lambda d, n, c=contrast: on_event(c, d, n)) if on_event else None
        if on_event:
            on_event(contrast, 0, n_perm)
        result = run_contrast(ds, model_name, repeats, n_perm, null_repeats, seed, n_jobs,
                              leakage_control=leakage_control, on_perm=notify)  # fmt: skip
        rows.append({"contrast": contrast, **result["summary"]})
        recordings.append(result["recordings"].assign(contrast=contrast))
        nulls[contrast] = pd.Series(result["null"])
        if result["pattern"] is not None:
            patterns[contrast] = result["pattern"]
    summary = pd.DataFrame(rows)
    if "p_value" in summary:
        adjusted = holm(dict(zip(summary["contrast"], summary["p_value"], strict=True)))
        summary["p_holm"] = summary["contrast"].map(adjusted)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    pd.concat(recordings).to_csv(out_dir / "recordings.tsv", sep="\t")
    pd.DataFrame(nulls).to_csv(out_dir / "null.tsv", sep="\t", index=False)
    if patterns:
        pd.DataFrame(patterns).to_csv(out_dir / "patterns.tsv", sep="\t")
    meta: dict[str, Any] = {
        "features": str(features_path),
        "model": model_name,
        "repeats": repeats,
        "n_perm": n_perm,
        "null_repeats": null_repeats,
        "include_excluded": include_excluded,
        "intersect_with": str(intersect_with) if intersect_with else None,
        "channels": list(channels) if channels else None,
        "run_context": get_run_context(seed=seed).as_dict(),
    }
    (out_dir / "run.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
