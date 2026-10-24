"""Pruebas de la capa de clasificación con tablas sintéticas de efecto conocido."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tesis_eeg_parkinson.classification.data import make_dataset
from tesis_eeg_parkinson.classification.evaluate import (
    bootstrap_ci,
    holm,
    permute_labels,
    run_contrast,
)
from tesis_eeg_parkinson.classification.report import write_report
from tesis_eeg_parkinson.classification.run import run_analysis

CHANNELS = ("AF3", "F7", "Cz", "Pz")


def _table(
    effect: float, fingerprint: float = 0.0, seed: int = 0, n: int = 10, metrics: int = 4
) -> pd.DataFrame:
    """n HC y n PD con dos sesiones; 8 épocas por registro; métricas × 4 canales.

    ``effect`` desplaza la métrica m0 en PD off; ``fingerprint`` da a cada sujeto un
    desplazamiento propio en todas las features (lo que una validación por épocas
    puede memorizar).
    """
    rng = np.random.default_rng(seed)
    cols = [f"m{k}__broadband__{ch}" for k in range(metrics) for ch in CHANNELS]
    rows = []
    subjects = [("hc", f"hc{i}", "hc") for i in range(n)]
    subjects += [(f"pd-{s}", f"pd{i}", s) for i in range(n) for s in ("off", "on")]
    offsets = {sid: rng.standard_normal(len(cols)) * fingerprint for _, sid, _ in subjects}
    for group, sid, session in subjects:
        for epoch in range(8):
            x = rng.standard_normal(len(cols)) + offsets[sid]
            if group == "pd-off":
                x[: len(CHANNELS)] += effect
            rows.append({"subject": sid, "session": session, "group": group, "epoch": epoch,
                         "exclude_primary": False, "exclude_reason": "",
                         **dict(zip(cols, x, strict=True))})  # fmt: skip
    table = pd.DataFrame(rows)
    table.loc[(table.subject == "pd9") & (table.session == "on"), "exclude_primary"] = True
    return table


def test_make_dataset_contrasts_exclusions_and_pairs() -> None:
    table = _table(0.0)
    off_hc = make_dataset(table, "off_vs_hc")
    assert off_hc.n_recordings == 20 and off_hc.y.sum() == 10 * 8
    on_hc = make_dataset(table, "on_vs_hc")
    assert on_hc.n_recordings == 19  # pd9 on excluido
    paired = make_dataset(table, "off_vs_on")
    assert paired.paired and paired.n_recordings == 18  # pd9 sale completo del pareado
    assert make_dataset(table, "off_vs_on", include_excluded=True).n_recordings == 20
    assert make_dataset(table, "off_vs_hc", channels=("AF3", "F7")).X.shape[1] == 8
    assert set(make_dataset(table, "off_vs_hc", subjects={"hc1", "pd1"}).subject) == {"hc1", "pd1"}


def test_permutations_respect_the_exchangeable_unit() -> None:
    table = _table(0.0)
    rng = np.random.default_rng(1)
    ds = make_dataset(table, "off_vs_hc")
    y = permute_labels(ds, rng)
    per_subject = pd.Series(y).groupby(ds.subject).nunique()
    assert per_subject.max() == 1 and y.sum() == ds.y.sum()  # una etiqueta por sujeto
    paired = make_dataset(table, "off_vs_on")
    y = permute_labels(paired, rng)
    per_patient = pd.Series(y).groupby(paired.subject).agg(lambda v: sorted(set(v)))
    assert all(v == [0, 1] for v in per_patient)  # cada paciente conserva una sesión de cada tipo


def test_detects_a_true_effect_and_its_pattern() -> None:
    ds = make_dataset(_table(effect=1.5), "off_vs_hc")
    result = run_contrast(ds, "lr", repeats=2, n_perm=19, null_repeats=1, seed=3,
                          n_boot=200, leakage_control=False)  # fmt: skip
    s = result["summary"]
    assert s["bacc"] > 0.85 and s["p_value"] == pytest.approx(0.05)  # mínimo posible con 19
    assert s["bacc_ci_low"] <= s["bacc"] <= s["bacc_ci_high"]
    top = result["pattern"].abs().sort_values(ascending=False).index[:4]
    assert all(name.startswith("m0__") for name in top)  # el patrón señala la métrica con efecto


def test_epoch_validation_inflates_through_subject_fingerprints() -> None:
    """Sin efecto de grupo pero con huellas por sujeto, solo la validación por épocas acierta."""
    # Con más features que sujetos, un modelo lineal puede memorizar a cada sujeto.
    ds = make_dataset(_table(effect=0.0, fingerprint=2.0, seed=5, metrics=16), "off_vs_hc")
    s = run_contrast(ds, "lr", repeats=2, n_perm=0, null_repeats=1, seed=4, n_boot=100)["summary"]
    assert s["epoch_bacc_epoch_cv"] > 0.85
    assert s["epoch_bacc_epoch_cv"] - s["epoch_bacc_subject_cv"] > 0.25


@pytest.mark.parametrize("model", ["slda", "rf"])
def test_sensitivity_models_run(model: str) -> None:
    ds = make_dataset(_table(effect=1.5, seed=6), "off_vs_hc")
    s = run_contrast(ds, model, repeats=1, n_perm=0, null_repeats=1, seed=2, n_boot=50,
                     leakage_control=False)["summary"]  # fmt: skip
    assert s["bacc"] > 0.7 and "p_value" not in s


def test_bootstrap_resamples_whole_patients() -> None:
    rec = pd.DataFrame({"subject": ["a", "a", "b", "b", "c", "c"], "y": [1, 0] * 3,
                        "proba": [0.9, 0.1, 0.8, 0.3, 0.4, 0.6]})  # fmt: skip
    ci = bootstrap_ci(rec, n_boot=300, seed=0)
    assert 0.0 <= ci["bacc"][0] <= ci["bacc"][1] <= 1.0


def test_holm() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_run_analysis_writes_tables(tmp_path: Path) -> None:
    path = tmp_path / "features.tsv"
    _table(effect=1.0).to_csv(path, sep="\t", index=False)
    summary = run_analysis(path, tmp_path / "out", "lr", ("off_vs_hc", "off_vs_on"), repeats=1,
                           n_perm=4, null_repeats=1, leakage_control=False)  # fmt: skip
    assert list(summary["contrast"]) == ["off_vs_hc", "off_vs_on"] and "p_holm" in summary
    for name in ("summary.tsv", "recordings.tsv", "null.tsv", "patterns.tsv", "run.json"):
        assert (tmp_path / "out" / name).exists()


def test_report_collects_analyses_and_draws_figures(tmp_path: Path) -> None:
    path = tmp_path / "features.tsv"
    _table(effect=1.0).to_csv(path, sep="\t", index=False)
    for name, model, perms in (("v1_v1_lr", "lr", 4), ("v1_v1_slda", "slda", 0)):
        run_analysis(path, tmp_path / "res" / name, model, repeats=1, n_perm=perms,
                     null_repeats=1)  # fmt: skip
    paths = write_report(tmp_path / "res")
    table = pd.read_csv(paths["resumen"], sep="\t")
    assert set(table["analysis"]) == {"v1_v1_lr", "v1_v1_slda"} and len(table) == 6
    assert paths["resultados"].exists() and paths["fuga"].exists()
