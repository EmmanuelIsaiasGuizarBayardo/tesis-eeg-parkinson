"""Pruebas de la interpretación con datos sintéticos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from test_classification import _table

from tesis_eeg_parkinson.classification.data import make_dataset
from tesis_eeg_parkinson.interpretation.figures import (
    FEATURES,
    plot_contrast_topomaps,
    plot_magnitude,
)
from tesis_eeg_parkinson.interpretation.patterns import bootstrap_patterns, resample_subjects
from tesis_eeg_parkinson.preprocessing.config import SCALP_CHANNELS


def test_resample_keeps_whole_subjects_and_both_classes() -> None:
    ds = make_dataset(_table(0.0), "off_vs_on")
    boot = resample_subjects(ds, np.random.default_rng(0))
    assert len(np.unique(boot.y)) == 2
    for s in np.unique(boot.subject):  # cada sujeto entra con sus 16 épocas (o múltiplos)
        assert (boot.subject == s).sum() % 16 == 0


def test_stability_separates_true_effect_from_noise() -> None:
    ds = make_dataset(_table(effect=1.5), "off_vs_hc")
    result = bootstrap_patterns(ds, n_boot=20, seed=1)
    effect = result[result.index.str.startswith("m0__")]
    noise = result[~result.index.str.startswith("m0__")]
    assert (effect["stability"] >= 0.95).all() and (effect["pattern"] > 0).all()
    assert noise["stability"].median() < effect["stability"].min()


def _fake_long() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for contrast in ("off_vs_hc", "on_vs_hc", "off_vs_on"):
        for metric, signal, _t, _f in FEATURES:
            for ch in SCALP_CHANNELS:
                rows.append({"contrast": contrast, "metric": metric, "signal": signal,
                             "channel": ch, "pattern": rng.normal(),
                             "stability": rng.uniform(0.5, 1.0)})  # fmt: skip
    return pd.DataFrame(rows)


def test_figures_are_written(tmp_path: Path) -> None:
    long = _fake_long()
    assert plot_contrast_topomaps(long, "off_vs_on", tmp_path / "t.pdf").exists()
    assert plot_magnitude(long, tmp_path / "m.pdf").exists()


def test_topomap_positions_fall_inside_the_head() -> None:
    """Con biosemi32, MNE proyecta los 32 electrodos dentro del contorno (radio 0.095 m).

    Con colin27_1005, Fp1 y Fp2 caían fuera: era la causa de los mapas desbordados.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    from tesis_eeg_parkinson.interpretation.figures import _info

    fig = mne.viz.plot_sensors(_info(list(SCALP_CHANNELS)), kind="topomap", show=False)
    radius = np.linalg.norm(np.asarray(fig.axes[0].collections[0].get_offsets()), axis=1)
    plt.close(fig)
    assert radius.max() < 0.095
