"""Pruebas de las figuras de la tesis con datos sintéticos (se construyen y se guardan)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tesis_eeg_parkinson.figures import tesis as T
from tesis_eeg_parkinson.figures.base import save
from tesis_eeg_parkinson.preprocessing.config import SCALP_CHANNELS

RNG = np.random.default_rng(0)
CONTRASTS = ("off_vs_hc", "on_vs_hc", "off_vs_on")


def _qc() -> pd.DataFrame:
    rows = []
    for i, session in enumerate(["hc"] * 6 + ["off"] * 5 + ["on"] * 5):
        rows.append(
            {
                "subject": f"s{i}",
                "session": session,
                "n_epochs_kept": 15 + i % 4,
                "n_epochs_total": 19,
                "n_ics_excluded": 3 + i % 5,
                "exclude_primary": i == 15,
                "prep_flagged": "T7,T8,FC5" if i % 2 else "F7",
                "excluded_labels": "eye blink,muscle artifact,other",
            }
        )
    return pd.DataFrame(rows)


def _patterns() -> pd.DataFrame:
    rows = [
        {
            "contrast": c,
            "metric": m,
            "signal": s,
            "channel": ch,
            "pattern": RNG.normal(),
            "stability": RNG.uniform(0.5, 1),
        }
        for c in CONTRASTS
        for m, s, _ in T.KEY_PATTERNS
        for ch in SCALP_CHANNELS
    ]
    return pd.DataFrame(rows)


def _features(names: list[str]) -> pd.DataFrame:
    cols = {f"{n}__{ch}": RNG.normal(size=40) for n in names for ch in SCALP_CHANNELS[:4]}
    return pd.DataFrame(cols)


def _resumen() -> pd.DataFrame:
    rows = [
        {
            "analysis": a,
            "contrast": c,
            "bacc": 0.6,
            "bacc_ci_low": 0.45,
            "bacc_ci_high": 0.75,
            "chance_p95": 0.68,
        }
        for a, _ in T.TRANSFER
        for c in CONTRASTS
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("estilo", ["tesis", "publicacion", "presentacion"])
def test_all_figures_build_and_save(tmp_path: Path, estilo: str) -> None:
    freqs = np.linspace(0.25, 45, 180)
    spectra = {
        n: (freqs, -10 * np.log10(freqs) + RNG.normal(0, 0.5, freqs.size))
        for n in ("pd3", "pd5", "pd6", "pd16")
    }
    amps = {
        b: np.abs(RNG.normal(m, m / 4, 500))
        for b, m in zip(
            ("cA4", "cD4", "cD3", "cD2", "cD1"), (2.7, 1.1, 0.36, 0.048, 0.0035), strict=True
        )
    }
    null = pd.DataFrame({c: RNG.normal(0.5, 0.1, 200) for c in CONTRASTS})
    summary = pd.DataFrame(
        {
            "contrast": CONTRASTS,
            "bacc": [0.61, 0.66, 0.69],
            "chance_p95": [0.68, 0.69, 0.65],
            "p_holm": [0.19, 0.19, 0.08],
        }
    )
    figuras = [
        T.fig_calidad(_qc(), estilo),
        T.fig_procedencia(spectra, estilo),
        T.fig_wavelet(amps, estilo),
        T.fig_redundancia(
            _features([f"{k}__cA4" for k in T.ALJALAL_NAMES]), _features(list(T.V1_NAMES)), estilo
        ),
        T.fig_nula(null, summary, estilo),
        T.fig_transferencia(_resumen(), estilo),
        T.fig_patrones_clave(_patterns(), estilo),
    ]
    for figura in figuras:
        pdf, png = save(figura, tmp_path)
        assert pdf.stat().st_size > 0 and png.stat().st_size > 0
        assert len(figura.leyenda) > 100


def test_band_responses_peak_in_their_band() -> None:
    freqs, resp = T.band_responses(128.0, ("cA4", "cD4", "cD3", "cD2"))
    for band, (lo, hi) in {"cA4": (0, 4), "cD4": (4, 8), "cD3": (8, 16), "cD2": (16, 32)}.items():
        peak = freqs[np.argmax(resp[band])]
        assert lo <= peak <= hi + 1, (band, peak)


def test_build_all_reports_missing_data(tmp_path: Path) -> None:
    status = T.build_all(
        tmp_path / "nada", tmp_path / "nada", tmp_path / "out", "publicacion", only=("nula",)
    )
    assert status["nula"].startswith("faltan datos")
