"""Tabla de features: una fila por época, una columna por métrica, señal y canal.

Nombre de columna: ``<métrica>__<señal>__<canal>``. Las columnas que empiezan con
``qc_`` son de control de calidad y no entran a la clasificación.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from tesis_eeg_parkinson.features.config import FeatureConfig
from tesis_eeg_parkinson.features.dwt import band_ranges, band_signals
from tesis_eeg_parkinson.features.metrics import (
    aljalal_metrics,
    aperiodic_and_alpha,
    permutation_entropy,
    relative_power,
    sample_entropy,
    time_asymmetry,
    wavelet_entropy,
)
from tesis_eeg_parkinson.reproducibility import get_run_context
from tesis_eeg_parkinson.units import volts_to_microvolts

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str], None]
META_COLUMNS = ("subject", "session", "group", "epoch", "exclude_primary", "exclude_reason")


def _resample(epochs: mne.Epochs, sfreq: float) -> mne.Epochs:
    """Decima a ``sfreq`` cuando el factor es entero.

    El Butterworth previo ya atenúa 65 dB a 64 Hz, así que tomar una de cada k
    muestras no introduce aliasing apreciable y evita los bordes del remuestreo FFT.
    """
    factor = epochs.info["sfreq"] / sfreq
    if np.isclose(factor, 1.0):
        return epochs
    if not np.isclose(factor, round(factor)):
        raise ValueError(f"Factor de decimación no entero: {factor}")
    return epochs.copy().decimate(int(round(factor)), verbose="error")


def epoch_features(
    data_uv: np.ndarray, ch_names: list[str], cfg: FeatureConfig
) -> dict[str, float]:
    """Features de una época.

    Parameters
    ----------
    data_uv : ndarray, shape (n_channels, n_times)
        Época en µV (las métricas de Aljalal están calibradas en µV; las de V1 son
        invariantes a la escala).
    ch_names : list of str
    cfg : FeatureConfig

    Returns
    -------
    dict
        ``<métrica>__<señal>__<canal>`` -> valor.
    """
    signals = band_signals(data_uv, cfg.wavelet, cfg.level, cfg.bands)
    values: dict[tuple[str, str], np.ndarray] = {}
    if cfg.family == "aljalal":
        if cfg.include_broadband:
            signals["original"] = data_uv
        for name, x in signals.items():
            for metric, v in aljalal_metrics(x).items():
                values[(metric, name)] = v
    else:
        for name, v in relative_power(signals).items():
            values[("relpow", name)] = v
        for name, x in signals.items():
            values[("wentropy", name)] = wavelet_entropy(x)
        for tau in cfg.permen_delays:
            values[(f"permen_tau{tau}", "broadband")] = permutation_entropy(
                data_uv, cfg.permen_order, tau
            )
        values[("sampen", "broadband")] = sample_entropy(data_uv, cfg.sampen_order, cfg.sampen_r)
        for tau in cfg.t3_delays:
            values[(f"t3_tau{tau}", "broadband")] = time_asymmetry(data_uv, tau)
        spectral = aperiodic_and_alpha(
            data_uv, cfg.sfreq, cfg.psd_window_s, cfg.aperiodic_range, cfg.alpha_range
        )
        for metric, v in spectral.items():
            values[(metric, "broadband")] = v
    return {
        f"{metric}__{sig}__{ch}": float(val)
        for (metric, sig), arr in values.items()
        for ch, val in zip(ch_names, arr, strict=True)
    }


def recording_features(epochs_path: Path, cfg: FeatureConfig, meta: dict[str, Any]) -> pd.DataFrame:
    """Features de todas las épocas de un registro, con sus metadatos."""
    epochs = _resample(mne.read_epochs(epochs_path, verbose="error"), cfg.sfreq)
    data = volts_to_microvolts(epochs.get_data(picks="eeg"))
    names = epochs.copy().pick("eeg").ch_names
    rows = [{**meta, "epoch": i, **epoch_features(ep, names, cfg)} for i, ep in enumerate(data)]
    return pd.DataFrame(rows)


def _meta_from_qc(epochs_path: Path) -> dict[str, Any]:
    qc_path = Path(str(epochs_path).replace("_desc-clean_epo.fif", "_desc-qc.json"))
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    rec = qc["recording"]
    session = rec["session"]
    return {
        "subject": rec["subject"],
        "session": session,
        "group": "hc" if session == "hc" else f"pd-{session}",
        "exclude_primary": qc["exclude_primary"],
        "exclude_reason": qc["exclude_reason"],
    }


def run_features(
    preproc_root: Path,
    cfg: FeatureConfig,
    out_root: Path,
    on_progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Calcula features para todos los registros de un perfil de preprocesamiento.

    Escribe ``features-<perfil>/preproc-<perfil>/features.tsv`` y ``features.json``
    (una carpeta por combinación, para que no se sobrescriban; configuración,
    bandas nominales, perfil de preprocesamiento y ``RunContext``).

    Raises
    ------
    FileNotFoundError
        Si no hay épocas en ``preproc_root``; nada se escribe.
    """
    paths = sorted(Path(preproc_root).rglob("*_desc-clean_epo.fif"))
    if not paths:
        raise FileNotFoundError(f"No hay épocas en {preproc_root}")
    notify = on_progress or (lambda *_args: None)
    tables = []
    for done, path in enumerate(paths):
        meta = _meta_from_qc(path)
        notify(done, len(paths), f"sub-{meta['subject']} ses-{meta['session']}")
        tables.append(recording_features(path, cfg, meta))
        notify(done + 1, len(paths), f"sub-{meta['subject']} ses-{meta['session']}")
    table = pd.concat(tables, ignore_index=True)
    preproc_name = Path(preproc_root).name
    out = Path(out_root) / f"features-{cfg.name}" / preproc_name
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "features.tsv", sep="\t", index=False)
    feature_cols = [c for c in table.columns if c not in META_COLUMNS and not c.startswith("qc_")]
    sidecar = {
        "config": cfg.as_dict(),
        "nominal_bands_hz": band_ranges(cfg.sfreq, cfg.level),
        "preproc_root": str(preproc_root),
        "n_epochs": len(table),
        "n_features": len(feature_cols),
        "naming": "<métrica>__<señal>__<canal>; columnas qc_ fuera de la clasificación",
        "run_context": get_run_context(seed=28).as_dict(),
    }
    (out / "features.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return table
