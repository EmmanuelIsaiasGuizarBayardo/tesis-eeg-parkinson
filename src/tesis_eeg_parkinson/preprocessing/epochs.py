"""Segmentación en épocas fijas y rechazo pico a pico con umbral a priori."""

from __future__ import annotations

import mne

from tesis_eeg_parkinson.preprocessing.config import PreprocessingConfig
from tesis_eeg_parkinson.units import microvolts_to_volts


def make_epochs(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> tuple[mne.Epochs, dict[str, int]]:
    """Épocas de duración fija, sin solapamiento, y rechazo opcional.

    Returns
    -------
    epochs : mne.Epochs
        Épocas conservadas; cada una mide exactamente duración × sfreq muestras.
    counts : dict
        ``n_epochs_total``, ``n_epochs_kept`` y ``n_dropped_ptp``.
    """
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=cfg.epoch_duration_s,
        overlap=0.0,
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )
    total = len(epochs)
    if cfg.reject_ptp_uv is not None:
        threshold_v = float(microvolts_to_volts(cfg.reject_ptp_uv))
        epochs.drop_bad(reject={"eeg": threshold_v}, verbose="error")
    return epochs, {
        "n_epochs_total": total,
        "n_epochs_kept": len(epochs),
        "n_dropped_ptp": total - len(epochs),
    }


def persistent_bad_channels(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> dict[str, float]:
    """Canales que superan el umbral pico a pico en la mayoría de las épocas.

    El rechazo pico a pico se evalúa sobre el máximo entre canales: un canal con
    falla persistente tiraría épocas cuyos otros 31 canales están limpios. Se mide,
    por canal, la fracción de épocas que supera el umbral.

    Returns
    -------
    dict of str to float
        Canal -> fracción de épocas sobre el umbral, solo para los que superan
        ``cfg.persistent_fraction``. Vacío si la regla está desactivada.
    """
    if cfg.persistent_fraction is None or cfg.reject_ptp_uv is None:
        return {}
    epochs = mne.make_fixed_length_epochs(
        raw, duration=cfg.epoch_duration_s, overlap=0.0, preload=True, verbose="error"
    )
    data = epochs.get_data(picks="eeg")  # (épocas, canales, muestras)
    ptp = data.max(axis=-1) - data.min(axis=-1)
    fraction = (ptp > microvolts_to_volts(cfg.reject_ptp_uv)).mean(axis=0)
    names = epochs.copy().pick("eeg").ch_names
    return {
        ch: round(float(f), 3)
        for ch, f in zip(names, fraction, strict=True)
        if f > cfg.persistent_fraction
    }
