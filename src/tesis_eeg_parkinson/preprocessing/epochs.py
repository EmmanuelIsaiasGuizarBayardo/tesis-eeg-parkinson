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
