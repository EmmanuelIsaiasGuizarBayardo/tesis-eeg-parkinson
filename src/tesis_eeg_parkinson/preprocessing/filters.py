"""Filtros: copia para ICA (FIR) y Butterworth final (SOS o [b,a] del legado)."""

from __future__ import annotations

import mne
import numpy as np
from numpy.typing import NDArray
from scipy import signal

from tesis_eeg_parkinson.preprocessing.config import PreprocessingConfig


def ica_copy(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> mne.io.BaseRaw:
    """Copia filtrada con FIR de fase cero, solo para ajustar la ICA.

    El ancho de transición ``"auto"`` de MNE sigue la misma regla que el
    ``pop_eegfiltnew`` de EEGLAB: min(max(25% del corte, 2 Hz), corte).
    """
    return raw.copy().filter(
        l_freq=cfg.ica_copy_l_freq,
        h_freq=cfg.ica_copy_h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose="error",
    )


def final_filter_coefficients(
    cfg: PreprocessingConfig, sfreq: float
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coeficientes del Butterworth pasabanda final.

    Returns
    -------
    ndarray or tuple of ndarray
        Matriz SOS si ``cfg.final_form == "sos"``; tupla (b, a) si es "ba".
    """
    band = [cfg.final_l_freq, cfg.final_h_freq]
    if cfg.final_form == "sos":
        return signal.butter(cfg.final_order, band, btype="bandpass", fs=sfreq, output="sos")
    return signal.butter(cfg.final_order, band, btype="bandpass", fs=sfreq, output="ba")


def apply_final_filter(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> mne.io.BaseRaw:
    """Aplica el Butterworth final con fase cero (forward-backward), en el lugar.

    Notes
    -----
    La forma [b,a] del diseño 0.5-32 Hz, N=5, a 512 Hz está mal condicionada:
    el redondeo al pasar a polinomios mueve el polo más lento de 0.99815 a
    0.99877 y el borde inferior queda en -1.77 dB por pasada en lugar de -3.01.
    MNE rechaza esa forma ("poles outside unit circle"), así que el perfil
    legacy la aplica con scipy y el relleno de MATLAB: 3·(orden+1-1) muestras.
    """
    sfreq = raw.info["sfreq"]
    if cfg.final_form == "sos":
        raw.filter(
            l_freq=cfg.final_l_freq,
            h_freq=cfg.final_h_freq,
            method="iir",
            iir_params={"order": cfg.final_order, "ftype": "butter", "output": "sos"},
            phase="zero",
            verbose="error",
        )
        return raw

    b, a = final_filter_coefficients(cfg, sfreq)
    padlen = 3 * (max(len(a), len(b)) - 1)  # nfact de filtfilt en MATLAB

    def _filtfilt_ba(data: NDArray[np.float64]) -> NDArray[np.float64]:
        return signal.filtfilt(b, a, data, axis=-1, padtype="odd", padlen=padlen)

    raw.apply_function(_filtfilt_ba, channel_wise=False, verbose="error")
    with raw.info._unlock():  # la API pública no expone registrar un filtro externo
        raw.info["highpass"] = cfg.final_l_freq
        raw.info["lowpass"] = cfg.final_h_freq
    return raw
