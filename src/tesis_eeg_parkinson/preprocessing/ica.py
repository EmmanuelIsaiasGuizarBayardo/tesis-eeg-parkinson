"""ICA con transferencia de pesos y rechazo de componentes con ICLabel."""

from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from numpy.typing import NDArray

from tesis_eeg_parkinson.preprocessing.config import ICLABEL_CLASSES, PreprocessingConfig

# Índices de ICLABEL_CLASSES que cuentan como artefacto conocido.
_KNOWN_ARTIFACTS = (1, 2, 3, 4, 5)
_BRAIN, _OTHER = 0, 6


def fit_ica(raw_copy: mne.io.BaseRaw, n_components: int, cfg: PreprocessingConfig) -> ICA:
    """Ajusta infomax extendido sobre la copia filtrada.

    Parameters
    ----------
    raw_copy : mne.io.BaseRaw
        Copia referenciada a promedio y filtrada para ICA.
    n_components : int
        Rango de los datos: 32 - 1 (CAR) - canales interpolados.
    cfg : PreprocessingConfig
        Perfil; aporta semilla e iteraciones.
    """
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},
        max_iter=cfg.ica_max_iter,
        random_state=cfg.seed,
    )
    ica.fit(raw_copy, verbose="error")
    return ica


def label_components(inst: mne.io.BaseRaw, ica: ICA) -> NDArray[np.float64]:
    """Probabilidades de ICLabel, forma (n_components, 7), en el orden de EEGLAB."""
    proba = iclabel_label_components(inst, ica, inplace=False)
    return np.asarray(proba, dtype=np.float64)


def select_artifact_components(
    proba: NDArray[np.float64], threshold: float, other_brain_max: float
) -> list[int]:
    """Criterio del manual: artefacto conocido, u "other" sin cerebro.

    Se rechaza un IC si su clase dominante es un artefacto conocido con
    probabilidad > ``threshold``, o si domina "other" con probabilidad
    > ``threshold`` y P(brain) < ``other_brain_max``.

    Parameters
    ----------
    proba : ndarray, shape (n_components, 7)
        Salida de `label_components`.
    threshold, other_brain_max : float
        Umbrales del criterio.

    Returns
    -------
    list of int
        Índices de los IC a excluir.
    """
    if proba.ndim != 2 or proba.shape[1] != len(ICLABEL_CLASSES):
        raise ValueError(f"Se esperaba (n, {len(ICLABEL_CLASSES)}); llegó {proba.shape}")
    top = proba.argmax(axis=1)
    top_p = proba.max(axis=1)
    known = np.isin(top, _KNOWN_ARTIFACTS) & (top_p > threshold)
    other = (top == _OTHER) & (top_p > threshold) & (proba[:, _BRAIN] < other_brain_max)
    return np.flatnonzero(known | other).astype(int).tolist()
