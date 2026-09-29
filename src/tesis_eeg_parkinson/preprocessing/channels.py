"""Canales malos: criterios deterministas de PREP e interpolación esférica."""

from __future__ import annotations

import mne
from pyprep.find_noisy_channels import NoisyChannels

from tesis_eeg_parkinson.preprocessing.config import PreprocessingConfig

# Criterios de PREP sin RANSAC: RANSAC es estocástico y no aporta como primera
# barrera en un montaje de 32 canales de laboratorio (decisión documentada).
CRITERIA = (
    "bad_by_nan",
    "bad_by_flat",
    "bad_by_deviation",
    "bad_by_correlation",
    "bad_by_hf_noise",
)


def detect_bad_channels(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> dict[str, list[str]]:
    """Detecta canales malos con los criterios deterministas de PREP.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Registro sin referenciar a promedio: un canal malo contaminaría la CAR
        antes de ser detectado.
    cfg : PreprocessingConfig
        Perfil; aporta la semilla.

    Returns
    -------
    dict of str to list of str
        Canales por criterio, más ``"bad_all"`` con la unión ordenada.

    Notes
    -----
    Umbrales por defecto de PREP (Bigdely-Shamlo et al., 2015): desviación
    robusta z > 5, correlación máxima < 0.4 en más del 1% de ventanas de 1 s,
    ruido de alta frecuencia z > 5. ``do_detrend`` quita la deriva antes de medir.
    """
    detector = NoisyChannels(raw.copy(), do_detrend=True, random_state=cfg.seed, ransac=False)
    detector.find_bad_by_nan_flat()
    detector.find_bad_by_deviation()
    detector.find_bad_by_correlation()
    detector.find_bad_by_hfnoise()
    found = detector.get_bads(as_dict=True)
    result = {key: sorted(found.get(key, [])) for key in CRITERIA}
    result["bad_all"] = sorted({ch for key in CRITERIA for ch in result[key]})
    return result


def interpolate_bad_channels(raw: mne.io.BaseRaw, bads: list[str]) -> mne.io.BaseRaw:
    """Interpola por splines esféricos los canales indicados, en el lugar.

    Cada canal interpolado es combinación lineal de los demás: reduce el rango
    de los datos en uno, lo que después fija el número de componentes de ICA.
    """
    if bads:
        raw.info["bads"] = list(bads)
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="error")
    return raw
