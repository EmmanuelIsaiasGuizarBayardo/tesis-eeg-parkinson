"""Emulación del Emotiv EPOC X a partir de los registros de BioSemi.

Orden, como en el equipo: referencia en el CMS (P3) -> 14 canales -> banda del
equipo (0.16 a 43 Hz) -> muestreo de salida (256 Hz) -> cuantización de salida
(0.1275 µV en el modo de 16 bits). Se aplica al registro completo, antes del
recorte, porque el equipo registra de forma continua.

No se emulan el ruido del electrodo salino, la impedancia ni la colocación del
casco: esos límites se declaran en docs/transferibilidad.md.
"""

from __future__ import annotations

import mne
import numpy as np

from tesis_eeg_parkinson.preprocessing.config import EMOTIV_CHANNELS, PreprocessingConfig
from tesis_eeg_parkinson.units import microvolts_to_volts


def quantize(raw: mne.io.BaseRaw, step_uv: float) -> mne.io.BaseRaw:
    """Redondea cada muestra al múltiplo más cercano del paso, en el lugar."""
    step = float(microvolts_to_volts(step_uv))
    raw.apply_function(lambda x: np.round(x / step) * step, channel_wise=False, verbose="error")
    return raw


def emulate_epoc_x(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> mne.io.BaseRaw:
    """Convierte un registro de 32 canales de BioSemi en uno del EPOC X, en el lugar.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Registro con los 32 canales de cuero cabelludo, sin referenciar.
    cfg : PreprocessingConfig
        Perfil con ``device="epoc_x"``.

    Returns
    -------
    mne.io.BaseRaw
        14 canales, referidos a P3, en la banda y el muestreo del equipo, cuantizados.

    Notes
    -----
    La CAR posterior sobre los 14 canales cancela cualquier referencia común, así
    que la elección de P3 solo importa para la cuantización, que ocurre antes.
    """
    raw.set_eeg_reference(ref_channels=[cfg.device_reference], projection=False, verbose="error")
    raw.pick(list(EMOTIV_CHANNELS))
    raw.reorder_channels(list(EMOTIV_CHANNELS))
    low, high = cfg.device_band
    raw.filter(l_freq=low, h_freq=high, method="fir", phase="zero", verbose="error")
    raw.resample(cfg.device_sfreq, verbose="error")
    return quantize(raw, cfg.device_step_uv)
