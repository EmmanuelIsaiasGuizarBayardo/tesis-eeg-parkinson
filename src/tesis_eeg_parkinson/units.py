"""Conversión de unidades en un solo lugar.

MNE trabaja en voltios. Toda conversión a microvoltios pasa por aquí, para que
exista un único punto auditable (en el legado se multiplicaba por 1e6 al cargar).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

VOLTS_PER_MICROVOLT = 1e-6


def volts_to_microvolts(data: ArrayLike) -> NDArray[np.float64]:
    """Convierte de V a µV.

    Parameters
    ----------
    data : array_like
        Señal en voltios.

    Returns
    -------
    ndarray
        Señal en microvoltios, como float64.
    """
    return np.asarray(data, dtype=np.float64) / VOLTS_PER_MICROVOLT


def microvolts_to_volts(data: ArrayLike) -> NDArray[np.float64]:
    """Convierte de µV a V (inversa de `volts_to_microvolts`)."""
    return np.asarray(data, dtype=np.float64) * VOLTS_PER_MICROVOLT
