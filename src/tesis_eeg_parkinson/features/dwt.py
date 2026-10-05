"""DWT diádica con reconstrucción de un solo coeficiente por banda."""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray


def band_ranges(sfreq: float, level: int) -> dict[str, tuple[float, float]]:
    """Banda nominal de cada coeficiente (partición diádica ideal).

    Con db4 las bandas reales se solapan en sus bordes; esta tabla es la nominal.
    """
    nyquist = sfreq / 2
    ranges = {f"cD{i}": (nyquist / 2**i, nyquist / 2 ** (i - 1)) for i in range(1, level + 1)}
    ranges[f"cA{level}"] = (0.0, nyquist / 2**level)
    return ranges


def band_signals(
    data: NDArray[np.float64], wavelet: str, level: int, bands: tuple[str, ...]
) -> dict[str, NDArray[np.float64]]:
    """Reconstruye en el tiempo cada coeficiente pedido, con los demás en cero.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    wavelet, level : str, int
    bands : tuple of str
        Nombres de PyWavelets ("cA4", "cD4", ..., "cD1").

    Returns
    -------
    dict of str to ndarray, shape (n_channels, n_times)

    Notes
    -----
    Es una DWT diádica (``wavedec``/``waverec``), no una descomposición en
    paquetes wavelet; el modo de borde es el simétrico, igual que el de MATLAB.
    """
    n_times = data.shape[-1]
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=-1)
    names = [f"cA{level}"] + [f"cD{i}" for i in range(level, 0, -1)]
    out: dict[str, NDArray[np.float64]] = {}
    for idx, name in enumerate(names):
        if name not in bands:
            continue
        kept = [c if i == idx else np.zeros_like(c) for i, c in enumerate(coeffs)]
        out[name] = pywt.waverec(kept, wavelet, axis=-1)[..., :n_times]
    return out
