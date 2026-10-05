"""Métricas por canal. Todas reciben (n_channels, n_times) y devuelven (n_channels,).

V1 (invariantes a la escala de la señal):
- potencia relativa por banda,
- entropía wavelet normalizada (cómo se reparte la energía de la banda en el tiempo),
- entropía de permutación (orden temporal; Bandt & Pompe, 2002),
- entropía muestral (regularidad temporal; Richman & Moorman, 2000),
- exponente aperiódico y centro de gravedad de alfa (Donoghue et al., 2020).

Aljalal et al. (2022): las ocho métricas tal como están escritas en el artículo,
calibradas para µV, más la lectura de TShEn sobre el histograma de valores únicos.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import signal
from scipy.spatial import cKDTree

with warnings.catch_warnings():  # fooof 1.1 avisa de su sucesor, specparam (aún preliminar)
    warnings.simplefilter("ignore")
    from fooof import FOOOFGroup
    from fooof.sim.gen import gen_aperiodic

Array = NDArray[np.float64]


# --------------------------------------------------------------------- V1
def relative_power(bands: dict[str, Array]) -> dict[str, Array]:
    """Energía de cada banda sobre la suma de las bandas dadas (suma 1 por canal)."""
    energy = {name: np.sum(x**2, axis=-1) for name, x in bands.items()}
    total = np.sum(list(energy.values()), axis=0)
    return {name: e / total for name, e in energy.items()}


def wavelet_entropy(x: Array) -> Array:
    """Entropía de Shannon normalizada de la energía en el tiempo, en [0, 1].

    Con pᵢ = xᵢ²/E, H = −Σ pᵢ log pᵢ / log N. Es invariante a la escala. La ShEn del
    artículo (Σ x² log x²) es igual a E·(log E − H·log N): sobre todo, energía.
    """
    energy = np.sum(x**2, axis=-1, keepdims=True)
    p = x**2 / energy
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)
    return h / np.log(x.shape[-1])


def permutation_entropy(x: Array, order: int, delay: int) -> Array:
    """Entropía de permutación normalizada, en [0, 1] (Bandt & Pompe, 2002).

    Invariante a cualquier transformación monótona creciente de la señal, así que
    no depende de ganancia ni de desplazamiento. Los empates se resuelven por orden
    de aparición (``argsort`` estable).
    """
    span = (order - 1) * delay + 1
    windows = sliding_window_view(x, span, axis=-1)[..., ::delay]  # (canales, ventanas, m)
    ranks = np.argsort(windows, axis=-1, kind="stable")
    codes = (ranks * order ** np.arange(order)).sum(axis=-1)
    out = np.empty(x.shape[0])
    for ch, row in enumerate(codes):
        counts = np.unique(row, return_counts=True)[1]
        p = counts / counts.sum()
        out[ch] = -np.sum(p * np.log(p)) / math.log(math.factorial(order))
    return out


def _sampen_1d(x: Array, order: int, r: float) -> float:
    n = x.size - order  # mismo número de plantillas para m y m + 1
    templates_m = sliding_window_view(x, order)[:n]
    templates_m1 = sliding_window_view(x, order + 1)[:n]
    # Pares a distancia de Chebyshev <= r, sin contar a cada plantilla consigo misma.
    b = cKDTree(templates_m).count_neighbors(cKDTree(templates_m), r, p=np.inf) - n
    a = cKDTree(templates_m1).count_neighbors(cKDTree(templates_m1), r, p=np.inf) - n
    return float(-np.log(a / b)) if a > 0 and b > 0 else float("nan")


def sample_entropy(x: Array, order: int, r_factor: float) -> Array:
    """Entropía muestral con r = r_factor·DE de cada canal (Richman & Moorman, 2000).

    Con r relativo a la DE es invariante a la escala. NaN si no hay coincidencias
    de longitud m + 1.
    """
    return np.array(
        [_sampen_1d(row, order, r_factor * row.std()) for row in np.asarray(x, dtype=float)]
    )


def aperiodic_and_alpha(
    x: Array,
    sfreq: float,
    window_s: float,
    fit_range: tuple[float, float],
    alpha_range: tuple[float, float],
) -> dict[str, Array]:
    """Exponente aperiódico y centro de gravedad de alfa, por canal.

    El espectro (Welch) se parametriza con FOOOF en modo "fixed" (Donoghue et al.,
    2020). El centro de gravedad de alfa se calcula sobre la potencia que queda por
    encima del componente aperiódico, para que la pendiente 1/f no lo sesgue hacia
    frecuencias bajas. Ambos son invariantes a la escala: una ganancia solo mueve el
    desplazamiento (offset) del ajuste.

    Returns
    -------
    dict
        ``aperiodic_exponent``, ``alpha_cog`` (NaN sin potencia periódica en alfa) y
        ``qc_fooof_r2`` (calidad del ajuste; no es una feature).
    """
    nperseg = int(window_s * sfreq)
    freqs, psd = signal.welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    group = FOOOFGroup(peak_width_limits=(1.0, 8.0), max_n_peaks=4, aperiodic_mode="fixed",
                       verbose=False)  # fmt: skip
    group.fit(freqs, psd, list(fit_range))
    aperiodic = group.get_params("aperiodic_params")  # (canales, [offset, exponente])
    fit_freqs = group.freqs
    flat = group.power_spectra - np.array([gen_aperiodic(fit_freqs, ap) for ap in aperiodic])
    mask = (fit_freqs >= alpha_range[0]) & (fit_freqs <= alpha_range[1])
    excess = np.clip(10 ** flat[:, mask] - 1, 0, None)  # razón de potencia sobre lo aperiódico
    with np.errstate(invalid="ignore", divide="ignore"):
        cog = (excess * fit_freqs[mask]).sum(axis=1) / excess.sum(axis=1)
    return {
        "aperiodic_exponent": aperiodic[:, 1],
        "alpha_cog": np.where(excess.sum(axis=1) > 0, cog, np.nan),
        "qc_fooof_r2": group.get_params("r_squared"),
    }


# --------------------------------------------------------------- Aljalal
# Parámetros del artículo, calibrados para µV.
THEN_ALPHA, NOEN_P, SUEN_EPSILON = 0.2, 1.1, 3.0
TSHEN_MIN, TSHEN_MAX = 0.0, 255.0


def _tshen_transform(s: Array) -> Array:
    """Ecuación 8: s > 1 -> 255; 0 <= s <= 1 -> 255·s; s < 0 -> 0 (umbrales en µV)."""
    return np.where(s > 1, TSHEN_MAX, np.where(s >= 0, TSHEN_MAX * s + TSHEN_MIN, TSHEN_MIN))


def _log_or_zero(v: Array) -> Array:
    with np.errstate(divide="ignore"):
        return np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), 0.0)


def aljalal_metrics(s: Array) -> dict[str, Array]:
    """Las ocho métricas de Aljalal et al. (2022), sobre amplitudes en µV.

    ``shen`` conserva el signo del artículo (ecuación 7, sin el menos), así que es
    el negativo de la entropía de Shannon convencional. ``tshen`` aplica la
    ecuación 9 sobre amplitudes transformadas, como el código que reprodujo el
    99.8%; ``tshen_hist`` la aplica sobre las probabilidades de los valores únicos,
    que es lo que describe el texto del artículo.
    """
    n = s.shape[-1]
    sq = s**2
    power = sq.sum(axis=-1) / n
    tr = _tshen_transform(s)
    tr_sq = tr**2
    k = np.maximum((tr_sq > 0).sum(axis=-1), 1)
    hist = np.empty(s.shape[0])
    for ch, row in enumerate(tr):
        counts = np.unique(row, return_counts=True)[1]
        p2 = (counts / n) ** 2
        hist[ch] = np.sum(p2 * np.log(p2)) / counts.size
    return {
        "energy": sq.sum(axis=-1),
        "lbp": _log_or_zero(power),
        "then": (np.abs(s) > THEN_ALPHA).sum(axis=-1).astype(float),
        "noen": (np.abs(s) ** NOEN_P).sum(axis=-1),
        "suen": n
        - (np.abs(s) <= SUEN_EPSILON).sum(axis=-1)
        + np.minimum(sq, SUEN_EPSILON**2).sum(axis=-1),
        "logen": _log_or_zero(sq).sum(axis=-1),
        "shen": (sq * _log_or_zero(sq)).sum(axis=-1),
        "tshen": (tr_sq * _log_or_zero(tr_sq)).sum(axis=-1) / k,
        "tshen_hist": hist,
    }
