"""Figuras de control de calidad con la biblioteca `figuras_cientificas`."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import figuras_cientificas as fc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from numpy.typing import NDArray

from tesis_eeg_parkinson.units import volts_to_microvolts

# Percentiles de la banda entre canales; se declaran en el pie de la figura.
BAND_PERCENTILES = (10.0, 90.0)
FMIN, FMAX = 0.25, 120.0
# Rango donde se lee la altura de cada curva para su etiqueta: evita que un
# notch (120 Hz) en el último punto la desplace.
LABEL_BAND = (85.0, 110.0)


def channel_psd_db(
    inst: mne.io.BaseRaw | mne.Epochs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """PSD de Welch por canal (ventanas de 4 s), en dB re 1 µV²/Hz.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
    psd_db : ndarray, shape (n_channels, n_freqs)
        En épocas, promediada sobre épocas antes de pasar a dB.
    """
    sfreq = inst.info["sfreq"]
    n_fft = int(4 * sfreq)
    kwargs = {"method": "welch", "fmin": FMIN, "fmax": min(FMAX, sfreq / 2 - 1)}
    if isinstance(inst, mne.io.BaseRaw):
        spectrum = inst.compute_psd(n_fft=n_fft, verbose="error", **kwargs)
        psd = spectrum.get_data()
    else:
        n_fft = min(n_fft, len(inst.times))
        spectrum = inst.compute_psd(n_fft=n_fft, verbose="error", **kwargs)
        psd = spectrum.get_data().mean(axis=0)
    # V²/Hz -> µV²/Hz: la conversión de amplitud entra al cuadrado.
    psd_uv2 = volts_to_microvolts(np.sqrt(psd)) ** 2
    return spectrum.freqs, 10 * np.log10(psd_uv2)


def _spread_labels(y: list[float], min_gap: float) -> list[float]:
    """Separa verticalmente etiquetas directas para que no se encimen."""
    order = np.argsort(y)
    placed = np.array(y, dtype=float)
    for prev, cur in zip(order[:-1], order[1:], strict=True):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed.tolist()


def plot_psd_stages(
    stages: list[tuple[str, mne.io.BaseRaw | mne.Epochs]],
    band: tuple[float, float],
    line_freq: float,
    fname: Path,
    title: str,
) -> Path:
    """Espectro por etapa del preprocesamiento, todas con la misma referencia.

    Parameters
    ----------
    stages : list of (str, Raw or Epochs)
        Etiqueta y datos de cada etapa, en orden (p. ej., tras CAR, tras ICA, final).
    band : tuple of float
        Banda del filtro final, marcada con líneas punteadas.
    line_freq : float
        Frecuencia de la red eléctrica, marcada con línea discontinua.
    fname : Path
        PDF de salida.
    title : str
        Título de la figura.

    Returns
    -------
    Path
        Ruta del PDF escrito.
    """
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # PDF idéntico byte a byte
    fc.usar_estilo("publicacion")
    # Márgenes fijos (layout="none"): el pie va dentro de la figura.
    fig, ax = plt.subplots(figsize=fc.tamano_figura("una_columna", alto_mm=85), layout="none")
    fig.subplots_adjust(left=0.15, right=0.78, top=0.92, bottom=0.38)
    colors = fc.categorica(len(stages))
    ends: list[float] = []
    for i, (_label, inst) in enumerate(stages):
        freqs, psd_db = channel_psd_db(inst)
        low, high = np.percentile(psd_db, BAND_PERCENTILES, axis=0)
        style = fc.estilo_serie(i, colors, marcador=False)
        ax.fill_between(freqs, low, high, color=style["color"], alpha=0.18, linewidth=0)
        median = np.median(psd_db, axis=0)
        ax.plot(freqs, median, **style)
        in_band = (freqs >= LABEL_BAND[0]) & (freqs <= LABEL_BAND[1])
        ends.append(float(np.median(median[in_band])))
    y_lo, y_hi = ax.get_ylim()
    gap = 0.08 * (y_hi - y_lo)  # alto aproximado de una etiqueta de 5 a 7 pt
    for y, (label, _), color in zip(_spread_labels(ends, gap), stages, colors, strict=True):
        ax.annotate(
            label,
            xy=(freqs[-1], y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=color,
            annotation_clip=False,
        )
    for f in band:
        ax.axvline(f, color="0.5", linestyle=":", linewidth=0.6)
    ax.axvline(line_freq, color="0.5", linestyle="--", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlim(FMIN, freqs[-1])
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD (dB re 1 µV²/Hz)")
    ax.set_title(title)
    caption = (
        "Welch, ventanas de 4 s. Línea: mediana entre canales; banda: percentiles "
        f"{BAND_PERCENTILES[0]:.0f} a {BAND_PERCENTILES[1]:.0f} entre canales. Punteadas: "
        f"banda final ({band[0]} a {band[1]} Hz); discontinua: red eléctrica "
        f"({line_freq:.0f} Hz). Eje de frecuencia logarítmico por el rango de {FMIN} a "
        f"{FMAX:.0f} Hz y la forma 1/f del espectro."
    )
    fig.text(0.02, 0.02, textwrap.fill(caption, width=60), va="bottom", ha="left")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)  # sin bbox_inches="tight": conserva el ancho de una columna
    plt.close(fig)
    return fname
