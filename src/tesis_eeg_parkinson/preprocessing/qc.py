"""Figuras de control de calidad con la biblioteca `figuras_cientificas`."""

from __future__ import annotations

from pathlib import Path

import figuras_cientificas as fc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from tesis_eeg_parkinson.units import volts_to_microvolts


def _mean_psd_db(inst: mne.io.BaseRaw | mne.Epochs, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD de Welch (ventanas de 4 s) promediada sobre canales, en dB re 1 µV²/Hz."""
    sfreq = inst.info["sfreq"]
    n_fft = int(4 * sfreq)
    fmax = min(fmax, sfreq / 2 - 1)
    kwargs = {"method": "welch", "fmin": 0.25, "fmax": fmax, "verbose": "error"}
    if isinstance(inst, mne.io.BaseRaw):
        spectrum = inst.compute_psd(n_fft=n_fft, **kwargs)
        psd = spectrum.get_data()
    else:
        spectrum = inst.compute_psd(n_fft=min(n_fft, len(inst.times)), **kwargs)
        psd = spectrum.get_data().mean(axis=0)
    # V²/Hz -> µV²/Hz: el factor de amplitud entra al cuadrado.
    psd_uv2 = volts_to_microvolts(np.sqrt(psd)) ** 2
    return spectrum.freqs, 10 * np.log10(psd_uv2.mean(axis=0))


def plot_psd_before_after(
    raw_before: mne.io.BaseRaw,
    epochs_after: mne.Epochs,
    band: tuple[float, float],
    line_freq: float | None,
    fname: Path,
    title: str,
) -> Path:
    """Espectro medio antes y después del preprocesamiento, en PDF vectorial.

    Sirve para verificar el filtro final y para detectar registros con otra
    procedencia (p. ej., sin el pico de la línea eléctrica).
    """
    fc.usar_estilo("publicacion")
    fig, ax = plt.subplots(figsize=fc.tamano_figura("una_columna", alto_mm=60))
    colors = fc.categorica(2)
    for i, (inst, label) in enumerate(((raw_before, "Cargado"), (epochs_after, "Preprocesado"))):
        freqs, db = _mean_psd_db(inst, fmax=120.0)
        ax.plot(freqs, db, label=label, **fc.estilo_serie(i, colors, marcador=False))
    for f in band:
        ax.axvline(f, color="0.5", linestyle=":", linewidth=0.6)
    if line_freq:
        ax.axvline(line_freq, color="0.5", linestyle="--", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD (dB re 1 µV²/Hz)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)  # sin bbox_inches="tight": conserva el ancho de una columna
    plt.close(fig)
    return fname
