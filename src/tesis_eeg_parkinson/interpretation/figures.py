"""Topografías de los patrones de Haufe y resumen de su magnitud por feature."""

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
import pandas as pd
from matplotlib.cm import ScalarMappable

from tesis_eeg_parkinson.interpretation.patterns import STABLE

# (métrica, señal, título, familia), en el orden del resumen.
FEATURES: tuple[tuple[str, str, str, str], ...] = (
    ("relpow", "cA4", "Pot. relativa 0–4 Hz", "Espectro"),
    ("relpow", "cD4", "Pot. relativa 4–8 Hz", "Espectro"),
    ("relpow", "cD3", "Pot. relativa 8–16 Hz", "Espectro"),
    ("relpow", "cD2", "Pot. relativa 16–32 Hz", "Espectro"),
    ("aperiodic_exponent", "broadband", "Exponente aperiódico", "Espectro"),
    ("alpha_cog", "broadband", "Centro de alfa", "Espectro"),
    ("wentropy", "cA4", "Ent. wavelet 0–4 Hz", "Entropía wavelet"),
    ("wentropy", "cD4", "Ent. wavelet 4–8 Hz", "Entropía wavelet"),
    ("wentropy", "cD3", "Ent. wavelet 8–16 Hz", "Entropía wavelet"),
    ("wentropy", "cD2", "Ent. wavelet 16–32 Hz", "Entropía wavelet"),
    ("sampen", "broadband", "SampEn", "Orden temporal"),
    ("permen_tau1", "broadband", "PermEn τ = 1", "Orden temporal"),
    ("permen_tau2", "broadband", "PermEn τ = 2", "Orden temporal"),
    ("permen_tau4", "broadband", "PermEn τ = 4", "Orden temporal"),
    ("t3_tau1", "broadband", "T3 τ = 1", "Forma de onda"),
    ("t3_tau2", "broadband", "T3 τ = 2", "Forma de onda"),
    ("t3_tau4", "broadband", "T3 τ = 4", "Forma de onda"),
)
FAMILIES = ("Espectro", "Entropía wavelet", "Orden temporal", "Forma de onda")
# Celdas de la rejilla 3 × 6; la celda libre lleva la barra de color.
GRID = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17)
COLORBAR_CELL = 11
POSITIVE = {"off_vs_hc": "PD off", "on_vs_hc": "PD on", "off_vs_on": "PD off"}
TITLES = {"off_vs_hc": "Off vs HC", "on_vs_hc": "On vs HC", "off_vs_on": "Off vs On"}


def _info(channels: list[str]) -> mne.Info:
    info = mne.create_info(channels, 128.0, "eeg")
    info.set_montage("colin27_1005")
    return info


def _caption(fig: plt.Figure, text: str, width: int = 128) -> None:
    fig.text(0.02, 0.02, textwrap.fill(text, width=width), va="bottom", ha="left")


def plot_contrast_topomaps(long: pd.DataFrame, contrast: str, fname: Path) -> Path:
    """Las 17 topografías de un contraste, con una sola escala divergente centrada en 0."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fc.usar_estilo("publicacion")
    sub = long[long["contrast"] == contrast]
    channels = list(dict.fromkeys(sub["channel"]))
    info = _info(channels)
    cmap, norm = fc.mapa_para(sub["pattern"].to_numpy(), centro=0)
    fig, axes = plt.subplots(3, 6, figsize=fc.tamano_figura("doble_columna", alto_mm=120),
                             layout="none")  # fmt: skip
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.24, wspace=0.15, hspace=0.45)
    flat = axes.ravel()
    for cell, (metric, signal, title, _family) in zip(GRID, FEATURES, strict=True):
        ax = flat[cell]
        rows = sub[(sub["metric"] == metric) & (sub["signal"] == signal)].set_index("channel")
        rows = rows.reindex(channels)
        mne.viz.plot_topomap(
            rows["pattern"].to_numpy(), info, axes=ax, cmap=cmap, cnorm=norm, contours=4,
            outlines="head", sensors=False, show=False,
            mask=(rows["stability"] >= STABLE).to_numpy(),
            mask_params={"marker": "o", "markerfacecolor": "w", "markeredgecolor": "k",
                         "linewidth": 0, "markersize": 2.5},
        )  # fmt: skip
        ax.set_title(title, fontsize=6)
    cax = flat[COLORBAR_CELL]
    cax.set_axis_off()
    bar_ax = cax.inset_axes((0.1, 0.35, 0.8, 0.12))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax, orientation="horizontal",
                 extend=fc.extension(sub["pattern"].to_numpy(), norm),
                 label=f"Patrón (u. a.)\n+ mayor en {POSITIVE[contrast]}")  # fmt: skip
    fig.suptitle(TITLES[contrast])
    _caption(fig, (
        "Patrones de activación (Haufe et al., 2014) del modelo principal ajustado con todos "
        "los registros, sobre features estandarizadas. Positivo (cálido): la feature es mayor en "
        f"{POSITIVE[contrast]}. Misma escala en los 17 mapas, saturada en el percentil 98. "
        f"Círculos blancos: canales cuyo signo se mantiene en al menos {STABLE:.0%} de los "
        "remuestreos de sujetos (bootstrap; exploratorio, fuera del prerregistro). Contornos "
        "continuos: positivo; discontinuos: negativo. Nariz arriba."
    ))  # fmt: skip
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    plt.close(fig)
    return fname


def plot_magnitude(long: pd.DataFrame, fname: Path) -> Path:
    """Media de |patrón| entre canales por feature, y canales con signo estable."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fc.usar_estilo("publicacion")
    colors = fc.categorica(len(FAMILIES))
    fig, axes = plt.subplots(1, 3, figsize=fc.tamano_figura("doble_columna", alto_mm=95),
                             sharey=True, layout="none")  # fmt: skip
    fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.27, wspace=0.35)
    ys = np.arange(len(FEATURES))
    for ax, letter, contrast in zip(axes, "abc", TITLES, strict=True):
        sub = long[long["contrast"] == contrast]
        for y, (metric, signal, _title, family) in zip(ys, FEATURES, strict=True):
            rows = sub[(sub["metric"] == metric) & (sub["signal"] == signal)]
            style = fc.estilo_serie(FAMILIES.index(family), colors)
            ax.plot(rows["pattern"].abs().mean(), y, linestyle="none", color=style["color"],
                    marker=style["marker"], markersize=3.5)  # fmt: skip
            n_stable = int((rows["stability"] >= STABLE).sum())
            where = ("axes fraction", "data")
            ax.annotate(f"{n_stable}/{len(rows)}", xy=(1.0, y), xycoords=where, xytext=(2, 0),
                        textcoords="offset points", va="center", fontsize=5)  # fmt: skip
        ax.set_title(TITLES[contrast])
        ax.set_xlabel("Media de |patrón| (u. a.)")
        ax.set_xlim(left=0)
        fc.letra_panel(ax, letter)
    axes[0].set_yticks(ys, [f[2] for f in FEATURES])
    axes[0].invert_yaxis()
    marker_names = {"o": "círculo", "s": "cuadrado", "^": "triángulo", "D": "rombo"}
    families = ", ".join(
        f"{fam} ({marker_names.get(fc.estilo_serie(i, colors)['marker'], '?')})"
        for i, fam in enumerate(FAMILIES)
    )
    _caption(fig, (
        "Media del valor absoluto del patrón de Haufe entre canales, por feature y contraste; "
        "comparable dentro de cada panel. A la derecha: canales cuyo signo se mantiene en al "
        f"menos {STABLE:.0%} de los remuestreos de sujetos (exploratorio). Familias por marcador: "
        f"{families}."
    ))  # fmt: skip
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    plt.close(fig)
    return fname
