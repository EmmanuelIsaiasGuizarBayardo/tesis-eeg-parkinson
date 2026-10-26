"""Utilidades comunes: estilo, tamaño, grupos y guardado (PDF vectorial y PNG de 300 dpi)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import figuras_cientificas as fc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Tres estilos con el mismo código:
# - tesis: todas las figuras a 160 mm (ancho de texto de una hoja carta), con texto de
#   9 a 10.5 pt y letras "A)" de 12 pt, como las figuras del artículo del CNIB;
# - publicacion: 89 o 183 mm con texto de 5 a 7 pt (guía de Nature);
# - presentacion: publicación escalada 1.8 veces, con el texto de 14 pt del estilo.
STYLES = ("tesis", "publicacion", "presentacion")
# Carpetas de salida en inglés, como el resto de results/.
STYLE_FOLDERS = {"tesis": "thesis", "publicacion": "publication", "presentacion": "presentation"}
SCALE = {"publicacion": 1.0, "presentacion": 1.8}
TESIS_WIDTH_MM = 160.0
TESIS_RC = {
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.3,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
}
GROUPS = ("hc", "pd-off", "pd-on")
GROUP_LABELS = {"hc": "HC", "pd-off": "PD off", "pd-on": "PD on"}
LSB_UV = 0.03125  # paso de cuantización de BioSemi ActiveTwo: 31.25 nV


@dataclass(frozen=True)
class Figura:
    """Una figura terminada y su leyenda para el documento."""

    nombre: str
    fig: plt.Figure
    leyenda: str


def setup(estilo: str) -> None:
    """Estilo de la biblioteca y fecha fija en el PDF (byte a byte reproducible)."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    if estilo == "tesis":
        fc.usar_estilo("publicacion")
        plt.rcParams.update(TESIS_RC)
    else:
        fc.usar_estilo(estilo)


def size(
    width_mm: float, height_mm: float, estilo: str, tesis_mm: float | None = None
) -> tuple[float, float]:
    """Tamaño en pulgadas.

    En tesis todas miden 160 mm de ancho y ``tesis_mm`` de alto (por omisión, 1.3 veces
    el alto de publicación); en publicación, 89 o 183 mm, y en presentación, 1.8 veces.
    """
    if estilo == "tesis":
        return TESIS_WIDTH_MM / 25.4, (tesis_mm or 1.3 * height_mm) / 25.4
    k = SCALE[estilo]
    return width_mm * k / 25.4, height_mm * k / 25.4


def panel(ax, letra: str, estilo: str) -> None:
    """Letra de panel: "A)" de 12 pt en tesis, como el artículo; minúscula de Nature si no."""
    if estilo == "tesis":
        fc.letra_panel(ax, letra, estilo="parentesis").set_fontsize(12)
    else:
        fc.letra_panel(ax, letra)


def text_pt(factor: float) -> float:
    """Tamaño de texto relativo al texto base del estilo activo."""
    return plt.rcParams["font.size"] * factor


def group_of(session: str) -> str:
    """Grupo a partir de la sesión BIDS."""
    return "hc" if session == "hc" else f"pd-{session}"


def group_styles() -> dict[str, dict]:
    """Color y marcador por grupo, iguales en todas las figuras."""
    colors = fc.categorica(len(GROUPS))
    return {g: fc.estilo_serie(i, colors) for i, g in enumerate(GROUPS)}


def strip(
    ax,
    values_by_group: dict[str, np.ndarray],
    hollow: dict[str, np.ndarray] | None = None,
    seed: int = 0,
) -> None:
    """Un punto por registro en cada grupo, con su mediana como barra horizontal."""
    rng = np.random.default_rng(seed)
    styles = group_styles()
    for i, g in enumerate(GROUPS):
        v = np.asarray(values_by_group.get(g, []), dtype=float)
        if v.size == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, v.size)
        empty = hollow.get(g, np.zeros(v.size, bool)) if hollow else np.zeros(v.size, bool)
        st = styles[g]
        ax.plot(
            x[~empty],
            v[~empty],
            linestyle="none",
            marker=st["marker"],
            color=st["color"],
            markersize=3.2,
            alpha=0.85,
        )
        if empty.any():
            ax.plot(
                x[empty],
                v[empty],
                linestyle="none",
                marker=st["marker"],
                color=st["color"],
                markerfacecolor="none",
                markersize=3.6,
            )
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", linewidth=0.9)
    ax.set_xticks(range(len(GROUPS)), [GROUP_LABELS[g] for g in GROUPS])
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)


def save(figura: Figura, out_dir: Path) -> list[Path]:
    """PDF vectorial (sin ``bbox_inches="tight"``) y PNG de 300 dpi para carteles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{figura.nombre}.pdf", out_dir / f"{figura.nombre}.png"]
    figura.fig.savefig(paths[0])
    figura.fig.savefig(paths[1], dpi=300)
    plt.close(figura.fig)
    return paths
