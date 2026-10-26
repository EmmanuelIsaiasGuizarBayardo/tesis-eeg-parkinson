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

# En presentación (diapositivas y carteles) todo crece 1.8 veces, junto con el texto
# de 14 pt del estilo; así las proporciones de la versión de publicación se conservan.
SCALE = {"publicacion": 1.0, "presentacion": 1.8}
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
    fc.usar_estilo(estilo)


def size(width_mm: float, height_mm: float, estilo: str) -> tuple[float, float]:
    """Tamaño en pulgadas; en publicación, el ancho es 89 o 183 mm (guía de Nature)."""
    k = SCALE[estilo]
    return width_mm * k / 25.4, height_mm * k / 25.4


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
