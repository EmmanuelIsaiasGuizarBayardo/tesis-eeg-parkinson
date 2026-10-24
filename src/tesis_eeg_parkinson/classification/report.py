"""Reúne los análisis de clasificación en una tabla y dos figuras.

- ``resumen.tsv``: una fila por análisis y contraste.
- ``resultados.pdf``: exactitud balanceada por registro con IC 95% y azar empírico.
- ``fuga.pdf``: exactitud por época con validación por sujeto y por épocas (misma
  unidad), para mostrar la inflación en cada análisis.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import figuras_cientificas as fc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carpeta del análisis -> (etiqueta, familia). El orden es el de las figuras.
ANALYSES: dict[str, tuple[str, str]] = {
    "v1_v1_lr": ("Principal (V1, regresión logística)", "Principal"),
    "v1_v1_slda": ("LDA con contracción", "Modelo"),
    "v1_v1_rf": ("Bosque aleatorio", "Modelo"),
    "aljalal_v1_lr": ("Features de Aljalal et al.", "Features"),
    "v1_v1-prep_lr": ("Interpolación de PREP", "Preprocesamiento"),
    "v1_v1-43_lr": ("Banda de 43 Hz", "Preprocesamiento"),
    "v1_v1_lr_incl-excluded": ("Con pd6 y pd16 ON", "Preprocesamiento"),
    "v1_v1_lr_inter-epocx": ("BioSemi, 32 canales", "Transferencia"),
    "v1_v1_lr_inter-epocx_ch-emotiv": ("BioSemi, 14 canales del EPOC X", "Transferencia"),
    "v1_epocx_lr_inter-v1": ("EPOC X emulado", "Transferencia"),
}
FAMILIES = ("Principal", "Modelo", "Features", "Preprocesamiento", "Transferencia")
CONTRAST_TITLES = {"off_vs_hc": "Off vs HC", "on_vs_hc": "On vs HC", "off_vs_on": "Off vs On"}


def collect(results_root: Path) -> pd.DataFrame:
    """Lee cada ``summary.tsv`` y agrega etiqueta, familia y orden."""
    frames = []
    for path in sorted(Path(results_root).glob("*/summary.tsv")):
        name = path.parent.name
        label, family = ANALYSES.get(name, (name, "Otro"))
        order = list(ANALYSES).index(name) if name in ANALYSES else len(ANALYSES)
        frames.append(pd.read_csv(path, sep="\t").assign(analysis=name, label=label,
                                                         family=family, order=order))  # fmt: skip
    if not frames:
        raise FileNotFoundError(f"No hay summary.tsv en {results_root}")
    return pd.concat(frames, ignore_index=True).sort_values(["order", "contrast"])


def _forest(table: pd.DataFrame, value: str, fname: Path, caption: str, xlabel: str,
            paired_value: str | None = None) -> Path:  # fmt: skip
    """Un panel por contraste; una fila por análisis."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fc.usar_estilo("publicacion")
    labels = table.drop_duplicates("analysis").sort_values("order")
    rows = {a: i for i, a in enumerate(labels["analysis"])}
    size = fc.tamano_figura("doble_columna", alto_mm=40 + 6 * len(rows))
    fig, axes = plt.subplots(1, 3, figsize=size, sharey=True, layout="none")
    fig.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.30, wspace=0.12)
    colors = fc.categorica(len(FAMILIES) if paired_value is None else 2)
    for ax, letter, contrast in zip(axes, "abc", CONTRAST_TITLES, strict=True):
        sub = table[table["contrast"] == contrast]
        for _, r in sub.iterrows():
            y = rows[r["analysis"]]
            if paired_value is None:
                style = fc.estilo_serie(FAMILIES.index(r["family"]) if r["family"] in FAMILIES
                                        else 0, colors)  # fmt: skip
                err = [[r[value] - r["bacc_ci_low"]], [r["bacc_ci_high"] - r[value]]]
                ax.errorbar(r[value], y, xerr=err, color=style["color"], marker=style["marker"],
                            markersize=3.5, linewidth=0.8, capsize=0)  # fmt: skip
                if pd.notna(r.get("chance_p95", np.nan)):
                    ax.plot(r["chance_p95"], y, marker="|", color="black", markersize=7)
            else:
                a, b = r[value], r[paired_value]
                ax.plot([a, b], [y, y], color="0.75", linewidth=0.8, zorder=1)
                for i, v in enumerate((a, b)):
                    style = fc.estilo_serie(i, colors)
                    ax.plot(v, y, linestyle="none", color=style["color"],
                            marker=style["marker"], markersize=3.5, zorder=2)  # fmt: skip
        ax.set_title(CONTRAST_TITLES[contrast])
        ax.set_xlim(0.2, 1.02)
        ax.set_xlabel(xlabel)
        fc.letra_panel(ax, letter)
    axes[0].set_yticks(list(rows.values()), labels["label"].tolist())
    axes[0].invert_yaxis()
    fig.text(0.02, 0.02, textwrap.fill(caption, width=128), va="bottom", ha="left")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)  # sin bbox_inches="tight"
    plt.close(fig)
    return fname


def write_report(results_root: Path) -> dict[str, Path]:
    """Escribe ``resumen.tsv``, ``resultados.pdf`` y ``fuga.pdf`` en ``results_root``."""
    table = collect(results_root)
    root = Path(results_root)
    table.drop(columns="order").to_csv(root / "resumen.tsv", sep="\t", index=False)
    marker_names = {"o": "círculo", "s": "cuadrado", "^": "triángulo", "D": "rombo",
                    "v": "triángulo invertido"}  # fmt: skip
    colors = fc.categorica(len(FAMILIES))
    families = ", ".join(
        f"{fam} ({marker_names.get(fc.estilo_serie(i, colors)['marker'], '?')})"
        for i, fam in enumerate(FAMILIES)
    )
    results = _forest(
        table, "bacc", root / "resultados.pdf",
        "Exactitud balanceada por registro: probabilidades de sus épocas promediadas en 20 "
        "repeticiones de validación por sujeto. Barra: IC 95% por bootstrap de sujetos. Marca "
        "vertical negra: percentil 95 de la nula por permutación (azar empírico); los modelos de "
        "sensibilidad no tienen permutaciones. Solo el análisis principal es confirmatorio "
        f"(Holm sobre tres contrastes). Familias por marcador: {families}.",
        "Exactitud balanceada (registro)",
    )  # fmt: skip
    leak = _forest(
        table, "epoch_bacc_subject_cv", root / "fuga.pdf",
        "Exactitud balanceada por época con el mismo modelo y las mismas épocas. Círculo: "
        "validación por sujeto (ningún sujeto aparece en entrenamiento y prueba). Cuadrado: "
        "validación por épocas, que reparte épocas de un mismo sujeto entre entrenamiento y "
        "prueba. La distancia entre ambos es la inflación por fuga; no es desempeño.",
        "Exactitud balanceada (época)",
        paired_value="epoch_bacc_epoch_cv",
    )  # fmt: skip
    return {"resumen": root / "resumen.tsv", "resultados": results, "fuga": leak}
