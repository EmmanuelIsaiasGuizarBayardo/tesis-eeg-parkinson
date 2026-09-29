"""Regresión: épocas del perfil ``legacy`` contra los ``*_clean.set`` de EEGLAB.

La ICA no es idéntica entre EEGLAB y MNE, así que se espera concordancia, no
igualdad. Se compara registro por registro, con las mismas épocas (ambos
segmentan desde la muestra 0, sin recorte ni rechazo) y los mismos canales.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import figuras_cientificas as fc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
GROUPS = ("hc", "pd-off", "pd-on")


def load_matlab_epochs(path: Path) -> mne.Epochs:
    """Lee un ``*_clean.set`` de EEGLAB (MNE convierte de µV a V al leer)."""
    return mne.read_epochs_eeglab(path, verbose="error")


def compare_epochs(ours: mne.Epochs, reference: mne.Epochs) -> dict[str, Any]:
    """Métricas de concordancia entre dos conjuntos de épocas del mismo registro.

    Parameters
    ----------
    ours : mne.Epochs
        Épocas del perfil ``legacy``.
    reference : mne.Epochs
        Épocas de EEGLAB; se reordenan a los canales de ``ours``.

    Returns
    -------
    dict
        Épocas de cada lado y comparadas; correlación de Pearson por canal sobre
        las épocas concatenadas (mediana, mínima y su canal); error RMS relativo
        a la referencia; diferencia de potencia por banda en dB (mediana entre
        canales; positivo = más potencia en MNE).

    Raises
    ------
    ValueError
        Si falta algún canal o las épocas no tienen la misma longitud.
    """
    missing = sorted(set(ours.ch_names) - set(reference.ch_names))
    if missing:
        raise ValueError(f"Canales ausentes en la referencia: {missing}")
    ref = reference.copy().pick(ours.ch_names)
    ref.reorder_channels(ours.ch_names)
    if len(ours.times) != len(ref.times):
        raise ValueError(f"Longitud de época distinta: {len(ours.times)} vs {len(ref.times)}")
    n = min(len(ours), len(ref))
    n_ch = len(ours.ch_names)
    a = ours.get_data()[:n].transpose(1, 0, 2).reshape(n_ch, -1)
    b = ref.get_data()[:n].transpose(1, 0, 2).reshape(n_ch, -1)
    r = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b, strict=True)])
    sfreq = ours.info["sfreq"]
    freqs, pa = signal.welch(a, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
    _, pb = signal.welch(b, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
    result: dict[str, Any] = {
        "n_epochs_ours": len(ours),
        "n_epochs_matlab": len(ref),
        "n_compared": n,
        "r_median": float(np.median(r)),
        "r_min": float(r.min()),
        "r_min_channel": ours.ch_names[int(r.argmin())],
        "rel_rms": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
    }
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        ratio = pa[:, mask].mean(axis=1) / pb[:, mask].mean(axis=1)
        result[f"{band}_db"] = float(np.median(10 * np.log10(ratio)))
    return result


def compare_dataset(ours_root: Path, matlab_root: Path) -> pd.DataFrame:
    """Compara todos los registros con par en ambas carpetas.

    Parameters
    ----------
    ours_root : Path
        ``data/processed/preproc-legacy``.
    matlab_root : Path
        Carpeta con ``sub-X_ses-Y_task-rest_eeg_clean.set``.

    Returns
    -------
    DataFrame
        Una fila por registro; ``error`` indica el par faltante o ilegible.
    """
    rows: list[dict[str, Any]] = []
    for ours_path in sorted(Path(ours_root).rglob("*_desc-clean_epo.fif")):
        stem = ours_path.name.replace("_desc-clean_epo.fif", "")
        subject, session = stem.split("_")[0][4:], stem.split("_")[1][4:]
        group = "hc" if session == "hc" else f"pd-{session}"
        row: dict[str, Any] = {"subject": subject, "session": session, "group": group}
        matlab_path = Path(matlab_root) / f"{stem}_eeg_clean.set"
        try:
            ours = mne.read_epochs(ours_path, verbose="error")
            row.update(compare_epochs(ours, load_matlab_epochs(matlab_path)))
            row["error"] = ""
        except Exception as err:  # se reporta y se sigue con el siguiente par
            row["error"] = f"{type(err).__name__}: {err}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_regression(table: pd.DataFrame, fname: Path, seed: int = 28) -> Path:
    """Correlación mediana y error RMS relativo por registro, con puntos por grupo."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fc.usar_estilo("publicacion")
    size = fc.tamano_figura("doble_columna", alto_mm=60)
    fig, axes = plt.subplots(1, 2, figsize=size, layout="none")  # márgenes fijos para el pie
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    colors = fc.categorica(len(GROUPS))
    rng = np.random.default_rng(seed)  # dispersión horizontal reproducible
    ok = table[table["error"] == ""]
    panels = (("r_median", "Correlación mediana por canal"), ("rel_rms", "Error RMS relativo"))
    for ax, letter, (column, label) in zip(axes, "ab", panels, strict=True):
        for i, group in enumerate(GROUPS):
            values = ok.loc[ok["group"] == group, column].to_numpy()
            x = i + rng.uniform(-0.15, 0.15, len(values))
            style = fc.estilo_serie(i, colors)
            ax.plot(
                x,
                values,
                linestyle="none",
                color=style["color"],
                marker=style.get("marker", "o"),
                markersize=3,
            )
            if len(values):
                ax.hlines(np.median(values), i - 0.3, i + 0.3, color="0.2", linewidth=0.8)
        ax.set_xticks(range(len(GROUPS)), ["HC", "PD off", "PD on"])
        ax.set_ylabel(label)
        fc.letra_panel(ax, letter)
    fig.text(
        0.02, 0.02,
        "Perfil legacy (MNE) contra los *_clean.set de EEGLAB. Un punto por registro; "
        "línea horizontal: mediana del grupo.\nCorrelación de Pearson por canal sobre las "
        "épocas concatenadas; error RMS relativo a EEGLAB.",
        va="bottom", ha="left",
    )  # fmt: skip
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    plt.close(fig)
    return fname
