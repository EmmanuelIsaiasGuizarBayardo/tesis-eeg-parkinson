"""De la tabla de features a los datos de un contraste."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tesis_eeg_parkinson.features.pipeline import META_COLUMNS

# Contraste -> (grupo positivo, grupo negativo). Off vs On es pareado.
CONTRASTS: dict[str, tuple[str, str]] = {
    "off_vs_hc": ("pd-off", "hc"),
    "on_vs_hc": ("pd-on", "hc"),
    "off_vs_on": ("pd-off", "pd-on"),
}


@dataclass(frozen=True)
class Dataset:
    """Épocas de un contraste.

    Attributes
    ----------
    X : ndarray, shape (n_epochs, n_features)
    y : ndarray of int, shape (n_epochs,)
        1 para el grupo positivo.
    subject : ndarray of str
        Unidad de agrupamiento y de permutación (en Off vs On, el paciente).
    recording : ndarray of str
        Unidad de decisión: sujeto y sesión.
    feature_names : list of str
    paired : bool
    """

    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    recording: np.ndarray
    feature_names: list[str]
    paired: bool

    @property
    def n_recordings(self) -> int:
        """Registros (unidades de decisión)."""
        return len(np.unique(self.recording))


def load_table(path: Path) -> pd.DataFrame:
    """Lee ``features.tsv`` conservando vacío el motivo de exclusión."""
    table = pd.read_csv(path, sep="\t", dtype={"subject": str, "session": str})
    table["exclude_reason"] = table["exclude_reason"].fillna("")
    return table


def feature_columns(table: pd.DataFrame, channels: tuple[str, ...] | None = None) -> list[str]:
    """Columnas de features (sin metadatos ni ``qc_``), opcionalmente de unos canales."""
    cols = [c for c in table.columns if c not in META_COLUMNS and not c.startswith("qc_")]
    if channels is not None:
        cols = [c for c in cols if c.rsplit("__", 1)[-1] in channels]
    return cols


def make_dataset(
    table: pd.DataFrame,
    contrast: str,
    include_excluded: bool = False,
    subjects: set[str] | None = None,
    channels: tuple[str, ...] | None = None,
) -> Dataset:
    """Selecciona las épocas de un contraste.

    Parameters
    ----------
    table : DataFrame
        Tabla de features.
    contrast : str
        Clave de `CONTRASTS`.
    include_excluded : bool
        Incluir los registros con ``exclude_primary`` (análisis de sensibilidad).
    subjects : set of str, optional
        Restringir a estos sujetos (p. ej., la intersección entre perfiles).
    channels : tuple of str, optional
        Restringir las features a estos canales.

    Notes
    -----
    En el contraste pareado solo quedan los pacientes con ambas sesiones, para que
    cada paciente sea su propio control.
    """
    positive, negative = CONTRASTS[contrast]
    rows = table[table["group"].isin([positive, negative])]
    if not include_excluded:
        rows = rows[~rows["exclude_primary"].astype(bool)]
    if subjects is not None:
        rows = rows[rows["subject"].isin(subjects)]
    paired = contrast == "off_vs_on"
    if paired:
        sessions = rows.groupby("subject")["group"].nunique()
        rows = rows[rows["subject"].isin(sessions[sessions == 2].index)]
    cols = feature_columns(table, channels)
    return Dataset(
        X=rows[cols].to_numpy(dtype=float),
        y=(rows["group"] == positive).to_numpy(dtype=int),
        subject=rows["subject"].to_numpy(dtype=str),
        recording=(rows["subject"] + "_" + rows["session"]).to_numpy(dtype=str),
        feature_names=cols,
        paired=paired,
    )
