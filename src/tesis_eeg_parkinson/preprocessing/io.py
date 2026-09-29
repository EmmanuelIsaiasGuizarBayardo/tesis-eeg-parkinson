"""Lectura de ds002778 desde BIDS: canales, montaje, recorte y procedencia."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from tesis_eeg_parkinson.preprocessing.config import (
    REST_ONSET_VALUE,
    SCALP_CHANNELS,
    PreprocessingConfig,
)


@dataclass(frozen=True)
class RecordingInfo:
    """Metadatos de un registro que se guardan en su JSON de QC.

    Attributes
    ----------
    subject, session : str
        Entidades BIDS.
    duration_s : float
        Duración del registro tal como se cargó, antes de recortar.
    rest_onset_s : float or None
        Inicio del reposo según events.tsv; None si el evento no existe.
    cropped_at_s : float
        Instante desde el que se conservó la señal (0 si no se recortó).
    line_freq_sidecar : float or None
        PowerLineFrequency declarada en el eeg.json.
    curator_notes : str
        Notas de participants.tsv para ese sujeto ("" si no hay).
    curator_preprocessed : bool
        True si los curadores declaran que la sesión no es dato crudo.
    """

    subject: str
    session: str
    duration_s: float
    rest_onset_s: float | None
    cropped_at_s: float
    line_freq_sidecar: float | None
    curator_notes: str
    curator_preprocessed: bool


def find_recordings(bids_root: Path, task: str = "rest", extension: str = ".bdf") -> list[BIDSPath]:
    """Lista los registros EEG del dataset, ordenados por sujeto y sesión.

    Parameters
    ----------
    bids_root : Path
        Raíz BIDS del dataset.
    task : str
        Tarea BIDS.
    extension : str
        Formato de los registros (".bdf" en ds002778).

    Returns
    -------
    list of BIDSPath
    """
    pattern = BIDSPath(root=bids_root, task=task, datatype="eeg", suffix="eeg", extension=extension)
    return sorted(pattern.match(), key=lambda p: (p.subject, p.session))


def read_rest_onset(bids_path: BIDSPath) -> float | None:
    """Lee el inicio del reposo directamente del events.tsv.

    Se lee el TSV y no las anotaciones de MNE porque ds002778 no trae la columna
    ``trial_type`` y el valor del evento es lo único que lo identifica.

    Returns
    -------
    float or None
        Onset en segundos del primer evento con valor `REST_ONSET_VALUE`.
    """
    events_path = bids_path.copy().update(suffix="events", extension=".tsv").fpath
    if not Path(events_path).exists():
        return None
    events = pd.read_csv(events_path, sep="\t", dtype={"value": str})
    rows = events[events["value"].astype(str).str.strip() == REST_ONSET_VALUE]
    return float(rows["onset"].iloc[0]) if len(rows) else None


def read_curator_notes(bids_root: Path, subject: str) -> str:
    """Devuelve la columna ``notes`` de participants.tsv para un sujeto."""
    table = pd.read_csv(Path(bids_root) / "participants.tsv", sep="\t", dtype=str)
    row = table[table["participant_id"] == f"sub-{subject}"]
    if row.empty or "notes" not in row:
        return ""
    note = str(row["notes"].iloc[0])
    return "" if note in ("n/a", "nan") else note


def load_raw(bids_path: BIDSPath, cfg: PreprocessingConfig) -> tuple[mne.io.BaseRaw, RecordingInfo]:
    """Carga un registro con los 32 canales de cuero cabelludo y su montaje.

    Selecciona por nombre (los EXG1-8 vienen tipados como EEG en channels.tsv)
    y recorta desde el inicio del reposo si el perfil lo pide.

    Parameters
    ----------
    bids_path : BIDSPath
        Registro a cargar.
    cfg : PreprocessingConfig
        Perfil de preprocesamiento.

    Returns
    -------
    raw : mne.io.BaseRaw
        Registro en memoria, en voltios.
    info : RecordingInfo
        Metadatos para el QC.

    Raises
    ------
    ValueError
        Si falta alguno de los 32 canales esperados.
    """
    raw = read_raw_bids(bids_path, verbose="error")
    raw.load_data(verbose="error")
    missing = [ch for ch in SCALP_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{bids_path.basename}: faltan canales {missing}")
    raw.pick(list(SCALP_CHANNELS))
    raw.reorder_channels(list(SCALP_CHANNELS))
    raw.set_channel_types(dict.fromkeys(SCALP_CHANNELS, "eeg"))
    raw.set_montage(cfg.montage, match_case=False, on_missing="raise")

    duration = raw.times[-1] + 1 / raw.info["sfreq"]
    onset = read_rest_onset(bids_path)
    cropped_at = 0.0
    if cfg.crop_to_rest_onset and onset is not None:
        raw.crop(tmin=onset)
        cropped_at = onset

    notes = read_curator_notes(bids_path.root, bids_path.subject)
    info = RecordingInfo(
        subject=bids_path.subject,
        session=bids_path.session,
        duration_s=float(duration),
        rest_onset_s=onset,
        cropped_at_s=float(cropped_at),
        line_freq_sidecar=raw.info.get("line_freq"),
        curator_notes=notes,
        curator_preprocessed=("preprocessed" in notes.lower() and bids_path.session == "on"),
    )
    return raw, info
