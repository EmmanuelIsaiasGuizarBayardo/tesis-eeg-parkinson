"""Canales malos: detección con la referencia robusta de PREP e interpolación esférica."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import mne
from pyprep import Reference

from tesis_eeg_parkinson.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True)
class BadChannelReport:
    """Resultado de la detección, para el JSON de QC.

    Attributes
    ----------
    prep_bads : list of str
        Todo lo que marca PREP: lo detectado con referencia robusta más lo
        inutilizable según pyprep (plano, NaN o SNR bajo).
    unusable : list of str
        Solo canales planos o con NaN, detectados sin referenciar: fallas del
        registro que no dependen de la referencia ni de artefactos fisiológicos.
    by_criterion : dict of str to list of str
        Canales por criterio, detectados sobre los datos con referencia robusta.
    by_criterion_unreferenced : dict of str to list of str
        Lo mismo sobre los datos sin referenciar. Solo es diagnóstico: muestra
        cuánto de la detección dependía de la referencia al CMS.
    correlation_bad_fraction : dict of str to float
        Fracción de ventanas de 1 s con correlación máxima < 0.4, por canal.
        Un canal se marca si supera 0.01; en 3 min, eso son unas 2 ventanas.
    """

    prep_bads: list[str]
    unusable: list[str]
    by_criterion: dict[str, list[str]]
    by_criterion_unreferenced: dict[str, list[str]]
    correlation_bad_fraction: dict[str, float]


@contextmanager
def _quiet_pyprep() -> Iterator[None]:
    """Silencia los avisos de pyprep (p. ej., el de RANSAC) y la salida de MNE."""
    pyprep_logger = logging.getLogger("pyprep")
    previous = pyprep_logger.level
    pyprep_logger.setLevel(logging.ERROR)
    try:
        with mne.use_log_level("error"):
            yield
    finally:
        pyprep_logger.setLevel(previous)


def _nonempty(criteria: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key: sorted(chs) for key, chs in criteria.items() if chs}


def detect_bad_channels(raw: mne.io.BaseRaw, cfg: PreprocessingConfig) -> BadChannelReport:
    """Detecta canales malos como PREP: contra una referencia robusta.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Registro sin referencia promedio y, en ``v2``, ya sin ruido de línea
        (PREP quita la línea antes de referenciar). No se modifica.
    cfg : PreprocessingConfig
        Perfil; aporta semilla e iteraciones.

    Returns
    -------
    BadChannelReport

    Notes
    -----
    Usa ``pyprep.Reference`` sin RANSAC y sin interpolar: la interpolación y la
    CAR se hacen después, explícitas, en el pipeline (Bigdely-Shamlo et al., 2015).
    """
    chs = list(raw.ch_names)
    with _quiet_pyprep():
        ref = Reference(
            raw.copy(),
            params={"ref_chs": chs, "reref_chs": chs},
            ransac=False,
            random_state=cfg.seed,
        )
        ref.perform_reference(max_iterations=cfg.prep_max_iterations, interpolate_bads=False)
    corr = ref._extra_info["interpolated"]["bad_by_correlation"]["bad_window_fractions"]
    original = ref.noisy_channels_original
    return BadChannelReport(
        prep_bads=sorted(ref.raw.info["bads"]),
        unusable=sorted(set(original.get("bad_by_nan", [])) | set(original.get("bad_by_flat", []))),
        by_criterion=_nonempty(ref.noisy_channels_before_interpolation),
        by_criterion_unreferenced=_nonempty(ref.noisy_channels_original),
        correlation_bad_fraction={ch: round(float(f), 4) for ch, f in zip(chs, corr, strict=True)},
    )


def channels_to_interpolate(report: BadChannelReport, cfg: PreprocessingConfig) -> list[str]:
    """Canales que se interpolan según el perfil.

    En ``v2`` solo los inutilizables. En ds002778, PREP marca sobre todo los
    canales laterales y frontales (T8, FC6, F8, FC5, T7, F7, Fp1, Fp2) por
    correlación y espectro: artefacto muscular y ocular que corresponde a la ICA,
    no fallas de electrodo. ``v2-prep`` interpola todo, como sensibilidad.
    """
    return list(report.prep_bads) if cfg.interpolate == "prep" else list(report.unusable)


def interpolate_bad_channels(raw: mne.io.BaseRaw, bads: list[str]) -> mne.io.BaseRaw:
    """Interpola por splines esféricos los canales indicados, en el lugar.

    Cada canal interpolado es combinación lineal de los demás: reduce el rango
    de los datos en uno, lo que después fija el número de componentes de ICA.
    """
    if bads:
        raw.info["bads"] = list(bads)
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="error")
    return raw
