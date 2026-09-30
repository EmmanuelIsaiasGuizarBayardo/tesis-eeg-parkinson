"""Preprocesamiento de ds002778: migración validada del pipeline de EEGLAB."""

from tesis_eeg_parkinson.preprocessing.config import (
    LEGACY,
    PROFILES,
    V2,
    V2_PREP,
    PreprocessingConfig,
)
from tesis_eeg_parkinson.preprocessing.pipeline import preprocess_recording, run_dataset

__all__ = [
    "LEGACY",
    "PROFILES",
    "V2",
    "V2_PREP",
    "PreprocessingConfig",
    "preprocess_recording",
    "run_dataset",
]
