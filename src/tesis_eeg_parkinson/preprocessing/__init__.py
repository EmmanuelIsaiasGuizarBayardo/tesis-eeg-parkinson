"""Preprocesamiento de ds002778: migración validada del pipeline de EEGLAB."""

from tesis_eeg_parkinson.preprocessing.config import (
    MATLAB,
    PROFILES,
    V1,
    V1_PREP,
    PreprocessingConfig,
)
from tesis_eeg_parkinson.preprocessing.pipeline import preprocess_recording, run_dataset

__all__ = [
    "MATLAB",
    "PROFILES",
    "V1",
    "V1_PREP",
    "PreprocessingConfig",
    "preprocess_recording",
    "run_dataset",
]
