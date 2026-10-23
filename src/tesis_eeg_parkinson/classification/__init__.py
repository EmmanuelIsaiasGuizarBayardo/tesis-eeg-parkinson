"""Clasificación con validación por sujeto, según el prerregistro (docs/clasificacion.md)."""

from tesis_eeg_parkinson.classification.data import CONTRASTS, Dataset, load_table, make_dataset
from tesis_eeg_parkinson.classification.evaluate import run_contrast

__all__ = ["CONTRASTS", "Dataset", "load_table", "make_dataset", "run_contrast"]
