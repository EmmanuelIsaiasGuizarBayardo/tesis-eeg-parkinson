"""Pruebas de la regresión contra EEGLAB, con épocas sintéticas."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from tesis_eeg_parkinson.validation.legacy_matlab import (
    compare_dataset,
    compare_epochs,
    load_matlab_epochs,
    plot_regression,
)

SFREQ = 512.0
CHANNELS = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2"]
mne.set_log_level("error")


def _epochs(seed: int, n_epochs: int = 4) -> mne.EpochsArray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(10 * SFREQ)) / SFREQ
    alpha = np.sin(2 * np.pi * 10 * t)
    data = rng.standard_normal((n_epochs, len(CHANNELS), t.size)) + alpha
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    return mne.EpochsArray(data * 1e-5, info)


def test_identical_epochs_agree_perfectly() -> None:
    ours = _epochs(0)
    result = compare_epochs(ours, ours.copy())
    assert result["r_median"] == pytest.approx(1.0)
    assert result["rel_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result["alpha_db"] == pytest.approx(0.0, abs=1e-9)


def test_channel_order_and_perturbation() -> None:
    ours = _epochs(1)
    reference = ours.copy().reorder_channels(CHANNELS[::-1])  # otro orden, mismos datos
    assert compare_epochs(ours, reference)["r_median"] == pytest.approx(1.0)
    noisy = ours.copy()
    noisy._data = (
        noisy.get_data() + np.random.default_rng(2).standard_normal(noisy._data.shape) * 1e-5
    )
    result = compare_epochs(ours, noisy)
    assert 0.3 < result["r_median"] < 0.95
    assert result["rel_rms"] > 0.3


def test_eeglab_roundtrip_and_dataset_table(tmp_path: Path) -> None:
    """Un .set escrito por EEGLAB-io se lee en voltios y se empareja por nombre."""
    ours = _epochs(3)
    ours_dir = tmp_path / "preproc-legacy" / "sub-hc1" / "ses-hc" / "eeg"
    ours_dir.mkdir(parents=True)
    ours.save(ours_dir / "sub-hc1_ses-hc_task-rest_desc-clean_epo.fif")
    matlab_dir = tmp_path / "legacy-matlab"
    matlab_dir.mkdir()
    set_path = matlab_dir / "sub-hc1_ses-hc_task-rest_eeg_clean.set"
    mne.export.export_epochs(set_path, ours, fmt="eeglab")
    loaded = load_matlab_epochs(set_path)
    np.testing.assert_allclose(loaded.get_data(), ours.get_data(), rtol=1e-5, atol=1e-12)
    table = compare_dataset(tmp_path / "preproc-legacy", matlab_dir)
    assert table.loc[0, "error"] == "" and table.loc[0, "r_median"] == pytest.approx(1.0)
    missing = pd.concat([table, pd.DataFrame([{"group": "pd-off", "error": "sin par"}])])
    assert plot_regression(missing, tmp_path / "fig.pdf").exists()
