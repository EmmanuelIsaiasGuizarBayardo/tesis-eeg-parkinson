"""Pruebas del análisis exploratorio del exponente aperiódico."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from tesis_eeg_parkinson.interpretation.aperiodic import (
    emg_index,
    exponents,
    paired_summary,
    run_aperiodic_check,
)
from tesis_eeg_parkinson.preprocessing.config import SCALP_CHANNELS

SFREQ = 512.0
mne.set_log_level("error")


def _power_law(n_ch: int, n: int, exponent: float, seed: int) -> np.ndarray:
    """Ruido con espectro 1/f^exponente."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1 / SFREQ)
    amp = np.zeros_like(freqs)
    amp[1:] = freqs[1:] ** (-exponent / 2)
    spec = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, (n_ch, freqs.size)))
    x = np.fft.irfft(spec, n=n, axis=-1)
    return x / x.std(axis=-1, keepdims=True) * 1e-5


def _epochs(exponent: float, seed: int, muscle: float = 0.0) -> mne.EpochsArray:
    data = np.stack([_power_law(32, 5120, exponent, seed + k) for k in range(6)])
    if muscle:  # ruido blanco: potencia de alta frecuencia, como la del músculo
        data += np.random.default_rng(seed).standard_normal(data.shape) * muscle * 1e-6
    epochs = mne.EpochsArray(data, mne.create_info(list(SCALP_CHANNELS), SFREQ, "eeg"))
    epochs.set_montage("colin27_1005")
    return epochs


def test_exponent_recovered_in_both_ranges() -> None:
    from scipy import signal

    x = _power_law(4, 128 * 60, 1.5, 0)[:, ::4]
    freqs, psd = signal.welch(x, fs=128.0, nperseg=256, axis=-1)
    for fit_range in ((2.0, 30.0), (2.0, 20.0)):
        np.testing.assert_allclose(exponents(freqs, psd, fit_range), 1.5, atol=0.25)


def test_emg_index_rises_with_high_frequency_noise() -> None:
    chans = ("T7", "T8")
    assert emg_index(_epochs(1.5, 1, muscle=3.0), chans) > emg_index(_epochs(1.5, 1), chans) + 0.1


def test_paired_summary_on_known_differences() -> None:
    rows = []
    for i in range(8):
        for session, exp in (("off", 1.0 - 0.1 * (i + 1)), ("on", 1.0)):
            rows.append({"subject": f"p{i}", "session": session, "reference": "car",
                         "range": "2-30", "channels": "central", "exponent": exp,
                         "emg_lateral": 0.1 * i if session == "off" else 0.0})  # fmt: skip
    s = paired_summary(pd.DataFrame(rows), n_boot=500).iloc[0]
    assert s["n"] == 8 and s["n_negative"] == 8
    assert s["delta_mean"] == pytest.approx(-0.45)
    assert s["p_wilcoxon"] == pytest.approx(2 / 2**8)  # bilateral exacta, todos negativos
    assert s["rho_emg"] == pytest.approx(-1.0)


def test_run_end_to_end(tmp_path: Path) -> None:
    rows = []
    for root in ("preproc-v1", "preproc-v1-43"):
        for k, subject in enumerate(("pd1", "pd2", "pd3")):
            for session, exp in (("off", 1.2), ("on", 1.6)):
                folder = tmp_path / root / f"sub-{subject}" / f"ses-{session}" / "eeg"
                folder.mkdir(parents=True, exist_ok=True)
                stem = f"sub-{subject}_ses-{session}_task-rest_desc-clean_epo.fif"
                _epochs(exp, 10 * k + (session == "on")).save(folder / stem)
                if root == "preproc-v1":
                    rows.append({"subject": subject, "session": session, "exclude_primary": False})
    pd.DataFrame(rows).to_csv(tmp_path / "preproc-v1" / "qc_summary.tsv", sep="\t", index=False)
    table, summary = run_aperiodic_check(tmp_path, tmp_path / "out")
    assert len(summary) == 8 and (summary["n"] == 3).all()
    assert (summary["delta_mean"] < 0).all()  # el exponente sintético es menor en OFF
    assert (tmp_path / "out" / "aperiodic_check.pdf").exists()
    assert json.dumps(summary.to_dict())  # serializable
