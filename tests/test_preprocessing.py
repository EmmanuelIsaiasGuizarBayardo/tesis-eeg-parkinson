"""Pruebas del preprocesamiento con señales sintéticas.

Ningún dato de personas entra al repositorio: todo se genera aquí, con fuentes
mezcladas para que los canales correlacionen como en un EEG real (los criterios
de PREP dependen de esa correlación).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, write_raw_bids
from scipy import signal

from tesis_eeg_parkinson.preprocessing.channels import (
    channels_to_interpolate,
    detect_bad_channels,
    interpolate_bad_channels,
)
from tesis_eeg_parkinson.preprocessing.config import LEGACY, SCALP_CHANNELS, V2, V2_PREP
from tesis_eeg_parkinson.preprocessing.epochs import make_epochs
from tesis_eeg_parkinson.preprocessing.filters import (
    apply_final_filter,
    apply_notch,
    final_filter_coefficients,
)
from tesis_eeg_parkinson.preprocessing.ica import select_artifact_components
from tesis_eeg_parkinson.preprocessing.io import find_recordings, load_raw, read_rest_onset
from tesis_eeg_parkinson.preprocessing.pipeline import assert_outside_raw, preprocess_recording
from tesis_eeg_parkinson.units import microvolts_to_volts, volts_to_microvolts

SFREQ = 512.0
mne.set_log_level("error")


def _synthetic_eeg(
    duration_s: float, seed: int = 0, extra: tuple[str, ...] = (), scale: float = 1e-5
) -> mne.io.RawArray:
    """EEG sintético: fuentes oscilatorias y de ruido rosa mezcladas en 32 canales."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SFREQ)
    t = np.arange(n) / SFREQ
    pink = np.cumsum(rng.standard_normal((2, n)), axis=1)
    pink -= pink.mean(axis=1, keepdims=True)
    pink /= pink.std(axis=1, keepdims=True)
    sources = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)),
            np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi)),
            np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi)),
            rng.standard_normal(n),
            pink,
        ]
    )
    mixing = rng.uniform(0.5, 1.5, (len(SCALP_CHANNELS), sources.shape[0]))
    data = mixing @ sources + 0.2 * rng.standard_normal((len(SCALP_CHANNELS), n))
    data *= scale  # 1e-5 V: desviación estándar de ~20 µV por canal
    names = list(SCALP_CHANNELS) + list(extra)
    if extra:
        data = np.vstack([data, 1e-5 * rng.standard_normal((len(extra), n))])
    info = mne.create_info(names, SFREQ, "eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage("colin27_1005", on_missing="ignore")
    raw.info["line_freq"] = 60.0
    return raw


@pytest.fixture()
def bids_dataset(tmp_path: Path) -> tuple[Path, BIDSPath]:
    """Dataset BIDS sintético con el formato de eventos de ds002778."""
    root = tmp_path / "bids"
    raw = _synthetic_eeg(45.0, extra=tuple(f"EXG{i}" for i in range(1, 9)))
    bids_path = BIDSPath(subject="pd6", session="on", task="rest", datatype="eeg", root=root)
    write_raw_bids(raw, bids_path, format="BrainVision", allow_preload=True, overwrite=True)
    events = pd.DataFrame(
        {"onset": [0.0, 7.25], "duration": [0.0, 0.0], "value": ["65536", "1"], "sample": [0, 3712]}
    )
    events.to_csv(
        bids_path.copy().update(suffix="events", extension=".tsv").fpath, sep="\t", index=False
    )
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants["notes"] = (
        "Used preprocessed data from EEGLAB .mat file instead of raw data for pd on"
    )
    participants.to_csv(root / "participants.tsv", sep="\t", index=False)
    return root, bids_path.copy().update(suffix="eeg", extension=".vhdr")


def test_units_roundtrip() -> None:
    assert volts_to_microvolts(1e-6) == pytest.approx(1.0)
    assert microvolts_to_volts(volts_to_microvolts(3.2e-5)) == pytest.approx(3.2e-5)


def test_final_filter_sos_matches_design() -> None:
    """Una pasada: -3.01 dB en los bordes; en fase cero, el doble."""
    sos = final_filter_coefficients(V2, SFREQ)
    _, h = signal.sosfreqz(sos, worN=[0.5, 32.0], fs=SFREQ)
    assert 20 * np.log10(np.abs(h)) == pytest.approx([-3.01, -3.01], abs=0.02)


def test_final_filter_passband_and_stopband() -> None:
    """10 Hz pasa intacto; 50 y 60 Hz quedan atenuados en fase cero."""
    n = int(60 * SFREQ)
    t = np.arange(n) / SFREQ
    tones = {10.0: None, 50.0: None, 60.0: None}
    data = np.vstack([np.sin(2 * np.pi * f * t) for f in tones]) * 1e-5
    raw = mne.io.RawArray(data.copy(), mne.create_info(3, SFREQ, "eeg"))  # RawArray no copia
    apply_final_filter(raw, V2)
    core = slice(int(10 * SFREQ), int(50 * SFREQ))  # lejos de los bordes
    gain_db = 20 * np.log10(raw.get_data()[:, core].std(axis=1) / data[:, core].std(axis=1))
    assert abs(gain_db[0]) < 0.09  # < 1% de error de amplitud
    assert gain_db[1] < -40
    assert gain_db[2] < -55


def test_legacy_filter_reproduces_matlab_filtfilt() -> None:
    """El perfil legacy aplica [b,a] con el relleno de filtfilt de MATLAB."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, int(30 * SFREQ))) * 1e-5
    raw = mne.io.RawArray(data.copy(), mne.create_info(2, SFREQ, "eeg"))
    apply_final_filter(raw, LEGACY)
    b, a = final_filter_coefficients(LEGACY, SFREQ)
    expected = signal.filtfilt(b, a, data, axis=-1, padtype="odd", padlen=3 * (len(a) - 1))
    np.testing.assert_allclose(raw.get_data(), expected, rtol=1e-10, atol=1e-18)
    assert raw.info["highpass"] == LEGACY.final_l_freq


def test_select_artifact_components_follows_manual_criterion() -> None:
    proba = np.array(
        [
            [0.90, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],  # cerebro: se conserva
            [0.05, 0.85, 0.02, 0.02, 0.02, 0.02, 0.02],  # músculo > 0.80: fuera
            [0.10, 0.02, 0.79, 0.02, 0.02, 0.02, 0.03],  # ojo en 0.79: se conserva (estricto)
            [0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.86],  # other sin cerebro: fuera
            [0.06, 0.02, 0.02, 0.02, 0.01, 0.02, 0.85],  # other con cerebro 0.06: se conserva
        ]
    )
    assert select_artifact_components(proba, 0.80, 0.05) == [1, 3]
    with pytest.raises(ValueError):
        select_artifact_components(proba[:, :6], 0.80, 0.05)


def test_notch_removes_line_and_keeps_band() -> None:
    n = int(60 * SFREQ)
    t = np.arange(n) / SFREQ
    data = np.vstack([np.sin(2 * np.pi * f * t) for f in (10.0, 60.0, 120.0)]) * 1e-5
    raw = mne.io.RawArray(data.copy(), mne.create_info(3, SFREQ, "eeg"))
    apply_notch(raw, V2)
    core = slice(int(10 * SFREQ), int(50 * SFREQ))
    gain_db = 20 * np.log10(raw.get_data()[:, core].std(axis=1) / data[:, core].std(axis=1))
    assert abs(gain_db[0]) < 0.01  # la banda de interés no cambia
    assert gain_db[1] < -30 and gain_db[2] < -30
    legacy = mne.io.RawArray(data.copy(), mne.create_info(3, SFREQ, "eeg"))
    apply_notch(legacy, LEGACY)  # sin notch en el perfil legacy
    np.testing.assert_array_equal(legacy.get_data(), data)


def test_bad_channel_detection_finds_injected_channels() -> None:
    raw = _synthetic_eeg(40.0, seed=3)
    assert detect_bad_channels(raw, V2).prep_bads == []
    data = raw.get_data()
    data[4] = 0.0  # FC1 plano
    data[20] = np.random.default_rng(9).standard_normal(data.shape[1]) * 2e-4  # CP6 ruidoso
    raw._data = data
    report = detect_bad_channels(raw, V2)
    assert {"FC1", "CP6"} <= set(report.prep_bads)
    assert report.unusable == ["FC1"]  # solo el plano es falla de registro
    assert channels_to_interpolate(report, V2) == ["FC1"]
    assert set(channels_to_interpolate(report, V2_PREP)) >= {"FC1", "CP6"}
    assert report.correlation_bad_fraction["CP6"] > 0.5
    assert report.correlation_bad_fraction["Fp1"] < 0.01


def test_interpolation_and_car_fix_rank() -> None:
    raw = _synthetic_eeg(30.0, seed=4)
    interpolate_bad_channels(raw, ["C3", "P4"])
    raw.set_eeg_reference("average", projection=False)
    data = raw.get_data()
    np.testing.assert_allclose(data.sum(axis=0), 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(data, tol=1e-10) == len(SCALP_CHANNELS) - 1 - 2


def test_epochs_have_exact_length_and_ptp_rejection() -> None:
    raw = _synthetic_eeg(65.0, seed=5, scale=3e-6)  # pico a pico muy por debajo de 150 µV
    raw._data[:, int(22 * SFREQ)] += microvolts_to_volts(400.0)  # espiga en la época 3
    epochs, counts = make_epochs(raw, V2)
    assert counts == {"n_epochs_total": 6, "n_epochs_kept": 5, "n_dropped_ptp": 1}
    assert len(epochs.times) == int(10 * SFREQ)
    starts = epochs.events[:, 0] - raw.first_samp
    assert np.all(starts % int(10 * SFREQ) == 0)  # rejilla fija de 10 s, sin solapamiento
    assert int(2 * 10 * SFREQ) not in starts  # la época con la espiga es la que se fue


def test_rest_onset_crop_and_provenance(bids_dataset: tuple[Path, BIDSPath]) -> None:
    root, bids_path = bids_dataset
    assert read_rest_onset(bids_path) == pytest.approx(7.25)
    raw, info = load_raw(bids_path, V2)
    assert raw.ch_names == list(SCALP_CHANNELS)  # EXG fuera, orden canónico
    assert info.cropped_at_s == pytest.approx(7.25)
    assert raw.times[-1] == pytest.approx(45.0 - 7.25 - 1 / SFREQ, abs=1e-6)
    assert info.curator_preprocessed
    _, info_legacy = load_raw(bids_path, LEGACY)
    assert info_legacy.cropped_at_s == 0.0
    assert find_recordings(root, extension=".vhdr")[0].subject == "pd6"


def test_preprocess_recording_end_to_end(
    bids_dataset: tuple[Path, BIDSPath], tmp_path: Path
) -> None:
    root, bids_path = bids_dataset
    cfg = replace(V2, ica_max_iter=200)
    qc = preprocess_recording(
        bids_path, cfg, out_root=tmp_path / "processed", figures_root=tmp_path / "qc"
    )
    stem = "sub-pd6_ses-on_task-rest"
    deriv = tmp_path / "processed" / "preproc-v2" / "sub-pd6" / "ses-on" / "eeg"
    epochs = mne.read_epochs(deriv / f"{stem}_desc-clean_epo.fif")
    assert len(epochs.times) == int(10 * SFREQ)
    assert qc["ica"]["n_components"] == len(SCALP_CHANNELS) - 1 - qc["n_interpolated"]
    assert qc["config"]["name"] == "v2" and "git_sha" in qc["run_context"]
    assert qc["exclude_primary"] and qc["exclude_reason"] == "curator_preprocessed"
    assert set(qc["bad_channels"]["correlation_bad_fraction"]) == set(SCALP_CHANNELS)
    assert (deriv / f"{stem}_desc-qc.json").exists()
    assert (tmp_path / "qc" / "preproc-v2" / f"{stem}_psd.pdf").exists()


def test_writing_inside_raw_is_forbidden(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    with pytest.raises(PermissionError):
        assert_outside_raw(raw_root / "derivatives", raw_root)
    assert_outside_raw(tmp_path / "processed", raw_root)
