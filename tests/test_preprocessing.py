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

from tesis_eeg_parkinson.features import V1 as V1_FEATURES
from tesis_eeg_parkinson.features import run_features
from tesis_eeg_parkinson.preprocessing.channels import (
    channels_to_interpolate,
    detect_bad_channels,
    interpolate_bad_channels,
)
from tesis_eeg_parkinson.preprocessing.config import (
    EMOTIV_CHANNELS,
    EPOC_X,
    MATLAB,
    SCALP_CHANNELS,
    V1,
    V1_PREP,
)
from tesis_eeg_parkinson.preprocessing.device import emulate_epoc_x
from tesis_eeg_parkinson.preprocessing.epochs import make_epochs, persistent_bad_channels
from tesis_eeg_parkinson.preprocessing.filters import (
    apply_final_filter,
    apply_notch,
    final_filter_coefficients,
)
from tesis_eeg_parkinson.preprocessing.ica import select_artifact_components
from tesis_eeg_parkinson.preprocessing.io import find_recordings, load_raw, read_rest_onset
from tesis_eeg_parkinson.preprocessing.pipeline import (
    _summary_row,
    assert_outside_raw,
    preprocess_recording,
    run_dataset,
)
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
    sos = final_filter_coefficients(V1, SFREQ)
    _, h = signal.sosfreqz(sos, worN=[0.5, 32.0], fs=SFREQ)
    assert 20 * np.log10(np.abs(h)) == pytest.approx([-3.01, -3.01], abs=0.02)


def test_final_filter_passband_and_stopband() -> None:
    """10 Hz pasa intacto; 50 y 60 Hz quedan atenuados en fase cero."""
    n = int(60 * SFREQ)
    t = np.arange(n) / SFREQ
    tones = {10.0: None, 50.0: None, 60.0: None}
    data = np.vstack([np.sin(2 * np.pi * f * t) for f in tones]) * 1e-5
    raw = mne.io.RawArray(data.copy(), mne.create_info(3, SFREQ, "eeg"))  # RawArray no copia
    apply_final_filter(raw, V1)
    core = slice(int(10 * SFREQ), int(50 * SFREQ))  # lejos de los bordes
    gain_db = 20 * np.log10(raw.get_data()[:, core].std(axis=1) / data[:, core].std(axis=1))
    assert abs(gain_db[0]) < 0.09  # < 1% de error de amplitud
    assert gain_db[1] < -40
    assert gain_db[2] < -55


def test_matlab_profile_reproduces_matlab_filtfilt() -> None:
    """El perfil matlab aplica [b,a] con el relleno de filtfilt de MATLAB."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, int(30 * SFREQ))) * 1e-5
    raw = mne.io.RawArray(data.copy(), mne.create_info(2, SFREQ, "eeg"))
    apply_final_filter(raw, MATLAB)
    b, a = final_filter_coefficients(MATLAB, SFREQ)
    expected = signal.filtfilt(b, a, data, axis=-1, padtype="odd", padlen=3 * (len(a) - 1))
    np.testing.assert_allclose(raw.get_data(), expected, rtol=1e-10, atol=1e-18)
    assert raw.info["highpass"] == MATLAB.final_l_freq


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
    apply_notch(raw, V1)
    core = slice(int(10 * SFREQ), int(50 * SFREQ))
    gain_db = 20 * np.log10(raw.get_data()[:, core].std(axis=1) / data[:, core].std(axis=1))
    assert abs(gain_db[0]) < 0.01  # la banda de interés no cambia
    assert gain_db[1] < -30 and gain_db[2] < -30
    matlab = mne.io.RawArray(data.copy(), mne.create_info(3, SFREQ, "eeg"))
    apply_notch(matlab, MATLAB)  # sin notch en el perfil matlab
    np.testing.assert_array_equal(matlab.get_data(), data)


def test_bad_channel_detection_finds_injected_channels() -> None:
    raw = _synthetic_eeg(40.0, seed=3)
    assert detect_bad_channels(raw, V1).prep_bads == []
    data = raw.get_data()
    data[4] = 0.0  # FC1 plano
    data[20] = np.random.default_rng(9).standard_normal(data.shape[1]) * 2e-4  # CP6 ruidoso
    raw._data = data
    report = detect_bad_channels(raw, V1)
    assert {"FC1", "CP6"} <= set(report.prep_bads)
    assert report.unusable == ["FC1"]  # solo el plano es falla de registro
    assert channels_to_interpolate(report, V1) == ["FC1"]
    assert set(channels_to_interpolate(report, V1_PREP)) >= {"FC1", "CP6"}
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
    epochs, counts = make_epochs(raw, V1)
    assert counts == {"n_epochs_total": 6, "n_epochs_kept": 5, "n_dropped_ptp": 1}
    assert len(epochs.times) == int(10 * SFREQ)
    starts = epochs.events[:, 0] - raw.first_samp
    assert np.all(starts % int(10 * SFREQ) == 0)  # rejilla fija de 10 s, sin solapamiento
    assert int(2 * 10 * SFREQ) not in starts  # la época con la espiga es la que se fue


def test_rest_onset_crop_and_provenance(bids_dataset: tuple[Path, BIDSPath]) -> None:
    root, bids_path = bids_dataset
    assert read_rest_onset(bids_path) == pytest.approx(7.25)
    raw, info = load_raw(bids_path, V1)
    assert raw.ch_names == list(SCALP_CHANNELS)  # EXG fuera, orden canónico
    assert info.cropped_at_s == pytest.approx(7.25)
    assert raw.times[-1] == pytest.approx(45.0 - 7.25 - 1 / SFREQ, abs=1e-6)
    assert info.curator_preprocessed
    _, info_matlab = load_raw(bids_path, MATLAB)
    assert info_matlab.cropped_at_s == 0.0
    assert find_recordings(root, extension=".vhdr")[0].subject == "pd6"


def test_preprocess_recording_end_to_end(
    bids_dataset: tuple[Path, BIDSPath], tmp_path: Path
) -> None:
    root, bids_path = bids_dataset
    cfg = replace(V1, ica_max_iter=200)
    qc = preprocess_recording(
        bids_path, cfg, out_root=tmp_path / "processed", figures_root=tmp_path / "qc"
    )
    stem = "sub-pd6_ses-on_task-rest"
    deriv = tmp_path / "processed" / "preproc-v1" / "sub-pd6" / "ses-on" / "eeg"
    epochs = mne.read_epochs(deriv / f"{stem}_desc-clean_epo.fif")
    assert len(epochs.times) == int(10 * SFREQ)
    assert qc["ica"]["n_components"] == len(SCALP_CHANNELS) - 1 - len(qc["interpolated"])
    assert qc["config"]["name"] == "v1" and "git_sha" in qc["run_context"]
    assert qc["exclude_primary"] and "curator_preprocessed" in qc["exclude_reason"]
    assert _summary_row(qc)["exclude_reason"] == qc["exclude_reason"]  # llega al resumen
    assert set(qc["bad_channels"]["correlation_bad_fraction"]) == set(SCALP_CHANNELS)
    assert (deriv / f"{stem}_desc-qc.json").exists()
    assert (tmp_path / "qc" / "preproc-v1" / f"{stem}_psd.pdf").exists()


def test_writing_inside_raw_is_forbidden(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    with pytest.raises(PermissionError):
        assert_outside_raw(raw_root / "derivatives", raw_root)
    assert_outside_raw(tmp_path / "processed", raw_root)


def test_empty_dataset_fails_before_writing(tmp_path: Path) -> None:
    """Sin registros no se escribe nada: un resumen vacío taparía el anterior."""
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError):
        run_dataset(tmp_path / "raw", V1, out_root=tmp_path / "processed")
    assert not (tmp_path / "processed").exists()


def test_persistent_channel_rule_separates_faults_from_intermittent_artifacts() -> None:
    """Un canal sobre el umbral en la mayoría de las épocas es falla; en pocas, artefacto."""
    raw = _synthetic_eeg(100.0, seed=11, scale=3e-6)  # 10 épocas limpias
    spike = microvolts_to_volts(400.0)
    for k in range(7):  # F3: espiga en 7 de 10 épocas
        raw._data[3, int((10 * k + 5) * SFREQ)] += spike
    for k in range(2):  # T8: espiga en 2 de 10 épocas
        raw._data[23, int((10 * k + 5) * SFREQ)] += spike
    found = persistent_bad_channels(raw, V1)
    assert list(found) == ["F3"] and found["F3"] == pytest.approx(0.7)
    assert persistent_bad_channels(raw, MATLAB) == {}  # sin la regla en el perfil matlab


def test_min_epochs_rule_marks_exclusion(
    bids_dataset: tuple[Path, BIDSPath], tmp_path: Path
) -> None:
    _, bids_path = bids_dataset
    cfg = replace(V1, ica_max_iter=200, min_epochs=50)  # 45 s nunca alcanzan 50 épocas
    qc = preprocess_recording(bids_path, cfg, out_root=tmp_path / "processed")
    assert qc["exclude_primary"]
    assert qc["exclude_reason"] == "curator_preprocessed;min_epochs"


def test_epoc_x_emulation() -> None:
    """14 canales, 256 Hz, banda del equipo, cuantización y rechazo del modo común."""
    base = _synthetic_eeg(40.0, seed=12)
    t = base.times
    common = 5e-5 * np.sin(2 * np.pi * 3 * t)  # señal común a los 32 canales, P3 incluido
    tone = np.zeros_like(base.get_data())
    tone[SCALP_CHANNELS.index("O1")] = 2e-5 * np.sin(2 * np.pi * 60 * t)
    with_common = mne.io.RawArray(base.get_data() + common + tone, base.info)
    without = mne.io.RawArray(base.get_data() + tone, base.info)
    a, b = emulate_epoc_x(with_common, EPOC_X), emulate_epoc_x(without, EPOC_X)
    step = microvolts_to_volts(EPOC_X.device_step_uv)
    assert a.ch_names == list(EMOTIV_CHANNELS) and a.info["sfreq"] == 256.0
    q = a.get_data() / step
    np.testing.assert_allclose(q, np.round(q), atol=1e-6)  # valores sobre la rejilla del ADC
    assert np.abs(a.get_data() - b.get_data()).max() <= step * 1.0001  # modo común rechazado
    spectrum = a.compute_psd(fmin=55, fmax=65).get_data()
    in_band = a.compute_psd(fmin=8, fmax=12).get_data()
    o1 = list(EMOTIV_CHANNELS).index("O1")
    assert 10 * np.log10(spectrum[o1].max() / in_band[o1].max()) < -20  # 60 Hz fuera de banda


def test_epoc_x_profile_end_to_end(bids_dataset: tuple[Path, BIDSPath], tmp_path: Path) -> None:
    _, bids_path = bids_dataset
    qc = preprocess_recording(bids_path, replace(EPOC_X, ica_max_iter=200), tmp_path / "processed")
    deriv = tmp_path / "processed" / "preproc-epocx" / "sub-pd6" / "ses-on" / "eeg"
    epochs = mne.read_epochs(deriv / "sub-pd6_ses-on_task-rest_desc-clean_epo.fif")
    assert epochs.ch_names == list(EMOTIV_CHANNELS)
    assert epochs.info["sfreq"] == 256.0 and len(epochs.times) == 2560
    assert qc["ica"]["n_components"] == 13 - len(qc["interpolated"])
    table = run_features(tmp_path / "processed" / "preproc-epocx", V1_FEATURES, tmp_path / "f")
    n_features = sum(1 for c in table.columns if "__" in c and not c.startswith("qc_"))
    assert n_features == 17 * 14  # V1 completo sobre los 14 canales, diezmado de 256 a 128 Hz


def test_prep_falls_back_when_robust_reference_is_impossible() -> None:
    """Con 14 canales sin correlación, PREP no puede referenciar; el registro no se pierde."""
    rng = np.random.default_rng(13)
    data = rng.standard_normal((14, int(40 * 256))) * 1e-5
    data[2] = 0.0  # F3 plano: falla del registro, independiente de la referencia
    raw = mne.io.RawArray(data, mne.create_info(list(EMOTIV_CHANNELS), 256.0, "eeg"))
    raw.set_montage("colin27_1005")
    report = detect_bad_channels(raw, EPOC_X)
    assert not report.robust_reference
    assert report.unusable == ["F3"]
    assert channels_to_interpolate(report, EPOC_X) == ["F3"]
