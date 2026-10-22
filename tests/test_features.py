"""Pruebas de las features con señales sintéticas.

Cada propiedad que justifica una métrica en docs/features.md se prueba aquí:
invariancia a la escala, sensibilidad al orden temporal y valores conocidos.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from tesis_eeg_parkinson.features import ALJALAL, V1, epoch_features, run_features
from tesis_eeg_parkinson.features.dwt import band_ranges, band_signals
from tesis_eeg_parkinson.features.metrics import (
    _sampen_1d,
    aljalal_metrics,
    aperiodic_and_alpha,
    permutation_entropy,
    relative_power,
    sample_entropy,
    time_asymmetry,
    wavelet_entropy,
)

SFREQ = 128.0
N = 1280  # 10 s a 128 Hz
mne.set_log_level("error")


def _pink(n_channels: int, exponent: float, seed: int, n: int = N, sfreq: float = SFREQ):
    """Ruido 1/f^exponente por forma espectral, más un pico de alfa en 10 Hz."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    amp = np.zeros_like(freqs)
    amp[1:] = freqs[1:] ** (-exponent / 2)
    spectrum = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, (n_channels, freqs.size)))
    x = np.fft.irfft(spectrum, n=n, axis=-1)
    x /= x.std(axis=-1, keepdims=True)
    t = np.arange(n) / sfreq
    return 10 * (x + 0.8 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28, (n_channels, 1))))


def test_band_ranges_at_128_hz_follow_classical_bands() -> None:
    assert band_ranges(128.0, 4) == {
        "cD1": (32.0, 64.0),
        "cD2": (16.0, 32.0),
        "cD3": (8.0, 16.0),
        "cD4": (4.0, 8.0),
        "cA4": (0.0, 4.0),
    }


def test_relative_power_locates_a_tone_and_sums_to_one() -> None:
    t = np.arange(N) / SFREQ
    x = np.vstack([np.sin(2 * np.pi * f * t) for f in (2.0, 6.0, 11.0, 24.0)])
    rel = relative_power(band_signals(x, "db4", 4, V1.bands))
    stacked = np.vstack([rel[b] for b in V1.bands])
    np.testing.assert_allclose(stacked.sum(axis=0), 1.0)
    assert stacked.argmax(axis=0).tolist() == [0, 1, 2, 3]  # cada tono en su banda


@pytest.mark.parametrize("gain", [1e-3, 7.3, 250.0])
def test_v1_features_are_scale_invariant(gain: float) -> None:
    """Una ganancia distinta (otro equipo, otra impedancia) no cambia las features V1."""
    x = _pink(4, 1.5, seed=1)
    names = [f"ch{i}" for i in range(4)]
    base = epoch_features(x, names, V1)
    scaled = epoch_features(gain * x, names, V1)
    for key, value in base.items():
        assert scaled[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key


def test_temporal_metrics_read_order_and_distributional_do_not() -> None:
    """Barajar las muestras conserva la distribución de amplitudes y destruye el orden."""
    x = _pink(3, 1.5, seed=2)
    shuffled = np.random.default_rng(3).permuted(x, axis=-1)
    np.testing.assert_allclose(wavelet_entropy(shuffled), wavelet_entropy(x))
    for metric in ("energy", "then", "noen", "suen", "logen", "shen"):
        np.testing.assert_allclose(aljalal_metrics(shuffled)[metric], aljalal_metrics(x)[metric])
    assert np.all(permutation_entropy(shuffled, 4, 1) > permutation_entropy(x, 4, 1) + 0.1)
    assert np.all(sample_entropy(shuffled, 2, 0.2) > sample_entropy(x, 2, 0.2) + 0.5)


def test_permutation_entropy_extremes() -> None:
    ramp = np.arange(N, dtype=float)[None, :]
    noise = np.random.default_rng(4).standard_normal((1, N))
    assert permutation_entropy(ramp, 4, 1)[0] == pytest.approx(0.0)
    assert permutation_entropy(noise, 4, 1)[0] > 0.98


def test_sample_entropy_matches_brute_force() -> None:
    x = np.random.default_rng(5).standard_normal(200)
    m, r = 2, 0.2 * x.std()
    n = x.size - m

    def matches(length: int) -> int:
        t = np.array([x[i : i + length] for i in range(n)])
        d = np.abs(t[:, None, :] - t[None, :, :]).max(axis=-1)
        return int((d <= r).sum() - n)

    assert _sampen_1d(x, m, r) == pytest.approx(-np.log(matches(m + 1) / matches(m)))


def test_aperiodic_exponent_and_alpha_cog_recover_known_values() -> None:
    x = _pink(4, 1.5, seed=6)
    out = aperiodic_and_alpha(x, SFREQ, 2.0, (2.0, 30.0), (7.0, 14.0))
    np.testing.assert_allclose(out["aperiodic_exponent"], 1.5, atol=0.3)
    np.testing.assert_allclose(out["alpha_cog"], 10.0, atol=0.6)


def test_paper_shannon_is_mostly_energy() -> None:
    """ShEn del artículo = E·(log E − H): explica su correlación de 0.998 con Energy."""
    x = _pink(3, 1.5, seed=7)
    m = aljalal_metrics(x)
    h = wavelet_entropy(x) * np.log(x.shape[-1])  # entropía sin normalizar
    np.testing.assert_allclose(m["shen"], m["energy"] * (np.log(m["energy"]) - h), rtol=1e-10)


def test_tshen_readings() -> None:
    """Amplitud: Σ s²log s²/k sobre valores no nulos. Histograma: valores únicos."""
    s = np.array([[-2.0, -1.0, 0.5, 0.5, 2.0, 3.0]])
    tr = np.array([0.0, 0.0, 127.5, 127.5, 255.0, 255.0])
    nonzero = tr[tr > 0] ** 2
    m = aljalal_metrics(s)
    assert m["tshen"][0] == pytest.approx(np.sum(nonzero * np.log(nonzero)) / nonzero.size)
    p2 = (np.array([2, 2, 2]) / 6) ** 2  # tres valores únicos con dos muestras cada uno
    assert m["tshen_hist"][0] == pytest.approx(np.sum(p2 * np.log(p2)) / 3)


def test_feature_counts_per_channel() -> None:
    names = ["Fp1", "Cz"]
    v1 = epoch_features(_pink(2, 1.5, seed=8), names, V1)
    per_channel = {k.rsplit("__", 1)[0] for k in v1 if not k.startswith("qc_")}
    assert len(per_channel) == 17  # 4 relpow + 4 wentropy + 3 permen + 3 t3 + sampen + 2 FOOOF
    x512 = _pink(2, 1.5, seed=9, n=5120, sfreq=512.0)
    alj = epoch_features(x512, names, ALJALAL)
    assert len(alj) == 9 * 6 * 2  # 8 métricas + tshen_hist, 5 coeficientes + original


def test_run_features_writes_table_with_metadata(tmp_path: Path) -> None:
    root = tmp_path / "preproc-v1" / "sub-hc1" / "ses-hc" / "eeg"
    root.mkdir(parents=True)
    names = ["Fp1", "Fz", "Cz", "Pz"]
    data = np.stack([_pink(4, 1.5, seed=s, n=5120, sfreq=512.0) for s in range(3)]) * 1e-6
    epochs = mne.EpochsArray(data, mne.create_info(names, 512.0, "eeg"))
    with epochs.info._unlock():  # como las épocas reales, ya filtradas a 32 Hz
        epochs.info["lowpass"] = 32.0
    epochs.save(root / "sub-hc1_ses-hc_task-rest_desc-clean_epo.fif")
    qc = {"recording": {"subject": "hc1", "session": "hc"}, "exclude_primary": False,
          "exclude_reason": ""}  # fmt: skip
    (root / "sub-hc1_ses-hc_task-rest_desc-qc.json").write_text(json.dumps(qc))
    cfg = replace(V1, permen_delays=(1,), t3_delays=(1,))
    table = run_features(tmp_path / "preproc-v1", cfg, tmp_path)
    assert len(table) == 3 and table["group"].eq("hc").all()
    assert (tmp_path / "features-v1" / "preproc-v1" / "features.tsv").exists()
    sidecar = json.loads(
        (tmp_path / "features-v1" / "preproc-v1" / "features.json").read_text(encoding="utf-8")
    )
    assert sidecar["n_features"] == 4 * (17 - 4)  # dos retrasos menos de PermEn y de T3
    written = pd.read_csv(tmp_path / "features-v1" / "preproc-v1" / "features.tsv", sep="\t")
    assert list(written.columns)[:4] == ["subject", "session", "group", "exclude_primary"]
    with pytest.raises(FileNotFoundError):
        run_features(tmp_path / "vacio", cfg, tmp_path)


def test_time_asymmetry_reads_the_direction_of_time() -> None:
    """Invertir la señal no cambia PermEn ni SampEn; T3 cambia de signo."""
    t = np.arange(N) / SFREQ
    rng = np.random.default_rng(11)
    wave = sum(np.sin(2 * np.pi * 10 * k * t) / k for k in (1, 2, 3))  # subida brusca
    x = (wave + 0.3 * rng.standard_normal(N))[None, :]
    rev = x[:, ::-1].copy()
    assert time_asymmetry(x, 1)[0] > 0.3
    np.testing.assert_allclose(time_asymmetry(rev, 1), -time_asymmetry(x, 1))
    np.testing.assert_allclose(permutation_entropy(rev, 4, 1), permutation_entropy(x, 4, 1))
    np.testing.assert_allclose(time_asymmetry(7.3 * x, 2), time_asymmetry(x, 2))  # escala
    noise = rng.standard_normal((3, N))  # proceso reversible
    assert np.all(np.abs(time_asymmetry(noise, 1)) < 0.1)
