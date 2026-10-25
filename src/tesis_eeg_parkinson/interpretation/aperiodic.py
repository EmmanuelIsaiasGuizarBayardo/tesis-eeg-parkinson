"""¿El espectro más plano en OFF es cortical o muscular? (análisis exploratorio)

Para cada paciente con ambas sesiones se ajusta el exponente aperiódico por registro
y se compara OFF menos ON en 8 condiciones, todas reportadas:

- referencia: CAR (la del pipeline) o CSD (laplaciano de superficie, que limita la
  difusión de la actividad de canales lejanos, incluida la que reparte la CAR);
- rango del ajuste: 2 a 30 Hz (el de V1) o 2 a 20 Hz (menos sensible al músculo,
  cuya potencia domina por encima de ~20 Hz);
- canales: laterales (sobre el músculo temporal) o centro-parietales (lejos de él).

Además, un índice muscular por registro, desde el preprocesamiento ``v1-43``: potencia
de 32 a 43 Hz sobre la de 2 a 30 Hz, en los canales laterales.

El plan y las reglas de lectura se fijaron antes de correrlo (docs/bitacora.md).
"""

from __future__ import annotations

import os
import textwrap
import warnings
from pathlib import Path

import figuras_cientificas as fc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats

with warnings.catch_warnings(record=True):  # fooof avisa de su sucesor (ver features)
    from fooof import FOOOFGroup

LATERAL = ("F7", "F8", "FC5", "FC6", "T7", "T8")
CENTRAL = ("C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4")
REFERENCES = ("car", "csd")
RANGES = ((2.0, 30.0), (2.0, 20.0))
SFREQ = 128.0
WINDOW_S = 2.0


def recording_psd(epochs: mne.Epochs, reference: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """PSD de Welch promediada sobre las épocas limpias de un registro, a 128 Hz.

    Returns
    -------
    freqs, psd (canales × frecuencias), nombres de canal
    """
    inst = epochs.copy()
    if reference == "csd":
        inst = mne.preprocessing.compute_current_source_density(inst, verbose="error")
    factor = int(round(inst.info["sfreq"] / SFREQ))
    data = inst.get_data()[..., ::factor]  # el filtro de 32 Hz ya evita el aliasing
    nperseg = int(WINDOW_S * SFREQ)
    freqs, psd = signal.welch(data, fs=SFREQ, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    return freqs, psd.mean(axis=0), inst.ch_names


def exponents(freqs: np.ndarray, psd: np.ndarray, fit_range: tuple[float, float]) -> np.ndarray:
    """Exponente aperiódico (FOOOF, modo fijo) por canal."""
    group = FOOOFGroup(peak_width_limits=(1.0, 8.0), max_n_peaks=4, aperiodic_mode="fixed",
                       verbose=False)  # fmt: skip
    group.fit(freqs, psd, list(fit_range))
    return group.get_params("aperiodic_params")[:, 1]


def emg_index(epochs_43: mne.Epochs, channels: tuple[str, ...]) -> float:
    """log10 de la potencia de 32 a 43 Hz sobre la de 2 a 30 Hz, media de ``channels``."""
    sfreq = epochs_43.info["sfreq"]
    nperseg = int(WINDOW_S * sfreq)
    data = epochs_43.copy().pick(list(channels)).get_data()
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    psd = psd.mean(axis=0)
    high = psd[:, (freqs >= 32) & (freqs <= 43)].mean(axis=1)
    low = psd[:, (freqs >= 2) & (freqs <= 30)].mean(axis=1)
    return float(np.mean(np.log10(high / low)))


def recording_table(epochs: mne.Epochs, epochs_43: mne.Epochs) -> list[dict]:
    """Exponente medio por grupo de canales en las 8 condiciones, e índice muscular."""
    rows = []
    for reference in REFERENCES:
        freqs, psd, names = recording_psd(epochs, reference)
        for low, high in RANGES:
            exp = pd.Series(exponents(freqs, psd, (low, high)), index=names)
            for group, chans in (("lateral", LATERAL), ("central", CENTRAL)):
                value = float(exp[list(chans)].mean())
                rows.append({"reference": reference, "range": f"{low:.0f}-{high:.0f}",
                             "channels": group, "exponent": value})  # fmt: skip
    emg = emg_index(epochs_43, LATERAL)
    return [{**r, "emg_lateral": emg} for r in rows]


def paired_summary(table: pd.DataFrame, n_boot: int = 5000, seed: int = 28) -> pd.DataFrame:
    """OFF menos ON por paciente: media, IC 95% por bootstrap, Wilcoxon exacto y Spearman
    entre el cambio del exponente y el cambio del índice muscular."""
    rng = np.random.default_rng(seed)
    keys = ["reference", "range", "channels"]
    wide = table.pivot_table(index=[*keys, "subject"], columns="session",
                             values=["exponent", "emg_lateral"])  # fmt: skip
    rows = []
    for cell, sub in wide.groupby(level=keys):
        delta = (sub[("exponent", "off")] - sub[("exponent", "on")]).to_numpy()
        d_emg = (sub[("emg_lateral", "off")] - sub[("emg_lateral", "on")]).to_numpy()
        boots = rng.choice(delta, (n_boot, delta.size)).mean(axis=1)
        rho, p_rho = stats.spearmanr(d_emg, delta)
        rows.append({
            **dict(zip(keys, cell, strict=True)),
            "n": delta.size,
            "delta_mean": float(delta.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "n_negative": int((delta < 0).sum()),
            "p_wilcoxon": float(stats.wilcoxon(delta, method="exact").pvalue),
            "rho_emg": float(rho),
            "p_rho": float(p_rho),
        })  # fmt: skip
    return pd.DataFrame(rows)


def _pairs(preproc_root: Path) -> pd.DataFrame:
    """Pacientes del análisis principal con ambas sesiones (los 13 pares de Off vs On)."""
    qc = pd.read_csv(preproc_root / "qc_summary.tsv", sep="\t", dtype={"subject": str})
    qc = qc[(qc["session"].isin(["off", "on"])) & (~qc["exclude_primary"].astype(bool))]
    complete = qc.groupby("subject")["session"].nunique()
    return qc[qc["subject"].isin(complete[complete == 2].index)]


def _epochs_path(root: Path, subject: str, session: str) -> Path:
    stem = f"sub-{subject}_ses-{session}_task-rest"
    return root / f"sub-{subject}" / f"ses-{session}" / "eeg" / f"{stem}_desc-clean_epo.fif"


def run_aperiodic_check(processed: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre el análisis sobre ``preproc-v1`` y ``preproc-v1-43``; escribe tablas y figura."""
    v1, v43 = Path(processed) / "preproc-v1", Path(processed) / "preproc-v1-43"
    rows = []
    for _, rec in _pairs(v1).iterrows():
        epochs = mne.read_epochs(_epochs_path(v1, rec["subject"], rec["session"]), verbose="error")
        epochs_43 = mne.read_epochs(_epochs_path(v43, rec["subject"], rec["session"]),
                                    verbose="error")  # fmt: skip
        for row in recording_table(epochs, epochs_43):
            rows.append({"subject": rec["subject"], "session": rec["session"], **row})
    table = pd.DataFrame(rows)
    summary = paired_summary(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "aperiodic_recordings.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "aperiodic_summary.tsv", sep="\t", index=False)
    plot_check(table, summary, out_dir / "aperiodic_check.pdf")
    return table, summary


def plot_check(table: pd.DataFrame, summary: pd.DataFrame, fname: Path) -> Path:
    """a: cambio OFF − ON del exponente por paciente en las 8 condiciones.
    b: cambio del exponente contra cambio del índice muscular, en dos condiciones."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fc.usar_estilo("publicacion")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=fc.tamano_figura("doble_columna", alto_mm=80),
                                     layout="none", width_ratios=(1.4, 1))  # fmt: skip
    fig.subplots_adjust(left=0.20, right=0.98, top=0.92, bottom=0.30, wspace=0.35)
    colors = fc.categorica(2)
    wide = table.pivot_table(index=["reference", "range", "channels", "subject"],
                             columns="session", values=["exponent", "emg_lateral"])  # fmt: skip
    order = summary.sort_values(["reference", "range", "channels"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    for y, r in order.iterrows():
        key = (r["reference"], r["range"], r["channels"])
        sub = wide.loc[key]
        delta = sub[("exponent", "off")] - sub[("exponent", "on")]
        style = fc.estilo_serie(0 if r["channels"] == "lateral" else 1, colors)
        jitter = y + rng.uniform(-0.15, 0.15, delta.size)
        ax_a.plot(delta, jitter, linestyle="none", color=style["color"],
                  marker=style["marker"], markersize=2.5, alpha=0.6)  # fmt: skip
        ax_a.errorbar(r["delta_mean"], y, xerr=[[r["delta_mean"] - r["ci_low"]],
                      [r["ci_high"] - r["delta_mean"]]], color="black", marker="|",
                      markersize=7, linewidth=0.9, capsize=0)  # fmt: skip
    labels = [f"{r['reference'].upper()}, {r['range']} Hz, "
              f"{'laterales' if r['channels'] == 'lateral' else 'centro-parietales'}"
              for _, r in order.iterrows()]  # fmt: skip
    ax_a.set_yticks(range(len(order)), labels)
    ax_a.invert_yaxis()
    ax_a.axvline(0, color="black", linewidth=0.6)
    ax_a.set_xlabel("Exponente OFF − ON")
    fc.letra_panel(ax_a, "a")
    for i, key in enumerate((("car", "2-30", "lateral"), ("csd", "2-20", "central"))):
        sub = wide.loc[key]
        style = fc.estilo_serie(i, colors)
        ax_b.plot(sub[("emg_lateral", "off")] - sub[("emg_lateral", "on")],
                  sub[("exponent", "off")] - sub[("exponent", "on")], linestyle="none",
                  color=style["color"], marker=style["marker"], markersize=3)  # fmt: skip
        ax_b.annotate(f"{key[0].upper()} {key[1]} Hz, "
                      f"{'laterales' if key[2] == 'lateral' else 'centro-parietales'}",
                      xy=(0.02, 0.95 - 0.07 * i), xycoords="axes fraction", color=style["color"],
                      fontsize=5.5, va="top")  # fmt: skip
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.axvline(0, color="black", linewidth=0.6)
    ax_b.set_xlabel("Índice muscular OFF − ON (log10)")
    ax_b.set_ylabel("Exponente OFF − ON")
    fc.letra_panel(ax_b, "b")
    caption = (
        "Análisis exploratorio, 13 pacientes con ambas sesiones. a: diferencia OFF − ON del "
        "exponente aperiódico por paciente (puntos; círculo: canales laterales, cuadrado: "
        "centro-parietales) y su media con IC 95% por bootstrap de pacientes (barra negra). "
        "CAR: referencia promedio; CSD: laplaciano de superficie. b: cambio del exponente contra "
        "cambio del índice muscular (potencia de 32 a 43 Hz sobre la de 2 a 30 Hz en canales "
        "laterales, del preprocesamiento a 43 Hz)."
    )
    fig.text(0.02, 0.02, textwrap.fill(caption, width=128), va="bottom", ha="left")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    plt.close(fig)
    return fname
