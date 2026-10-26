"""Figuras clave por sección. Cada función recibe datos ya cargados y devuelve `Figura`;
los cargadores leen el repositorio. Las leyendas van aparte, en ``leyendas.md``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import figuras_cientificas as fc
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy import signal

from tesis_eeg_parkinson.features.dwt import band_signals
from tesis_eeg_parkinson.figures.base import (
    GROUP_LABELS,
    GROUPS,
    LSB_UV,
    Figura,
    group_of,
    group_styles,
    panel,
    save,
    setup,
    size,
    strip,
    text_pt,
)
from tesis_eeg_parkinson.interpretation.figures import _info
from tesis_eeg_parkinson.preprocessing.config import SCALP_CHANNELS, V1
from tesis_eeg_parkinson.preprocessing.filters import final_filter_coefficients

IC_LABELS = ("muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other")
IC_NAMES = ("Músculo", "Ojo", "Corazón", "Línea", "Canal", "Otro")
CONTRAST_TITLES = {"off_vs_hc": "Off vs HC", "on_vs_hc": "On vs HC", "off_vs_on": "Off vs On"}
TOPO_KW = {
    "outlines": "head",
    "extrapolate": "head",
    "border": "mean",
    "image_interp": "cubic",
    "sensors": False,
    "show": False,
}


# ---------------------------------------------------------------- 1. Preprocesamiento
def fig_calidad(qc: pd.DataFrame, estilo: str) -> Figura:
    """Calidad por registro: épocas, componentes, canales que marca PREP y etiquetas de ICA."""
    setup(estilo)
    qc = qc.assign(group=qc["session"].map(group_of))
    fig, axes = plt.subplots(2, 2, figsize=size(183, 120, estilo, 170), layout="constrained")
    (ax_a, ax_b), (ax_c, ax_d) = axes
    frac = qc["n_epochs_kept"] / qc["n_epochs_total"]
    by = {g: frac[qc["group"] == g].to_numpy() for g in GROUPS}
    hollow = {
        g: qc.loc[qc["group"] == g, "exclude_primary"].astype(bool).to_numpy() for g in GROUPS
    }
    strip(ax_a, by, hollow)
    ax_a.set_ylabel("Épocas conservadas (fracción)")
    ax_a.set_ylim(max(0.0, np.floor(frac.min() * 10) / 10 - 0.05), 1.03)
    panel(ax_a, "a", estilo)
    strip(ax_b, {g: qc.loc[qc["group"] == g, "n_ics_excluded"].to_numpy() for g in GROUPS}, hollow)
    ax_b.set_ylabel("Componentes excluidos por registro")
    ax_b.yaxis.set_major_locator(MaxNLocator(integer=True))
    panel(ax_b, "b", estilo)
    flags = qc["prep_flagged"].fillna("").str.split(",")
    rate = np.array([flags.map(lambda f, c=ch: c in f).mean() for ch in SCALP_CHANNELS])
    cmap, norm = fc.mapa_para(rate)
    mne.viz.plot_topomap(
        rate,
        _info(list(SCALP_CHANNELS)),
        axes=ax_c,
        cmap=cmap,
        cnorm=norm,
        contours=0,
        **{**TOPO_KW, "sensors": True},
    )
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_c,
        shrink=0.7,
        extend=fc.extension(rate, norm),
        label="Fracción de registros marcados",
    )
    ax_c.set_title("Marcados por PREP\n(diagnóstico)")
    panel(ax_c, "c", estilo)
    styles = group_styles()
    ys = np.arange(len(IC_LABELS))
    for i, g in enumerate(GROUPS):
        sub = qc[qc["group"] == g]
        labels = sub["excluded_labels"].fillna("").str.split(",")
        mean = [labels.map(lambda v, lab=lab: v.count(lab)).mean() for lab in IC_LABELS]
        ax_d.plot(
            mean,
            ys + (i - 1) * 0.22,
            linestyle="none",
            marker=styles[g]["marker"],
            color=styles[g]["color"],
            markersize=3.5,
            label=GROUP_LABELS[g],
        )
    ax_d.set_yticks(ys, IC_NAMES)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Componentes excluidos\n(media por registro)")
    ax_d.legend(loc="best")
    panel(ax_d, "d", estilo)
    leyenda = (
        "Calidad del preprocesamiento V1 por registro (46 registros). a, fracción de épocas de "
        "10 s conservadas tras el rechazo pico a pico de 150 µV; b, componentes de ICA "
        "excluidos por ICLabel. Un punto por registro; línea: mediana del grupo; marcadores "
        "vacíos: sesiones fuera del análisis principal (pd6 y pd16 ON). c, fracción de "
        "registros en que la detección de PREP con referencia robusta marca cada canal; es "
        "diagnóstico, porque V1 solo interpola canales planos o con NaN. Los canales "
        "laterales y frontales concentran las marcas, como corresponde a actividad muscular "
        "y ocular. d, etiqueta dominante de los componentes excluidos, media por registro."
    )
    return Figura("01_preprocesamiento_calidad", fig, leyenda)


def fig_procedencia(spectra: dict[str, tuple[np.ndarray, np.ndarray]], estilo: str) -> Figura:
    """Espectro de cada sesión ON; pd6 y pd16 ON frente a las demás."""
    setup(estilo)
    fig, ax = plt.subplots(figsize=size(89, 70, estilo, 95), layout="constrained")
    others = [db for name, (_, db) in spectra.items() if name not in ("pd6", "pd16")]
    freqs = next(iter(spectra.values()))[0]
    if others:
        stack = np.vstack(others)
        lo, hi = np.percentile(stack, (10, 90), axis=0)
        ax.fill_between(freqs, lo, hi, color="0.85", linewidth=0)
        ax.plot(freqs, np.median(stack, axis=0), color="black", linewidth=1.0)
        ax.annotate(
            "Otras sesiones ON",
            xy=(freqs[-1], np.median(stack, axis=0)[-1]),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
        )
    colors = fc.categorica(2)
    for i, name in enumerate(("pd6", "pd16")):
        if name in spectra:
            st = fc.estilo_serie(i, colors, marcador=False)
            ax.plot(*spectra[name], **st)
            ax.annotate(
                f"sub-{name} ON",
                xy=(freqs[3], spectra[name][1][3]),
                xytext=(4, -10 - 9 * i),
                textcoords="offset points",
                color=st["color"],
            )
    ax.set_xscale("log")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD (dB re 1 µV²/Hz)")
    leyenda = (
        "Procedencia distinta de dos sesiones. Espectro de las épocas limpias de V1 (Welch, "
        "ventanas de 4 s; mediana entre canales) de cada sesión ON. Línea negra y banda gris: "
        "mediana y percentiles 10 a 90 de las otras 13 sesiones ON. Las sesiones ON de sub-pd6 "
        "y sub-pd16, preprocesadas por los curadores antes de publicarse, carecen de contenido "
        "por debajo de ~2 Hz. Eje de frecuencia logarítmico por el rango de 0.25 a 45 Hz."
    )
    return Figura("02_preprocesamiento_procedencia", fig, leyenda)


# ---------------------------------------------------------------------- 2. Wavelets
def band_responses(fs: float, bands: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Respuesta en potencia (dB) de cada banda reconstruida, medida con ruido blanco."""
    x = np.random.default_rng(0).standard_normal((1, 2**16))
    out = band_signals(x, "db4", 4, bands)
    nper = 1024
    freqs, p_in = signal.welch(x[0], fs=fs, nperseg=nper)
    return freqs, {
        b: 10 * np.log10(signal.welch(out[b][0], fs=fs, nperseg=nper)[1] / p_in) for b in bands
    }


def fig_wavelet(amplitudes: dict[str, np.ndarray], estilo: str) -> Figura:
    """Bandas de la DWT a 128 y 512 Hz frente al filtro final, y su amplitud real."""
    setup(estilo)
    fig = plt.figure(figsize=size(183, 130, estilo, 175), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax_a, ax_b, ax_c = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
    )
    colors = fc.categorica(5)
    names = ("cA4", "cD4", "cD3", "cD2", "cD1")
    sos = final_filter_coefficients(V1, 512.0)
    for ax, fs, letter in ((ax_a, 128.0, "a"), (ax_b, 512.0, "b")):
        freqs, resp = band_responses(fs, names)
        for i, b in enumerate(names):
            st = fc.estilo_serie(i, colors, marcador=False)
            ax.plot(freqs[1:], resp[b][1:], color=st["color"], linestyle=st["linestyle"], label=b)
        f, h = signal.sosfreqz(sos, worN=freqs[1:], fs=512.0)
        ax.plot(f, 40 * np.log10(np.abs(h) + 1e-12), color="black", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlim(0.5, fs / 2)
        ax.set_ylim(-60, 5)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Ganancia (dB)")
        ax.set_title(f"DWT db4 a {fs:.0f} Hz")
        panel(ax, letter, estilo)
    ax_a.legend(loc="lower left", ncol=2)
    styles = fc.categorica(5)
    for i, b in enumerate(names):
        v = np.asarray(amplitudes[b])
        med, p5, p95 = np.median(v), np.percentile(v, 5), np.percentile(v, 95)
        ax_c.plot([i, i], [p5, p95], color=styles[i], linewidth=2.5, solid_capstyle="butt")
        ax_c.plot(
            i, med, marker=fc.estilo_serie(i, styles)["marker"], color=styles[i], markersize=5
        )
        ax_c.annotate(
            f"{med:.3g} µV", xy=(i, med), xytext=(8, 0), textcoords="offset points", va="center"
        )
    ax_c.axhline(LSB_UV, color="black", linestyle="--", linewidth=0.8)
    ax_c.annotate(
        "Paso de cuantización de BioSemi (31.25 nV)",
        xy=(-0.45, LSB_UV),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
    )
    ax_c.set_yscale("log")
    ax_c.set_xticks(
        range(5),
        ["cA4\n0–16 Hz", "cD4\n16–32 Hz", "cD3\n32–64 Hz", "cD2\n64–128 Hz", "cD1\n128–256 Hz"],
    )
    ax_c.set_xlim(-0.5, 5.0)
    ax_c.set_ylabel("Mediana de |amplitud| por época y canal (µV)")
    panel(ax_c, "c", estilo)
    leyenda = (
        "Qué es señal y qué es residuo en la descomposición wavelet. a y b, ganancia de cada "
        "banda reconstruida de la DWT diádica (db4, 4 niveles), medida con ruido blanco, a 128 "
        "y 512 Hz; línea negra: filtro final de V1 (Butterworth de 0.5 a 32 Hz en fase cero). "
        "A 512 Hz, cD3, cD2 y cD1 caen por encima del corte; a 128 Hz, las cuatro bandas que "
        "usa V1 cubren delta, theta, alfa con beta baja, y beta. c, mediana de |amplitud| por "
        "época y canal en las 797 épocas a 512 Hz (punto: mediana; barra: percentiles 5 a 95). "
        "cD2 queda del orden del paso de cuantización del amplificador y cD1 por debajo: son "
        "residuo numérico del filtrado, no actividad neural. Ejes de frecuencia logarítmicos "
        "por el rango de 0.5 a 256 Hz; eje de amplitud logarítmico por los cuatro órdenes de "
        "magnitud entre bandas."
    )
    return Figura("03_wavelet_senal_residuo", fig, leyenda)


# ------------------------------------------------------------------------ 3. Features
ALJALAL_NAMES = {
    "energy": "Energía",
    "lbp": "LBP",
    "then": "ThEn",
    "noen": "NoEn",
    "suen": "SuEn",
    "logen": "LogEn",
    "shen": "ShEn",
    "tshen": "TShEn",
}
V1_NAMES = {
    "relpow__cA4": "PR 0–4",
    "relpow__cD4": "PR 4–8",
    "relpow__cD3": "PR 8–16",
    "relpow__cD2": "PR 16–32",
    "aperiodic_exponent__broadband": "Exponente",
    "alpha_cog__broadband": "Centro α",
    "wentropy__cA4": "EW 0–4",
    "wentropy__cD4": "EW 4–8",
    "wentropy__cD3": "EW 8–16",
    "wentropy__cD2": "EW 16–32",
    "sampen__broadband": "SampEn",
    "permen_tau1__broadband": "PermEn τ1",
    "permen_tau2__broadband": "PermEn τ2",
    "permen_tau4__broadband": "PermEn τ4",
    "t3_tau1__broadband": "T3 τ1",
    "t3_tau2__broadband": "T3 τ2",
    "t3_tau4__broadband": "T3 τ4",
}


def channel_median(table: pd.DataFrame, prefixes: dict[str, str]) -> pd.DataFrame:
    """Mediana entre canales de cada feature, por época."""
    out = {}
    for prefix, label in prefixes.items():
        cols = [c for c in table.columns if c.rsplit("__", 1)[0] == prefix]
        out[label] = table[cols].median(axis=1)
    return pd.DataFrame(out)


def fig_redundancia(aljalal: pd.DataFrame, v1: pd.DataFrame, estilo: str) -> Figura:
    """Correlación de Spearman entre features: las ocho originales frente a V1."""
    setup(estilo)
    a = channel_median(aljalal, {f"{k}__cA4": v for k, v in ALJALAL_NAMES.items()}).corr("spearman")
    b = channel_median(v1, V1_NAMES).corr("spearman")
    values = np.concatenate([a.to_numpy().ravel(), b.to_numpy().ravel()])
    cmap, norm = fc.mapa_para(values, centro=0)
    # Celdas más grandes en a (lleva los valores escritos) que en b.
    fig, (ax_a, ax_b, cax) = plt.subplots(
        1,
        3,
        figsize=size(183, 95, estilo, 100),
        layout="constrained",
        width_ratios=(8, 11, 0.45),
    )
    for ax, m, letter, title in ((ax_a, a, "a", "Aljalal et al., cA4"), (ax_b, b, "b", "V1")):
        ax.imshow(m.to_numpy(), cmap=cmap, norm=norm)
        ax.set_xticks(range(len(m)), m.columns, rotation=90)
        ax.set_yticks(range(len(m)), m.index)
        ax.set_title(title)
        panel(ax, letter, estilo)
    for (i, j), v in np.ndenumerate(a.to_numpy()):
        ax_a.text(
            j,
            i,
            f"{v:.2f}".replace("0.", ".").replace("-.", "−."),
            ha="center",
            va="center",
            fontsize=text_pt(0.65),
            color="white" if abs(v) > 0.6 else "black",
        )
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        extend=fc.extension(values, norm),
        label="ρ de Spearman",
    )
    leyenda = (
        "Redundancia entre features. Correlación de Spearman entre épocas (n = 797) de la "
        "mediana entre canales de cada feature. a, las ocho métricas de Aljalal et al. (2022) "
        "en cA4 (512 Hz): casi todas miden lo mismo, la amplitud. b, las 17 features de V1: "
        "potencia relativa (PR), entropía wavelet normalizada (EW), exponente aperiódico, "
        "centro de alfa, SampEn, PermEn y T3 por retraso. Misma escala divergente en ambos "
        "paneles, centrada en 0."
    )
    return Figura("04_features_redundancia", fig, leyenda)


# -------------------------------------------------------------------- 4. Clasificación
def fig_nula(null: pd.DataFrame, summary: pd.DataFrame, estilo: str) -> Figura:
    """Distribución nula por permutación, azar empírico y valor observado por contraste."""
    setup(estilo)
    fig, axes = plt.subplots(
        1, 3, figsize=size(183, 60, estilo, 80), layout="constrained", sharey=True
    )
    color = fc.categorica(1)[0]
    bins = np.arange(0.2, 0.92, 0.025)
    for ax, letter, contrast in zip(axes, "abc", CONTRAST_TITLES, strict=True):
        r = summary.set_index("contrast").loc[contrast]
        ax.hist(null[contrast].dropna(), bins=bins, color="0.75", linewidth=0)
        ax.axvline(0.5, color="0.4", linestyle=":", linewidth=0.8)
        ax.axvline(r["chance_p95"], color="black", linewidth=1.0)
        ax.axvline(r["bacc"], color=color, linewidth=1.6)
        ymax = ax.get_ylim()[1]
        ax.annotate(
            "P95",
            xy=(r["chance_p95"], ymax * 0.95),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
        )
        ax.annotate(
            f"{r['bacc']:.3f}\np Holm {r['p_holm']:.3f}",
            xy=(r["bacc"], ymax * 0.75),
            xytext=(3, 0),
            textcoords="offset points",
            color=color,
        )
        ax.set_title(CONTRAST_TITLES[contrast])
        ax.set_xlabel("Exactitud balanceada (registro)")
        panel(ax, letter, estilo)
    axes[0].set_ylabel("Permutaciones")
    leyenda = (
        "Azar empírico. Distribución nula de la exactitud balanceada por registro en 1,000 "
        "permutaciones de etiquetas por sujeto (en Off vs On, intercambio de sesiones dentro "
        "de cada paciente), repitiendo todo el procedimiento con validación anidada. Línea "
        "punteada: azar teórico (0.5); línea negra: percentil 95 de la nula; línea de color: "
        "exactitud observada del análisis principal, con su p corregido por Holm. Con unos 30 "
        "registros, superar el azar exige alrededor de 0.68, no 0.5."
    )
    return Figura("05_clasificacion_azar_empirico", fig, leyenda)


TRANSFER = (
    ("v1_v1_lr_inter-epocx", "32 canales"),
    ("v1_v1_lr_inter-epocx_ch-emotiv", "14 canales"),
    ("v1_epocx_lr_inter-v1", "EPOC X\nemulado"),
)


def fig_transferencia(resumen: pd.DataFrame, estilo: str) -> Figura:
    """De 32 canales a 14 y al EPOC X emulado, con los mismos sujetos."""
    setup(estilo)
    fig, axes = plt.subplots(
        1, 3, figsize=size(183, 65, estilo, 85), layout="constrained", sharey=True
    )
    color = fc.categorica(1)[0]
    for ax, letter, contrast in zip(axes, "abc", CONTRAST_TITLES, strict=True):
        rows = [
            resumen[(resumen["analysis"] == a) & (resumen["contrast"] == contrast)]
            for a, _ in TRANSFER
        ]
        xs = [i for i, r in enumerate(rows) if len(r)]
        vals = [rows[i].iloc[0] for i in xs]
        ax.plot(xs, [v["bacc"] for v in vals], color=color, linewidth=0.8)
        for x, v in zip(xs, vals, strict=True):
            ax.errorbar(
                x,
                v["bacc"],
                yerr=[[v["bacc"] - v["bacc_ci_low"]], [v["bacc_ci_high"] - v["bacc"]]],
                color=color,
                marker="o",
                markersize=4,
                capsize=0,
            )
            ax.plot(x + 0.18, v["chance_p95"], marker="_", color="black", markersize=9)
        ax.set_xticks(range(len(TRANSFER)), [t for _, t in TRANSFER])
        ax.set_xlim(-0.5, len(TRANSFER) - 0.5)
        ax.set_title(CONTRAST_TITLES[contrast])
        panel(ax, letter, estilo)
    axes[0].set_ylabel("Exactitud balanceada (registro)")
    leyenda = (
        "Transferencia al Emotiv EPOC X con las mismas personas. Exactitud balanceada por "
        "registro (punto) con IC 95% por bootstrap de sujetos, con los 32 canales de BioSemi, "
        "con solo los 14 canales del EPOC X y con los registros emulados como si vinieran del "
        "EPOC X (referencia, banda, muestreo de 256 Hz, cuantización de 16 bits y una ICA de 14 "
        "canales). Raya negra: percentil 95 de la nula por permutación (azar empírico). "
        "Descriptivo; la emulación es un límite superior del equipo real."
    )
    return Figura("06_clasificacion_transferencia", fig, leyenda)


# -------------------------------------------------------------------- 5. Interpretación
KEY_PATTERNS = (
    ("relpow", "cD4", "Potencia relativa\n4–8 Hz"),
    ("alpha_cog", "broadband", "Centro\nde alfa"),
    ("aperiodic_exponent", "broadband", "Exponente\naperiódico"),
)


def fig_patrones_clave(patterns: pd.DataFrame, estilo: str) -> Figura:
    """Tres features clave en los tres contrastes; escala compartida por fila."""
    setup(estilo)
    fig, axes = plt.subplots(
        3,
        4,
        figsize=size(183, 140, estilo, 200),
        layout="constrained",
        width_ratios=(1, 1, 1, 0.08),
    )
    info = _info(list(SCALP_CHANNELS))
    positive = {"off_vs_hc": "PD off", "on_vs_hc": "PD on", "off_vs_on": "PD off"}
    for row, contrast in enumerate(CONTRAST_TITLES):
        sub = patterns[patterns["contrast"] == contrast]
        pieces = [
            sub[(sub["metric"] == m) & (sub["signal"] == s)]
            .set_index("channel")
            .reindex(list(SCALP_CHANNELS))
            for m, s, _ in KEY_PATTERNS
        ]
        values = np.concatenate([p["pattern"].to_numpy() for p in pieces])
        cmap, norm = fc.mapa_para(values, centro=0)
        for col, (piece, (_, _, title)) in enumerate(zip(pieces, KEY_PATTERNS, strict=True)):
            ax = axes[row, col]
            mne.viz.plot_topomap(
                piece["pattern"].to_numpy(),
                info,
                axes=ax,
                cmap=cmap,
                cnorm=norm,
                contours=4,
                mask=(piece["stability"] >= 0.95).to_numpy(),
                mask_params={
                    "marker": "o",
                    "markerfacecolor": "w",
                    "markeredgecolor": "k",
                    "linewidth": 0,
                    "markersize": 2.5,
                },
                **TOPO_KW,
            )
            if row == 0:
                ax.set_title(title)
        axes[row, 0].annotate(
            CONTRAST_TITLES[contrast],
            xy=(-0.12, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="right",
            va="center",
        )
        panel(axes[row, 0], "abc"[row], estilo)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=axes[row, 3],
            extend=fc.extension(values, norm),
            label=f"Patrón (u. a.)\n+ mayor en {positive[contrast]}",
        )
    leyenda = (
        "Patrones de activación (Haufe et al., 2014) de tres features clave en los tres "
        "contrastes del análisis principal, sobre features estandarizadas. Positivo (cálido): "
        "la feature es mayor en el grupo indicado en la barra de color de cada fila, con escala "
        "compartida por fila. Círculos blancos: canales cuyo signo se mantiene en al menos 95% "
        "de 200 remuestreos de sujetos (exploratorio). Contornos continuos: positivo; "
        "discontinuos: negativo. Off vs HC y On vs HC muestran lentificación oscilatoria "
        "difusa (más potencia relativa de 4 a 8 Hz y centro de alfa menor) sin cambio del "
        "exponente; en Off vs On, el exponente más plano en OFF se explica mejor por "
        "actividad muscular (análisis exploratorio de la etapa 5b). Descriptivo: ningún "
        "contraste superó el criterio prerregistrado."
    )
    return Figura("07_interpretacion_patrones_clave", fig, leyenda)


# ------------------------------------------------------------------------- Cargadores
def _epochs_files(root: Path) -> list[Path]:
    return sorted(Path(root).rglob("*_desc-clean_epo.fif"))


def load_spectra(preproc: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Espectro (mediana entre canales, dB re 1 µV²/Hz) de cada sesión ON."""
    out = {}
    for f in _epochs_files(preproc):
        if "_ses-on_" not in f.name:
            continue
        ep = mne.read_epochs(f, verbose="error")
        fs = ep.info["sfreq"]
        freqs, psd = signal.welch(ep.get_data() * 1e6, fs=fs, nperseg=int(4 * fs), axis=-1)
        keep = (freqs >= 0.25) & (freqs <= 45)
        out[f.name.split("_")[0][4:]] = (
            freqs[keep],
            10 * np.log10(np.median(psd.mean(0), 0)[keep]),
        )
    return out


def load_amplitudes(preproc: Path) -> dict[str, np.ndarray]:
    """Mediana de |amplitud| por época y canal en cada subbanda a 512 Hz (µV)."""
    names = ("cA4", "cD4", "cD3", "cD2", "cD1")
    vals: dict[str, list[np.ndarray]] = {b: [] for b in names}
    for f in _epochs_files(preproc):
        for ep in mne.read_epochs(f, verbose="error").get_data() * 1e6:
            for b, s in band_signals(ep, "db4", 4, names).items():
                vals[b].append(np.median(np.abs(s), axis=-1))
    return {b: np.concatenate(v) for b, v in vals.items()}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"subject": str})


def builders(processed: Path, results: Path) -> dict[str, Callable[[str], Figura]]:
    """Cada figura con su cargador; se construye solo si sus datos existen."""
    pre = processed / "preproc-v1"
    cls = results / "classification"
    return {
        "calidad": lambda e: fig_calidad(_read(pre / "qc_summary.tsv"), e),
        "procedencia": lambda e: fig_procedencia(load_spectra(pre), e),
        "wavelet": lambda e: fig_wavelet(load_amplitudes(pre), e),
        "redundancia": lambda e: fig_redundancia(
            _read(processed / "features-aljalal" / "preproc-v1" / "features.tsv"),
            _read(processed / "features-v1" / "preproc-v1" / "features.tsv"),
            e,
        ),
        "nula": lambda e: fig_nula(
            _read(cls / "v1_v1_lr" / "null.tsv"), _read(cls / "v1_v1_lr" / "summary.tsv"), e
        ),
        "transferencia": lambda e: fig_transferencia(_read(cls / "resumen.tsv"), e),
        "patrones": lambda e: fig_patrones_clave(
            _read(results / "interpretation" / "v1_v1" / "patterns.tsv"), e
        ),
    }


FIGURES = ("calidad", "procedencia", "wavelet", "redundancia", "nula", "transferencia", "patrones")


def build_all(
    processed: Path, results: Path, out_dir: Path, estilo: str, only: tuple[str, ...] = FIGURES
) -> dict[str, str]:
    """Construye las figuras pedidas; devuelve nombre -> ruta o motivo del fallo.

    Escribe ``leyendas.md`` con la leyenda de cada figura construida.
    """
    status, captions = {}, []
    funcs = builders(processed, results)
    for key in only:
        try:
            figura = funcs[key](estilo)
            paths = save(figura, out_dir)
            captions.append(f"## {figura.nombre}\n\n{figura.leyenda}\n")
            status[key] = str(paths[0])
        except FileNotFoundError as err:
            status[key] = f"faltan datos: {err.filename}"
    if captions:
        (out_dir / "leyendas.md").write_text(
            f"# Leyendas de las figuras ({estilo})\n\n" + "\n".join(captions), encoding="utf-8"
        )
    return status
