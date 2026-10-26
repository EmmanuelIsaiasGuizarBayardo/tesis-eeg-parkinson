"""Figuras complementarias: conceptos para la defensa y el material suplementario.

Mismo estilo y tamaño que las figuras clave (ver base.py). Las que no necesitan datos
(pipeline, filtro, orden temporal, fuga simulada) son deterministas.
"""

from __future__ import annotations

from pathlib import Path

import figuras_cientificas as fc
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import signal

from tesis_eeg_parkinson.classification.data import make_dataset
from tesis_eeg_parkinson.classification.evaluate import run_contrast
from tesis_eeg_parkinson.features.dwt import band_signals
from tesis_eeg_parkinson.features.metrics import (
    aljalal_metrics,
    permutation_entropy,
    sample_entropy,
    time_asymmetry,
    wavelet_entropy,
)
from tesis_eeg_parkinson.figures.base import (
    GROUP_LABELS,
    GROUPS,
    Figura,
    group_of,
    group_styles,
    panel,
    setup,
    size,
    text_pt,
)
from tesis_eeg_parkinson.preprocessing.config import V1

# ------------------------------------------------------------------------ Pipeline
# (etapa, paso, variante de sensibilidad que cambia ese paso o None)
STEPS = (
    ("Datos", "ds002778 v1.0.5\n31 personas, 46 registros", None),
    ("Datos", "BioSemi: 32 canales, 512 Hz", None),
    (
        "Preprocesamiento",
        "Recorte al inicio del reposo",
        "epocx: emulación del\nEPOC X antes de todo",
    ),
    ("Preprocesamiento", "Notch en 60 Hz y armónicos", None),
    (
        "Preprocesamiento",
        "PREP como diagnóstico;\nsolo se interpolan planos o NaN",
        "v1-prep: interpola\ntodo lo que marca PREP",
    ),
    ("Preprocesamiento", "Referencia promedio (CAR)", None),
    ("Preprocesamiento", "ICA infomax extendida\n(copia de 1 a 100 Hz) e ICLabel", None),
    (
        "Preprocesamiento",
        "Butterworth de 0.5 a 32 Hz\n(SOS, fase cero)",
        "v1-43: corte en 43 Hz\n(banda del EPOC X)",
    ),
    ("Preprocesamiento", "Épocas de 10 s; rechazo > 150 µV;\nfalla persistente; mínimo 6", None),
    ("Features", "Diezmado a 128 Hz;\nDWT db4 de 4 niveles", "aljalal: 8 métricas\na 512 Hz"),
    ("Features", "17 features por canal (544)", None),
    (
        "Clasificación",
        "Regresión logística L2;\nvalidación anidada por sujeto",
        "Control de fuga:\nvalidación por épocas",
    ),
    ("Clasificación", "Exactitud por registro;\n1,000 permutaciones; Holm", None),
    ("Interpretación", "Patrones de Haufe;\nestabilidad por bootstrap", None),
)


def fig_pipeline(estilo: str) -> Figura:
    """Diagrama del pipeline V1; a la derecha, la variante que cambia cada paso."""
    setup(estilo)
    fig, ax = plt.subplots(figsize=size(183, 150, estilo, 215), layout="none")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_axis_off()
    stages = list(dict.fromkeys(st for st, _, _ in STEPS))
    colors = dict(zip(stages, fc.categorica(len(stages)), strict=True))
    step_h, stage_gap = 1.0, 0.75
    y, ys = 0.0, []
    for k, (stage, _, _) in enumerate(STEPS):
        if k and stage != STEPS[k - 1][0]:
            y -= stage_gap
        ys.append(y)
        y -= step_h
    box_x, box_w, var_x, var_w, h = 14.0, 52.0, 72.0, 27.0, 0.78
    for k, ((stage, text, variant), yk) in enumerate(zip(STEPS, ys, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (box_x, yk - h / 2),
                box_w,
                h,
                boxstyle="round,pad=0,rounding_size=0.15",
                facecolor="white",
                edgecolor=colors[stage],
                linewidth=1.3,
                mutation_aspect=1 / 12,
            )
        )
        ax.text(
            box_x + box_w / 2,
            yk,
            text,
            ha="center",
            va="center",
            fontsize=text_pt(0.85),
            linespacing=1.15,
        )
        if k + 1 < len(STEPS):
            ax.add_patch(
                FancyArrowPatch(
                    (box_x + box_w / 2, yk - h / 2),
                    (box_x + box_w / 2, ys[k + 1] + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color="0.35",
                    linewidth=0.8,
                    shrinkA=0,
                    shrinkB=0,
                )
            )
        if variant:
            ax.add_patch(
                FancyBboxPatch(
                    (var_x, yk - h / 2),
                    var_w,
                    h,
                    boxstyle="round,pad=0,rounding_size=0.15",
                    facecolor="0.95",
                    edgecolor="0.6",
                    linewidth=0.7,
                    linestyle="--",
                    mutation_aspect=1 / 12,
                )
            )
            ax.text(
                var_x + var_w / 2,
                yk,
                variant,
                ha="center",
                va="center",
                fontsize=text_pt(0.8),
                color="0.25",
                linespacing=1.15,
            )
            ax.plot([box_x + box_w, var_x], [yk, yk], color="0.6", linestyle="--", linewidth=0.7)
    for stage in stages:
        idx = [i for i, (st, _, _) in enumerate(STEPS) if st == stage]
        top, bottom = ys[idx[0]] + h / 2, ys[idx[-1]] - h / 2
        ax.plot([11, 11], [bottom, top], color=colors[stage], linewidth=3, solid_capstyle="butt")
        # Encabezado de etapa sobre su primera caja: legible aunque la etapa tenga un paso.
        ax.text(box_x, top + 0.08, stage, ha="left", va="bottom", fontweight="bold")
    ax.text(
        var_x + var_w / 2,
        ys[0] + h / 2 + 0.25,
        "Variantes de sensibilidad",
        ha="center",
        va="bottom",
        fontweight="bold",
        color="0.25",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(ys[-1] - h, ys[0] + h + 0.6)
    leyenda = (
        "Pipeline V1. Cada caja es un paso, en orden de ejecución, agrupado por etapa (barra "
        "de color). A la derecha, en gris y unidas al paso que modifican, las variantes que se "
        "corrieron para el análisis de sensibilidad: cada una cambia un solo paso respecto de "
        "V1. Todo el pipeline está en el repositorio del proyecto, con pruebas automáticas."
    )
    return Figura("08_pipeline_v1", fig, leyenda)


# --------------------------------------------------------------- Filtro SOS y [b,a]
def filter_responses(n: int = 2**20, fs: float = 512.0) -> dict[str, np.ndarray]:
    """Respuesta efectiva de ambas formas, medida filtrando un impulso en float64.

    Se mide la respuesta del filtro tal como se aplica (``lfilter`` y ``sosfilt``), no la
    del polinomio evaluado con ``freqz``: con [b, a] mal condicionado, ``freqz`` y
    ``np.roots`` también pierden precisión.
    """
    band = [V1.final_l_freq, V1.final_h_freq]
    b, a = signal.butter(V1.final_order, band, btype="bandpass", fs=fs, output="ba")
    sos = signal.butter(V1.final_order, band, btype="bandpass", fs=fs, output="sos")
    imp = np.zeros(n)
    imp[0] = 1.0
    h_ba, h_sos = signal.lfilter(b, a, imp), signal.sosfilt(sos, imp)
    return {
        "freqs": np.fft.rfftfreq(n, 1 / fs),
        "ba": np.abs(np.fft.rfft(h_ba)),
        "sos": np.abs(np.fft.rfft(h_sos)),
        "t": np.arange(n) / fs,
        "imp_ba": h_ba,
        "imp_sos": h_sos,
    }


def _envelope_db(x: np.ndarray, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    m = (x.size // block) * block
    env = np.abs(x[:m]).reshape(-1, block).max(axis=1)
    return np.arange(env.size) * block, 20 * np.log10(env + 1e-300)


def fig_filtro(r: dict[str, np.ndarray], estilo: str) -> Figura:
    """Por qué el Butterworth final se aplica en SOS y no en forma [b,a]."""
    setup(estilo)
    fig = plt.figure(figsize=size(183, 120, estilo, 150), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    colors = fc.categorica(2)
    st = {k: fc.estilo_serie(i, colors, marcador=False) for i, k in enumerate(("sos", "ba"))}
    names = {"sos": "SOS (diseño)", "ba": "[b, a]"}
    f = r["freqs"]
    sel = (f >= 0.05) & (f <= 255)
    for k in ("sos", "ba"):
        db = 20 * np.log10(r[k] + 1e-300)
        ax_a.plot(f[sel], db[sel], label=names[k], **st[k])
        zoom = (f >= 0.2) & (f <= 2.0)
        ax_b.plot(f[zoom], db[zoom], **st[k])
        g = float(np.interp(0.5, f, db))
        ax_b.plot(0.5, g, marker="o", color=st[k]["color"], markersize=4)
        ax_b.annotate(
            f"{g:.2f} dB".replace("-", "−"),
            xy=(0.5, g),
            xytext=(8, 4 if k == "ba" else -12),
            textcoords="offset points",
            color=st[k]["color"],
        )
        samples, env = _envelope_db(r[f"imp_{k}"])
        t = samples / 512.0
        keep = t <= 12
        ax_c.plot(t[keep], env[keep], **st[k])
    ax_a.set_xscale("log")
    ax_a.set_ylim(-60, 3)
    ax_a.axvspan(V1.final_l_freq, V1.final_h_freq, color="0.92", zorder=0)
    ax_a.set_xlabel("Frecuencia (Hz)")
    ax_a.set_ylabel("Ganancia por pasada (dB)")
    ax_a.legend(loc="lower center")
    panel(ax_a, "a", estilo)
    ax_b.axvline(0.5, color="0.5", linestyle=":", linewidth=0.8)
    ax_b.set_xlim(0.2, 2.0)
    ax_b.set_ylim(-14, 1)
    ax_b.set_xlabel("Frecuencia (Hz)")
    ax_b.set_ylabel("Ganancia por pasada (dB)")
    panel(ax_b, "b", estilo)
    ax_c.set_xlabel("Tiempo tras el impulso (s)")
    ax_c.set_ylabel("Envolvente de |h| (dB)")
    ax_c.set_ylim(-160, 0)
    panel(ax_c, "c", estilo)
    leyenda = (
        "Forma numérica del filtro final. Butterworth de orden 5, pasabanda de 0.5 a 32 Hz a "
        "512 Hz, en secciones de segundo orden (SOS) o como cociente de polinomios [b, a]. Las "
        "respuestas se midieron filtrando un impulso tal como se aplica cada forma, no "
        "evaluando el polinomio. a, ganancia por pasada; la banda de paso está sombreada. En "
        "[b, a], la atenuación por debajo de ~0.3 Hz se estanca cerca de −20 dB por pasada, "
        "porque el redondeo deja de anular la ganancia en 0 Hz. b, "
        "borde inferior: en SOS, la ganancia en 0.5 Hz es la del diseño (−3.01 dB); en "
        "[b, a], −1.75 dB, porque el redondeo de los coeficientes desplaza el polo más lento. "
        "c, envolvente de la respuesta al impulso: [b, a] decae más lento. V1 usa SOS; la "
        "reproducción del procesamiento previo conserva [b, a]. Eje de frecuencia de a "
        "logarítmico por el rango de 0.05 a 255 Hz."
    )
    return Figura("09_filtro_sos_frente_a_ba", fig, leyenda)


# ------------------------------------------------------------- Descomposición DWT
BAND_TEXT = {
    "cA4": "cA4, 0–4 Hz",
    "cD4": "cD4, 4–8 Hz",
    "cD3": "cD3, 8–16 Hz",
    "cD2": "cD2, 16–32 Hz",
    "cD1": "cD1, 32–64 Hz (residuo)",
}


def fig_descomposicion(x_uv: np.ndarray, sfreq: float, label: str, estilo: str) -> Figura:
    """Una época real a 128 Hz y su reconstrucción por banda."""
    setup(estilo)
    names = tuple(BAND_TEXT)
    bands = band_signals(x_uv[None, :], "db4", 4, names)
    t = np.arange(x_uv.size) / sfreq
    rows = [("Señal (0.5 a 32 Hz)", x_uv)] + [(BAND_TEXT[b], bands[b][0]) for b in names]
    fig, axes = plt.subplots(
        len(rows), 1, figsize=size(183, 130, estilo, 175), sharex=True, layout="constrained"
    )
    colors = fc.categorica(len(names))
    for i, (ax, (name, y)) in enumerate(zip(axes, rows, strict=True)):
        color = "black" if i == 0 else colors[i - 1]
        ax.plot(t, y, color=color, linewidth=0.7)
        ax.set_ylabel("µV")
        ax.annotate(
            name,
            xy=(0.005, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontweight="bold" if i == 0 else "normal",
        )
        lim = np.max(np.abs(y)) * 1.15
        ax.set_ylim(-lim, lim)
    axes[-1].set_xlabel("Tiempo (s)")
    axes[-1].set_xlim(0, t[-1])
    leyenda = (
        f"Descomposición de una época real ({label}; 10 s, diezmada a 128 Hz). Arriba, la "
        "señal limpia de V1; debajo, su reconstrucción en el tiempo por coeficiente de la DWT "
        "diádica (db4, 4 niveles), con los demás coeficientes en cero. Las cinco bandas suman "
        "la señal. cD1 (32 a 64 Hz) queda por encima del filtro final y V1 no la usa. Cada "
        "fila tiene su propia escala de amplitud."
    )
    return Figura("10_wavelet_descomposicion", fig, leyenda)


# ------------------------------------------------------------- Orden y dirección
def demo_signal(seed: int = 0, sfreq: float = 128.0, n: int = 1280) -> np.ndarray:
    """Onda asimétrica de 10 Hz (subida brusca) más ruido 1/f, en µV."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sfreq
    wave = sum(np.sin(2 * np.pi * 10 * k * t) / k for k in (1, 2, 3))
    pink = np.cumsum(rng.standard_normal(n))
    pink = (pink - pink.mean()) / pink.std()
    return 8 * wave + 4 * pink + rng.standard_normal(n)


def order_effects(x: np.ndarray, seed: int = 1) -> pd.DataFrame:
    """Cambio relativo (%) de cada métrica al barajar las muestras y al invertir el tiempo."""
    shuffled = np.random.default_rng(seed).permutation(x)
    reversed_ = x[::-1].copy()

    def metrics(v: np.ndarray) -> dict[str, float]:
        row = v[None, :]
        alj = aljalal_metrics(row)
        return {
            "Energía (Aljalal)": alj["energy"][0],
            "ShEn (Aljalal)": alj["shen"][0],
            "LogEn (Aljalal)": alj["logen"][0],
            "Entropía wavelet": wavelet_entropy(row)[0],
            "PermEn τ = 1": permutation_entropy(row, 4, 1)[0],
            "SampEn": sample_entropy(row, 2, 0.2)[0],
            "T3 τ = 1": time_asymmetry(row, 1)[0],
        }

    base, sh, rv = metrics(x), metrics(shuffled), metrics(reversed_)
    rows = [
        {"metric": k, "barajar": 100 * (sh[k] - v) / abs(v), "invertir": 100 * (rv[k] - v) / abs(v)}
        for k, v in base.items()
    ]
    return pd.DataFrame(rows)


def fig_orden(effects: pd.DataFrame, estilo: str) -> Figura:
    """Qué métricas leen el orden de las muestras y cuáles su dirección."""
    setup(estilo)
    fig, axes = plt.subplots(
        1, 2, figsize=size(183, 70, estilo, 95), sharey=True, layout="constrained"
    )
    colors = fc.categorica(2)
    ys = np.arange(len(effects))
    for ax, col, letter, title, c in (
        (axes[0], "barajar", "a", "Barajar las muestras", colors[0]),
        (axes[1], "invertir", "b", "Invertir el tiempo", colors[1]),
    ):
        ax.barh(ys, effects[col], color=c, height=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        for y, v in zip(ys, effects[col], strict=True):
            right = v >= 0 or abs(v) < 0.5
            label = "sin cambio" if abs(v) < 0.5 else f"{v:+.0f}%".replace("-", "−")
            ax.annotate(
                label,
                xy=(v, y),
                xytext=(4 if right else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if right else "right",
                fontsize=text_pt(0.85),
            )
        ax.set_xlim(-260, 200)
        ax.set_xlabel("Cambio relativo (%)")
        ax.set_title(title)
        panel(ax, letter, estilo)
    axes[0].set_yticks(ys, effects["metric"])
    axes[0].invert_yaxis()
    leyenda = (
        "Qué lee cada métrica, en una señal sintética de 10 s a 128 Hz (onda de 10 Hz con "
        "subida brusca más ruido 1/f). a, cambio relativo al barajar las muestras, que conserva "
        "la distribución de amplitudes y destruye el orden: las métricas de Aljalal et al. y la "
        "entropía wavelet no cambian; PermEn y SampEn sí. b, cambio al invertir la señal en el "
        "tiempo: solo T3 cambia, de signo (−200%). Una señal de 10 s; los porcentajes "
        "ilustran la dirección del efecto, no su tamaño en datos reales."
    )
    return Figura("11_features_orden_y_direccion", fig, leyenda)


# ------------------------------------------------------------- Simulación de fuga
def synthetic_table(
    fingerprint: float, seed: int, n: int = 10, epochs: int = 8, n_features: int = 64
) -> pd.DataFrame:
    """n HC y n PD off sin efecto de grupo; cada sujeto con una huella propia."""
    rng = np.random.default_rng(seed)
    cols = [f"m{k}__broadband__ch{k % 4}" for k in range(n_features)]
    rows = []
    for group, prefix in (("hc", "hc"), ("pd-off", "pd")):
        for i in range(n):
            offset = rng.standard_normal(n_features) * fingerprint
            for e in range(epochs):
                x = rng.standard_normal(n_features) + offset
                rows.append(
                    {
                        "subject": f"{prefix}{i}",
                        "session": group.split("-")[-1],
                        "group": group,
                        "epoch": e,
                        "exclude_primary": False,
                        "exclude_reason": "",
                        **dict(zip(cols, x, strict=True)),
                    }
                )
    return pd.DataFrame(rows)


def leakage_curve(
    strengths: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    repeats: int = 2,
) -> pd.DataFrame:
    """Exactitud por época con validación por épocas y por sujeto, sin efecto real."""
    rows = []
    for s in strengths:
        for seed in seeds:
            ds = make_dataset(synthetic_table(s, seed), "off_vs_hc")
            r = run_contrast(
                ds, "lr", repeats=repeats, n_perm=0, null_repeats=1, seed=seed, n_boot=50
            )["summary"]
            rows.append(
                {
                    "fingerprint": s,
                    "seed": seed,
                    "epoch_cv": r["epoch_bacc_epoch_cv"],
                    "subject_cv": r["epoch_bacc_subject_cv"],
                }
            )
    return pd.DataFrame(rows)


def fig_fuga_simulada(curve: pd.DataFrame, estilo: str) -> Figura:
    """La validación por épocas acierta sin efecto real; la validación por sujeto, no."""
    setup(estilo)
    fig, ax = plt.subplots(figsize=size(89, 70, estilo, 95), layout="constrained")
    colors = fc.categorica(2)
    for i, (col, label) in enumerate(
        (("epoch_cv", "Validación por épocas"), ("subject_cv", "Validación por sujeto"))
    ):
        st = fc.estilo_serie(i, colors)
        g = curve.groupby("fingerprint")[col]
        ax.fill_between(g.min().index, g.min(), g.max(), color=st["color"], alpha=0.2, linewidth=0)
        ax.plot(g.mean().index, g.mean(), **st, markersize=4)
        ax.annotate(
            label,
            xy=(g.mean().index[-1], g.mean().iloc[-1]),
            xytext=(-4, 8 if i == 0 else -12),
            textcoords="offset points",
            ha="right",
            color=st["color"],
        )
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=0.8)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Huella de cada sujeto (DE entre sujetos / DE dentro)")
    ax.set_ylabel("Exactitud balanceada (época)")
    leyenda = (
        "Simulación de la fuga. Datos sintéticos sin ninguna diferencia entre grupos: 20 "
        "sujetos (10 por grupo), 8 épocas cada uno y 64 features, donde cada sujeto tiene un "
        "desplazamiento propio en todas sus features (su huella). Mismo modelo que la tesis "
        "(regresión logística L2 anidada). Con validación por épocas, la exactitud crece con la "
        "huella hasta casi 1: el modelo reconoce a las personas, no a los grupos. Con validación "
        "por sujeto se queda en el azar. Línea: media de 3 semillas; banda: mínimo y máximo; "
        "punteada: 0.5."
    )
    return Figura("12_clasificacion_fuga_simulada", fig, leyenda)


# ------------------------------------------------------------- Espectros por grupo
def fig_espectros_grupo(
    spectra: pd.DataFrame, freqs: np.ndarray, estilo: str, n_boot: int = 2000
) -> Figura:
    """Espectro medio por grupo con IC 95% por bootstrap de registros, y la diferencia con HC."""
    setup(estilo)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=size(183, 75, estilo, 100), layout="constrained")
    styles = group_styles()
    rng = np.random.default_rng(0)
    data = {g: np.vstack(spectra.loc[spectra["group"] == g, "db"].to_list()) for g in GROUPS}

    def boot(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(m), (n_boot, len(m)))
        means = m[idx].mean(axis=1)
        return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)

    for g in GROUPS:
        st = {**styles[g], "marker": None}
        lo, hi = boot(data[g])
        ax_a.fill_between(freqs, lo, hi, color=st["color"], alpha=0.2, linewidth=0)
        ax_a.plot(
            freqs, data[g].mean(axis=0), label=f"{GROUP_LABELS[g]} (n = {len(data[g])})", **st
        )
        if g != "hc":
            d_idx = rng.integers(0, len(data[g]), (n_boot, len(data[g])))
            h_idx = rng.integers(0, len(data["hc"]), (n_boot, len(data["hc"])))
            d = data[g][d_idx].mean(1) - data["hc"][h_idx].mean(1)
            ax_b.fill_between(
                freqs,
                *np.percentile(d, (2.5, 97.5), axis=0),
                color=st["color"],
                alpha=0.2,
                linewidth=0,
            )
            ax_b.plot(
                freqs, data[g].mean(0) - data["hc"].mean(0), label=f"{GROUP_LABELS[g]} − HC", **st
            )
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_xlim(freqs[0], freqs[-1])
    ax_a.set_ylabel("PSD (dB re 1 µV²/Hz)")
    ax_b.set_ylabel("Diferencia (dB)")
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_a.legend(loc="lower left")
    ax_b.legend(loc="lower left")
    panel(ax_a, "a", estilo)
    panel(ax_b, "b", estilo)
    leyenda = (
        "Espectro por grupo (análisis principal: sin pd6 y pd16 ON). a, media entre registros "
        "de la PSD de cada registro (Welch, ventanas de 4 s; mediana entre canales), con IC 95% "
        "por bootstrap de registros. b, diferencia de cada grupo de PD con HC, con IC 95% por "
        "bootstrap independiente de ambos grupos. Descriptivo. Eje de frecuencia logarítmico por "
        "el rango de 1 a 32 Hz."
    )
    return Figura("13_espectros_por_grupo", fig, leyenda)


# ------------------------------------------------------------------------ Cargadores
def load_example_epoch(
    preproc: Path, subject: str = "hc1", session: str = "hc", channel: str = "O1", epoch: int = 0
) -> tuple[np.ndarray, float, str]:
    """Una época limpia de V1, diezmada a 128 Hz, en µV."""
    f = (
        preproc
        / f"sub-{subject}"
        / f"ses-{session}"
        / "eeg"
        / f"sub-{subject}_ses-{session}_task-rest_desc-clean_epo.fif"
    )
    ep = mne.read_epochs(f, verbose="error").pick([channel])
    x = ep.get_data()[epoch, 0] * 1e6
    factor = int(round(ep.info["sfreq"] / 128.0))
    return x[::factor], ep.info["sfreq"] / factor, f"sub-{subject}, {channel}, época {epoch}"


def load_group_spectra(preproc: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """PSD por registro (dB; mediana entre canales) de 1 a 32 Hz, sin las excluidas."""
    qc = pd.read_csv(preproc / "qc_summary.tsv", sep="\t", dtype={"subject": str})
    keep = qc[~qc["exclude_primary"].astype(bool)]
    rows, freqs = [], None
    for _, r in keep.iterrows():
        f = (
            preproc
            / f"sub-{r['subject']}"
            / f"ses-{r['session']}"
            / "eeg"
            / f"sub-{r['subject']}_ses-{r['session']}_task-rest_desc-clean_epo.fif"
        )
        ep = mne.read_epochs(f, verbose="error")
        fs = ep.info["sfreq"]
        fr, psd = signal.welch(ep.get_data() * 1e6, fs=fs, nperseg=int(4 * fs), axis=-1)
        sel = (fr >= 1) & (fr <= 32)
        freqs = fr[sel]
        db = 10 * np.log10(np.median(psd.mean(axis=0), axis=0)[sel])
        rows.append({"group": group_of(r["session"]), "db": db})
    return pd.DataFrame(rows), freqs
