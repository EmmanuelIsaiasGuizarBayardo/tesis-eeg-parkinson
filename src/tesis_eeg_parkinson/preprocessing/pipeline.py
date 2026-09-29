"""Orquestación por registro y por dataset; escribe derivados BIDS y QC."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mne
import pandas as pd
from mne_bids import BIDSPath

from tesis_eeg_parkinson.preprocessing.channels import (
    BadChannelReport,
    detect_bad_channels,
    interpolate_bad_channels,
)
from tesis_eeg_parkinson.preprocessing.config import (
    GRID_LINE_FREQ,
    ICLABEL_CLASSES,
    SCALP_CHANNELS,
    PreprocessingConfig,
)
from tesis_eeg_parkinson.preprocessing.epochs import make_epochs
from tesis_eeg_parkinson.preprocessing.filters import apply_final_filter, apply_notch, ica_copy
from tesis_eeg_parkinson.preprocessing.ica import (
    fit_ica,
    label_components,
    select_artifact_components,
)
from tesis_eeg_parkinson.preprocessing.io import find_recordings, load_raw
from tesis_eeg_parkinson.preprocessing.qc import plot_psd_stages
from tesis_eeg_parkinson.reproducibility import get_run_context, set_seed

logger = logging.getLogger(__name__)

StageCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str, str], None]

# Decisión tomada antes de ver cualquier clasificación (docs/preprocesamiento.md):
# los curadores declaran que estas sesiones no son dato crudo, y su espectro lo
# confirma (sin pico de 60 Hz, sin contenido por debajo de ~2 Hz).
EXCLUSION_REASON = "curator_preprocessed"


def assert_outside_raw(path: Path, raw_root: Path) -> None:
    """Falla si ``path`` cae dentro de ``raw_root`` (data/raw es inmutable)."""
    if Path(path).resolve().is_relative_to(Path(raw_root).resolve()):
        raise PermissionError(f"Prohibido escribir en datos crudos: {path}")


def derivative_root(out_root: Path, cfg: PreprocessingConfig) -> Path:
    """Carpeta de derivados del perfil, con su dataset_description.json."""
    root = Path(out_root) / f"preproc-{cfg.name}"
    root.mkdir(parents=True, exist_ok=True)
    description = {
        "Name": f"Preprocesamiento ds002778, perfil {cfg.name}",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "tesis_eeg_parkinson.preprocessing"}],
        "SourceDatasets": [{"DOI": "doi:10.18112/openneuro.ds002778.v1.0.5"}],
    }
    (root / "dataset_description.json").write_text(
        json.dumps(description, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return root


def preprocess_recording(
    bids_path: BIDSPath,
    cfg: PreprocessingConfig,
    out_root: Path,
    figures_root: Path | None = None,
    on_stage: StageCallback | None = None,
) -> dict[str, Any]:
    """Preprocesa un registro y escribe sus épocas y su JSON de QC.

    Orden: carga y recorte -> notch -> canales malos (PREP) e interpolación ->
    CAR -> copia FIR -> ICA -> ICLabel -> aplicar al maestro -> Butterworth
    final -> épocas.

    Parameters
    ----------
    on_stage : callable, optional
        Recibe el nombre de cada etapa al iniciarla (para la barra de progreso).

    Returns
    -------
    dict
        El QC del registro (lo mismo que se escribe en el JSON).
    """
    stage = on_stage or (lambda _name: None)
    assert_outside_raw(out_root, bids_path.root)
    set_seed(cfg.seed)

    stage("carga")
    raw, rec = load_raw(bids_path, cfg)
    apply_notch(raw, cfg)

    report = BadChannelReport([], {}, {}, {})
    if cfg.detect_bad_channels:
        stage("canales (PREP)")
        report = detect_bad_channels(raw, cfg)
        interpolate_bad_channels(raw, report.bads)
    raw.set_eeg_reference("average", projection=False, verbose="error")
    raw_car = raw.copy() if figures_root is not None else None
    rank = len(SCALP_CHANNELS) - 1 - len(report.bads)

    stage("ICA")
    copy = ica_copy(raw, cfg)
    ica = fit_ica(copy, n_components=rank, cfg=cfg)
    stage("ICLabel")
    proba = label_components(copy if cfg.iclabel_on == "ica_copy" else raw, ica)
    exclude = select_artifact_components(proba, cfg.iclabel_threshold, cfg.iclabel_other_brain_max)
    ica.apply(raw, exclude=exclude, verbose="error")
    raw_ica = raw.copy() if figures_root is not None else None

    stage("filtro y épocas")
    apply_final_filter(raw, cfg)
    epochs, counts = make_epochs(raw, cfg)

    stage("guardar")
    deriv = derivative_root(out_root, cfg) / f"sub-{rec.subject}" / f"ses-{rec.session}" / "eeg"
    deriv.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{rec.subject}_ses-{rec.session}_task-rest"
    epochs.save(deriv / f"{stem}_desc-clean_epo.fif", overwrite=True, verbose="error")

    qc: dict[str, Any] = {
        "recording": asdict(rec),
        "line_freq_sidecar_mismatch": rec.line_freq_sidecar != GRID_LINE_FREQ,
        "exclude_primary": rec.curator_preprocessed,
        "exclude_reason": EXCLUSION_REASON if rec.curator_preprocessed else "",
        "bad_channels": asdict(report),
        "n_interpolated": len(report.bads),
        "needs_review": len(report.bads) > cfg.max_bad_channels,
        "ica": {
            "n_components": rank,
            "excluded": exclude,
            "excluded_labels": [ICLABEL_CLASSES[int(proba[i].argmax())] for i in exclude],
            "probabilities": proba.round(4).tolist(),
        },
        "epochs": counts,
        "config": cfg.as_dict(),
        "run_context": get_run_context(seed=cfg.seed).as_dict(),
        "versions": {"mne": mne.__version__},
    }
    (deriv / f"{stem}_desc-qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if figures_root is not None:
        plot_psd_stages(
            [("Tras CAR", raw_car), ("Tras ICA", raw_ica), ("Final", epochs)],
            band=(cfg.final_l_freq, cfg.final_h_freq),
            line_freq=GRID_LINE_FREQ,
            fname=Path(figures_root) / f"preproc-{cfg.name}" / f"{stem}_psd.pdf",
            title=f"sub-{rec.subject} ses-{rec.session} ({cfg.name})",
        )
    return qc


def _summary_row(qc: dict[str, Any]) -> dict[str, Any]:
    rec = qc["recording"]
    return {
        "rest_onset_s": rec["rest_onset_s"],
        "curator_preprocessed": rec["curator_preprocessed"],
        "exclude_primary": qc["exclude_primary"],
        "line_freq_sidecar": rec["line_freq_sidecar"],
        "bad_channels": ",".join(qc["bad_channels"]["bads"]),
        "n_interpolated": qc["n_interpolated"],
        "needs_review": qc["needs_review"],
        "n_ics_excluded": len(qc["ica"]["excluded"]),
        "excluded_labels": ",".join(qc["ica"]["excluded_labels"]),
        **qc["epochs"],
        "error": "",
    }


def run_dataset(
    bids_root: Path,
    cfg: PreprocessingConfig,
    out_root: Path,
    figures_root: Path | None = None,
    subjects: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Preprocesa todos los registros y escribe ``qc_summary.tsv``.

    Un registro que falla se registra con su error y no detiene a los demás.

    Parameters
    ----------
    on_progress : callable, optional
        Recibe (completados, total, registro, etapa) en cada cambio de etapa.
    """
    recordings = [
        p for p in find_recordings(Path(bids_root)) if not subjects or p.subject in subjects
    ]
    total = len(recordings)
    notify = on_progress or (lambda *_args: None)
    rows: list[dict[str, Any]] = []
    for done, bids_path in enumerate(recordings):
        name = f"sub-{bids_path.subject} ses-{bids_path.session}"
        base = {"subject": bids_path.subject, "session": bids_path.session}
        try:
            qc = preprocess_recording(
                bids_path, cfg, out_root, figures_root,
                on_stage=lambda s, d=done, n=name: notify(d, total, n, s),
            )  # fmt: skip
            rows.append({**base, **_summary_row(qc)})
        except Exception as err:  # se reporta y se sigue con el siguiente registro
            logger.exception("Falló %s", bids_path.basename)
            rows.append({**base, "error": f"{type(err).__name__}: {err}"})
        notify(done + 1, total, name, "listo")
    summary = pd.DataFrame(rows)
    summary.to_csv(derivative_root(out_root, cfg) / "qc_summary.tsv", sep="\t", index=False)
    return summary
