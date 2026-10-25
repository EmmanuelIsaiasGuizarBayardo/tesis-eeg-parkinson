"""Punto de entrada de Tesis_EEG_Parkinson.

Uso::

    uv run tesis-eeg-parkinson preprocess --profile matlab
    uv run tesis-eeg-parkinson preprocess --profile v1 --subject hc1 pd3
    uv run tesis-eeg-parkinson compare-matlab
    uv run tesis-eeg-parkinson features --profile v1
    uv run tesis-eeg-parkinson classify --features v1 --preproc v1
    uv run tesis-eeg-parkinson report
    uv run tesis-eeg-parkinson interpret
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tesis_eeg_parkinson.classification.data import CONTRASTS
from tesis_eeg_parkinson.classification.models import MODELS
from tesis_eeg_parkinson.classification.report import write_report
from tesis_eeg_parkinson.classification.run import run_analysis
from tesis_eeg_parkinson.features import PROFILES as FEATURE_PROFILES
from tesis_eeg_parkinson.features import run_features
from tesis_eeg_parkinson.interpretation import run_interpretation
from tesis_eeg_parkinson.interpretation.figures import plot_contrast_topomaps, plot_magnitude
from tesis_eeg_parkinson.preprocessing import PROFILES, run_dataset
from tesis_eeg_parkinson.preprocessing.io import find_recordings
from tesis_eeg_parkinson.preprocessing.pipeline import EXPECTED_RECORDINGS
from tesis_eeg_parkinson.validation.matlab import (
    compare_dataset,
    ica_association,
    plot_regression,
)

RAIZ = Path.cwd()
console = Console()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tesis-eeg-parkinson")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preprocess", help="Preprocesa ds002778 con un perfil")
    pre.add_argument("--profile", choices=sorted(PROFILES), default="v1")
    pre.add_argument("--bids-root", type=Path, default=RAIZ / "data" / "raw" / "ds002778")
    pre.add_argument("--out", type=Path, default=RAIZ / "data" / "processed")
    pre.add_argument("--figures", type=Path, default=RAIZ / "results" / "qc")
    pre.add_argument("--no-figures", action="store_true", help="No generar figuras de QC")
    pre.add_argument("--subject", nargs="*", default=None, help="Solo estos sujetos (sin 'sub-')")
    cmp = sub.add_parser("compare-matlab", help="Compara el perfil matlab contra EEGLAB")
    cmp.add_argument("--ours", type=Path, default=RAIZ / "data" / "processed" / "preproc-matlab")
    cmp.add_argument("--eeglab", type=Path, default=RAIZ / "data" / "processed" / "eeglab")
    cmp.add_argument("--figures", type=Path, default=RAIZ / "results" / "qc")
    feat = sub.add_parser("features", help="Calcula features sobre un preprocesamiento")
    feat.add_argument("--profile", choices=sorted(FEATURE_PROFILES), default="v1")
    feat.add_argument("--preproc", choices=sorted(PROFILES), default="v1")
    feat.add_argument("--processed", type=Path, default=RAIZ / "data" / "processed")
    cls = sub.add_parser("classify", help="Clasificación prerregistrada (docs/clasificacion.md)")
    cls.add_argument("--features", choices=sorted(FEATURE_PROFILES), default="v1")
    cls.add_argument("--preproc", choices=sorted(PROFILES), default="v1")
    cls.add_argument("--model", choices=MODELS, default="lr")
    cls.add_argument("--contrast", choices=[*CONTRASTS, "all"], default="all")
    cls.add_argument("--repeats", type=int, default=20)
    cls.add_argument("--n-perm", type=int, default=None, help="Por defecto 1000 en lr, 0 en otros")
    cls.add_argument("--null-repeats", type=int, default=2)
    cls.add_argument("--jobs", type=int, default=-1)
    cls.add_argument("--include-excluded", action="store_true")
    cls.add_argument("--intersect-preproc", choices=sorted(PROFILES), default=None)
    cls.add_argument("--channels", choices=["emotiv"], default=None)
    cls.add_argument("--no-leakage-control", action="store_true")
    cls.add_argument("--processed", type=Path, default=RAIZ / "data" / "processed")
    cls.add_argument("--out", type=Path, default=RAIZ / "results" / "classification")
    rep = sub.add_parser("report", help="Tabla y figuras de todos los análisis de clasificación")
    rep.add_argument("--results", type=Path, default=RAIZ / "results" / "classification")
    itp = sub.add_parser("interpret", help="Patrones de Haufe, estabilidad y topografías")
    itp.add_argument("--features", choices=sorted(FEATURE_PROFILES), default="v1")
    itp.add_argument("--preproc", choices=sorted(PROFILES), default="v1")
    itp.add_argument("--n-boot", type=int, default=200)
    itp.add_argument("--jobs", type=int, default=-1)
    itp.add_argument("--processed", type=Path, default=RAIZ / "data" / "processed")
    itp.add_argument("--out", type=Path, default=RAIZ / "results" / "interpretation")
    return parser


def _configure_logging() -> None:
    """Solo avisos de terceros; el progreso lo muestra la barra."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    mne.set_log_level("error")


def _group(row: pd.Series) -> str:
    return "HC" if row["session"] == "hc" else f"PD {row['session']}"


def _print_summary(summary: pd.DataFrame, profile: str) -> None:
    """Tabla por grupo: lo que conviene revisar antes de abrir los TSV."""
    table = Table(title=f"Preprocesamiento, perfil {profile}")
    for col in (
        "Grupo",
        "Reg.",
        "Errores",
        "PREP",
        "Interp.",
        "ICs",
        "Épocas",
        "Revisar",
        "Excluir",
    ):
        table.add_column(col, justify="right")
    data = summary.assign(grupo=summary.apply(_group, axis=1))
    ok = data[data["error"].fillna("") == ""]
    for group, rows in data.groupby("grupo"):
        good = ok[ok["grupo"] == group]
        table.add_row(
            str(group),
            str(len(rows)),
            str(len(rows) - len(good)),
            f"{good['n_prep_flagged'].mean():.1f}" if len(good) else "-",
            f"{good['n_interpolated'].mean():.1f}" if len(good) else "-",
            f"{good['n_ics_excluded'].mean():.1f}" if len(good) else "-",
            f"{int(good['n_epochs_kept'].sum())}/{int(good['n_epochs_total'].sum())}",
            str(int(good["needs_review"].sum())),
            str(int(good["exclude_primary"].sum())),
        )
    console.print(table)
    console.print(
        "PREP, Interp. e ICs: media por registro (canales que marca PREP, canales "
        "interpolados, componentes excluidos). Revisar: más de 3 interpolados. "
        "Excluir: fuera del análisis principal."
    )


def _preprocess(args: argparse.Namespace) -> None:
    found = len(find_recordings(args.bids_root))
    if found == 0:
        console.print(f"[red]No hay registros .bdf en {args.bids_root}.[/red] ¿Falta el dataset?")
        raise SystemExit(1)
    if found != EXPECTED_RECORDINGS and not args.subject:
        console.print(
            f"[yellow]Aviso: {found} registros; ds002778 v1.0.5 tiene "
            f"{EXPECTED_RECORDINGS}.[/yellow]"
        )
    columns = (
        SpinnerColumn("line"),
        TextColumn("{task.description:<34}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("preparando", total=None)

        def on_progress(done: int, total: int, name: str, stage: str) -> None:
            progress.update(task, completed=done, total=total, description=f"{name} · {stage}")

        summary = run_dataset(
            bids_root=args.bids_root,
            cfg=PROFILES[args.profile],
            out_root=args.out,
            figures_root=None if args.no_figures else args.figures,
            subjects=args.subject,
            on_progress=on_progress,
        )
    _print_summary(summary, args.profile)


def _compare_matlab(args: argparse.Namespace) -> None:
    with console.status("Comparando contra EEGLAB..."):
        table = compare_dataset(args.ours, args.eeglab)
    out = args.ours / "regression_matlab.tsv"
    table.to_csv(out, sep="\t", index=False)
    fig = plot_regression(table, args.figures / "regression_matlab.pdf")
    ok = table[table["error"] == ""]
    console.print(f"{len(ok)}/{len(table)} registros comparados. Tabla: {out}. Figura: {fig}")
    if len(ok):
        console.print(
            f"Correlación mediana por registro: mediana {ok['r_median'].median():.3f}, "
            f"mínima {ok['r_median'].min():.3f}; error RMS relativo: mediana "
            f"{ok['rel_rms'].median():.3f}."
        )
        for metric, (rho, p) in ica_association(ok).items():
            console.print(
                f"Spearman |EEGLAB − MNE componentes| vs {metric}: rho = {rho:.2f}, p = {p:.2g}"
            )


def _features(args: argparse.Namespace) -> None:
    cfg = FEATURE_PROFILES[args.profile]
    columns = (
        SpinnerColumn("line"),
        TextColumn("{task.description:<34}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("preparando", total=None)

        def on_progress(done: int, total: int, name: str) -> None:
            progress.update(task, completed=done, total=total, description=f"{name} · features")

        table = run_features(
            preproc_root=args.processed / f"preproc-{args.preproc}",
            cfg=cfg,
            out_root=args.processed,
            on_progress=on_progress,
        )
    n_feat = sum(1 for c in table.columns if "__" in c and not c.startswith("qc_"))
    nan = table[[c for c in table.columns if "__" in c and not c.startswith("qc_")]].isna()
    console.print(
        f"features-{cfg.name}/preproc-{args.preproc}: {len(table)} épocas × {n_feat} features. "
        f"Valores NaN: {int(nan.to_numpy().sum())}."
    )


def _classify(args: argparse.Namespace) -> None:
    from tesis_eeg_parkinson.preprocessing.config import EMOTIV_CHANNELS

    features = (
        args.processed / f"features-{args.features}" / f"preproc-{args.preproc}" / "features.tsv"
    )
    n_perm = args.n_perm if args.n_perm is not None else (1000 if args.model == "lr" else 0)
    name = f"{args.features}_{args.preproc}_{args.model}"
    intersect = None
    if args.intersect_preproc:
        intersect = features.parent.parent / f"preproc-{args.intersect_preproc}" / "features.tsv"
        name += f"_inter-{args.intersect_preproc}"
    if args.include_excluded:
        name += "_incl-excluded"
    if args.channels:
        name += f"_ch-{args.channels}"
    contrasts = tuple(CONTRASTS) if args.contrast == "all" else (args.contrast,)
    columns = (SpinnerColumn("line"), TextColumn("{task.description:<34}"), BarColumn(),
               MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn())  # fmt: skip
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("preparando", total=None)

        def on_event(contrast: str, done: int, total: int) -> None:
            stage = "validación" if done == 0 else "permutaciones"
            progress.update(
                task, completed=done, total=max(total, 1), description=f"{contrast} · {stage}"
            )

        summary = run_analysis(
            features, args.out / name, args.model, contrasts, args.repeats, n_perm,
            args.null_repeats, n_jobs=args.jobs, include_excluded=args.include_excluded,
            intersect_with=intersect,
            channels=EMOTIV_CHANNELS if args.channels == "emotiv" else None,
            leakage_control=not args.no_leakage_control, on_event=on_event,
        )  # fmt: skip
    table = Table(title=f"Clasificación {name}")
    for col in (
        "Contraste",
        "Registros",
        "Exact. bal. [IC 95%]",
        "AUC",
        "Azar (P95)",
        "p Holm",
        "Fuga",
    ):
        table.add_column(col, justify="right")
    for _, r in summary.iterrows():
        table.add_row(
            r["contrast"],
            f"{int(r['n_recordings'])}",
            f"{r['bacc']:.3f} [{r['bacc_ci_low']:.3f}, {r['bacc_ci_high']:.3f}]",
            f"{r['auc']:.3f}",
            f"{r['chance_p95']:.3f}" if "chance_p95" in r and pd.notna(r["chance_p95"]) else "-",
            f"{r['p_holm']:.4f}" if "p_holm" in r and pd.notna(r["p_holm"]) else "-",
            f"{r['epoch_bacc_epoch_cv']:.3f}" if "epoch_bacc_epoch_cv" in r else "-",
        )
    console.print(table)
    console.print(
        "Exactitud balanceada por registro. Azar: percentil 95 de la nula por permutación. "
        "Fuga: exactitud por época con validación por épocas (control, no desempeño)."
    )


def _report(args: argparse.Namespace) -> None:
    paths = write_report(args.results)
    table = pd.read_csv(paths["resumen"], sep="\t")
    inflation = table["epoch_bacc_epoch_cv"] - table["epoch_bacc_subject_cv"]
    console.print(
        f"{table['analysis'].nunique()} análisis, {len(table)} filas. Inflación por fuga "
        f"(época, misma unidad): mediana {inflation.median():.3f}, "
        f"rango {inflation.min():.3f} a {inflation.max():.3f}."
    )
    for name, path in paths.items():
        console.print(f"{name}: {path}")


def _interpret(args: argparse.Namespace) -> None:
    features = (
        args.processed / f"features-{args.features}" / f"preproc-{args.preproc}" / "features.tsv"
    )
    out = args.out / f"{args.features}_{args.preproc}"
    columns = (SpinnerColumn("line"), TextColumn("{task.description:<34}"), BarColumn(),
               MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn())  # fmt: skip
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("preparando", total=None)

        def on_event(contrast: str, done: int, total: int) -> None:
            progress.update(
                task, completed=done, total=total, description=f"{contrast} · bootstrap"
            )

        long = run_interpretation(features, out, args.n_boot, n_jobs=args.jobs, on_event=on_event)
    for contrast in long["contrast"].unique():
        plot_contrast_topomaps(long, contrast, out / f"patrones_{contrast}.pdf")
    plot_magnitude(long, out / "magnitud.pdf")
    stable = long.assign(estable=long["stability"] >= 0.95).groupby("contrast")["estable"].sum()
    console.print(f"Features × canales con signo estable (≥95%): {stable.to_dict()}")
    console.print(f"Tabla y figuras en {out}")


def main(argv: list[str] | None = None) -> None:
    """Ejecuta el subcomando indicado."""
    args = _parser().parse_args(argv)
    _configure_logging()
    if args.command == "preprocess":
        _preprocess(args)
    elif args.command == "compare-matlab":
        _compare_matlab(args)
    elif args.command == "features":
        _features(args)
    elif args.command == "classify":
        _classify(args)
    elif args.command == "report":
        _report(args)
    elif args.command == "interpret":
        _interpret(args)


if __name__ == "__main__":
    main()
