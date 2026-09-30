"""Punto de entrada de Tesis_EEG_Parkinson.

Uso::

    uv run tesis-eeg-parkinson preprocess --profile legacy
    uv run tesis-eeg-parkinson preprocess --profile v2 --subject hc1 pd3
    uv run tesis-eeg-parkinson compare-legacy
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

from tesis_eeg_parkinson.preprocessing import PROFILES, run_dataset
from tesis_eeg_parkinson.validation.legacy_matlab import (
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
    pre.add_argument("--profile", choices=sorted(PROFILES), default="v2")
    pre.add_argument("--bids-root", type=Path, default=RAIZ / "data" / "raw" / "ds002778")
    pre.add_argument("--out", type=Path, default=RAIZ / "data" / "processed")
    pre.add_argument("--figures", type=Path, default=RAIZ / "results" / "qc")
    pre.add_argument("--no-figures", action="store_true", help="No generar figuras de QC")
    pre.add_argument("--subject", nargs="*", default=None, help="Solo estos sujetos (sin 'sub-')")
    cmp = sub.add_parser("compare-legacy", help="Compara el perfil legacy contra EEGLAB")
    cmp.add_argument("--ours", type=Path, default=RAIZ / "data" / "processed" / "preproc-legacy")
    cmp.add_argument("--matlab", type=Path, default=RAIZ / "data" / "processed" / "legacy-matlab")
    cmp.add_argument("--figures", type=Path, default=RAIZ / "results" / "qc")
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


def _compare_legacy(args: argparse.Namespace) -> None:
    with console.status("Comparando contra EEGLAB..."):
        table = compare_dataset(args.ours, args.matlab)
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


def main(argv: list[str] | None = None) -> None:
    """Ejecuta el subcomando indicado."""
    args = _parser().parse_args(argv)
    _configure_logging()
    if args.command == "preprocess":
        _preprocess(args)
    elif args.command == "compare-legacy":
        _compare_legacy(args)


if __name__ == "__main__":
    main()
