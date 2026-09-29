"""Punto de entrada de Tesis_EEG_Parkinson.

Uso::

    uv run tesis-eeg-parkinson preprocess --profile legacy
    uv run tesis-eeg-parkinson preprocess --profile v2 --subject hc1 pd3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tesis_eeg_parkinson.preprocessing import PROFILES, run_dataset

RAIZ = Path.cwd()


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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Ejecuta el subcomando indicado."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "preprocess":
        summary = run_dataset(
            bids_root=args.bids_root,
            cfg=PROFILES[args.profile],
            out_root=args.out,
            figures_root=None if args.no_figures else args.figures,
            subjects=args.subject,
        )
        failed = summary["error"].astype(bool).sum() if "error" in summary else 0
        print(f"{len(summary)} registros procesados; {failed} con error.")


if __name__ == "__main__":
    main()
