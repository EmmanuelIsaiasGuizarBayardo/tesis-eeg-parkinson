"""Ingreso único de ds002778 v1.0.5 a data/raw (paso de ingreso, no de procesamiento).

Es el único script que escribe en data/raw, y se niega a hacerlo si el destino ya
tiene contenido: después del ingreso, data/raw es inmutable.

Uso::

    uv run python tools/download_ds002778.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import openneuro

DATASET, TAG = "ds002778", "1.0.5"
DESTINO = Path(__file__).resolve().parent.parent / "data" / "raw" / DATASET


def main() -> None:
    """Descarga la versión fijada y muestra la fila de procedencia."""
    if DESTINO.exists() and any(DESTINO.iterdir()):
        raise SystemExit(f"{DESTINO} ya tiene contenido; data/raw es inmutable.")
    DESTINO.mkdir(parents=True, exist_ok=True)
    openneuro.download(dataset=DATASET, tag=TAG, target_dir=DESTINO)
    hoy = dt.date.today().isoformat()
    print("\nAgrega esta fila a la tabla de procedencia de data/raw/README.md:")
    doi = f"doi:10.18112/openneuro.{DATASET}.v{TAG}"
    print(f"| {DATASET} v{TAG} | OpenNeuro, {doi} | {hoy} | CC0 |")


if __name__ == "__main__":
    main()
