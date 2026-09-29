"""Punto de entrada de Tesis_EEG_Parkinson."""

from __future__ import annotations

from tesis_eeg_parkinson.reproducibility import get_run_context, set_seed


def main() -> None:
    """Ejecuta el pipeline principal."""
    set_seed(42)
    ctx = get_run_context(seed=42)
    print("Entorno de Tesis_EEG_Parkinson listo.")
    print(f"commit={ctx.git_sha[:8]} dirty={ctx.git_dirty} seed={ctx.seed}")


if __name__ == "__main__":
    main()
