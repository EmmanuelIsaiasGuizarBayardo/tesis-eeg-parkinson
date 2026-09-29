"""Configuración del preprocesamiento: dos perfiles completos y explícitos.

`V2` es el pipeline de la tesis. `LEGACY` reproduce el pipeline de MATLAB/EEGLAB
del manual original y existe solo para validar la migración: se compara contra
los resultados del legado y después se cambia un factor a la vez hacia `V2`.
Cada derivado guarda `as_dict()` en su JSON de QC, así que el perfil exacto que
lo produjo queda registrado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

# Montaje BioSemi 32 en el orden del archivo BDF (channels.tsv de ds002778).
SCALP_CHANNELS: tuple[str, ...] = (
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
)  # fmt: skip

# Subconjunto del Emotiv EPOC X presente en BioSemi 32.
EMOTIV_CHANNELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)  # fmt: skip

# Orden de columnas de ICLabel (idéntico en EEGLAB y en mne-icalabel).
ICLABEL_CLASSES: tuple[str, ...] = (
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
)  # fmt: skip

# Valor del evento que, en ds002778, marca el inicio del reposo (inferido: no
# está documentado; ver docs/preprocesamiento.md).
REST_ONSET_VALUE = "1"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parámetros de un perfil de preprocesamiento.

    Attributes
    ----------
    name : str
        Nombre del perfil; nombra la carpeta de derivados.
    montage : str
        Montaje estándar de MNE para las posiciones de los electrodos.
    crop_to_rest_onset : bool
        Recortar el registro desde el evento de inicio del reposo.
    detect_bad_channels : bool
        Detectar canales malos (criterios deterministas de PREP) e interpolarlos
        antes de la CAR.
    max_bad_channels : int
        Por encima de este número, el registro se marca para revisión manual.
    ica_copy_l_freq, ica_copy_h_freq : float, float or None
        Banda FIR de la copia sobre la que se ajusta la ICA.
    ica_max_iter : int or "auto"
        Iteraciones máximas de infomax extendido.
    iclabel_on : {"ica_copy", "master"}
        Instancia sobre la que se calculan las features de ICLabel.
    iclabel_threshold : float
        Probabilidad mínima (estricta) de la clase dominante para rechazar.
    iclabel_other_brain_max : float
        Un IC "other" dominante se rechaza solo si P(brain) es menor que esto.
    final_l_freq, final_h_freq : float
        Banda del Butterworth final (Hz).
    final_order : int
        Orden de diseño del Butterworth (el pasabanda resultante tiene 2N polos).
    final_form : {"sos", "ba"}
        Forma numérica del filtro final. "ba" replica MATLAB [b,a] + filtfilt.
    epoch_duration_s : float
        Duración de las épocas, sin solapamiento.
    reject_ptp_uv : float or None
        Umbral pico a pico fijo, en µV, fijado a priori. None = sin rechazo.
    seed : int
        Semilla de la ICA y de toda operación aleatoria.
    """

    name: str
    montage: str = "standard_1005"
    crop_to_rest_onset: bool = True
    detect_bad_channels: bool = True
    max_bad_channels: int = 3
    ica_copy_l_freq: float = 1.0
    ica_copy_h_freq: float | None = 100.0
    ica_max_iter: int | Literal["auto"] = "auto"
    iclabel_on: Literal["ica_copy", "master"] = "ica_copy"
    iclabel_threshold: float = 0.80
    iclabel_other_brain_max: float = 0.05
    final_l_freq: float = 0.5
    final_h_freq: float = 32.0
    final_order: int = 5
    final_form: Literal["sos", "ba"] = "sos"
    epoch_duration_s: float = 10.0
    reject_ptp_uv: float | None = 150.0
    seed: int = 28

    def as_dict(self) -> dict[str, Any]:
        """Devuelve el perfil como diccionario serializable."""
        return asdict(self)


V2 = PreprocessingConfig(name="v2")

LEGACY = PreprocessingConfig(
    name="legacy",
    crop_to_rest_onset=False,
    detect_bad_channels=False,
    ica_copy_h_freq=None,
    iclabel_on="master",
    final_form="ba",
    reject_ptp_uv=None,
)

PROFILES: dict[str, PreprocessingConfig] = {p.name: p for p in (V2, LEGACY)}
