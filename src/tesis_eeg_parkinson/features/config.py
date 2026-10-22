"""Configuración de features: perfiles completos y explícitos.

- `V1`: features de la tesis. Todas invariantes a la escala de la señal (ganancia,
  impedancia, unidades), porque el objetivo incluye transferir a otro equipo. Tres
  leen el orden temporal, y una de ellas, su dirección (forma de onda). Se eligen
  por sus propiedades, no por su desempeño: escogerlas por su separación entre
  grupos con estos mismos datos sería fuga de selección.
- `ALJALAL`: la configuración publicada por Aljalal et al. (2022), para la
  comparación con el método original. No es un método de la tesis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class FeatureConfig:
    """Parámetros de un perfil de features.

    Attributes
    ----------
    name : str
        Nombre del perfil; nombra la carpeta de salida.
    family : {"v1", "aljalal"}
        Conjunto de métricas.
    sfreq : float
        Frecuencia de muestreo de trabajo. 128 Hz coloca el árbol db4 de 4 niveles
        sobre delta, theta, alfa y beta; es además un modo del EPOC X.
    wavelet, level : str, int
        DWT diádica.
    bands : tuple of str
        Coeficientes que se conservan como señales de banda.
    include_broadband : bool
        Agregar la señal completa como señal adicional.
    permen_order : int
        Orden de la entropía de permutación (m).
    permen_delays : tuple of int
        Retrasos (τ): cada uno mira una escala temporal distinta.
    t3_delays : tuple of int
        Retrasos de la asimetría temporal de los incrementos (forma de onda).
    sampen_order : int
        Orden de la entropía muestral (m).
    sampen_r : float
        Tolerancia de la entropía muestral, como fracción de la DE de la señal.
    psd_window_s : float
        Ventana de Welch para el espectro.
    aperiodic_range : tuple of float
        Rango del ajuste aperiódico (lejos de los bordes del filtro de 0.5 a 32 Hz).
    alpha_range : tuple of float
        Rango del centro de gravedad de alfa.
    """

    name: str
    family: Literal["v1", "aljalal"]
    sfreq: float
    wavelet: str = "db4"
    level: int = 4
    bands: tuple[str, ...] = ("cA4", "cD4", "cD3", "cD2")
    include_broadband: bool = True
    permen_order: int = 4
    permen_delays: tuple[int, ...] = (1, 2, 4)
    t3_delays: tuple[int, ...] = (1, 2, 4)
    sampen_order: int = 2
    sampen_r: float = 0.2
    psd_window_s: float = 2.0
    aperiodic_range: tuple[float, float] = (2.0, 30.0)
    alpha_range: tuple[float, float] = (7.0, 14.0)

    def as_dict(self) -> dict[str, Any]:
        """Devuelve el perfil como diccionario serializable."""
        return asdict(self)


V1 = FeatureConfig(name="v1", family="v1", sfreq=128.0)

# Aljalal et al. (2022): 512 Hz, cinco coeficientes más la señal original, ocho métricas.
ALJALAL = FeatureConfig(
    name="aljalal",
    family="aljalal",
    sfreq=512.0,
    bands=("cA4", "cD4", "cD3", "cD2", "cD1"),
)

PROFILES: dict[str, FeatureConfig] = {p.name: p for p in (V1, ALJALAL)}
