# Bitácora del proyecto

Una entrada por etapa: qué se decidió, qué se obtuvo y qué queda pendiente. El
método completo vive en `preprocesamiento.md`, `features.md` y `transferibilidad.md`;
las cifras salen de los archivos que genera el código (`qc_summary.tsv`,
`features.json`) y se citan con el commit que las produjo. Al cerrar cada etapa se
agrega su entrada; nada de lo que aquí aparece se recuerda "de memoria".

## Etapa 1. Preprocesamiento V1

Versión `v0.2.0`; reglas de épocas desde `8ee8843`. Cifras de la corrida posterior a
`8ee8843`.

**Decisiones.** ds002778 v1.0.5; 32 canales por nombre; recorte en el evento "1";
notch de 60 Hz y armónicos; PREP solo como diagnóstico y se interpolan solo canales
planos o con NaN; CAR; ICA infomax extendida (n = rango) sobre una copia de 1 a
100 Hz, con ICLabel sobre la copia; Butterworth de 0.5 a 32 Hz en SOS; épocas de
10 s; rechazo pico a pico de 150 µV; un canal sobre el umbral en más de la mitad de
las épocas se interpola; mínimo de 6 épocas. Fuera del análisis principal: pd6 on y
pd16 on, preprocesadas por los curadores.

| Grupo | Épocas | ICs excluidos por registro |
|---|---|---|
| HC | 276/290 | 7.6 |
| PD off | 261/284 | 8.0 |
| PD on | 260/271 | 5.3 |

Única interpolación: F3 en sub-hc1 (falla persistente; recuperó sus 11 épocas).

**Hallazgos para la tesis.**

- PREP marca canales laterales y frontales por correlación y espectro: artefacto
  muscular y ocular, no fallas. Interpolarlos debilitaba la ICA, reconstruía canales
  del EPOC X desde electrodos ajenos y afectaba más a HC.
- La forma [b,a] del Butterworth original está mal condicionada (−1.77 dB en 0.5 Hz
  en lugar de −3.01).
- 9 registros declaran 50 Hz de red; el campo no es confiable.
- Apéndice de concordancia con el procesamiento previo, listo en `preprocesamiento.md`.

## Etapa 2. Features V1

Commits `8ee8843` y `c09bc08` (T3).

**Decisiones.** 17 features por canal, todas invariantes a la escala y elegidas por
sus propiedades, no por su desempeño: potencia relativa y entropía wavelet
normalizada en cuatro bandas (128 Hz, db4), PermEn y T3 en tres retrasos, SampEn,
exponente aperiódico y centro de gravedad de alfa. Configuración `aljalal` para la
comparación con el método publicado.

| Tabla | Épocas | Features | NaN |
|---|---|---|---|
| `v1/preproc-v1` | 797 | 544 | 11 |
| `aljalal/preproc-v1` | 797 | 1,728 | 0 |

**Hallazgos para la tesis.**

- La ShEn del artículo es E·(log E − H): sobre todo energía (ρ = 0.998 con Energy).
- TShEn, en cualquiera de sus dos lecturas, mide la fracción de muestras pequeñas.
- PermEn y SampEn no distinguen la dirección del tiempo; T3 sí.

## Etapa 3. Variantes y transferibilidad

Perfiles `v1-prep`, `v1-43`, `matlab` y `epocx` (este último desde `b64e6b4`).

| Perfil | HC | PD off | PD on | ICs por registro (HC/off/on) |
|---|---|---|---|---|
| `v1-prep` | 280/290 | 269/284 | 263/271 | 2.2 / 3.7 / 4.1 |
| `v1-43` | 276/290 | 253/284 | 256/271 | 7.6 / 8.0 / 5.3 |
| `epocx` | 260/290 | 250/284 | 255/271 | 2.2 / 1.9 / 1.5 |
| `matlab` | 300/300 | 293/293 | 291/291 | 7.8 / 5.8 / 6.3 |

| Tabla de features | Épocas | Features | NaN |
|---|---|---|---|
| `v1/preproc-v1-43` | 785 | 544 | 16 |
| `v1/preproc-epocx` | 765 | 238 | 71, todos en `alpha_cog` |

**Hallazgos para la tesis.**

- `v1-43`: 12 épocas más rechazadas, todas en PD; 7 por FC5, F7 o T8 (músculo
  temporal) y en canales del EPOC X. Respalda el corte de 32 Hz.
- `epocx`: la referencia robusta de PREP abortó en hc33, hc4, hc8 y pd9 (marcó al
  menos 13 de 14 canales); la ICA excluye menos componentes; sub-hc1 conserva 5 de
  18 épocas y sale del análisis principal, aunque en V1 sí entra.
- `matlab`: concordancia con EEGLAB de 0.958 (mediana por registro); la diferencia
  la explica la ICA.

## Etapa 4. Clasificación

Prerregistro `7d80c45`, commiteado antes de cualquier corrida; código `bf1d203`. Se
corrieron el análisis principal y los nueve secundarios prerregistrados, sin
desviaciones. Tabla y figuras: `uv run tesis-eeg-parkinson report`.

**Principal** (V1, regresión logística L2 anidada; exactitud balanceada por registro):

| Contraste | Registros | Exactitud [IC 95%] | AUC | Azar (P95) | p Holm | Fuga (época) |
|---|---|---|---|---|---|---|
| Off vs HC | 31 | 0.613 [0.443, 0.777] | 0.571 | 0.677 | 0.192 | 0.944 |
| On vs HC | 29 | 0.659 [0.477, 0.829] | 0.688 | 0.690 | 0.192 | 0.961 |
| Off vs On | 26 | 0.692 [0.577, 0.808] | 0.675 | 0.654 | 0.081 | 0.875 |

**Hallazgos para la tesis.**

- Ningún contraste supera el criterio prerregistrado (Holm < 0.05). Off vs On supera
  el percentil 95 de su nula sin corregir (p = 0.027): exploratorio.
- El azar empírico (P95) va de 0.65 a 0.71 en todos los análisis: con unos 30
  registros, superar el azar exige alrededor de 0.68, no 0.5.
- La fuga se reproduce en los diez análisis: con validación por épocas, de 0.81 a
  0.995; por sujeto, de 0.44 a 0.75.
- Con las features de Aljalal, la fuga es mayor (0.978 a 0.995) que con V1 (0.875 a
  0.961). Hipótesis: las features dependientes de la amplitud codifican la identidad
  del sujeto; también son tres veces más.
- Con pd6 y pd16 ON incluidas, Off vs On sube a 0.733 (p Holm = 0.036 dentro de su
  propia familia, no confirmatoria): sus sesiones ON tienen otra procedencia. Respalda
  la exclusión a priori.
- Transferencia, mismos sujetos (32 canales, 14 canales, EPOC X emulado): Off vs HC
  0.600, 0.567 y 0.467; On vs HC 0.679, 0.579 y 0.436; Off vs On 0.692, 0.654 y
  0.654. Descriptivo: las diferencias caben en los IC, y la emulación es un límite
  superior.
- Sin señal no lineal consistente: el bosque aleatorio da el valor más alto de los 30
  (On vs HC, 0.745), pero sin permutaciones y elegido entre muchos; no se interpreta.
- Off vs On compara dos sesiones (día, colocación del gorro): la medicación es una
  causa posible de la diferencia, no la única.

## Pendientes para el final

**Para la interpretación (siguiente etapa).**

- Topografías de los patrones de Haufe del análisis principal (`patterns.tsv`).
- Piloto con un EPOC X real: la emulación es un límite superior.

**Para el texto de la tesis.**

- Corregir el 0.589 de la §7.3 (es KNN + ShEn con 14 canales; KNN + LogEn con 32
  da 0.471), aclarar que la Tabla 6 salía de canales elegidos por *forward-addition*,
  invertir la lectura de ShEn y TShEn y llamar DWT a lo que se llamaba
  *wavelet packet*.
- Citar a Appelhoff et al. (2019), como piden los curadores del dataset.

**Fuera del repositorio.**

- Escribir a los curadores del dataset antes de publicar en revistas arbitradas.
- Camera-ready del CNIB: los "p < 0.001" sin permutación válida, "no loss of
  accuracy" y "within 0.3%".
- Migrar de `fooof` a `specparam` cuando publique una versión estable.
