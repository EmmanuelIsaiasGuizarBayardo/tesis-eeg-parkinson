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
- La fuga se reproduce en los diez análisis. En la misma unidad (exactitud por época),
  la validación por épocas suma de 21 a 46 puntos porcentuales (mediana 36). En el
  análisis principal: Off vs HC 0.54 contra 0.94; On vs HC 0.59 contra 0.96; Off vs On
  0.58 contra 0.88.
- Promediar las épocas de un registro mejora la exactitud por sujeto (0.54–0.59 por
  época, 0.61–0.69 por registro): una decisión necesita minutos, no una época de 10 s.
- Con las features de Aljalal, la validación por épocas da más (0.978 a 0.995), pero
  la validación por sujeto también; la inflación es comparable a la de V1 (mediana
  0.380 contra 0.369), salvo en Off vs On (0.463 contra 0.295). Corrige una primera
  lectura de esta bitácora que decía que inflaban más.
- Con pd6 y pd16 ON incluidas, Off vs On sube a 0.733 (p Holm = 0.036 dentro de su
  propia familia, no confirmatoria): un resultado atribuible a la procedencia de esas
  sesiones, no a la medicación. Verificado registro por registro: los 4 registros
  añadidos se clasificaron bien (pd6 on con probabilidad 0.07 de ser off) y los otros
  26 conservaron sus 18 aciertos. El control de calidad previo a la clasificación
  evitó reportarlo como efecto de la medicación.
- Transferencia, mismos sujetos (32 canales, 14 canales, EPOC X emulado): Off vs HC
  0.600, 0.567 y 0.467; On vs HC 0.679, 0.579 y 0.436; Off vs On 0.692, 0.654 y
  0.654. Descriptivo: las diferencias caben en los IC, y la emulación es un límite
  superior.
- Sin señal no lineal consistente: el bosque aleatorio da el valor más alto de los 30
  (On vs HC, 0.745), pero sin permutaciones y elegido entre muchos; no se interpreta.
- Off vs On compara dos sesiones (día, colocación del gorro): la medicación es una
  causa posible de la diferencia, no la única.

## Etapa 5. Interpretación

Commit `27b2813` (patrones y estabilidad); topografías corregidas en el commit de esta
entrada. `uv run tesis-eeg-parkinson interpret`.

**Método.** Patrones de activación de Haufe et al. (2014), A = Cov(Z)·w, del modelo
principal ajustado con todos los registros, sobre features estandarizadas. Estabilidad
del signo: 200 remuestreos de sujetos por contraste, cada uno con su validación anidada;
un canal es estable si su signo se mantiene en al menos 95% de ellos. Adición
exploratoria, decidida antes de mirar los patrones (ver Etapa 4).

**Alcance.** Son patrones de modelos que no superaron la prueba confirmatoria. Un signo
estable indica que la asociación se repite al remuestrear a estas personas; no que
generalice ni que permita clasificar individuos. En la tesis van como resultados
descriptivos y como hipótesis para la discusión.

**Qué mide cada feature.** El centro de alfa es periódico (potencia por encima del
ajuste aperiódico de FOOOF); el exponente es aperiódico; la potencia relativa mezcla
ambos componentes y sus cuatro bandas suman 1.

**Resultados.**

| Componente (features) | Off vs HC | On vs HC | Off vs On |
|---|---|---|---|
| Potencia relativa, mixta (4) | 35.8% (39) | 34.3% (35) | 26.7% (26) |
| Orden temporal: PermEn ×3, SampEn (4) | 25.3% (12) | 27.0% (12) | 26.8% (12) |
| Entropía wavelet (4) | 16.2% (31) | 16.1% (24) | 15.8% (9) |
| Periódico: centro de alfa (1) | 10.4% (18) | 12.0% (15) | 3.9% (5) |
| Forma de onda: T3 (3) | 7.0% (1) | 7.2% (11) | 15.3% (22) |
| Aperiódico: exponente (1) | 5.3% (1) | 3.5% (1) | 11.5% (11) |

Porcentaje de la suma de |patrón| del contraste; entre paréntesis, pares feature-canal
con signo estable (102, 98 y 85 en total, de 544).

1. **Off vs HC y On vs HC: lentificación oscilatoria difusa.** Mayor potencia relativa
   de 4 a 8 Hz en PD (30 y 32 de 32 canales estables), menor centro de alfa (18 y 15,
   sobre todo posteriores y temporales) y exponente casi sin participación (1 canal).
   Lectura: el pico de alfa se desplaza hacia theta sin cambio de la pendiente 1/f,
   coherente con Stoffers et al. (2007). Aparece igual con y sin medicación.
2. **Off vs On: un perfil distinto.** El exponente es la feature de mayor peso (11
   canales estables; espectro más plano en OFF); la potencia de 4 a 8 Hz es mayor en ON
   (21 canales); T3 tiene 22 canales estables, con signos opuestos entre la región
   posterior (P3, PO3, Oz, O2) y la frontal (Fp1, F3, FC1); y PermEn, SampEn y la
   potencia de 16 a 32 Hz son mayores en OFF en T7, T8 y FC5.

**Discusión: dos explicaciones para el espectro más plano en OFF.**

- *Cortical.* El exponente aperiódico se ha propuesto como índice del balance entre
  excitación e inhibición (Gao et al., 2017); un espectro más plano sería más excitación
  relativa sin medicación.
- *Muscular.* La actividad del músculo eleva la potencia de alta frecuencia en canales
  temporales (Muthukumaraswamy, 2013) y, con ello, aplana el espectro. OFF tiene más
  complejidad y más potencia de 16 a 32 Hz justo en T7, T8 y FC5, los canales que
  causaron las épocas extra rechazadas a 43 Hz, todas en PD.
- *Lo que pesa en cada lado.* 5 de los 11 canales estables del exponente son
  centro-parietales (C3, C4, CP2, Pz, PO4), lejos del músculo temporal, lo que no apoya
  una explicación solo muscular; pero la CAR reparte el artefacto de los canales
  temporales entre todos los canales, así que tampoco la descarta. Con estos datos no se
  puede separar. Además, Off vs On compara dos sesiones: la medicación es una causa
  posible, no la única.
- *T3.* Con el corte de 32 Hz lee la forma de onda de alfa, no la de beta.

**Figuras.** Una por contraste (`patrones_<contraste>.pdf`) y `magnitud.pdf`. Las
topografías usan el montaje `biosemi32`, el del gorro del registro: con `colin27_1005`,
Fp1 y Fp2 se proyectaban fuera del contorno y el mapa se desbordaba. Se fijaron
`extrapolate="head"` y `border="mean"`; una prueba verifica que los 32 electrodos caen
dentro del contorno.

## Etapa 5b. Exponente aperiódico y músculo (exploratorio)

**Plan, fijado antes de correrlo (25 de octubre de 2026).** Pregunta: ¿el espectro más
plano en OFF que sugieren los patrones de Off vs On es cortical o muscular? Es una
comparación directa por registro, no un patrón de clasificador:

- 13 pacientes con ambas sesiones (los pares del análisis principal), épocas de V1.
- Exponente aperiódico (FOOOF, modo fijo) por registro y canal, promediado en dos grupos
  fijados de antemano: laterales (F7, F8, FC5, FC6, T7, T8), sobre el músculo temporal,
  y centro-parietales (C3, Cz, C4, CP1, CP2, P3, Pz, P4), lejos de él.
- 8 condiciones, todas reportadas: referencia CAR o CSD (laplaciano de superficie, que
  limita la difusión desde canales lejanos, incluida la que reparte la CAR) × ajuste de
  2 a 30 Hz (el de V1) o de 2 a 20 Hz (menos sensible al músculo) × grupo de canales.
- Índice muscular por registro, del preprocesamiento a 43 Hz: log10 de la potencia de
  32 a 43 Hz sobre la de 2 a 30 Hz, en los canales laterales.
- Por condición: media de OFF − ON con IC 95% por bootstrap de pacientes, Wilcoxon
  exacto y Spearman entre el cambio del exponente y el del índice muscular. Valores p
  sin corregir: es exploratorio.

**Reglas de lectura, fijadas antes de correrlo.**

- Apoya la explicación **cortical** si el exponente es menor en OFF en los canales
  centro-parietales con CSD y con el ajuste de 2 a 20 Hz (IC sin el 0), y su cambio no
  se asocia con el del índice muscular.
- Apoya la explicación **muscular** si la diferencia se concentra en los canales
  laterales, se desvanece en los centro-parietales con CSD o con 2 a 20 Hz (IC con el
  0), y se asocia negativamente con el cambio del índice muscular.
- Cualquier otro patrón es **no concluyente** y se reporta tal cual. Con 13 pares, un
  IC que incluye el 0 no demuestra ausencia de efecto.

`uv run tesis-eeg-parkinson aperiodic-check`

## Pendientes para el final

**Análisis posibles, fuera del pipeline actual.**

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
