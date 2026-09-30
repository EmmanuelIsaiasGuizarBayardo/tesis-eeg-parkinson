# Preprocesamiento de ds002778

Migración validada a Python del pipeline de EEGLAB del manual original. El código
vive en `src/tesis_eeg_parkinson/preprocessing/` y cada parámetro, en
`preprocessing/config.py`. Hay dos perfiles completos:

- **`legacy`** reproduce el pipeline de MATLAB. Existe solo para validar la
  migración contra los resultados anteriores.
- **`v2`** es el pipeline de la tesis. Se llega a él desde `legacy` cambiando un
  factor a la vez, para poder atribuir cada diferencia a una decisión.
- **`v2-prep`** es igual a `v2` pero interpola todo lo que marca PREP. Existe solo
  como variante del análisis de sensibilidad.

Cada derivado guarda en su JSON de QC el perfil completo y el `RunContext`
(semilla, commit y estado del árbol).

## Lo que el dataset obliga a considerar

Revisado en los metadatos de la versión 1.0.5 (Rockhill et al., 2021).

- **Canales.** `channels.tsv` declara 40 canales EEG (los 32 del cuero cabelludo y
  EXG1 a EXG8) y un canal Status. Se selecciona por nombre, no por índice.
- **Referencia.** `EEGReference` es "n/a": BioSemi registra contra el CMS y los
  archivos no traen referencia previa. La CAR es necesaria, no redundante.
- **Inicio del reposo.** Un evento de valor "1" aparece segundos después del inicio
  (entre 2.9 y 21.8 s en los registros revisados; 95.6 s en sub-pd6 ON; ausente en
  sub-hc4). Se interpreta como inicio del reposo: con él, sub-pd6 ON queda en unos
  193 s, como los demás. La interpretación es inferida, no documentada.
- **Procedencia heterogénea.** Según `participants.tsv`, las sesiones ON de sub-pd6
  y sub-pd16 son datos preprocesados por los curadores, no crudos. Su espectro lo
  confirma (ver "Exclusión del análisis principal").
- **Metadatos de la red eléctrica.** 9 registros declaran `PowerLineFrequency` de
  50 Hz (hc8, hc21, hc25, hc31, hc32, pd6 on, pd13 on, pd16 on y pd22 on), aunque
  la red de San Diego es de 60 Hz. El campo no es confiable: el pipeline usa 60 Hz
  y registra la discrepancia en el QC.
- **Duración.** Unos 3.2 min por registro.
- **Advertencia de los curadores.** El README del dataset desaconseja clasificar PD
  contra controles con aprendizaje automático por el tamaño de la muestra y pide
  contactar a los curadores antes de publicar en revistas arbitradas.

## Pipeline

| Paso | `legacy` | `v2` | Motivo |
|---|---|---|---|
| Lectura | 32 canales, montaje colin27_1005 | Igual | Selección por nombre |
| Recorte | Ninguno | Desde el evento "1", si existe | Excluir el tramo previo al reposo |
| Notch | Ninguno | 60, 120, 180 y 240 Hz | PREP quita la línea antes de referenciar |
| Canales malos | Ninguno | PREP como diagnóstico; se interpolan solo planos o con NaN | Ver abajo |
| Referencia | CAR | CAR | Necesaria |
| Copia para ICA | FIR pasa-altas 1 Hz | FIR 1 a 100 Hz | Especificación de ICLabel |
| ICA | Infomax extendido | Igual; n = 31 menos interpolados | La CAR reduce el rango en 1 |
| ICLabel | Sobre el maestro | Sobre la copia | Especificación de ICLabel |
| Limpieza | Pesos aplicados al maestro | Igual | Conserva 0.5 a 1 Hz |
| Filtro final | Butterworth N=5, forma [b,a] | Butterworth N=5, forma SOS | Ver abajo |
| Épocas | 10 s, sin rechazo | 10 s, pico a pico > 150 µV fuera | Criterio explícito y fijo |

## Decisiones y su justificación

**Canales malos.** Se probaron dos versiones de la detección de PREP, y ambas
marcaron sobre todo artefacto fisiológico, no fallas de electrodo:

- Sobre datos referenciados al CMS y con ruido de línea, 25 de 46 registros quedaron
  para revisión (Cz en 29, C3 en 26), casi siempre por `bad_by_correlation`.
- Con la referencia robusta de PREP tras el notch (Bigdely-Shamlo et al., 2015), la
  interpolación subió a 8.8 canales por registro en HC, 5.9 en PD off y 4.1 en PD on
  (16 de 32 en sub-hc4). Los canales más marcados fueron T8 (31 registros), FC6 (28),
  FC5 (27), F8 (26), T7 (23) y F7 (21) por `bad_by_correlation`, y Fp1 y Fp2 por
  `bad_by_psd`.

Son los canales sobre los músculos temporal y frontal y los que captan parpadeos.
Una hipótesis adicional, no verificada, es que en un montaje de 32 canales los
laterales tienen a sus vecinos más lejos y su correlación máxima es menor de por sí;
el criterio marca con más del 1% de ventanas de 1 s bajo 0.4, que en 3 min son unas
2 ventanas. Interpolarlos tenía cuatro costos: la ICA se quedaba sin artefactos que
separar (los IC musculares excluidos bajaron de 115 a 56, y sub-hc4 quedó con
rango 15 y ningún IC excluido); 8 de los 10 canales más interpolados pertenecen al
EPOC X, que se reconstruirían desde electrodos que el dispositivo no tiene; HC
recibía más del doble de interpolación que PD on; y la topografía se suavizaba en un
tercio del montaje.

Decisión (tomada tras ver el QC, por el patrón espacial y no por las etiquetas):
`v2` interpola solo los canales planos o con NaN, que son fallas del registro, y
deja el artefacto muscular y ocular a la ICA; un canal ruidoso aislado lo separa la
ICA y lo rechaza ICLabel como "channel noise". La detección completa de PREP se
guarda en el QC, y `v2-prep` la aplica como variante de sensibilidad.

**Notch.** FIR de fase cero en 60 Hz y armónicos, sobre el registro completo. No
altera la banda final y evita que la ICA gaste componentes en ruido de línea: en la
primera versión, 82 de los IC excluidos eran "line noise".

**ICA.** Infomax extendido con una semilla fija. El número de componentes es el
rango de los datos: 31 por la CAR, menos uno por cada canal interpolado. La ICA se
ajusta sobre una copia filtrada y sus pesos se aplican al registro sin filtrar, de
modo que la banda de 0.5 a 1 Hz se conserva (estrategia de transferencia del manual).

**ICLabel.** En `legacy` (etiquetas sobre el maestro) 194 de los IC excluidos
fueron "other"; sobre la copia filtrada dominan "muscle" y "line noise". La limpieza
original de la tesis probablemente quitaba, sobre todo, componentes de contenido
desconocido. El modelo está diseñado para componentes de infomax extendido sobre
datos con referencia promedio y filtrados entre 1 y 100 Hz, y recibe la misma
instancia con la que se ajustó la ICA (Pion-Tonachini et al., 2019). Por eso en
`v2` se etiqueta sobre la copia; el legado lo hacía sobre el maestro, con deriva de
baja frecuencia en el espectro. El criterio de rechazo es el del manual: clase
dominante de artefacto conocido con probabilidad mayor que 0.80, o "other" dominante
con probabilidad mayor que 0.80 y P(brain) menor que 0.05.

**ICA por registro y fuga.** Cada ICA usa solo su registro y ninguna etiqueta. Con
validación por sujeto, ningún dato de prueba entra al entrenamiento; en cualquier
partición dentro del sujeto sí sería fuga.

**Filtro final en SOS.** El diseño Butterworth 0.5 a 32 Hz, N=5, a 512 Hz está mal
condicionado en la forma [b,a] que usaba el manual. Al pasar a polinomios, el polo
más lento se mueve de 0.99815 a 0.99877 (verificado con raíces en 60 dígitos) y el
borde inferior queda en −1.77 dB por pasada en lugar de los −3.01 dB del diseño; la
diferencia contra SOS es de 2.2% RMS en ruido blanco. MNE rechaza esa forma. `v2`
usa SOS; `legacy` reproduce [b,a] con scipy y el relleno de `filtfilt` de MATLAB.

**Rechazo de épocas.** Umbral pico a pico fijo de 150 µV sobre la señal ya limpia y
filtrada, decidido antes de ver cualquier resultado de clasificación. El número de
épocas descartadas por registro se reporta.

**Unidades.** Todo el pipeline trabaja en voltios, como MNE. La conversión a µV
ocurre en un solo lugar, `tesis_eeg_parkinson/units.py`.

**Intensidad de limpieza por grupo.** PD off pierde más IC (8.2 contra 6.1 en HC)
y más épocas por el umbral pico a pico (1.4 contra 0.6 por registro). Es plausible
por temblor y actividad muscular sin medicación; se reporta por grupo y entra al
análisis de sensibilidad.

**Exclusión del análisis principal.** Regla fijada antes de ver cualquier
clasificación: una sesión sin pico de red eléctrica o con forma espectral distinta
sale del análisis principal y queda en el de sensibilidad. Las sesiones ON de sub-pd6
y sub-pd16 la cumplen: no tienen pico de 60 Hz, tienen una meseta cerca de −30 dB por
debajo de ~1.5 Hz con subida brusca en ~2 Hz (pasa-altas previo) y la CAR casi no
cambia su espectro (probable referencia promedio previa). El QC las marca con
`exclude_primary`. Consecuencia: Off vs On queda con 13 pares, On vs HC con 13
sesiones ON y Off vs HC conserva los 15 OFF.

**Fuera a propósito.** ASR, autoreject, los canales EXG (su colocación no está
documentada) y dipfit.

## Salidas

```
data/processed/preproc-<perfil>/
├── dataset_description.json          derivado BIDS
├── qc_summary.tsv                    una fila por registro
└── sub-XX/ses-YY/eeg/
    ├── sub-XX_ses-YY_task-rest_desc-clean_epo.fif
    └── sub-XX_ses-YY_task-rest_desc-qc.json
results/qc/preproc-<perfil>/sub-XX_ses-YY_task-rest_psd.pdf
results/qc/regression_matlab.pdf
```

La figura de QC muestra tres etapas con la misma referencia (tras CAR, tras ICA y
final): mediana entre canales con banda de percentiles 10 a 90, etiquetas directas y
la red eléctrica en 60 Hz. Sigue las reglas de `figuras_cientificas`.

El JSON de QC registra el inicio del reposo, la procedencia, la marca de exclusión,
los canales malos por criterio (con y sin referencia robusta) y su fracción de
ventanas con baja correlación, las probabilidades de ICLabel de cada componente, los componentes
excluidos, las épocas conservadas y descartadas, el perfil y el `RunContext`.

## Validación

**Automática** (`tests/test_preprocessing.py`, solo señales sintéticas):

- el filtro SOS da −3.01 dB por pasada en 0.5 y 32 Hz;
- el notch atenúa 60 y 120 Hz más de 30 dB sin tocar 10 Hz, y `legacy` no lo aplica;
- en fase cero, un tono de 10 Hz se conserva dentro de 1%, y los de 50 y 60 Hz se
  atenúan más de 40 y 55 dB;
- el perfil `legacy` reproduce `filtfilt` de MATLAB;
- el criterio de ICLabel se aplica tal como en el manual;
- PREP detecta un canal plano y uno ruidoso inyectados, y ninguno en datos limpios;
  `v2` interpola solo el plano y `v2-prep` ambos;
- la regresión contra EEGLAB da concordancia perfecta con datos idénticos, empareja
  canales por nombre y lee un `.set` escrito por EEGLAB-io en voltios;
- tras interpolar y referenciar, la suma entre canales es cero y el rango es el esperado;
- cada época mide 5,120 muestras sobre una rejilla fija, y el rechazo quita la época
  con una espiga de 400 µV;
- el recorte empieza en el evento "1", y se marca la procedencia de los curadores;
- el registro completo produce épocas, JSON y figura de QC;
- nada puede escribirse dentro de `data/raw/`.

**Contra el legado.** `compare-legacy` compara las épocas del perfil `legacy`
contra los `*_clean.set` del manual en `data/processed/legacy-matlab/`: correlación
de Pearson por canal, error RMS relativo y diferencia de potencia por banda (delta,
theta, alfa y beta, en dB). Después, las features y exactitudes contra
`results_full.csv`.

Resultado. La segmentación se reproduce exactamente: las 46 parejas tienen el mismo
número de épocas. El filtro coincide con `filtfilt` de MATLAB por construcción
(prueba automática). La limpieza concuerda con una correlación mediana por registro
de 0.958 (mínima 0.777) y un error RMS relativo mediano de 0.41. La diferencia se
explica por la ICA: según las filas de `icaweights` de cada `.set`, y suponiendo 31
componentes (el rango tras la CAR), EEGLAB quitó en la mediana 4 componentes, el valor
que reportaba el manual, y MNE 6; MNE quita más en 35 de 46 registros. La discrepancia crece con la diferencia
absoluta de componentes conservados (Spearman con la correlación mediana
ρ = −0.39, p = 0.008; con la potencia delta ρ = −0.49, p < 0.001; análisis
exploratorio, las sesiones de un paciente no son independientes). Con el mismo
número de componentes, la correlación mediana sube a 0.98; lo que queda se atribuye
a que infomax es estocástico y las implementaciones de EEGLAB y MNE no son idénticas.
Los canales que más difieren son frontales y laterales (F7, Fp2, AF3, FC5), donde
actúan los componentes oculares y musculares. La ICA no es idéntica entre EEGLAB
y MNE, así que se espera concordancia, no igualdad. Después se cambia un factor a la
vez: filtro SOS, ICLabel sobre la copia, recorte, canales malos y rechazo de épocas.

**Procedencia.** Las figuras de espectro de sub-pd6 y sub-pd16 ON se revisan antes de
decidir su tratamiento: si carecen del pico de 60 Hz o difieren en forma, se excluyen
del análisis principal y se conservan en el de sensibilidad. La decisión se toma
antes de ver cualquier clasificación.

## Cómo correr

```powershell
uv run python tools/download_ds002778.py
uv run tesis-eeg-parkinson preprocess --profile legacy
uv run tesis-eeg-parkinson preprocess --profile v2
uv run tesis-eeg-parkinson preprocess --profile v2-prep
uv run tesis-eeg-parkinson preprocess --profile v2 --subject pd6 pd16
uv run tesis-eeg-parkinson compare-legacy
```

## Referencias

Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The
PREP pipeline: Standardized preprocessing for large-scale EEG analysis. *Frontiers in
Neuroinformatics, 9*, 16. https://doi.org/10.3389/fninf.2015.00016

Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated
electroencephalographic independent component classifier, dataset, and website.
*NeuroImage, 198*, 181–197. https://doi.org/10.1016/j.neuroimage.2019.05.026

Rockhill, A. P., Jackson, N., George, J., Aron, A., & Swann, N. C. (2021). *UC San
Diego resting state EEG data from patients with Parkinson's disease* (Versión 1.0.5)
[Conjunto de datos]. OpenNeuro. https://doi.org/10.18112/openneuro.ds002778.v1.0.5
