# Preprocesamiento de ds002778

Migración validada a Python del pipeline de EEGLAB del manual original. El código
vive en `src/tesis_eeg_parkinson/preprocessing/` y cada parámetro, en
`preprocessing/config.py`. Hay dos perfiles completos:

- **`legacy`** reproduce el pipeline de MATLAB. Existe solo para validar la
  migración contra los resultados anteriores.
- **`v2`** es el pipeline de la tesis. Se llega a él desde `legacy` cambiando un
  factor a la vez, para poder atribuir cada diferencia a una decisión.

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
  y sub-pd16 son datos preprocesados por los curadores, no crudos. Sus metadatos
  difieren del resto (línea de 50 Hz en vez de 60; corte superior de 256 Hz en vez
  de 104). Se marcan en el QC y se evalúan con análisis de sensibilidad.
- **Duración.** Unos 3.2 min por registro; línea eléctrica de 60 Hz.
- **Advertencia de los curadores.** El README del dataset desaconseja clasificar PD
  contra controles con aprendizaje automático por el tamaño de la muestra y pide
  contactar a los curadores antes de publicar en revistas arbitradas.

## Pipeline

| Paso | `legacy` | `v2` | Motivo |
|---|---|---|---|
| Lectura | 32 canales, montaje standard_1005 | Igual | Selección por nombre |
| Recorte | Ninguno | Desde el evento "1", si existe | Excluir el tramo previo al reposo |
| Canales malos | Ninguno | PREP determinista + interpolación | Un canal ruidoso contamina la CAR |
| Referencia | CAR | CAR | Necesaria |
| Copia para ICA | FIR pasa-altas 1 Hz | FIR 1 a 100 Hz | Especificación de ICLabel |
| ICA | Infomax extendido | Igual; n = 31 menos interpolados | La CAR reduce el rango en 1 |
| ICLabel | Sobre el maestro | Sobre la copia | Especificación de ICLabel |
| Limpieza | Pesos aplicados al maestro | Igual | Conserva 0.5 a 1 Hz |
| Filtro final | Butterworth N=5, forma [b,a] | Butterworth N=5, forma SOS | Ver abajo |
| Épocas | 10 s, sin rechazo | 10 s, pico a pico > 150 µV fuera | Criterio explícito y fijo |

## Decisiones y su justificación

**Canales malos.** Se usan los criterios deterministas de PREP (Bigdely-Shamlo et
al., 2015) mediante pyprep: señal plana o ausente, desviación robusta (z > 5),
correlación máxima menor que 0.4 en más del 1% de ventanas de 1 s, y ruido de alta
frecuencia (z > 5). RANSAC se excluye porque es estocástico y no aporta como primera
barrera. Los canales detectados se interpolan por splines esféricos antes de la CAR.
Con más de 3 canales malos (≈10%), el registro se marca para revisión manual.

**ICA.** Infomax extendido con una semilla fija. El número de componentes es el
rango de los datos: 31 por la CAR, menos uno por cada canal interpolado. La ICA se
ajusta sobre una copia filtrada y sus pesos se aplican al registro sin filtrar, de
modo que la banda de 0.5 a 1 Hz se conserva (estrategia de transferencia del manual).

**ICLabel.** El modelo está diseñado para componentes de infomax extendido sobre
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

**Fuera a propósito.** ASR, autoreject, filtro notch (la línea de 60 Hz queda por
encima del corte de 32 Hz), los canales EXG (su colocación no está documentada) y
dipfit.

## Salidas

```
data/processed/preproc-<perfil>/
├── dataset_description.json          derivado BIDS
├── qc_summary.tsv                    una fila por registro
└── sub-XX/ses-YY/eeg/
    ├── sub-XX_ses-YY_task-rest_desc-clean_epo.fif
    └── sub-XX_ses-YY_task-rest_desc-qc.json
results/qc/preproc-<perfil>/sub-XX_ses-YY_task-rest_psd.pdf
```

El JSON de QC registra el inicio del reposo, la procedencia, los canales malos por
criterio, las probabilidades de ICLabel de cada componente, los componentes
excluidos, las épocas conservadas y descartadas, el perfil y el `RunContext`.

## Validación

**Automática** (`tests/test_preprocessing.py`, solo señales sintéticas):

- el filtro SOS da −3.01 dB por pasada en 0.5 y 32 Hz;
- en fase cero, un tono de 10 Hz se conserva dentro de 1%, y los de 50 y 60 Hz se
  atenúan más de 40 y 55 dB;
- el perfil `legacy` reproduce `filtfilt` de MATLAB;
- el criterio de ICLabel se aplica tal como en el manual;
- PREP detecta un canal plano y uno ruidoso inyectados, y ninguno en datos limpios;
- tras interpolar y referenciar, la suma entre canales es cero y el rango es el esperado;
- cada época mide 5,120 muestras sobre una rejilla fija, y el rechazo quita la época
  con una espiga de 400 µV;
- el recorte empieza en el evento "1", y se marca la procedencia de los curadores;
- el registro completo produce épocas, JSON y figura de QC;
- nada puede escribirse dentro de `data/raw/`.

**Contra el legado.** Con el perfil `legacy` se comparan las épocas contra los
`*_clean.set` del manual, si se conservan (correlación por canal y espectro), y las
features y exactitudes contra `results_full.csv`. La ICA no es idéntica entre EEGLAB
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
uv run tesis-eeg-parkinson preprocess --profile v2 --subject pd6 pd16
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
