# Preprocesamiento V1 de ds002778

El código vive en `src/tesis_eeg_parkinson/preprocessing/` y cada parámetro, en
`preprocessing/config.py`. Hay tres perfiles completos:

- **`v1`**: el preprocesamiento de la tesis.
- **`v1-prep`**: igual a `v1`, pero interpola todo lo que marca PREP. Solo para el
  análisis de sensibilidad.
- **`matlab`**: reproduce el procesamiento previo en EEGLAB. No es un método de la
  tesis; existe para la verificación que resume el apéndice de este documento.

Cada derivado guarda en su JSON de QC el perfil completo y el `RunContext`
(semilla, commit y estado del árbol).

## Lo que el dataset obliga a considerar

Revisado en los metadatos de la versión 1.0.5 (Rockhill et al., 2021).

- **Canales.** `channels.tsv` declara 40 canales EEG (los 32 del cuero cabelludo y
  EXG1 a EXG8) y un canal Status. Se selecciona por nombre, no por índice.
- **Referencia.** `EEGReference` es "n/a": BioSemi registra contra el CMS y los
  archivos no traen referencia previa, así que la CAR es necesaria.
- **Inicio del reposo.** Un evento de valor "1" aparece segundos después del inicio
  (entre 2.9 y 21.8 s en los registros revisados; 95.6 s en sub-pd6 ON; ausente en
  sub-hc4). Se interpreta como inicio del reposo: con él, sub-pd6 ON queda en unos
  193 s, como los demás. La interpretación es inferida, no documentada.
- **Procedencia heterogénea.** Según `participants.tsv`, las sesiones ON de sub-pd6
  y sub-pd16 son datos preprocesados por los curadores, no crudos (ver "Exclusión
  del análisis principal").
- **Metadatos de la red eléctrica.** 9 registros declaran `PowerLineFrequency` de
  50 Hz (hc8, hc21, hc25, hc31, hc32, pd6 on, pd13 on, pd16 on y pd22 on), aunque
  la red de San Diego es de 60 Hz. El campo no es confiable: el pipeline usa 60 Hz
  y registra la discrepancia en el QC.
- **Duración.** Unos 3.2 min por registro.
- **Advertencia de los curadores.** El README del dataset desaconseja clasificar PD
  contra controles con aprendizaje automático por el tamaño de la muestra y pide
  contactar a los curadores antes de publicar en revistas arbitradas.

## Pipeline V1

| Paso | Decisión | Motivo |
|---|---|---|
| Lectura | 32 canales por nombre, montaje colin27_1005 | EXG1-8 vienen tipados como EEG |
| Recorte | Desde el evento "1", si existe | Excluir el tramo previo al reposo |
| Notch | 60, 120, 180 y 240 Hz, FIR de fase cero | Quitar la línea antes de detectar canales |
| Canales malos | PREP como diagnóstico; se interpolan solo planos o con NaN | Ver abajo |
| Referencia | CAR | Necesaria |
| Copia para ICA | FIR de 1 a 100 Hz | Especificación de ICLabel |
| ICA | Infomax extendido; n = 31 menos interpolados; semilla fija | La CAR reduce el rango en 1 |
| ICLabel | Sobre la copia; criterio de rechazo fijo | Especificación de ICLabel |
| Limpieza | Pesos aplicados al registro sin filtrar | Conserva la banda de 0.5 a 1 Hz |
| Filtro final | Butterworth N=5, 0.5 a 32 Hz, SOS, fase cero | Estabilidad numérica |
| Épocas | 10 s sin solapamiento; fuera las de pico a pico > 150 µV | Criterio explícito y fijo |

## Decisiones y su justificación

**Notch.** FIR de fase cero en 60 Hz y armónicos, sobre el registro completo, como
en PREP (Bigdely-Shamlo et al., 2015). No altera la banda final y evita que la ICA
gaste componentes en ruido de línea: en una corrida preliminar sin notch, 82 de los
IC excluidos eran "line noise".

**Canales malos.** La detección completa de PREP, con su referencia robusta y sus
umbrales por defecto (perfil `v1-prep`), interpoló 8.8 canales por registro en HC,
5.9 en PD off y 4.1 en PD on, y 16 de 32 en sub-hc4. Los canales más marcados fueron
T8 (31 registros), FC6 (28), FC5 (27), F8 (26), T7 (23) y F7 (21), por
`bad_by_correlation`, y Fp1 y Fp2, por `bad_by_psd`: los canales sobre los músculos
temporal y frontal y los que captan parpadeos. Una hipótesis adicional, no
verificada, es que en 32 canales los electrodos laterales tienen a sus vecinos más
lejos y su correlación máxima es menor de por sí; el criterio marca con más del 1% de
ventanas de 1 s bajo 0.4, que en 3 min son unas 2 ventanas.

Interpolarlos tenía cuatro costos. La ICA se quedaba sin artefactos que separar
(sub-hc4 quedó con rango 15 y ningún IC excluido). Ocho de los diez canales más
interpolados pertenecen al EPOC X, que se reconstruirían desde electrodos que el
dispositivo no tiene. HC recibía más del doble de interpolación que PD on. Y la
topografía se suavizaba en un tercio del montaje.

Decisión, tomada tras ver el QC por el patrón espacial y no por las etiquetas: `v1`
interpola solo canales planos o con NaN, que son fallas del registro, y deja el
artefacto muscular y ocular a la ICA; un canal ruidoso aislado lo separa la ICA y lo
rechaza ICLabel como "channel noise". El QC guarda la detección completa de PREP, con
y sin referencia robusta, y la fracción de ventanas con baja correlación por canal.
Con más de 3 canales interpolados, el registro se marca para revisión.

**ICA.** Infomax extendido con semilla fija. El número de componentes es el rango de
los datos: 31 por la CAR, menos uno por canal interpolado. La ICA se ajusta sobre una
copia filtrada y sus pesos se aplican al registro sin filtrar, de modo que la banda de
0.5 a 1 Hz se conserva.

**ICLabel.** El modelo está diseñado para componentes de infomax extendido sobre
datos con referencia promedio y filtrados entre 1 y 100 Hz, y recibe la misma
instancia con la que se ajustó la ICA (Pion-Tonachini et al., 2019). Se rechaza un
componente si su clase dominante es un artefacto conocido con probabilidad mayor que
0.80, o si domina "other" con probabilidad mayor que 0.80 y P(brain) menor que 0.05.

**ICA por registro y fuga.** Cada ICA usa solo su registro y ninguna etiqueta. Con
validación por sujeto, ningún dato de prueba entra al entrenamiento; en cualquier
partición dentro del sujeto sí sería fuga.

**Filtro final en SOS.** Un Butterworth de 0.5 a 32 Hz, N=5, a 512 Hz, está mal
condicionado en forma de polinomios [b,a]: el redondeo mueve el polo más lento de
0.99815 a 0.99877 (verificado con raíces en 60 dígitos) y el borde inferior queda en
−1.77 dB por pasada en lugar de −3.01 dB. MNE rechaza esa forma. La forma SOS
conserva el diseño.

**Rechazo de épocas.** Umbral pico a pico fijo de 150 µV sobre la señal limpia y
filtrada, decidido antes de ver cualquier resultado de clasificación. Las épocas
descartadas se reportan por registro y por grupo.

**Intensidad de limpieza por grupo.** Los componentes excluidos, los canales
interpolados y las épocas descartadas se reportan por grupo tras la corrida
definitiva. Una diferencia entre grupos es plausible (temblor y actividad muscular
sin medicación) y entra al análisis de sensibilidad.

**Exclusión del análisis principal.** Regla fijada antes de ver cualquier
clasificación: una sesión sin pico de red eléctrica o con forma espectral distinta
sale del análisis principal y queda en el de sensibilidad. Las sesiones ON de sub-pd6
y sub-pd16 la cumplen: no tienen pico de 60 Hz, tienen una meseta cerca de −30 dB por
debajo de ~1.5 Hz con subida brusca en ~2 Hz (pasa-altas previo) y la CAR casi no
cambia su espectro (probable referencia promedio previa). El QC las marca con
`exclude_primary`. Consecuencia: Off vs On queda con 13 pares, On vs HC con 13
sesiones ON y Off vs HC conserva los 15 OFF.

**Unidades.** Todo el pipeline trabaja en voltios, como MNE. La conversión a µV
ocurre en un solo lugar, `tesis_eeg_parkinson/units.py`.

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
```

El JSON de QC registra el inicio del reposo, la procedencia, la marca de exclusión,
los canales que marca PREP por criterio y los que se interpolaron, las
probabilidades de ICLabel de cada componente, los componentes excluidos, las épocas
conservadas y descartadas, el perfil y el `RunContext`. La figura de QC muestra tres
etapas con la misma referencia (tras CAR, tras ICA y final): mediana entre canales
con banda de percentiles 10 a 90, etiquetas directas y la red eléctrica en 60 Hz.

## Verificación automática

`tests/test_preprocessing.py` y `tests/test_validation.py`, solo con señales
sintéticas:

- el filtro SOS da −3.01 dB por pasada en 0.5 y 32 Hz; en fase cero, un tono de 10 Hz
  se conserva dentro de 1% y los de 50 y 60 Hz se atenúan más de 40 y 55 dB;
- el notch atenúa 60 y 120 Hz más de 30 dB sin tocar 10 Hz;
- el criterio de ICLabel se aplica tal como está definido;
- PREP detecta un canal plano y uno ruidoso inyectados, y ninguno en datos limpios;
  `v1` interpola solo el plano y `v1-prep` ambos;
- tras interpolar y referenciar, la suma entre canales es cero y el rango es el esperado;
- cada época mide 5,120 muestras sobre una rejilla fija, y el rechazo quita la época
  con una espiga de 400 µV;
- el recorte empieza en el evento "1", y se marca la procedencia de los curadores;
- el registro completo produce épocas, JSON y figura de QC;
- nada puede escribirse dentro de `data/raw/`, y una corrida sin registros falla
  antes de escribir.

## Cómo correr

```powershell
uv run tesis-eeg-parkinson preprocess --profile v1
uv run tesis-eeg-parkinson preprocess --profile v1-prep
uv run tesis-eeg-parkinson preprocess --profile v1 --subject pd6 pd16
```

## Apéndice: concordancia con el procesamiento previo

Los resultados preliminares (paper del CNIB2026, reporte semestral y borrador de
tesis) provenían de un preprocesamiento anterior en EEGLAB. Los de la tesis lo
sustituyen. Para que esa sustitución sea verificable, el perfil `matlab` reproduce
el procesamiento previo en MNE y `compare-matlab` lo compara contra sus archivos
`*_clean.set`:

```powershell
uv run tesis-eeg-parkinson preprocess --profile matlab
uv run tesis-eeg-parkinson compare-matlab
```

| Aspecto | Resultado |
|---|---|
| Épocas | Idénticas en los 46 registros |
| Filtro | Idéntico a `filtfilt` de MATLAB (prueba automática) |
| Correlación mediana por canal, por registro | Mediana 0.958; mínima 0.777 |
| Error RMS relativo | Mediana 0.41 |
| Componentes de ICA quitados | EEGLAB: mediana 4; MNE: mediana 6 (MNE quita más en 35 de 46) |
| Diferencia de componentes y concordancia (Spearman) | Correlación: ρ = −0.39, p = 0.008; potencia delta: ρ = −0.49, p < 0.001 |

La discrepancia se explica por la ICA, que es estocástica y no está implementada
igual en EEGLAB que en MNE: crece con la diferencia de componentes quitados y, cuando
ambos quitan los mismos, la correlación mediana sube a 0.98. El conteo de EEGLAB
supone 31 componentes (el rango tras la CAR), y el análisis de Spearman es
exploratorio porque las sesiones de un paciente no son independientes. Los canales
que más difieren son frontales y laterales (F7, Fp2, AF3, FC5), donde actúan los
componentes oculares y musculares. La figura está en `results/qc/regression_matlab.pdf`.

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
