# Features de ds002778

El código vive en `src/tesis_eeg_parkinson/features/` y cada parámetro, en
`features/config.py`. Hay dos perfiles:

- **`v1`**: las features de la tesis.
- **`aljalal`**: la configuración publicada por Aljalal et al. (2022), para la
  comparación con el método original. No es un método de la tesis.

Ambos parten de las épocas del preprocesamiento V1 (512 Hz, 0.5 a 32 Hz).

## Criterios de V1

Las features se eligen por sus propiedades, no por su desempeño. Escogerlas por su
separación entre grupos con estos mismos datos sería fuga de selección; por eso, al
fijar este conjunto, ninguna métrica se había evaluado entre grupos.

1. **Invariancia a la escala.** La tesis incluye transferir a otro equipo (Emotiv
   EPOC X), donde cambian la ganancia, la impedancia y la cuantización. Una feature
   que depende de la amplitud absoluta mide, en parte, el equipo.
2. **Dimensión temporal.** Las ocho métricas de Aljalal et al. dependen solo del
   multiconjunto de amplitudes: barajar las muestras no las cambia. V1 incluye
   medidas que sí leen el orden temporal.
3. **Información distinta.** Las ocho métricas originales correlacionan entre sí con
   ρ ≥ 0.906 en cA4 (Spearman). V1 cubre cuatro aspectos con pocas features.
4. **Parsimonia.** Con 31 personas, cada feature adicional cuesta.

## Conjunto V1

Por canal, sobre épocas decimadas a 128 Hz:

| Feature | Señales | Mide | Escala | Orden |
|---|---|---|---|---|
| `relpow` | cA4, cD4, cD3, cD2 | Reparto de energía entre bandas (lentificación) | Invariante | No |
| `wentropy` | cA4, cD4, cD3, cD2 | Cómo se reparte la energía de la banda en el tiempo | Invariante | No |
| `permen_tau1/2/4` | Banda completa | Estructura ordinal a tres escalas temporales | Invariante | Sí |
| `sampen` | Banda completa | Regularidad de patrones de amplitud | Invariante | Sí |
| `aperiodic_exponent` | Banda completa | Pendiente 1/f del espectro | Invariante | No |
| `alpha_cog` | Banda completa | Frecuencia del ritmo alfa | Invariante | No |

Son 14 features por canal (448 con 32 canales; 196 con los 14 del EPOC X). La
invariancia a la escala está probada para cada una con ganancias de 0.001 a 250.

**Frecuencia de trabajo y bandas.** A 128 Hz, el árbol db4 de 4 niveles da cA4
(0 a 4 Hz), cD4 (4 a 8), cD3 (8 a 16) y cD2 (16 a 32): delta, theta, alfa con beta
baja, y beta. cD1 (32 a 64 Hz) es la falda del filtro final y se descarta. 128 Hz es
además uno de los dos modos del EPOC X. Las épocas se diezman por 4 sin remuestreo
FFT: el Butterworth final ya atenúa 65 dB a 64 Hz, así que no hay aliasing apreciable.

**`relpow`.** Energía de cada banda sobre la suma de las cuatro. Sustituye a Energy y
LBP, que son absolutas. La lentificación espectral es el biomarcador EEG más
replicado en PD (Stoffers et al., 2007). Las cuatro suman 1: una es redundante, y la
etapa de clasificación debe tenerlo en cuenta.

**`wentropy`.** Entropía de Shannon normalizada de pᵢ = xᵢ²/E en cada banda. Es la
forma correcta de la familia de Coifman y Wickerhauser (1992): la ShEn del artículo
(Σ x² log x²) es igual a E·(log E − H), dominada por la energía, lo que explica su
correlación de 0.998 con Energy. Distingue energía sostenida (H alta) de energía en
ráfagas (H baja). No lee el orden.

**`permen`.** Entropía de permutación de orden 4 (Bandt & Pompe, 2002). Es invariante
a cualquier transformación monótona creciente: ganancia, desplazamiento o curva de
respuesta del amplificador no la cambian. Los retrasos 1, 2 y 4 miran ventanas de
31, 55 y 102 ms a 128 Hz; hacen de versión multiescala sin acortar la serie. Con
1,280 muestras y 24 patrones posibles hay unas 53 ventanas por patrón.

**`sampen`.** Entropía muestral con m = 2 y r = 0.2·DE (Richman & Moorman, 2000). Es
la medida que en análisis previos cambió +258% al barajar las muestras mientras las
ocho originales no cambiaban. Con r relativo a la DE es invariante a la escala; a
128 Hz, r ronda 2 µV, muy por encima del paso de cuantización del EPOC X.

**`aperiodic_exponent` y `alpha_cog`.** El espectro de Welch (ventanas de 2 s) se
parametriza con FOOOF entre 2 y 30 Hz (Donoghue et al., 2020). El exponente separa la
pendiente 1/f de las oscilaciones; sin esa separación, un cambio aperiódico se lee
como lentificación oscilatoria. El centro de gravedad de alfa (7 a 14 Hz; Klimesch,
1999) se calcula sobre la potencia que queda por encima del componente aperiódico.
Ambos son invariantes a la escala: una ganancia solo mueve el desplazamiento del
ajuste. `alpha_cog` es NaN si no hay potencia periódica en alfa; `qc_fooof_r2`
registra la calidad del ajuste y no entra a la clasificación. Se usa `fooof` 1.1,
estable; su sucesor, `specparam`, solo tiene versiones preliminares.

**Considerado y fuera:**

- Lempel-Ziv: binariza la señal, y su información se solapa con PermEn.
- Entropía multiescala (Costa et al., 2002): el *coarse-graining* deja series de 320
  muestras a escala 4, demasiado cortas para SampEn.
- Parámetros de Hjorth: la movilidad es casi el centroide espectral, ya cubierto.
- Potencia absoluta y las ocho métricas originales: dependen de la escala.
- PermEn y SampEn por banda: multiplicarían las features sin una hipótesis previa.

**Vacío de evidencia.** Ninguna de estas features se ha evaluado entre grupos. Que
discriminen mejor o peor que las originales es una pregunta abierta, y se responde en
la etapa de clasificación, con la misma validación por sujetos para todas.

## Configuración `aljalal`

512 Hz, db4 a 4 niveles, cinco coeficientes (cA4, cD4, cD3, cD2, cD1) más la señal
original, y las ocho métricas tal como están escritas en el artículo, calibradas en
µV. Dos detalles:

- `shen` conserva el signo de su ecuación 7, sin el menos: es el negativo de la
  entropía de Shannon convencional. A los clasificadores les da igual; a la
  interpretación de "más entropía en PD", no.
- El artículo admite dos lecturas de TShEn: sobre amplitudes (`tshen`, la que
  reprodujo el 99.8%) y sobre las probabilidades de los valores únicos
  (`tshen_hist`, la que describe su texto). Se calculan ambas. Con umbrales absolutos
  en 0 y 1 µV, las dos terminan midiendo la fracción de muestras pequeñas.

Son 1,728 features por época con 32 canales.

## Salidas

```
data/processed/features-<perfil>/
├── features.tsv     una fila por época; metadatos + <métrica>__<señal>__<canal>
└── features.json    configuración, bandas nominales, preprocesamiento y RunContext
```

Los metadatos incluyen `exclude_primary` y `exclude_reason`, copiados del QC del
preprocesamiento. Las columnas `qc_` no entran a la clasificación.

## Verificación automática

`tests/test_features.py`, solo con señales sintéticas:

- las bandas nominales a 128 Hz son las esperadas, y un tono cae en su banda;
- las features V1 no cambian con ganancias de 0.001, 7.3 y 250;
- barajar las muestras no cambia `wentropy` ni las métricas originales, y sí sube
  PermEn y SampEn;
- PermEn vale 0 en una rampa y más de 0.98 en ruido blanco;
- SampEn coincide con un cálculo por fuerza bruta;
- el exponente y `alpha_cog` recuperan 1.5 y 10 Hz en ruido 1/f con un pico de alfa;
- la ShEn del artículo cumple E·(log E − H);
- ambas lecturas de TShEn coinciden con su definición sobre un vector conocido;
- la tabla trae los metadatos y el número esperado de features, y una carpeta sin
  épocas falla antes de escribir.

## Cómo correr

```powershell
uv run tesis-eeg-parkinson features --profile v1
uv run tesis-eeg-parkinson features --profile aljalal
```

## Referencias

Aljalal, M., Aldosari, S. A., Molinas, M., AlSharabi, K., & Alturki, F. A. (2022).
Detection of Parkinson's disease from EEG signals using discrete wavelet transform,
different entropy measures, and machine learning techniques. *Scientific Reports,
12*(1), 22547. https://doi.org/10.1038/s41598-022-26644-7

Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for
time series. *Physical Review Letters, 88*(17), 174102.
https://doi.org/10.1103/PhysRevLett.88.174102

Coifman, R. R., & Wickerhauser, M. V. (1992). Entropy-based algorithms for best basis
selection. *IEEE Transactions on Information Theory, 38*(2), 713–718.
https://doi.org/10.1109/18.119732

Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of
complex physiologic time series. *Physical Review Letters, 89*(6), 068102.
https://doi.org/10.1103/PhysRevLett.89.068102

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T.,
Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020).
Parameterizing neural power spectra into periodic and aperiodic components. *Nature
Neuroscience, 23*(12), 1655–1665. https://doi.org/10.1038/s41593-020-00744-x

Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory
performance: A review and analysis. *Brain Research Reviews, 29*(2–3), 169–195.
https://doi.org/10.1016/S0165-0173(98)00056-3

Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using
approximate entropy and sample entropy. *American Journal of Physiology-Heart and
Circulatory Physiology, 278*(6), H2039–H2049.
https://doi.org/10.1152/ajpheart.2000.278.6.H2039

Stoffers, D., Bosboom, J. L. W., Deijen, J. B., Wolters, E. C., Berendse, H. W., &
Stam, C. J. (2007). Slowing of oscillatory brain activity is a stable characteristic
of Parkinson's disease without dementia. *Brain, 130*(7), 1847–1860.
https://doi.org/10.1093/brain/awm034
