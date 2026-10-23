# Prerregistro de la clasificación

Escrito y commiteado antes de la primera clasificación con las features V1. Ningún
análisis de este documento se ha corrido al registrarlo. Lo que sí se conocía: los
resultados del procesamiento previo (k-NN + TShEn con 32 canales, Off vs HC: 0.998
con validación por épocas y 0.468 por sujeto). Toda desviación posterior se registra
en `docs/bitacora.md` con su motivo y su fecha.

## Preguntas

Con validación por sujeto, ¿las features V1 distinguen por encima del azar empírico:

1. **Off vs HC**: PD sin medicación contra controles (diagnóstico)?
2. **On vs HC**: PD con medicación contra controles?
3. **Off vs On**: las dos sesiones de un mismo paciente (efecto de la medicación)?

Además, de forma descriptiva: ¿cuánto infla la validación por épocas?, ¿cuánto se
pierde al pasar al EPOC X emulado?, y ¿dependen las respuestas del modelo o del
preprocesamiento?

## Datos

`data/processed/features-v1/preproc-v1/features.tsv`: 544 features por época.
Análisis principal sin los registros marcados con `exclude_primary` (pd6 on y pd16 on).
En Off vs On solo entran los pacientes con ambas sesiones: 13 pares.

## Modelo principal

Imputación por la mediana, estandarización y regresión logística L2 con
`class_weight="balanced"`, todo ajustado solo con los datos de entrenamiento de cada
pliegue. La regularización C se elige entre 10⁻⁴ y 10¹ (6 valores, escala log) con
validación anidada: 3 pliegues internos estratificados por sujeto, criterio AUC.

## Validación

5 pliegues externos estratificados por sujeto, 20 repeticiones con sujetos barajados
(semilla 28 + repetición). En Off vs On, las dos sesiones de un paciente van siempre
al mismo pliegue. El clasificador ve épocas; la decisión es por registro: promedio de
las probabilidades de sus épocas, positivo si es ≥ 0.5.

## Métricas e inferencia

- **Primaria:** exactitud balanceada por registro, sobre las probabilidades
  promediadas entre repeticiones.
- **Secundaria:** AUC por registro.
- **IC 95%:** *bootstrap* de 2,000 remuestreos de sujetos (en Off vs On, de pacientes
  con sus dos sesiones), por percentiles.
- **Prueba:** 1,000 permutaciones. Entre grupos se permutan las etiquetas entre
  sujetos; en Off vs On se intercambian on y off dentro de cada paciente con
  probabilidad 1/2. Cada permutación repite todo el procedimiento, incluida la
  anidación, con 2 repeticiones: una nula con menos repeticiones es más variable que
  el estadístico observado, así que la prueba resulta conservadora.
  p = (1 + nulas ≥ observado) / 1,001.
- **Azar empírico:** media y percentil 95 de la nula (Combrisson & Jerbi, 2015).
- **Multiplicidad:** Holm sobre los tres contrastes del análisis principal; α = 0.05.
  Solo esta familia es confirmatoria.

## Control de fuga (no es desempeño)

El mismo modelo con validación por épocas, que mezcla épocas de un sujeto entre
entrenamiento y prueba. Se reporta solo para medir la inflación, junto a la
exactitud por época de la validación por sujeto.

## Análisis secundarios y de sensibilidad (descriptivos)

Se reportan todos, sin elegir el mejor:

| Pregunta | Comando (`uv run tesis-eeg-parkinson classify ...`) |
|---|---|
| Principal | `--features v1 --preproc v1` |
| ¿Depende del modelo? | `--features v1 --preproc v1 --model slda` y `--model rf` |
| Método publicado | `--features aljalal --preproc v1` |
| Canales interpolados por PREP | `--features v1 --preproc v1-prep` |
| Banda de 43 Hz | `--features v1 --preproc v1-43` |
| Con pd6 on y pd16 on | `--features v1 --preproc v1 --include-excluded` |
| BioSemi, 32 canales | `--features v1 --preproc v1 --intersect-preproc epocx` |
| BioSemi, los 14 canales del EPOC X | `--features v1 --preproc v1 --channels emotiv --intersect-preproc epocx` |
| EPOC X emulado | `--features v1 --preproc epocx --intersect-preproc v1` |

Las tres últimas usan los mismos sujetos y separan dos pérdidas: la de pasar de 32 a
14 canales y la del equipo (referencia, banda, cuantización y una ICA de 14 canales).

Modelos de sensibilidad: LDA con contracción de Ledoit-Wolf (analítica, sin
hiperparámetros) y bosque aleatorio, los dos que Lotte et al. (2018) señalan como útiles
con pocas muestras de entrenamiento; el bosque, además, sondea señal no lineal. Con 500 árboles y valores por defecto. Sin permutaciones; IC por *bootstrap*.

## Interpretación

Patrones de Haufe et al. (2014), A = Cov(Z)·w, del modelo lineal ajustado con todos
los datos del análisis principal. Son descriptivos: no se someten a prueba ni se usan
para elegir features.

## Por qué regresión logística L2

- **Régimen:** unos 25 sujetos de entrenamiento y 544 features. Un modelo lineal con
  regularización fuerte es el sesgo adecuado; los modelos no lineales necesitan más
  sujetos de los que hay.
- **Probabilidades:** las da calibradas por construcción, y la decisión por registro
  las promedia.
- **Costo:** es convexa y determinista; 1,000 permutaciones con anidación son viables.
- **Interpretación:** admite patrones de Haufe.
- **Tiempo casi real:** predecir es un producto punto.

Varoquaux et al. (2017) encontraron que, en decodificadores no dispersos, los valores
por defecto pueden predecir tan bien como la anidación, y que esta mejora la
estabilidad de los mapas. Aquí se usa la anidación por esa estabilidad, que importa
para los patrones, y porque no introduce sesgo.

## Referencias

Combrisson, E., & Jerbi, K. (2015). Exceeding chance level by chance: The caveat of
theoretical chance levels in brain signal classification and statistical assessment
of decoding accuracy. *Journal of Neuroscience Methods, 250*, 126–136.
https://doi.org/10.1016/j.jneumeth.2015.01.010

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., &
Bießmann, F. (2014). On the interpretation of weight vectors of linear models in
multivariate neuroimaging. *NeuroImage, 87*, 96–110.
https://doi.org/10.1016/j.neuroimage.2013.10.067

Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., &
Yger, F. (2018). A review of classification algorithms for EEG-based brain–computer
interfaces: A 10 year update. *Journal of Neural Engineering, 15*(3), 031005.
https://doi.org/10.1088/1741-2552/aab2f2

Varoquaux, G., Raamana, P. R., Engemann, D. A., Hoyos-Idrobo, A., Schwartz, Y., &
Thirion, B. (2017). Assessing and tuning brain decoders: Cross-validation, caveats,
and guidelines. *NeuroImage, 145*, 166–179.
https://doi.org/10.1016/j.neuroimage.2016.10.038
