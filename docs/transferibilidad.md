# Transferibilidad e implicaciones

Notas para el capítulo de transferibilidad al Emotiv EPOC X. Cada punto dice qué se
pierde o qué cambia, por qué y cómo lo trata el proyecto.

## Lo que transfiere por diseño

Las 17 features de V1 son invariantes a la escala de la señal (`docs/features.md`):
ganancia, impedancia y unidades no las cambian. Es la condición mínima para comparar
un equipo de investigación (BioSemi, 24 bits, electrodos activos con gel) con uno de
consumo (EPOC X, 14 o 16 bits, fieltro con solución salina). No basta: la invariancia
no protege del ruido de los electrodos ni de la cobertura espacial.

## Lo que el hardware no puede medir

**Acoplamiento fase-amplitud beta-gamma.** Swann et al. (2015) lo encontraron elevado
en PD sin medicación, en este mismo dataset, con gamma de 50 a 150 Hz. El EPOC X
registra de 0.16 a 43 Hz, así que ese biomarcador no es transferible; V1 tampoco lo
conserva, porque su banda final llega a 32 Hz. Es un hallazgo del capítulo: no todo
biomarcador documentado sobrevive al cambio de equipo.

**Forma de onda de beta.** Su asimetría vive en los armónicos: un beta de 20 Hz
necesita su armónico de 40 Hz. El EPOC X llega a ~43 Hz, V1 a 32 Hz. La variante
`v1-43` mide si T3 recupera esa forma dentro de la banda del EPOC X. V1 mantiene 32 Hz
por la actividad muscular, y `v1-43` lo respalda: con los mismos componentes de ICA,
subir el corte descartó 12 épocas más, todas en PD (8 off, 4 on, 0 en HC); 7 de las 12
por FC5, F7 o T8, sobre el músculo temporal, y 2 por Fp1. Es consistente con actividad
muscular, sin demostrarla. Y 7 de esas 12 caen en canales del EPOC X: la banda del
equipo deja entrar contenido que aquí afecta más a PD que a HC, un atajo no cortical
para un clasificador.

**Cobertura espacial.** Los 14 canales del EPOC X no incluyen la línea media ni los
centro-parietales (Cz, CP1, CP2).

## Emulación del EPOC X (perfil `epocx`)

Sobre cada registro completo de BioSemi, en el orden del equipo:

1. Referencia en P3, la posición de su CMS.
2. Sus 14 canales; ni la CAR ni la ICA ven electrodos que el equipo no tiene.
3. Su banda, 0.16 a 43 Hz.
4. Su muestreo de 256 Hz y su paso de 16 bits (0.1275 µV), el modo del piloto
   planeado. A 256 Hz la entrada de ICLabel conserva una frecuencia de Nyquist por
   encima de 100 Hz, como en su entrenamiento; las features se diezman a 128 Hz.
5. El resto de V1 igual: CAR sobre 14 canales, ICA con 13 componentes, ICLabel,
   filtros y reglas de épocas.

La CAR cancela cualquier referencia común, así que P3 solo importa para la
cuantización.

**Lo que muestra la primera corrida.** La limpieza automática pierde fuerza con 14
canales laterales:

- **PREP no transfiere.** Su referencia robusta abortó en 4 de 46 registros: marcó
  como malos al menos 13 de los 14 canales. En los demás marcó entre 3.4 y 4.5 por
  registro (24% a 32% del montaje). Sus criterios de correlación suponen vecinos
  cercanos, y el EPOC X no tiene línea media. La detección queda como diagnóstico,
  sin referencia robusta cuando esta no es posible; solo se interpolan canales planos
  o con NaN, igual que en V1.
- **La ICA separa menos.** Con 13 componentes se excluyen 1.4 a 2.2 por registro,
  frente a 5.3 a 8.0 en V1, y el umbral pico a pico descarta más épocas. Lo que la
  ICA no separa en el equipo de consumo, el clasificador lo recibe. **No se emulan** el ruido del electrodo salino, la impedancia ni la
colocación del casco: la emulación es un límite superior de lo que el equipo real
puede conservar.

## Límites para un análisis casi en tiempo real

- **Filtros:** `filtfilt` (fase cero) usa el futuro de la señal; en línea haría falta
  un filtro causal, con su retardo de fase.
- **ICA:** se ajusta sobre el registro completo. En línea haría falta una
  descomposición preajustada o un método adaptativo, con sus propias garantías.
- **Features:** compatibles. V1 tarda ~0.56 s por época de 10 s con 32 canales.

## Referencias

Jackson, N., Cole, S. R., Voytek, B., & Swann, N. C. (2019). Characteristics of
waveform shape in Parkinson's disease detected with scalp electroencephalography.
*eNeuro, 6*(3), ENEURO.0151-19.2019. https://doi.org/10.1523/ENEURO.0151-19.2019

Swann, N. C., de Hemptinne, C., Aron, A. R., Ostrem, J. L., Knight, R. T., & Starr,
P. A. (2015). Elevated synchrony in Parkinson disease detected with
electroencephalography. *Annals of Neurology, 78*(5), 742–750.
https://doi.org/10.1002/ana.24507
