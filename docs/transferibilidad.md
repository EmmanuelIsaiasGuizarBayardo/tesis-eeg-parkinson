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
`v1-43` mide si T3 recupera esa forma dentro de la banda del EPOC X; V1 mantiene
32 Hz porque un corte más alto deja pasar más actividad muscular, frecuente en PD, y
el modelo podría aprender esa contaminación en lugar de la dinámica cortical.

**Cobertura espacial.** Los 14 canales del EPOC X no incluyen la línea media ni los
centro-parietales (Cz, CP1, CP2). Pendiente: perfil de simulación del EPOC X.

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
