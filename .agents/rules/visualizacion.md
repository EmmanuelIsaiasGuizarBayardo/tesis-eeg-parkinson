---
trigger: always_on
description: "Reglas de visualización científica (figuras-cientificas 0.1.4)."
---

<!-- Generado por figuras-cientificas 0.1.4. No se edita aquí: se regenera con
     'uv run python -m figuras_cientificas regla' tras actualizar la biblioteca. -->

# Reglas de visualización científica

Aplican al generar cualquier figura con color en Python: Matplotlib, MNE-Python o
Seaborn. El código de estas reglas vive en `figuras_cientificas`; se usa en lugar
de reescribir paletas, tamaños o estilos a mano.

## Principios

- Toda figura sigue siendo legible sin color: con daltonismo, en escala de grises
  e impresa. El color nunca es la única codificación (Crameri et al., 2020).
- Mapas continuos perceptualmente uniformes y de luminosidad ordenada. Nunca `jet`,
  `rainbow`, `hsv`, `gist_rainbow` ni `turbo` para datos.
- Toda la tinta informa: sin rejilla pesada, sin 3D decorativo ni marcos de más
  (Tufte, 2001).

## Elegir el mapa de color

**REGLA DURA.** `cmap, norm = mapa_para(datos, centro=...)` decide por los datos:

- Con ambos signos o con una referencia natural (cero, línea base, azar):
  divergente con escala simétrica alrededor del centro. Por ejemplo, un ERSP en dB
  respecto a la línea base, o una exactitud con `centro=0.5`.
- De un solo signo, aunque la métrica pudiera tener otro en teoría: secuencial en
  su rango real. Imponer simetría a datos de un lado desperdicia medio mapa.
- El extremo se fija en el percentil 98, para que un atípico no comprima la escala.
  Si la escala recorta datos, la barra de color lo marca:
  `fig.colorbar(..., extend=extension(datos, norm))`.

| Dato | Función | Mapa |
|---|---|---|
| Ordenado, sin centro | `secuencial()` | batlow |
| Con centro, fondo claro | `divergente()` | roma, de centro claro |
| Con centro, fondo oscuro | `divergente("oscuro")` | managua, de centro oscuro |
| Angular: fase, dirección | `ciclico()` | romaO |

Sobre fondo claro el centro del divergente debe ser claro: un centro oscuro hace
que "sin efecto" sea lo más visible de la figura. En ambos, positivo es cálido.

En escala de grises un divergente conserva la magnitud pero no el signo: sus dos
extremos tienen la misma luminosidad a propósito, para que ningún lado del cero se
vea más intenso. Si la figura debe leerse impresa en grises, el signo va también en
contornos: continuos para lo positivo y discontinuos para lo negativo.

## Categorías

- `categorica(n)`: `petroff6` hasta 6 categorías y `petroff10` de 7 a 10 (Petroff,
  2024). Con más de 10 el color ya no distingue: facetar, agrupar o resaltar unas
  pocas en gris.
- `categorica(n, paleta="okabe_ito")` cuando se requiera Okabe-Ito por nombre (Okabe &
  Ito, 2008; Wong, 2011). Su orden pone primero los cuatro colores con contraste de
  3:1 contra blanco.
- El amarillo de Okabe-Ito (1.3:1 contra blanco) va solo en rellenos con borde
  oscuro; el negro se reserva para una referencia o el promedio general.
- `estilo_serie(i, colores)` agrega tipo de línea y marcador; `trama(i)`, tramas de
  relleno para barras. En texto y leyendas, las series se nombran por su forma
  ("línea discontinua"), no por su color.
- Etiqueta directa sobre la gráfica cuando se pueda, en vez de solo una leyenda.

## Tamaño y formato

Según la guía de figuras de Nature, que sirve de referencia aunque la revista sea
otra (Nature, s. f.-a, s. f.-b):

- `usar_estilo("publicacion")` antes de crear la figura: texto de 5 a 7 pt en una
  sola tipografía sans-serif, líneas finas y texto editable en el PDF.
- `figsize=tamano_figura("una_columna")` para 89 mm, o `"doble_columna"` para 183 mm.
  El alto no pasa de 170 mm, para que la leyenda quepa debajo.
- `letra_panel(ax, "a")`: minúscula, negrita, vertical y de 8 pt.
- Se guarda en PDF vectorial. Nunca `bbox_inches="tight"`: cambia el ancho diseñado.
- Si hace falta un formato de mapa de bits, PNG a 300 dpi como mínimo. Nunca JPG: su
  compresión con pérdida degrada líneas y texto.
- Para diapositivas, `usar_estilo("presentacion")`.

## Ejes y escalas

- Toda transformación (dB, % de cambio, z-score) aparece en la etiqueta del eje o de la
  barra de color, y la barra lleva siempre unidades y, si es divergente, el signo.
- Una escala logarítmica o `symlog` se usa solo con una razón (rango dinámico amplio,
  ley de potencia esperada) y se declara en el pie.
- Amplitudes de EEG en µV cuando los valores caen entre -100 y 100. Si un eje cruza
  tres órdenes de magnitud, notación científica:
  `ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))`.

## Estadística visible

- Toda media lleva su incertidumbre, y el pie dice cuál es. El IC 95% es preferible
  al SEM (Cumming & Finch, 2005; Krzywinski & Altman, 2013).
- Datos continuos: mostrar la distribución, con puntos por sujeto o violín, no solo
  barras, que la ocultan (Weissgerber et al., 2015).
- Exactitud de decodificación: como proporción y con el nivel de azar empírico,
  obtenido por permutación, no el teórico (Combrisson & Jerbi, 2015).
- Valor p exacto, prueba usada y corrección por comparaciones múltiples, junto con
  el tamaño del efecto.

## EEG

- Tiempo-frecuencia: declarar método, parámetros, línea base y transformación (dB o
  %). Relativo a la línea base tiene signo, y `mapa_para` lo vuelve divergente. La
  frecuencia va en escala logarítmica.
- ERP: declarar la polaridad del eje, marcar t = 0 y dibujar bandas de confianza
  por condición.
- Topografías: nariz arriba y orejas marcadas para orientar; con MNE-Python, se
  conserva el contorno de cabeza.
- Topografías y matrices comparables comparten `norm`; la conectividad se ordena por
  región (frontal, central, temporal, parietal, occipital).

## Coherencia entre paneles

Ejes compartidos cuando se comparan paneles, la misma `norm` en mapas comparables,
la misma notación numérica y una sola leyenda para varios paneles (`fig.legend`).

## Reproducibilidad

- Semilla fija en todo proceso aleatorio (`np.random.default_rng(42)`), y cada figura se
  regenera desde un script, nunca a mano. Las versiones exactas de las bibliotecas
  las fija `uv.lock`.
- Con los mismos datos, semilla y versiones, la figura sale igual, pero el PDF cambia
  de bytes porque lleva la fecha de creación. Para un PDF idéntico byte a byte, se
  fija `SOURCE_DATE_EPOCH` antes de guardar.

## Antes de enviar

`revisar_figura(fig)` muestra la figura con deuteranopia, protanopia y tritanopia
simuladas (Machado et al., 2009) y en luminosidad L*. Si algo se pierde en alguna
vista, la figura depende del color. Para una paleta propia, `separacion_minima` y
`contraste` dan las cifras.

## Cómo citar

> The Scientific colour maps roma and batlow (Crameri, 2018) were used to prevent
> visual distortion of the data and exclusion of readers with colour-vision
> deficiencies (Crameri et al., 2020). Categorical colours follow the accessible
> colour sequences of Petroff (2024).

Si se usó Okabe-Ito:

> Categorical colours follow the Okabe-Ito palette (Okabe & Ito, 2008; Wong, 2011),
> designed to be distinguishable for viewers with colour-vision deficiency.

## Referencias

Combrisson, E., & Jerbi, K. (2015). Exceeding chance level by chance: The caveat of
theoretical chance levels in brain signal classification and statistical assessment
of decoding accuracy. *Journal of Neuroscience Methods, 250*, 126–136.
https://doi.org/10.1016/j.jneumeth.2015.01.010

Crameri, F. (2018). *Scientific colour maps* (Versión 8.0.1) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.1243862

Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in science
communication. *Nature Communications, 11*(1), 5444.
https://doi.org/10.1038/s41467-020-19160-7

Cumming, G., & Finch, S. (2005). Inference by eye: Confidence intervals and how to
read pictures of data. *American Psychologist, 60*(2), 170–180.
https://doi.org/10.1037/0003-066X.60.2.170

Krzywinski, M., & Altman, N. (2013). Error bars. *Nature Methods, 10*(10), 921–922.
https://doi.org/10.1038/nmeth.2659

Machado, G. M., Oliveira, M. M., & Fernandes, L. A. F. (2009). A physiologically-based
model for simulation of color vision deficiency. *IEEE Transactions on Visualization
and Computer Graphics, 15*(6), 1291–1298. https://doi.org/10.1109/TVCG.2009.113

Nature. (s. f.-a). *Building and exporting figure panels*. Nature Research Figure
Guide. https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/

Nature. (s. f.-b). *Final submission*. https://www.nature.com/nature/for-authors/final-submission

Okabe, M., & Ito, K. (2008). *Color universal design (CUD): How to make figures and
presentations that are friendly to colorblind people* [Sitio web]. J*FLY Data Depository
for Drosophila Researchers. https://jfly.uni-koeln.de/color/

Petroff, M. A. (2024). *Accessible color sequences for data visualization* (Versión 3)
[Preprint]. arXiv. https://doi.org/10.48550/arXiv.2107.02270

Tufte, E. R. (2001). *The visual display of quantitative information* (2.ª ed.).
Graphics Press.

Weissgerber, T. L., Milic, N. M., Winham, S. J., & Garovic, V. D. (2015). Beyond bar
and line graphs: Time for a new data presentation paradigm. *PLOS Biology, 13*(4),
e1002128. https://doi.org/10.1371/journal.pbio.1002128

Wong, B. (2011). Points of view: Color blindness. *Nature Methods, 8*(6), 441.
https://doi.org/10.1038/nmeth.1618
