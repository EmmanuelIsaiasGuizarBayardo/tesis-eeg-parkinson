# Créditos

Créditos de Tesis_EEG_Parkinson.

Los roles siguen la taxonomía [CRediT](https://credit.niso.org/), la que usan las
revistas académicas. Cada persona tiene sus roles y una narrativa que coincide
con ellos. La narrativa describe aportaciones, no cargos.

---

## Contribuciones

### Emmanuel Isaías Guízar-Bayardo

*Conceptualization · Methodology · Software · Formal analysis · Visualization · Writing – original draft*

Diseñó la réplica del pipeline de Aljalal et al. (2022) y su validación por sujetos, implementó el preprocesamiento y el análisis, y redactó la tesis.

### Miguel Serrano-Reyes

*Supervision*

Dirige la tesis.

<!-- Por cada persona más: una sección "### Nombres Apellidos", sus roles en
     cursiva y su narrativa. Si además es autora, va en CITATION.cff con el mismo
     nombre. Lo que CRediT no cubre, como operar demostraciones o facilitar
     talleres, va en una seccion "## Operación y divulgación". -->

---

## Trabajo de terceros

<!-- Describe en prosa cada pieza ajena y si el proyecto la deriva (la adapta o
     la reimplementa) o la integra (la usa tal cual). La referencia formal va
     solo en 'references' de CITATION.cff; las bibliotecas y sus versiones ya
     están en pyproject.toml y uv.lock. -->

El preprocesamiento reimplementa en Python el pipeline de EEGLAB del manual propio del autor (se deriva). Integra tal cual el modelo ICLabel (Pion-Tonachini et al., 2019) mediante mne-icalabel, los criterios de canales ruidosos de PREP (Bigdely-Shamlo et al., 2015) mediante pyprep, y la lectura BIDS de MNE-BIDS. Los datos son de Rockhill et al. (2021).

---

## Cómo citar

GitHub genera la referencia desde `CITATION.cff` con el botón **Cite this
repository**. No se mantiene una cita escrita a mano: se desfasaría con cada versión.

## Cómo se actualiza

Quien contribuya agrega su sección, con los roles que correspondan, en el mismo
*pull request* que su aportación.
