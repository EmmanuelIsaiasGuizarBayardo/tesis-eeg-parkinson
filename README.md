# Tesis_EEG_Parkinson

Detección de enfermedad de Parkinson en EEG: validación por sujetos y transferibilidad a Emotiv EPOC X.

## Estructura

```
data/raw/          INMUTABLE. Datos crudos, idealmente en layout BIDS-EEG. No versionado.
data/processed/    Derivados: épocas, features, matrices filtradas. No versionado.
notebooks/         Exploración interactiva. No contiene lógica reutilizable.
results/           Figuras y métricas. No versionado.
src/tesis_eeg_parkinson/   Código fuente. Paquete instalable.
tests/             Pruebas.
tools/             Utilidades del repositorio.
```

## Puesta en marcha

Requisitos: Git y [uv](https://docs.astral.sh/uv/). Funciona igual en Windows, Mac y Linux.

```
uv sync
uv run pre-commit install
```

`uv sync` crea `.venv`, instala desde `uv.lock` e instala el paquete en modo editable.

**Sin uv, solo con pip:**

```
python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Importar desde libretas

El paquete está instalado en modo editable, así que no hace falta tocar `sys.path`:

```python
from tesis_eeg_parkinson.reproducibility import set_seed, get_run_context
```

## Dependencias

`pyproject.toml` → `uv.lock` (fuente de verdad) → `requirements.txt` (export).

Para agregar una dependencia: `uv add <paquete>`. Al hacer commit, un hook regenera
`uv.lock` y `requirements.txt`; si los modifica, el commit se detiene a propósito y
basta con volver a agregar los archivos.

## Licencia

El código se distribuye bajo la licencia MIT (`LICENSE`).

## Datos de personas

<!-- Declaración obligatoria: qué se capta, de quién, dónde vive y cuánto dura.
     Si el proyecto sí guarda datos, reemplaza el párrafo por qué se guarda,
     dónde, por cuánto tiempo y con qué consentimiento. -->

**Señal de personas.** Registros de EEG en reposo de 31 personas adultas (15 con
enfermedad de Parkinson y 16 controles) del conjunto público ds002778 v1.0.5 de
OpenNeuro (licencia CC0), desidentificados por sus curadores; el consentimiento
informado corresponde al estudio original que recolectó los datos (Swann et al.,
2015). Los crudos viven en `data/raw/` y los derivados en `data/processed/`, en la
máquina de quien corre el análisis y fuera del control de versiones, durante el
desarrollo de la tesis. Ningún componente los transmite por red. Este repositorio
no contiene datos personales.

**Marco legal.** Esta declaración es técnica, no un aviso de privacidad. Revisó si
hace falta uno: pendiente.

## Decisiones documentadas

- **Ingreso de datos.** `tools/download_ds002778.py` es el único script que escribe
  en `data/raw/`, una sola vez, y se niega si el destino ya tiene contenido. Después
  del ingreso, `data/raw/` es inmutable.
- **ICA por registro.** Cada ICA se ajusta con su propio registro y sin etiquetas;
  con validación por sujeto, ningún dato de prueba entra al entrenamiento. Sería
  fuga (modo 3 del módulo de investigación) en cualquier partición dentro del sujeto.
- **Detalle del preprocesamiento** y justificación de cada paso: `docs/preprocesamiento.md`.

## Créditos

Los roles de cada persona, en taxonomía CRediT, están en `CREDITS.md`. Para citar
el proyecto, GitHub genera la referencia desde `CITATION.cff` con el botón
**Cite this repository**. Para contribuir, ver `CONTRIBUTING.md`.

## Estándar DUNNE

Este proyecto nació de la plantilla DUNNE. Las reglas que le aplican están en
`docs/estandar/` y los comandos de uso frecuente en `docs/comandos.md`.
`AGENTS.md` le entrega ese contexto a Claude Code y a la mayoría de los
asistentes de código.

Para traer las mejoras más recientes del estándar, con el árbol de trabajo limpio:

```
uvx copier update --trust
```
