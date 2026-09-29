<!-- Generado por la plantilla DUNNE; se actualiza con 'uvx copier update --trust'. -->

# Comandos de Tesis_EEG_Parkinson

Solo aparecen las secciones que aplican a este proyecto. Donde un comando cambia
entre sistemas operativos, se muestran ambas variantes.

---

<details>
<summary><b>Trabajo diario</b></summary>

Sincronizar el entorno, por ejemplo después de un `git pull` que cambió el lock.

```
uv sync
```

Ejecutar el punto de entrada o cualquier script, sin activar nada.

```
uv run python -m tesis_eeg_parkinson.main
uv run python ruta/al/script.py
```

Activar el entorno para una terminal interactiva.

```
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Mac y Linux
source .venv/bin/activate
```

</details>

<details>
<summary><b>Dependencias</b></summary>

Agregar o quitar. Nunca con `pip install` suelto: no quedaría en `uv.lock`.

```
uv add mne-bids
uv add --dev ipdb
uv remove pandas
```

Subir todo a la versión más reciente compatible, y ver el árbol resuelto.

```
uv lock --upgrade
uv tree
```

Regenerar `requirements.txt` a mano. Siempre con este script, nunca con `uv export`.

```
uv run python tools/export_requirements.py
```

</details>

<details>
<summary><b>Libretas</b></summary>

Abrir Jupyter dentro del entorno del proyecto.

```
uv run jupyter lab
```

Volver a registrar el kernel si desaparece de la lista.

```
uv run python -m ipykernel install --user --name tesis_eeg_parkinson
```

Dentro de la libreta: importar el paquete y recargar los cambios de `src/` sin reiniciar el kernel.

```python
%load_ext autoreload
%autoreload 2
from tesis_eeg_parkinson.reproducibility import set_seed
```

</details>

<details>
<summary><b>Calidad y pruebas</b></summary>

```
uv run pytest -q
uv run ruff check . --fix
uv run ruff format .
uv run pre-commit run --all-files
```

</details>

<details>
<summary><b>Git y GitHub</b></summary>

Antes de cada push: confirmar que ningún archivo de datos se coló.

```
git status --porcelain
git ls-files data/
```

Commit normal. Si el hook regenera `uv.lock` o `requirements.txt`, vuelve a agregar y a commitear.

```
git add .
git commit -m "Mensaje"
```

Saltarse los hooks cuando no hay red.

```
git commit --no-verify -m "Mensaje"
```

Ver cambios sin paginador.

```
git --no-pager diff --stat
```

</details>

<details>
<summary><b>Publicar una versión</b></summary>

Antes de cada presentación pública. Primero sube el número en `pyproject.toml` y en
`CITATION.cff` (`version`), y fija `date-released` en el CFF con la fecha de hoy.

```
uv run pytest -q
git commit -am "Publicar la versión 1.0.0"
git tag -a v1.0.0 -m "Evento donde se presentó"
git push --follow-tags
```

La prueba de gobernanza falla si las dos versiones no coinciden.

</details>

<details>
<summary><b>Estándar DUNNE</b></summary>

Traer las mejoras del estándar, con el árbol limpio. Sin `--defaults`, pregunta por los rasgos nuevos que haya agregado la plantilla.

```
uvx copier update --trust
git --no-pager diff --stat
git add -A
git commit -m "Actualizar a la plantilla DUNNE"
```

El commit final no es opcional: sin él, la siguiente actualización encuentra el árbol sucio y se niega a correr.

Ver qué versión de la plantilla tiene este proyecto.

```
git --no-pager grep -h _commit .copier-answers.yml
```

</details>

<details>
<summary><b>Replicar en otra máquina</b></summary>

Con uv, resolución exacta desde el lock.

```
git clone URL
cd tesis-eeg-parkinson
uv sync
uv run pre-commit install
```

Solo con pip.

```
python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

</details>

<details>
<summary><b>Reparaciones y desmontaje</b></summary>

Reconstruir el entorno desde cero.

```
# Windows (PowerShell)
Remove-Item -Recurse -Force .venv
# Mac y Linux
rm -rf .venv
# Después, en ambos
uv sync
```

Desmontar el proyecto por completo, incluido su kernel de Jupyter.

```
uv run jupyter kernelspec uninstall tesis_eeg_parkinson
gh repo delete CUENTA/tesis-eeg-parkinson --yes
```

Y después borra la carpeta del proyecto.

</details>
