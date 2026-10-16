import json
from pathlib import Path

from app.domain.modelo_lectura import ModeloDeLectura


class CatalogoLecturaFichaJson:
    """Lee el reconocedor vigente de `ficha_reconocedor.json`, que genera el notebook 11 (`ml/`).

    Se lee en cada consulta: publicar una versión nueva es reemplazar el archivo, sin reiniciar el servicio.
    La ficha se publica en el almacén con su propio nombre de archivo.
    """

    def __init__(self, ruta: str) -> None:
        self._ruta = ruta

    def lectura_activo(self) -> ModeloDeLectura | None:
        if not self._ruta or not Path(self._ruta).is_file():
            return None
        ficha = json.loads(Path(self._ruta).read_text(encoding="utf-8"))
        return ModeloDeLectura(
            nombre=ficha["modelo"],
            version=ficha["version"],
            formato=ficha["formato"],
            archivo=ficha["archivo"],
            sha256=ficha["sha256"],
            ficha=Path(self._ruta).name,
        )
