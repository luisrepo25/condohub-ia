import json
from pathlib import Path

from app.domain.modelos import ModeloDePlacas


class CatalogoFichaJson:
    """Lee el modelo vigente de la ficha `modelo_placas.json` que genera el notebook 07 (`ml/`).

    Se lee en cada consulta: publicar una versión nueva es reemplazar el archivo, sin reiniciar el servicio.
    """

    def __init__(self, ruta: str) -> None:
        self._ruta = ruta

    def placas_activo(self) -> ModeloDePlacas | None:
        if not self._ruta or not Path(self._ruta).is_file():
            return None
        ficha = json.loads(Path(self._ruta).read_text(encoding="utf-8"))
        return ModeloDePlacas(
            nombre=ficha["modelo"],
            version=ficha["version"],
            formato=ficha["formato"],
            archivo=ficha["archivo"],
            sha256=ficha["sha256"],
        )
