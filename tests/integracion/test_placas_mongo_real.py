"""La predicción de una lectura de placa pasa el validador $jsonSchema de `db/mongo-init.js` (MongoDB real).

Preparación y comandos: ver `test_mongo_real.py`.
"""

from typing import Any

import pytest

pymongo = pytest.importorskip("pymongo")

from app.application.leer_placa import LecturaCruda, LeerPlaca  # noqa: E402
from app.infrastructure.predicciones_placas import RepositorioPrediccionesPlacasMongo  # noqa: E402
from tests.dobles import AHORA, RelojFijo  # noqa: E402

pytestmark = pytest.mark.integracion


class _Lector:
    nombre, version = "lector-prueba", "1.0.0"

    def __init__(self, lectura: LecturaCruda | None) -> None:
        self._lectura = lectura

    def leer(self, imagen: bytes) -> LecturaCruda | None:
        return self._lectura


def test_I5_la_lectura_de_una_placa_queda_guardada_con_modelo_version_y_confianza(base: Any) -> None:
    caso = LeerPlaca(
        _Lector(LecturaCruda("2602SIT", 0.97, (10.0, 20.0, 110.0, 60.0))),
        RepositorioPrediccionesPlacasMongo(base["predicciones"]),
        RelojFijo(),
    )
    r = caso.ejecutar(b"img", camara="porteria-1")
    doc = base["predicciones"].find_one({"tipo": "PLACA"})
    assert str(doc["_id"]) == r.prediccion_id
    assert doc["resultado"]["texto"] == "2602SIT"
    assert doc["modelo"] == {"nombre": "lector-prueba", "version": "1.0.0"}
    assert (doc["confianza"], doc["fecha"], doc["camara"]) == (0.97, AHORA, "porteria-1")


def test_una_lectura_SIN_PLACA_tambien_pasa_el_validador(base: Any) -> None:
    caso = LeerPlaca(
        _Lector(None), RepositorioPrediccionesPlacasMongo(base["predicciones"]), RelojFijo()
    )
    caso.ejecutar(b"img")
    doc = base["predicciones"].find_one({"tipo": "PLACA"})
    assert doc["resultado"]["estado"] == "SIN_PLACA" and doc["resultado"]["texto"] is None
