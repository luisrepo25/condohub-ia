"""La vista previa de un reporte contra el validador `$jsonSchema` REAL de MongoDB (G4).

Cómo correrlas: ver test_mongo_real.py.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

pymongo = pytest.importorskip("pymongo")

from app.domain.reportes import FormatoReporte, Reporte, vista_previa_desde  # noqa: E402
from app.infrastructure.mongo.reportes import documento_a_reporte, reporte_a_documento  # noqa: E402


@pytest.fixture(autouse=True)
def limpia_reportes(base: Any) -> Any:
    base["reportes"].delete_many({})
    yield
    base["reportes"].delete_many({})


def _reporte_listo() -> Reporte:
    reporte = Reporte(
        id=uuid4(), admin_sub="sub-admin", formato=FormatoReporte.EXCEL, creado_en=datetime.now(UTC)
    )
    reporte.registrar_vista_previa(
        vista_previa_desde(
            [
                {"codigo": "B-101", "titular.nombre": "Ana", "deuda": "540.00"},
                {"codigo": "B-102", "deuda": None},
            ]
        )
    )
    return reporte


def test_G4_el_validador_acepta_la_vista_previa_y_se_recupera_igual(base: Any) -> None:
    original = _reporte_listo()

    base["reportes"].insert_one(reporte_a_documento(original))
    guardado = base["reportes"].find_one({"reporteId": str(original.id)})

    assert guardado is not None
    assert guardado["vistaPrevia"]["columnas"] == [
        "codigo",
        "titular.nombre",
        "deuda",
    ]  # los puntos van en valores, no en claves
    assert documento_a_reporte(guardado).vista_previa == original.vista_previa


def test_G4_el_validador_rechaza_una_vista_previa_mal_formada(base: Any) -> None:
    doc = reporte_a_documento(_reporte_listo())
    doc["vistaPrevia"]["filas"] = [[1, 2, 3]]  # las celdas son texto o null, nunca números

    with pytest.raises(pymongo.errors.WriteError):
        base["reportes"].insert_one(doc)
