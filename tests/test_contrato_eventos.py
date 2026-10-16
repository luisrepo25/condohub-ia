"""El evento que publica el job de I4 cumple `contracts/events/` (lo que consume n8n)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

from app.application.agrupar_desconocidos import evento_visitante_recurrente  # noqa: E402
from app.domain.agrupamiento import Aparicion, GrupoRecurrente  # noqa: E402
from tests.dobles import eje  # noqa: E402

CONTRATOS = Path(__file__).resolve().parents[3] / "contracts" / "events"
AHORA = datetime(2026, 10, 20, 2, 0, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not CONTRATOS.exists(), reason="repo abierto sin el espacio de trabajo condohub"
)


def _validador() -> "jsonschema.Draft202012Validator":
    recursos = [
        (p.name, referencing.Resource.from_contents(json.loads(p.read_text(encoding="utf-8"))))
        for p in CONTRATOS.glob("*.schema.json")
    ]
    registro = referencing.Registry().with_resources(recursos)
    esquema = json.loads(
        (CONTRATOS / "ia.VisitanteRecurrenteDetectado.schema.json").read_text(encoding="utf-8")
    )
    return jsonschema.Draft202012Validator(esquema, registry=registro)


def _grupo(camara: str | None = "porteria") -> GrupoRecurrente:
    base = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    return GrupoRecurrente(
        tuple(
            Aparicion(
                f"d{i}",
                eje(0),
                (base + timedelta(days=3 * i)).strftime("%Y-%m-%d"),
                base + timedelta(days=3 * i),
                camara,
            )
            for i in range(3)
        )
    )


@pytest.mark.parametrize("camara", ["porteria", None])
def test_I4_el_evento_visitante_recurrente_cumple_el_contrato(camara: str | None) -> None:
    sobre = evento_visitante_recurrente("cluster-1", _grupo(camara), AHORA).como_sobre()
    errores = list(_validador().iter_errors(sobre))
    assert errores == [], [e.message for e in errores]


def test_el_contrato_rechaza_un_evento_con_datos_incompletos() -> None:
    sobre = evento_visitante_recurrente("cluster-1", _grupo(), AHORA).como_sobre()
    del sobre["data"]["diasDistintos"]
    assert list(_validador().iter_errors(sobre))  # la prueba vale solo si el validador sabe rechazar


def test_el_contrato_rechaza_un_visitante_con_menos_de_3_dias() -> None:
    sobre = evento_visitante_recurrente("cluster-1", _grupo(), AHORA).como_sobre()
    sobre["data"]["diasDistintos"] = 2
    assert list(_validador().iter_errors(sobre))
