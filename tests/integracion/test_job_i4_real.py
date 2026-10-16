"""El job nocturno de I4 con MongoDB y RabbitMQ REALES: del desconocido guardado al aviso que recibe n8n."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pymongo = pytest.importorskip("pymongo")

from app.config import Config  # noqa: E402
from app.domain.rostros import Desconocido  # noqa: E402
from app.infrastructure.composicion import construir_job_agrupamiento  # noqa: E402
from app.infrastructure.mongo.repositorios import RepositorioDesconocidosMongo  # noqa: E402
from app.jobs.agrupar_desconocidos import ejecutar  # noqa: E402
from tests.dobles import con_similitud  # noqa: E402
from tests.integracion.conftest import BASE, RABBIT_URL, URI  # noqa: E402

pytestmark = pytest.mark.integracion
CONTRATOS = Path(__file__).resolve().parents[4] / "contracts" / "events"


def _config(rabbit: str = RABBIT_URL) -> Config:
    return Config("i", "j", persistencia="mongo", mongo_uri=URI, mongo_db=BASE, rabbitmq_url=rabbit)


def _guardar_visitante(base: Any, dias_atras: tuple[int, ...] = (6, 3, 1)) -> None:
    repo = RepositorioDesconocidosMongo(base["desconocidos"])
    ahora = datetime.now(UTC)
    for k, dias in enumerate(dias_atras):
        repo.guardar(
            Desconocido.capturar(
                con_similitud(0.97, otro=100 + k),
                "facenet-vggface2",
                "1.0.0",
                "porteria",
                ahora - timedelta(days=dias),
                timedelta(days=30),
            )
        )


def test_I4_el_visitante_recurrente_llega_al_exchange_con_el_sobre_del_contrato(
    base: Any, cola_de_eventos: Any
) -> None:
    _guardar_visitante(base)
    resumen = ejecutar(*construir_job_agrupamiento(_config()))
    assert resumen["visitantes_recurrentes"] == 1 and len(resumen["alertas_nuevas"]) == 1
    assert (resumen["avisos_publicados"], resumen["avisos_pendientes"]) == (1, 0)

    # en MongoDB: el visitante (validador $jsonSchema), los desconocidos marcados y el outbox publicado
    cluster = base["clusters_desconocidos"].find_one({})
    assert cluster["alertado"] is True and cluster["apariciones"] == 3 and cluster["diasDistintos"] == 3
    assert cluster["clusterId"] == resumen["alertas_nuevas"][0] and len(cluster["miembros"]) == 3
    assert {d["clusterId"] for d in base["desconocidos"].find({})} == {cluster["clusterId"]}
    aviso = base["outbox"].find_one({})
    assert aviso["publicadoEn"] is not None and aviso["type"] == "ia.VisitanteRecurrenteDetectado"

    # lo que recibió el broker
    sobre = cola_de_eventos()
    assert sobre is not None and sobre["eventId"] == aviso["eventId"] and sobre["type"] == aviso["type"]
    assert sobre["data"]["clusterId"] == cluster["clusterId"] and sobre["data"]["diasDistintos"] == 3
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    if CONTRATOS.exists():
        recursos = [
            (p.name, referencing.Resource.from_contents(json.loads(p.read_text(encoding="utf-8"))))
            for p in CONTRATOS.glob("*.schema.json")
        ]
        esquema = json.loads(
            (CONTRATOS / "ia.VisitanteRecurrenteDetectado.schema.json").read_text(encoding="utf-8")
        )
        validador = jsonschema.Draft202012Validator(
            esquema, registry=referencing.Registry().with_resources(recursos)
        )
        assert list(validador.iter_errors(sobre)) == []


def test_I4_la_noche_siguiente_el_mismo_visitante_no_genera_otro_aviso(
    base: Any, cola_de_eventos: Any
) -> None:
    _guardar_visitante(base)
    construir = construir_job_agrupamiento(_config())
    ejecutar(*construir)
    assert cola_de_eventos() is not None
    segunda = ejecutar(*construir_job_agrupamiento(_config()))
    assert segunda["alertas_nuevas"] == [] and segunda["avisos_publicados"] == 0
    assert cola_de_eventos(espera_s=1.0) is None
    assert base["outbox"].count_documents({}) == 1 and base["clusters_desconocidos"].count_documents({}) == 1


def test_S4_si_rabbitmq_esta_caido_el_aviso_espera_en_el_outbox_y_se_envia_despues(
    base: Any, cola_de_eventos: Any
) -> None:
    _guardar_visitante(base)
    caido = ejecutar(
        *construir_job_agrupamiento(
            _config("amqp://condohub:condohub@localhost:1/%2F?connection_attempts=1&socket_timeout=1")
        )
    )
    assert (caido["avisos_publicados"], caido["avisos_pendientes"]) == (0, 1)
    assert base["outbox"].find_one({})["publicadoEn"] is None  # no se perdió: espera
    assert cola_de_eventos(espera_s=0.5) is None

    recuperado = ejecutar(
        *construir_job_agrupamiento(_config())
    )  # la noche siguiente, con el broker de vuelta
    assert (recuperado["avisos_publicados"], recuperado["avisos_pendientes"]) == (1, 0)
    assert recuperado["alertas_nuevas"] == []  # el visitante ya estaba registrado: no se duplica
    assert cola_de_eventos() is not None and base["outbox"].count_documents({}) == 1


def test_I3_un_visitante_con_apariciones_en_solo_dos_dias_no_genera_aviso(
    base: Any, cola_de_eventos: Any
) -> None:
    _guardar_visitante(base, dias_atras=(5, 5, 2))
    resumen = ejecutar(*construir_job_agrupamiento(_config()))
    assert resumen["visitantes_recurrentes"] == 0 and base["outbox"].count_documents({}) == 0
    assert cola_de_eventos(espera_s=0.5) is None
