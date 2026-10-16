"""Publicar lo pendiente del outbox en RabbitMQ (S4): en orden, sin perder ni repetir mensajes."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from app.application.agrupar_desconocidos import EventoSaliente
from app.application.publicar_outbox import MensajeriaNoDisponible, PublicarOutbox
from app.infrastructure.memoria import OutboxMemoria
from app.infrastructure.rabbitmq import PublicadorRabbitMq
from tests.dobles import AHORA, RelojFijo


def _evento(n: int) -> EventoSaliente:
    return EventoSaliente(
        f"00000000-0000-4000-8000-00000000000{n}", "ia.VisitanteRecurrenteDetectado", AHORA, {"n": n}
    )


class PublicadorFalso:
    def __init__(self) -> None:
        self.enviados: list[str] = []
        self.falla_en: set[str] = set()

    def publicar(self, evento: EventoSaliente) -> None:
        if evento.event_id in self.falla_en:
            raise MensajeriaNoDisponible("el broker no responde")
        self.enviados.append(evento.event_id)


def _caso(outbox: OutboxMemoria, publicador: PublicadorFalso) -> PublicarOutbox:
    return PublicarOutbox(outbox, publicador, RelojFijo())


def test_S4_se_publican_los_pendientes_en_el_orden_en_que_se_guardaron_y_quedan_marcados() -> None:
    outbox, publicador = OutboxMemoria(), PublicadorFalso()
    for n in (1, 2, 3):
        outbox.agregar(_evento(n))
    r = _caso(outbox, publicador).ejecutar()
    assert (r.publicados, r.pendientes) == (3, 0)
    assert publicador.enviados == [_evento(n).event_id for n in (1, 2, 3)]
    assert outbox.pendientes() == [] and set(outbox.publicados) == set(publicador.enviados)


def test_S4_lo_ya_publicado_no_se_vuelve_a_enviar() -> None:
    outbox, publicador = OutboxMemoria(), PublicadorFalso()
    outbox.agregar(_evento(1))
    _caso(outbox, publicador).ejecutar()
    outbox.agregar(_evento(2))
    r = _caso(outbox, publicador).ejecutar()
    assert r.publicados == 1 and publicador.enviados == [_evento(1).event_id, _evento(2).event_id]


def test_S4_si_el_broker_falla_el_mensaje_se_queda_pendiente_y_se_respeta_el_orden() -> None:
    outbox, publicador = OutboxMemoria(), PublicadorFalso()
    for n in (1, 2, 3):
        outbox.agregar(_evento(n))
    publicador.falla_en = {_evento(2).event_id}
    r = _caso(outbox, publicador).ejecutar()
    assert (r.publicados, r.pendientes) == (1, 2)  # el 3 no se adelanta al 2
    assert [e.event_id for e in outbox.pendientes()] == [_evento(2).event_id, _evento(3).event_id]
    publicador.falla_en = set()
    assert (
        _caso(outbox, publicador).ejecutar().publicados == 2
    )  # el reintento de la noche siguiente los envía


def test_sin_pendientes_no_hace_nada() -> None:
    r = _caso(OutboxMemoria(), PublicadorFalso()).ejecutar()
    assert (r.publicados, r.pendientes) == (0, 0)


# ------------------------------------------------ adaptador de RabbitMQ (con pika de mentira)


class CanalFalso:
    def __init__(self, falla: Exception | None = None) -> None:
        self.mensajes: list[dict[str, Any]] = []
        self.confirmando = False
        self._falla = falla

    def confirm_delivery(self) -> None:
        self.confirmando = True

    def basic_publish(
        self, exchange: str, routing_key: str, body: bytes, properties: Any, mandatory: bool
    ) -> None:
        if self._falla:
            raise self._falla
        self.mensajes.append(
            {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
                "mandatory": mandatory,
            }
        )


class ConexionFalsa:
    def __init__(self, canal: CanalFalso) -> None:
        self._canal = canal
        self.cerrada = False

    def channel(self) -> CanalFalso:
        return self._canal

    def close(self) -> None:
        self.cerrada = True


def test_el_mensaje_va_al_exchange_con_la_routing_key_del_tipo_y_el_sobre_como_cuerpo() -> None:
    pytest.importorskip("pika")
    canal = CanalFalso()
    conexion = ConexionFalsa(canal)
    PublicadorRabbitMq("amqp://x", "condohub.events", conectar=lambda: conexion).publicar(_evento(1))
    [m] = canal.mensajes
    assert (m["exchange"], m["routing_key"], m["mandatory"]) == (
        "condohub.events",
        "ia.VisitanteRecurrenteDetectado",
        True,
    )
    assert json.loads(m["body"]) == _evento(1).como_sobre()
    assert m["properties"].delivery_mode == 2 and m["properties"].message_id == _evento(1).event_id
    assert m["properties"].content_type == "application/json"
    assert canal.confirmando and conexion.cerrada


def test_un_fallo_del_broker_se_traduce_a_mensajeria_no_disponible_y_cierra_la_conexion() -> None:
    pytest.importorskip("pika")
    error_de_pika = type("UnroutableError", (Exception,), {"__module__": "pika.exceptions"})
    canal = CanalFalso(falla=error_de_pika("sin ruta"))
    conexion = ConexionFalsa(canal)
    with pytest.raises(MensajeriaNoDisponible):
        PublicadorRabbitMq("amqp://x", "condohub.events", conectar=lambda: conexion).publicar(_evento(1))
    assert conexion.cerrada


def test_si_no_se_puede_conectar_se_avisa_que_la_mensajeria_no_esta_disponible() -> None:
    def sin_conexion() -> Any:
        raise OSError("connection refused")

    with pytest.raises(MensajeriaNoDisponible):
        PublicadorRabbitMq("amqp://x", "condohub.events", conectar=sin_conexion).publicar(_evento(1))


def test_un_error_que_no_es_del_broker_no_se_disfraza() -> None:
    canal = CanalFalso(falla=KeyError("bug"))
    with pytest.raises(KeyError):
        PublicadorRabbitMq("amqp://x", "condohub.events", conectar=lambda: ConexionFalsa(canal)).publicar(
            _evento(1)
        )
    assert datetime.now(UTC)
