"""Publicador de eventos en RabbitMQ (pika): exchange `condohub.events` (topic), routing key = tipo."""

import importlib
import json
from collections.abc import Callable
from typing import Any

from app.application.agrupar_desconocidos import EventoSaliente
from app.application.publicar_outbox import MensajeriaNoDisponible


class PublicadorRabbitMq:
    """Puerto `PublicadorEventos`. Una conexión por mensaje: el volumen es de unas pocas alertas por noche."""

    def __init__(self, url: str, exchange: str, conectar: Callable[[], Any] | None = None) -> None:
        self._url, self._exchange, self._conectar = url, exchange, conectar

    def _abrir(self) -> Any:
        if self._conectar is not None:
            return self._conectar()
        pika = importlib.import_module("pika")
        return pika.BlockingConnection(pika.URLParameters(self._url))

    def publicar(self, evento: EventoSaliente) -> None:
        try:
            conexion = self._abrir()
        except Exception as e:
            if isinstance(e, OSError) or type(e).__module__.startswith("pika"):
                raise MensajeriaNoDisponible("No se pudo conectar con RabbitMQ") from e
            raise
        try:
            pika = importlib.import_module("pika")
            canal = conexion.channel()
            canal.confirm_delivery()  # el broker confirma que recibió (y enrutó) el mensaje
            canal.basic_publish(
                exchange=self._exchange,
                routing_key=evento.tipo,
                body=json.dumps(evento.como_sobre(), ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json", delivery_mode=2, message_id=evento.event_id
                ),
                mandatory=True,  # si ninguna cola lo recibe, falla en lugar de perderse en silencio
            )
        except Exception as e:
            if isinstance(e, OSError) or type(e).__module__.startswith("pika"):
                raise MensajeriaNoDisponible(f"RabbitMQ no aceptó el mensaje: {type(e).__name__}") from e
            raise
        finally:
            conexion.close()
