"""Publica en RabbitMQ lo pendiente del outbox (S4). Los consumidores son idempotentes por `eventId` (S5)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.application.agrupar_desconocidos import EventoSaliente
from app.application.modelos import Reloj


class MensajeriaNoDisponible(Exception):
    """El broker no responde o rechazó el mensaje: queda pendiente para el próximo intento."""


class OutboxPendiente(Protocol):
    def pendientes(self) -> list[EventoSaliente]:
        """Lo aún no publicado, en el orden en que se guardó."""
        ...

    def marcar_publicado(self, event_id: str, cuando: datetime) -> None: ...


class PublicadorEventos(Protocol):
    def publicar(self, evento: EventoSaliente) -> None:
        """Entrega el evento al broker (con confirmación). Lanza `MensajeriaNoDisponible` si no se logra."""
        ...


@dataclass(frozen=True)
class ResultadoPublicacion:
    publicados: int
    pendientes: int


class PublicarOutbox:
    def __init__(self, outbox: OutboxPendiente, publicador: PublicadorEventos, reloj: Reloj) -> None:
        self._outbox, self._publicador, self._reloj = outbox, publicador, reloj

    def ejecutar(self) -> ResultadoPublicacion:
        pendientes = self._outbox.pendientes()
        publicados = 0
        for evento in pendientes:
            try:
                self._publicador.publicar(evento)
            except MensajeriaNoDisponible:
                break  # el orden se respeta: no se adelanta ningún mensaje posterior
            self._outbox.marcar_publicado(evento.event_id, self._reloj.ahora())
            publicados += 1
        return ResultadoPublicacion(publicados, len(pendientes) - publicados)
