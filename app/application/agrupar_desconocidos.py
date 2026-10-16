"""Job nocturno de I4: agrupa a los desconocidos de 30 días y avisa de cada visitante una sola vez.

Orden de escritura (a prueba de caídas a mitad): 1) marcar a los miembros con su visitante, 2) dejar el
aviso en el outbox (idempotente por `eventId`), 3) registrar el visitante como ya avisado.
Si el proceso se cae entre dos pasos,
el reintento reutiliza el mismo visitante y el mismo `eventId`: el aviso no se pierde ni se duplica (S4, S5).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.application.modelos import Reloj
from app.domain.agrupamiento import Aparicion, GrupoRecurrente, ParametrosAgrupamiento, grupos_recurrentes
from app.domain.rostros import Embedding

TIPO_EVENTO = "ia.VisitanteRecurrenteDetectado"


def _iso(momento: datetime) -> str:
    return momento.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class EventoSaliente:
    """Un evento listo para el outbox, con el sobre común de `contracts/events/_sobre.schema.json`."""

    event_id: str
    tipo: str
    ocurrido_en: datetime
    datos: dict[str, Any]
    version: int = 1

    def como_sobre(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "type": self.tipo,
            "occurredAt": _iso(self.ocurrido_en),
            "version": self.version,
            "data": self.datos,
        }


@dataclass(frozen=True)
class ClusterRegistrado:
    """El visitante recurrente tal como se guarda en `clusters_desconocidos`."""

    cluster_id: str
    ejecucion_id: str
    apariciones: int
    dias_distintos: int
    primera_vez: datetime
    ultima_vez: datetime
    alertado: bool
    alertado_en: datetime | None
    miembros: tuple[str, ...]
    eps: float
    min_samples: int


@dataclass(frozen=True)
class ResultadoAgrupamiento:
    desconocidos: int  # cuántos se agruparon
    grupos: int  # cuántos visitantes recurrentes hay hoy (nuevos y ya conocidos)
    alertas: list[str]  # clusterId de los avisos nuevos de esta noche


class Agrupador(Protocol):
    def agrupar(self, vectores: Sequence[Embedding], eps: float, min_samples: int) -> list[int]:
        """Una etiqueta por vector (distancia coseno); -1 es ruido."""
        ...


class LectorDesconocidos(Protocol):
    def vigentes(self, ahora: datetime) -> list[Aparicion]:
        """Los desconocidos que aún no vencen (I3)."""
        ...

    def asignar_cluster(self, ids: list[str], cluster_id: str) -> None: ...


class RegistroClusters(Protocol):
    def obtener(self, cluster_id: str) -> ClusterRegistrado | None: ...
    def guardar(self, cluster: ClusterRegistrado) -> None: ...


class Outbox(Protocol):
    def agregar(self, evento: EventoSaliente) -> None:
        """Idempotente: si ya hay un evento con ese `event_id`, no hace nada."""
        ...


def evento_visitante_recurrente(cluster_id: str, grupo: GrupoRecurrente, ahora: datetime) -> EventoSaliente:
    """El `eventId` sale del visitante, así que el mismo visitante nunca genera dos eventos distintos."""
    datos: dict[str, Any] = {
        "clusterId": cluster_id,
        "apariciones": grupo.apariciones,
        "diasDistintos": grupo.dias_distintos,
        "primeraVez": _iso(grupo.primera_vez),
        "ultimaVez": _iso(grupo.ultima_vez),
    }
    if grupo.camara is not None:
        datos["camara"] = grupo.camara
    event_id = str(uuid5(NAMESPACE_URL, f"condohub/ia/VisitanteRecurrenteDetectado/{cluster_id}"))
    return EventoSaliente(event_id, TIPO_EVENTO, ahora, datos)


class AgruparDesconocidos:
    def __init__(
        self,
        desconocidos: LectorDesconocidos,
        clusters: RegistroClusters,
        outbox: Outbox,
        agrupador: Agrupador,
        reloj: Reloj,
        parametros: ParametrosAgrupamiento,
        retencion: timedelta,
        nuevo_id: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        if retencion <= timedelta(0):
            raise ValueError("La retención debe ser positiva")
        self._desconocidos, self._clusters, self._outbox = desconocidos, clusters, outbox
        self._agrupador, self._reloj, self._p, self._nuevo_id = agrupador, reloj, parametros, nuevo_id
        self._retencion = retencion

    def ejecutar(self) -> ResultadoAgrupamiento:
        ahora = self._reloj.ahora()
        ejecucion_id = self._nuevo_id()
        apariciones = self._desconocidos.vigentes(ahora)
        if len(apariciones) < self._p.min_samples:
            return ResultadoAgrupamiento(len(apariciones), 0, [])

        etiquetas = self._agrupador.agrupar([a.vector for a in apariciones], self._p.eps, self._p.min_samples)
        grupos = grupos_recurrentes(etiquetas, apariciones, self._p)

        alertas: list[str] = []
        for grupo in grupos:
            cluster_id = (
                grupo.cluster_previo() or self._nuevo_id()
            )  # el mismo visitante conserva su identidad
            previo = self._clusters.obtener(cluster_id)
            ya_avisado = previo is not None and previo.alertado
            miembros = [a.id for a in grupo.miembros]

            self._desconocidos.asignar_cluster(miembros, cluster_id)  # 1) identidad
            if not ya_avisado:
                self._outbox.agregar(evento_visitante_recurrente(cluster_id, grupo, ahora))  # 2) aviso
            self._clusters.guardar(  # 3) registro
                ClusterRegistrado(
                    cluster_id=cluster_id,
                    ejecucion_id=ejecucion_id,
                    apariciones=grupo.apariciones,
                    dias_distintos=grupo.dias_distintos,
                    primera_vez=grupo.primera_vez,
                    ultima_vez=grupo.ultima_vez,
                    alertado=True,
                    alertado_en=previo.alertado_en if previo and previo.alertado_en else ahora,
                    miembros=tuple(miembros),
                    eps=self._p.eps,
                    min_samples=self._p.min_samples,
                )
            )
            if not ya_avisado:
                alertas.append(cluster_id)
        return ResultadoAgrupamiento(len(apariciones), len(grupos), alertas)
