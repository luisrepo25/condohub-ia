"""MongoDB para el job nocturno de I4: lectura de desconocidos, visitantes registrados y outbox (S4)."""

import importlib
from datetime import datetime
from typing import Any

from app.application.agrupar_desconocidos import ClusterRegistrado, EventoSaliente
from app.domain.agrupamiento import Aparicion


class LectorDesconocidosMongo:
    """Colección `desconocidos`: lee los que aún no vencen (I3) y les anota a qué visitante pertenecen."""

    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def vigentes(self, ahora: datetime) -> list[Aparicion]:
        return [
            Aparicion(
                id=str(d["_id"]),
                vector=tuple(float(x) for x in d["vector"]),
                dia=d.get("dia") or d["capturadoEn"].strftime("%Y-%m-%d"),
                capturado_en=d["capturadoEn"],
                camara=d.get("camara"),
                cluster_id=d.get("clusterId"),
            )
            for d in self._c.find({"expiraEn": {"$gt": ahora}})
        ]

    def asignar_cluster(self, ids: list[str], cluster_id: str) -> None:
        object_id = importlib.import_module("bson").ObjectId
        self._c.update_many(
            {"_id": {"$in": [object_id(i) for i in ids]}}, {"$set": {"clusterId": cluster_id}}
        )


class RegistroClustersMongo:
    """Colección `clusters_desconocidos`: un documento por visitante (su identidad dura entre noches)."""

    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def obtener(self, cluster_id: str) -> ClusterRegistrado | None:
        d = self._c.find_one({"clusterId": cluster_id})
        if d is None:
            return None
        return ClusterRegistrado(
            cluster_id=d["clusterId"],
            ejecucion_id=d["ejecucionId"],
            apariciones=d["apariciones"],
            dias_distintos=d["diasDistintos"],
            primera_vez=d["primeraVez"],
            ultima_vez=d["ultimaVez"],
            alertado=d["alertado"],
            alertado_en=d.get("alertadoEn"),
            miembros=tuple(str(m) for m in d.get("miembros", [])),
            eps=d["parametros"]["eps"],
            min_samples=d["parametros"]["minSamples"],
        )

    def guardar(self, cluster: ClusterRegistrado) -> None:
        object_id = importlib.import_module("bson").ObjectId
        documento = {
            "clusterId": cluster.cluster_id,
            "ejecucionId": cluster.ejecucion_id,
            "parametros": {"eps": cluster.eps, "minSamples": cluster.min_samples, "metrica": "cosine"},
            "miembros": [object_id(m) for m in cluster.miembros],
            "apariciones": cluster.apariciones,
            "diasDistintos": cluster.dias_distintos,
            "primeraVez": cluster.primera_vez,
            "ultimaVez": cluster.ultima_vez,
            "alertado": cluster.alertado,  # I4: true cuando el aviso ya quedó en el outbox
            "alertadoEn": cluster.alertado_en,
        }
        self._c.replace_one({"clusterId": cluster.cluster_id}, documento, upsert=True)


class OutboxMongo:
    """Colección `outbox` (S4): los eventos esperan aquí hasta que se publican en RabbitMQ."""

    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def agregar(self, evento: EventoSaliente) -> None:
        """Idempotente: el índice único sobre `eventId` y `$setOnInsert` evitan duplicados."""
        documento = {
            "eventId": evento.event_id,
            "type": evento.tipo,
            "occurredAt": evento.ocurrido_en,
            "version": evento.version,
            "data": evento.datos,
            "publicadoEn": None,
        }
        self._c.update_one({"eventId": evento.event_id}, {"$setOnInsert": documento}, upsert=True)

    def pendientes(self) -> list[EventoSaliente]:
        return [
            EventoSaliente(d["eventId"], d["type"], d["occurredAt"], d["data"], d["version"])
            for d in self._c.find({"publicadoEn": None}).sort([("occurredAt", 1), ("_id", 1)])
        ]

    def marcar_publicado(self, event_id: str, cuando: datetime) -> None:
        self._c.update_one({"eventId": event_id}, {"$set": {"publicadoEn": cuando}})
