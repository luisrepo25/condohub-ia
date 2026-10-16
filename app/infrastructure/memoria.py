"""Adaptadores en memoria de los puertos de rostros: para pruebas y desarrollo (no sobreviven al reinicio)."""

from datetime import datetime
from uuid import UUID, uuid4

from app.application.agrupar_desconocidos import ClusterRegistrado, EventoSaliente
from app.domain.agrupamiento import Aparicion
from app.domain.rostros import Desconocido, PerfilFacial, Prediccion


class RepositorioPerfilesMemoria:
    def __init__(self) -> None:
        self._perfiles: dict[UUID, PerfilFacial] = {}

    def guardar(self, perfil: PerfilFacial) -> None:
        self._perfiles[perfil.residente_id] = perfil

    def por_residente(self, residente_id: UUID) -> PerfilFacial | None:
        return self._perfiles.get(residente_id)

    def con_consentimiento_vigente(self) -> list[PerfilFacial]:
        return [p for p in self._perfiles.values() if p.vigente]


class RepositorioDesconocidosMemoria:
    def __init__(self) -> None:
        self.guardados: list[Desconocido] = []
        self._clusters: dict[str, str] = {}   # id del desconocido -> su visitante (job de I4)

    def guardar(self, desconocido: Desconocido) -> None:
        self.guardados.append(desconocido)

    # --- lectura para el job nocturno de I4 (puerto `LectorDesconocidos`)
    def vigentes(self, ahora: datetime) -> list[Aparicion]:
        return [
            Aparicion(f"d{i}", d.vector, d.dia, d.capturado_en, d.camara, self._clusters.get(f"d{i}"))
            for i, d in enumerate(self.guardados)
            if d.expira_en > ahora
        ]

    def asignar_cluster(self, ids: list[str], cluster_id: str) -> None:
        for id_ in ids:
            self._clusters[id_] = cluster_id


class RegistroClustersMemoria:
    def __init__(self) -> None:
        self._clusters: dict[str, ClusterRegistrado] = {}

    def obtener(self, cluster_id: str) -> ClusterRegistrado | None:
        return self._clusters.get(cluster_id)

    def guardar(self, cluster: ClusterRegistrado) -> None:
        self._clusters[cluster.cluster_id] = cluster

    def todos(self) -> list[ClusterRegistrado]:
        return list(self._clusters.values())


class OutboxMemoria:
    """Outbox en memoria: idempotente por `event_id`."""

    def __init__(self) -> None:
        self.eventos: list[EventoSaliente] = []
        self.publicados: dict[str, datetime] = {}

    def agregar(self, evento: EventoSaliente) -> None:
        if all(e.event_id != evento.event_id for e in self.eventos):
            self.eventos.append(evento)

    def pendientes(self) -> list[EventoSaliente]:
        return [e for e in self.eventos if e.event_id not in self.publicados]

    def marcar_publicado(self, event_id: str, cuando: datetime) -> None:
        self.publicados[event_id] = cuando


class RepositorioPrediccionesMemoria:
    def __init__(self) -> None:
        self.guardadas: list[tuple[str, Prediccion]] = []

    def guardar(self, prediccion: Prediccion) -> str:
        prediccion_id = str(uuid4())
        self.guardadas.append((prediccion_id, prediccion))
        return prediccion_id


class AlmacenFotosMemoria:
    def __init__(self) -> None:
        self.objetos: dict[str, bytes] = {}
        self.borrados = 0

    def guardar(self, residente_id: UUID, fotos: list[bytes]) -> list[str]:
        claves = [f"rostros/{residente_id}/{uuid4()}.jpg" for _ in fotos]
        self.objetos.update(zip(claves, fotos, strict=True))
        return claves

    def borrar(self, claves: list[str]) -> None:
        self.borrados += 1
        for clave in claves:
            self.objetos.pop(clave, None)
