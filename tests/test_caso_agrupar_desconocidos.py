"""El job nocturno de I4: agrupar a los desconocidos y avisar una sola vez de cada visitante recurrente."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from app.application.agrupar_desconocidos import AgruparDesconocidos, ClusterRegistrado
from app.domain.agrupamiento import ParametrosAgrupamiento
from app.domain.rostros import Desconocido
from app.infrastructure.dbscan import AgrupadorDbscan
from app.infrastructure.memoria import (
    OutboxMemoria,
    RegistroClustersMemoria,
    RepositorioDesconocidosMemoria,
)
from tests.dobles import RelojFijo, con_similitud

INICIO = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
TIPO = "ia.VisitanteRecurrenteDetectado"


def _vector(persona: int, k: int) -> tuple[float, ...]:
    """Un desconocido por persona y aparición: misma persona = cercanos; personas distintas = ortogonales."""
    return con_similitud(0.97, base=persona, otro=100 + persona * 10 + k)


class Escena:
    def __init__(self, parametros: ParametrosAgrupamiento | None = None) -> None:
        self.desconocidos = RepositorioDesconocidosMemoria()
        self.clusters = RegistroClustersMemoria()
        self.outbox = OutboxMemoria()
        self.reloj = RelojFijo(INICIO)
        self._vistos: dict[int, int] = {}
        self.caso = AgruparDesconocidos(
            self.desconocidos,
            self.clusters,
            self.outbox,
            AgrupadorDbscan(),
            self.reloj,
            parametros or ParametrosAgrupamiento(),
            retencion=timedelta(days=30),
        )

    def ver(self, persona: int, dia: int, camara: str | None = "porteria") -> None:
        """La persona `persona` aparece en portería el día `dia` (contado desde INICIO) y no es reconocida."""
        k = self._vistos.get(persona, 0)
        self._vistos[persona] = k + 1
        momento = INICIO + timedelta(days=dia)
        self.desconocidos.guardar(
            Desconocido.capturar(
                _vector(persona, k), "facenet-vggface2", "1.0.0", camara, momento, timedelta(days=30)
            )
        )

    def noche(self, dia: int) -> Any:
        """Corre el job en la noche del día `dia`."""
        self.reloj.avanzar(INICIO + timedelta(days=dia, hours=14) - self.reloj.ahora())
        return self.caso.ejecutar()


def test_I4_solo_se_avisa_del_visitante_que_cumple_3_apariciones_en_3_dias_distintos() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia)  # recurrente: 3 días distintos
    for _ in range(3):
        e.ver(1, 2)  # viene 3 veces pero el mismo día
    e.ver(2, 1)
    e.ver(2, 5)  # solo dos días
    e.ver(3, 4)  # una sola aparición
    r = e.noche(8)
    assert r.desconocidos == 9 and r.grupos == 1 and len(r.alertas) == 1
    assert len(e.outbox.eventos) == 1 and e.outbox.eventos[0].tipo == TIPO


def test_I4_el_evento_lleva_los_datos_del_contrato_y_un_id_estable() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia, "porteria")
    r = e.noche(8)
    [evento] = e.outbox.eventos
    cluster_id = r.alertas[0]
    assert evento.event_id == str(
        uuid5(NAMESPACE_URL, f"condohub/ia/VisitanteRecurrenteDetectado/{cluster_id}")
    )
    sobre = evento.como_sobre()
    assert set(sobre) == {"eventId", "type", "occurredAt", "version", "data"}
    assert sobre["type"] == TIPO and sobre["version"] == 1 and sobre["occurredAt"].endswith("Z")
    assert sobre["data"] == {
        "clusterId": cluster_id,
        "apariciones": 3,
        "diasDistintos": 3,
        "primeraVez": "2026-10-01T09:00:00Z",
        "ultimaVez": "2026-10-08T09:00:00Z",
        "camara": "porteria",
    }


def test_el_evento_no_lleva_camara_si_ninguna_aparicion_la_indica() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia, None)
    e.noche(8)
    assert "camara" not in e.outbox.eventos[0].como_sobre()["data"]


def test_I4_el_visitante_que_sigue_viniendo_no_vuelve_a_generar_alerta() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia)
    primera = e.noche(8)
    e.ver(0, 9)  # sigue viniendo
    segunda = e.noche(10)
    assert len(primera.alertas) == 1 and segunda.alertas == [] and segunda.grupos == 1
    assert len(e.outbox.eventos) == 1
    [cluster] = e.clusters.todos()
    assert (
        cluster.cluster_id == primera.alertas[0] and cluster.apariciones == 4 and cluster.dias_distintos == 4
    )


def test_I4_correr_el_job_dos_veces_la_misma_noche_no_duplica_la_alerta() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia)
    e.noche(8)
    repetida = e.noche(8)
    assert repetida.alertas == [] and len(e.outbox.eventos) == 1


def test_I4_otro_visitante_recurrente_distinto_genera_su_propia_alerta() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia)
    primera = e.noche(8)
    for dia in (9, 10, 11):
        e.ver(5, dia)
    segunda = e.noche(12)
    assert len(segunda.alertas) == 1 and segunda.alertas[0] != primera.alertas[0]
    assert len(e.outbox.eventos) == 2


def test_I3_los_desconocidos_vencidos_no_participan() -> None:
    e = Escena()
    for dia in (-40, -35, -31):  # ya pasaron los 30 días: el índice TTL los borraría
        e.ver(0, dia)
    r = e.noche(5)
    assert r.desconocidos == 0 and r.alertas == [] and e.outbox.eventos == []


def test_con_menos_desconocidos_que_min_samples_no_se_hace_nada() -> None:
    e = Escena()
    e.ver(0, 0)
    e.ver(0, 1)
    r = e.noche(2)
    assert (r.desconocidos, r.grupos, r.alertas) == (2, 0, []) and e.outbox.eventos == []


def test_I4_si_el_job_se_cae_a_mitad_el_reintento_no_duplica_ni_pierde_el_aviso() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia)

    class RegistroQueFallaUnaVez(RegistroClustersMemoria):
        fallos_pendientes = 1

        def guardar(self, cluster: ClusterRegistrado) -> None:
            if self.fallos_pendientes:
                self.fallos_pendientes -= 1
                raise RuntimeError("se cayó el proceso")
            super().guardar(cluster)

    e.clusters = RegistroQueFallaUnaVez()
    e.caso = AgruparDesconocidos(
        e.desconocidos,
        e.clusters,
        e.outbox,
        AgrupadorDbscan(),
        e.reloj,
        ParametrosAgrupamiento(),
        retencion=timedelta(days=30),
    )
    with pytest.raises(RuntimeError):
        e.noche(8)
    assert len(e.outbox.eventos) == 1  # el aviso ya quedó en el outbox: no se pierde
    r = e.noche(8)
    assert len(e.outbox.eventos) == 1  # el reintento reutiliza el mismo visitante y el mismo eventId
    assert len(e.clusters.todos()) == 1 and e.clusters.todos()[0].alertado
    assert r.alertas == [e.clusters.todos()[0].cluster_id] or r.alertas == []


def test_los_parametros_de_i4_se_respetan() -> None:
    e = Escena(ParametrosAgrupamiento(min_apariciones=2, min_dias=2, min_samples=2))
    e.ver(0, 0)
    e.ver(0, 3)
    assert len(e.noche(4).alertas) == 1


def test_el_outbox_es_idempotente_por_event_id() -> None:
    e = Escena()
    for dia in (0, 3, 7):
        e.ver(0, dia)
    e.noche(8)
    e.outbox.agregar(e.outbox.eventos[0])
    assert len(e.outbox.eventos) == 1
    assert UUID(e.outbox.eventos[0].event_id)
