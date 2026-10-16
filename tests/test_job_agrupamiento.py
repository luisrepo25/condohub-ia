"""El comando nocturno: resumen, código de salida y requisitos de configuración."""

from datetime import timedelta
from typing import Any

import pytest

from app.application.agrupar_desconocidos import AgruparDesconocidos
from app.application.publicar_outbox import MensajeriaNoDisponible, PublicarOutbox
from app.config import Config
from app.domain.agrupamiento import ParametrosAgrupamiento
from app.domain.rostros import Desconocido
from app.infrastructure.composicion import construir_job_agrupamiento
from app.infrastructure.dbscan import AgrupadorDbscan
from app.infrastructure.memoria import OutboxMemoria, RegistroClustersMemoria, RepositorioDesconocidosMemoria
from app.jobs.agrupar_desconocidos import ejecutar
from tests.dobles import AHORA, RelojFijo, con_similitud


class _Publicador:
    def __init__(self, funciona: bool) -> None:
        self.funciona, self.enviados = funciona, 0

    def publicar(self, evento: Any) -> None:
        if not self.funciona:
            raise MensajeriaNoDisponible("sin broker")
        self.enviados += 1


def _job(publicador: _Publicador) -> tuple[AgruparDesconocidos, PublicarOutbox]:
    desconocidos = RepositorioDesconocidosMemoria()
    for dia in (7, 4, 1):  # el mismo visitante en tres días distintos
        momento = AHORA - timedelta(days=dia)
        desconocidos.guardar(
            Desconocido.capturar(
                con_similitud(0.97, otro=100 + dia),
                "facenet-vggface2",
                "1.0.0",
                "porteria",
                momento,
                timedelta(days=30),
            )
        )
    outbox = OutboxMemoria()
    reloj = RelojFijo()
    return (
        AgruparDesconocidos(
            desconocidos,
            RegistroClustersMemoria(),
            outbox,
            AgrupadorDbscan(),
            reloj,
            ParametrosAgrupamiento(),
            retencion=timedelta(days=30),
        ),
        PublicarOutbox(outbox, publicador, reloj),
    )


def test_I4_el_resumen_cuenta_desconocidos_visitantes_alertas_y_avisos_publicados() -> None:
    publicador = _Publicador(funciona=True)
    resumen = ejecutar(*_job(publicador))
    assert resumen["desconocidos"] == 3 and resumen["visitantes_recurrentes"] == 1
    assert len(resumen["alertas_nuevas"]) == 1
    assert (resumen["avisos_publicados"], resumen["avisos_pendientes"]) == (1, 0) and publicador.enviados == 1


def test_S4_si_el_broker_no_responde_el_aviso_queda_pendiente_y_el_resumen_lo_dice() -> None:
    resumen = ejecutar(*_job(_Publicador(funciona=False)))
    assert (resumen["avisos_publicados"], resumen["avisos_pendientes"]) == (0, 1)
    assert len(resumen["alertas_nuevas"]) == 1  # el visitante se detectó igual: el aviso espera en el outbox


def test_el_resumen_no_incluye_vectores_ni_fotos() -> None:
    resumen = ejecutar(*_job(_Publicador(funciona=True)))
    assert set(resumen) == {
        "desconocidos",
        "visitantes_recurrentes",
        "alertas_nuevas",
        "avisos_publicados",
        "avisos_pendientes",
    }


def test_el_job_exige_mongo_porque_los_desconocidos_viven_ahi() -> None:
    with pytest.raises(ValueError, match="PERSISTENCIA=mongo"):
        construir_job_agrupamiento(Config("i", "j"))


def test_con_mongo_el_job_se_arma_sin_conectar_todavia() -> None:
    pytest.importorskip("pymongo")
    agrupar, publicar = construir_job_agrupamiento(
        Config("i", "j", persistencia="mongo", mongo_uri="mongodb://servidor-que-no-existe:27017")
    )
    assert isinstance(agrupar, AgruparDesconocidos) and isinstance(publicar, PublicarOutbox)


def test_los_parametros_de_i4_salen_de_la_configuracion() -> None:
    c = Config("i", "j", dbscan_eps=0.4, dbscan_min_samples=4, i4_min_apariciones=5, i4_min_dias=4)
    p = ParametrosAgrupamiento(c.dbscan_eps, c.dbscan_min_samples, c.i4_min_apariciones, c.i4_min_dias)
    assert (p.eps, p.min_samples, p.min_apariciones, p.min_dias) == (0.4, 4, 5, 4)
