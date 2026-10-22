"""I4 con re-aviso (propuesta): el visitante que sigue apareciendo vuelve a avisar cada N apariciones más."""

import pytest

from app.domain.agrupamiento import ParametrosAgrupamiento
from tests.test_caso_agrupar_desconocidos import Escena


def _escena(reaviso_cada: int) -> Escena:
    return Escena(ParametrosAgrupamiento(min_apariciones=3, min_dias=1, reaviso_cada=reaviso_cada))


def test_I4_sin_reaviso_un_visitante_avisa_una_sola_vez_aunque_siga_apareciendo() -> None:
    e = _escena(0)
    for _ in range(3):
        e.ver(0, 0)
    assert len(e.noche(0).alertas) == 1
    for _ in range(10):
        e.ver(0, 0)
    assert e.noche(0).alertas == []
    assert len(e.outbox.eventos) == 1


def test_I4_con_reaviso_avisa_de_nuevo_al_cruzar_cada_tramo() -> None:
    e = _escena(5)
    for _ in range(3):
        e.ver(0, 0)
    primera = e.noche(0)
    assert len(primera.alertas) == 1  # el primer aviso, al llegar a 3
    for _ in range(2):
        e.ver(0, 0)  # 5 apariciones: empieza el tramo 1
    segunda = e.noche(0)
    assert segunda.alertas == primera.alertas  # el mismo visitante, avisado otra vez
    e.ver(0, 0)  # 6: sigue en el tramo 1
    assert e.noche(0).alertas == []
    assert len(e.outbox.eventos) == 2


def test_S5_cada_tramo_tiene_su_event_id_y_repetir_la_noche_no_duplica() -> None:
    e = _escena(5)
    for _ in range(5):
        e.ver(0, 0)
    e.noche(0)
    e.noche(0)  # reintento de la misma noche
    assert len(e.outbox.eventos) == 1
    for _ in range(5):
        e.ver(0, 0)  # 10 apariciones: tramo 2
    e.noche(0)
    ids = [ev.event_id for ev in e.outbox.eventos]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_I4_el_reaviso_actualiza_la_fecha_del_aviso() -> None:
    e = _escena(5)
    for _ in range(3):
        e.ver(0, 0)
    cluster_id = e.noche(0).alertas[0]
    primero = e.clusters.obtener(cluster_id)
    assert primero is not None
    for _ in range(2):
        e.ver(0, 1)
    e.noche(1)
    segundo = e.clusters.obtener(cluster_id)
    assert segundo is not None and primero.alertado_en is not None
    assert segundo.alertado_en is not None and segundo.alertado_en > primero.alertado_en


def test_el_reaviso_no_puede_ser_negativo() -> None:
    with pytest.raises(ValueError):
        ParametrosAgrupamiento(reaviso_cada=-1)
