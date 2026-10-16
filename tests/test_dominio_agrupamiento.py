"""I4 en el dominio: qué grupo es un «visitante recurrente» (≥ 3 apariciones en ≥ 3 días distintos)."""

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.agrupamiento import Aparicion, GrupoRecurrente, ParametrosAgrupamiento, grupos_recurrentes
from tests.dobles import eje

INICIO = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def _aparicion(i: int, dia: int, camara: str | None = "porteria", cluster_id: str | None = None) -> Aparicion:
    momento = INICIO + timedelta(days=dia)
    return Aparicion(f"d{i}", eje(0), momento.strftime("%Y-%m-%d"), momento, camara, cluster_id)


def test_I4_los_parametros_por_defecto_son_los_de_la_regla_y_se_validan() -> None:
    p = ParametrosAgrupamiento()
    assert (p.eps, p.min_samples, p.min_apariciones, p.min_dias) == (0.30, 3, 3, 3)
    for malo in ({"eps": 0}, {"eps": 2.5}, {"min_samples": 0}, {"min_apariciones": 0}, {"min_dias": 0}):
        with pytest.raises(ValueError):
            ParametrosAgrupamiento(**malo)


def test_I4_tres_apariciones_en_tres_dias_distintos_es_un_visitante_recurrente() -> None:
    aps = [_aparicion(0, 0), _aparicion(1, 3), _aparicion(2, 7)]
    [g] = grupos_recurrentes([0, 0, 0], aps, ParametrosAgrupamiento())
    assert (g.apariciones, g.dias_distintos) == (3, 3)
    assert g.primera_vez == aps[0].capturado_en and g.ultima_vez == aps[2].capturado_en


def test_I4_quien_viene_varias_veces_el_mismo_dia_no_es_recurrente() -> None:
    aps = [_aparicion(i, 0) for i in range(5)]
    assert grupos_recurrentes([0] * 5, aps, ParametrosAgrupamiento()) == []


def test_I4_dos_dias_distintos_no_alcanzan_aunque_haya_muchas_apariciones() -> None:
    aps = [_aparicion(0, 0), _aparicion(1, 0), _aparicion(2, 1), _aparicion(3, 1)]
    assert grupos_recurrentes([0, 0, 0, 0], aps, ParametrosAgrupamiento()) == []


def test_I4_el_ruido_nunca_es_un_visitante_recurrente() -> None:
    aps = [_aparicion(0, 0), _aparicion(1, 3), _aparicion(2, 7)]
    assert grupos_recurrentes([-1, -1, -1], aps, ParametrosAgrupamiento()) == []


def test_I4_cada_grupo_se_evalua_por_separado_y_salen_en_orden_de_aparicion() -> None:
    aps = [
        _aparicion(0, 0),
        _aparicion(1, 0),
        _aparicion(2, 1),
        _aparicion(3, 2),
        _aparicion(4, 5),
        _aparicion(5, 9),
    ]
    # grupo 0: días 0, 1 y 2; grupo 1: día 0 más los días 5 y 9: los dos cumplen
    grupos = grupos_recurrentes([0, 1, 0, 0, 1, 1], aps, ParametrosAgrupamiento())
    assert [sorted(a.id for a in g.miembros) for g in grupos] == [["d0", "d2", "d3"], ["d1", "d4", "d5"]]


def test_los_umbrales_de_i4_son_parametros() -> None:
    aps = [_aparicion(0, 0), _aparicion(1, 3)]
    assert grupos_recurrentes([0, 0], aps, ParametrosAgrupamiento()) == []
    assert len(grupos_recurrentes([0, 0], aps, ParametrosAgrupamiento(min_apariciones=2, min_dias=2))) == 1


def test_el_grupo_reporta_la_camara_mas_frecuente_o_ninguna() -> None:
    con_camara = GrupoRecurrente(
        (_aparicion(0, 0, "norte"), _aparicion(1, 1, "sur"), _aparicion(2, 2, "sur"))
    )
    assert con_camara.camara == "sur"
    sin_camara = GrupoRecurrente((_aparicion(0, 0, None), _aparicion(1, 1, None)))
    assert sin_camara.camara is None


def test_el_grupo_recuerda_a_que_visitante_pertenecia_en_una_noche_anterior() -> None:
    nuevo = GrupoRecurrente((_aparicion(0, 0), _aparicion(1, 1), _aparicion(2, 2)))
    assert nuevo.cluster_previo() is None
    conocido = GrupoRecurrente(
        (
            _aparicion(0, 0, cluster_id="A"),
            _aparicion(1, 1, cluster_id="A"),
            _aparicion(2, 2, cluster_id="B"),
            _aparicion(3, 3),
        )
    )
    assert conocido.cluster_previo() == "A"  # el id que más apariciones del grupo ya tenían


def test_etiquetas_y_apariciones_deben_ser_del_mismo_tamano() -> None:
    with pytest.raises(ValueError):
        grupos_recurrentes([0, 0], [_aparicion(0, 0)], ParametrosAgrupamiento())
