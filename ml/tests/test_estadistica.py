import math

import pytest

from ml.src import estadistica


def test_una_mejora_constante_es_concluyente() -> None:
    r = estadistica.resumen_pareado([0.05] * 5, [0.04] * 5)
    assert r["mejora_media"] == pytest.approx(0.01) and r["repeticiones_donde_mejora"] == 5
    assert r["concluyente"] and r["ic95"][0] > 0


def test_una_mejora_que_cambia_de_signo_no_es_concluyente() -> None:
    base = [0.05, 0.05, 0.05, 0.05, 0.05]
    ajustado = [0.04, 0.06, 0.045, 0.058, 0.048]  # a veces mejora y a veces empeora
    r = estadistica.resumen_pareado(base, ajustado)
    assert not r["concluyente"] and r["ic95"][0] < 0 < r["ic95"][1]


def test_un_empeoramiento_no_es_concluyente_de_mejora() -> None:
    r = estadistica.resumen_pareado([0.04] * 4, [0.05] * 4)
    assert not r["concluyente"] and r["mejora_media"] < 0 and r["repeticiones_donde_mejora"] == 0


def test_el_intervalo_se_ensancha_con_menos_repeticiones() -> None:
    muchos = estadistica.resumen_pareado(
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [0.04, 0.045, 0.043, 0.047, 0.041, 0.046]
    )
    pocos = estadistica.resumen_pareado([0.05, 0.05, 0.05], [0.04, 0.045, 0.043])
    assert (pocos["ic95"][1] - pocos["ic95"][0]) > (muchos["ic95"][1] - muchos["ic95"][0])


def test_con_una_sola_repeticion_no_hay_intervalo_ni_conclusion() -> None:
    r = estadistica.resumen_pareado([0.05], [0.04])
    assert r["ic95"] is None and r["desviacion"] is None and not r["concluyente"]


def test_formas_distintas_se_rechazan() -> None:
    with pytest.raises(ValueError):
        estadistica.resumen_pareado([0.05, 0.05], [0.04])
    with pytest.raises(ValueError):
        estadistica.resumen_pareado([], [])


def test_la_mejora_media_es_la_diferencia_de_las_medias() -> None:
    r = estadistica.resumen_pareado([0.06, 0.04], [0.05, 0.03])
    assert math.isclose(r["mejora_media"], r["eer_base_media"] - r["eer_ajustado_media"])
