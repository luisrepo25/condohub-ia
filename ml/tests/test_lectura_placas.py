"""Lectura de la placa por votación entre franjas (ml/src/lectura_placas.py)."""

import pytest

from ml.src import lectura_placas as lp


def test_A4_candidatas_corrige_letra_digito_y_penaliza():
    c = lp.candidatas("5280Y1N", 0.8)
    assert list(c) == ["5280YIN"]
    assert c["5280YIN"] == pytest.approx(0.8 * 0.9)


def test_A4_candidatas_quita_ruido_de_los_bordes():
    assert "6658CAR" in lp.candidatas("L6658CARI", 0.6)
    assert lp.candidatas("6658CAR", 0.7) == {"6658CAR": pytest.approx(0.7)}


def test_A4_formato_antiguo_o_rotulo_no_producen_placa():
    assert lp.candidatas("BOLIVIA510MDA", 0.9) == {}
    assert lp.candidatas("BOLIVIA", 0.9) == {}
    assert lp.candidatas("", 0.9) == {}


def test_A4_se_acepta_la_placa_con_tres_votos():
    lecturas = [("16658CARE", 0.6), ("6658CAR", 0.7), ("L6658CARI", 0.5), ("BOLIVIA", 0.9)]
    placa, confianza = lp.elegir_por_votos(lecturas)
    assert placa == "6658CAR"
    assert confianza == pytest.approx(0.7)


def test_A4_con_menos_votos_que_el_minimo_no_se_acepta_nada():
    assert lp.elegir_por_votos([("6658CAR", 0.9), ("6658CAR", 0.9)]) == (None, 0.0)
    assert lp.elegir_por_votos([("6658CAR", 0.9), ("6658CAR", 0.9)], votos_minimos=2)[0] == "6658CAR"


def test_A4_una_ventana_desfasada_que_aparece_en_una_sola_franja_no_pasa():
    """Una lectura ruidosa puede contener una «placa» corrida un carácter: sin votos repetidos no se acepta."""
    lecturas = [("12225FTU", 0.8), ("2225", 0.7), ("BOLIVIA", 0.9), ("", 0.0), ("FTU", 0.6), ("2225F", 0.5)]
    assert lp.elegir_por_votos(lecturas)[0] is None


def test_A4_gana_la_placa_con_mas_votos_aunque_otra_tenga_mas_confianza():
    lecturas = [("1234ABC", 0.99), ("5678DEF", 0.5), ("5678DEF", 0.5), ("5678DEF", 0.5)]
    assert lp.elegir_por_votos(lecturas)[0] == "5678DEF"


def test_votos_minimos_invalido():
    with pytest.raises(ValueError):
        lp.elegir_por_votos([("6658CAR", 0.9)], votos_minimos=0)
