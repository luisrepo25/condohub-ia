import random

import pytest

from ml.src import placas

VOC = """<annotation><size><width>400</width><height>200</height></size>
<object><name>licence</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>300</xmax><ymax>150</ymax></bndbox></object>
</annotation>"""


@pytest.mark.parametrize("texto", ["1234ABC", "1234 ABC", "1234-abc", "0000ZZZ"])
def test_A4_placa_con_cuatro_digitos_y_tres_letras_es_valida(texto: str) -> None:
    assert placas.es_placa_valida(texto)


@pytest.mark.parametrize("texto", ["ABC1234", "123ABC", "12345ABC", "1234AB", "", "1234ABCD"])
def test_A4_lectura_que_no_cumple_el_formato_no_es_placa(texto: str) -> None:
    assert not placas.es_placa_valida(texto)


def test_A4_normalizar_deja_mayusculas_sin_separadores() -> None:
    assert placas.normalizar("12-34 abc.") == "1234ABC"


def test_A4_correccion_por_posicion_arregla_confusiones_letra_digito() -> None:
    assert placas.corregir_por_posicion("I23OABC") == "1230ABC"      # I→1, O→0 en la zona de dígitos
    assert placas.corregir_por_posicion("1234A8C") == "1234ABC"      # 8→B en la zona de letras
    assert placas.corregir_por_posicion("5B34SBC") == "5834SBC"      # B→8 en dígitos; S es letra válida


def test_A4_correccion_no_inventa_si_el_largo_no_es_siete() -> None:
    assert placas.corregir_por_posicion("12ABC") == "12ABC"


@pytest.mark.parametrize(
    ("leido", "esperado"),
    [
        ("18271WWFL", "8271WWF"),    # marco leído como "1" y "L"
        ("4616MGRI", "4616MGR"),
        ("J0502WXM", "0502WXM"),
        ("1234ABC", "1234ABC"),      # ya válida: no se toca
        ("ZDN1771", "ZDN1771"),      # sin ventana válida: no se inventa nada
        ("12AB", "12AB"),
    ],
)
def test_A4_extraer_placa_descarta_sobrantes_sin_inventar(leido: str, esperado: str) -> None:
    assert placas.extraer_placa(leido) == esperado


def test_A4_elegir_placa_real_con_z_leida_como_2_y_ruido_del_marco() -> None:
    # Lectura real (foto de placa boliviana): rótulo, letra del departamento y código del sticker alrededor
    bloques = [("26082YU", 0.76), ("PBOLIVIA", 0.82), ("PC", 0.26), ("382085", 0.7)]
    assert placas.elegir_placa(bloques) == ("2608ZYU", 0.76)


def test_A4_elegir_placa_prefiere_el_bloque_valido_de_mayor_confianza() -> None:
    assert placas.elegir_placa([("608ZYU", 0.53), ("2608ZYU", 0.72), ("382085", 1.0)]) == ("2608ZYU", 0.72)


def test_A4_elegir_placa_une_bloques_si_el_ocr_parte_la_placa() -> None:
    assert placas.elegir_placa([("BOLIVIA", 0.9), ("2608", 0.9), ("ZYU", 0.8)]) == ("2608ZYU", 0.8)


def test_A4_el_rotulo_bolivia_nunca_se_convierte_en_placa() -> None:
    assert placas.elegir_placa([("BOLIVIA", 0.95)]) == (None, 0.0)
    assert placas.elegir_placa([("PALABRA", 0.95)]) == (None, 0.0)     # 7 letras: demasiadas correcciones


def test_A4_elegir_placa_sin_formato_valido_no_acepta_nada() -> None:
    assert placas.elegir_placa([("PBOLIVIA", 0.8), ("PC", 0.3), ("382085", 1.0)]) == (None, 0.0)
    assert placas.elegir_placa([]) == (None, 0.0)


def test_voc_a_yolo_normaliza_la_caja() -> None:
    assert placas.voc_a_yolo(VOC) == ["0 0.500000 0.500000 0.500000 0.500000"]


def test_voc_sin_tamano_se_rechaza() -> None:
    with pytest.raises(ValueError):
        placas.voc_a_yolo("<annotation><object><bndbox/></object></annotation>")


def test_la_division_de_imagenes_es_reproducible_y_sin_solapes() -> None:
    nombres = [f"img{i}.png" for i in range(100)]
    a, b = placas.dividir_imagenes(nombres, semilla=3), placas.dividir_imagenes(nombres, semilla=3)
    assert a == b
    todas = a["train"] + a["val"] + a["test"]
    assert sorted(todas) == sorted(nombres)
    assert len(a["test"]) == 15


def test_division_con_pocas_imagenes_se_rechaza() -> None:
    with pytest.raises(ValueError):
        placas.dividir_imagenes(["a", "b", "c"])


def test_distancia_de_edicion() -> None:
    assert placas.distancia_edicion("1234ABC", "1234ABC") == 0
    assert placas.distancia_edicion("1234ABC", "1Z34ABC") == 1
    assert placas.distancia_edicion("1234ABC", "1234AB") == 1


def test_metricas_de_lectura() -> None:
    m = placas.metricas_lectura(["1234ABC", "5678XYZ"], ["1234ABC", "5678 XYS"])
    assert m["placa_completa"] == 0.5
    assert m["precision_por_caracter"] == pytest.approx(1 - 1 / 14)


def test_placas_sinteticas_cumplen_el_formato() -> None:
    pytest.importorskip("PIL")
    from ml.src import placas_sinteticas

    rng = random.Random(1)
    for _ in range(20):
        texto = placas_sinteticas.texto_aleatorio(rng)
        assert placas.es_placa_valida(texto)
    assert placas_sinteticas.dibujar("1234ABC", rng).size == (360, 180)
