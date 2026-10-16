import numpy as np
import pytest

from ml.src import onnx_placas as op
from ml.src import placas


def _salida(candidatas: list[tuple[float, float, float, float, float]], n: int = 8400) -> np.ndarray:
    """Salida cruda de YOLOv8 con una sola clase: [1, 5, n] con las candidatas dadas y el resto en cero."""
    s = np.zeros((1, 5, n), dtype=np.float32)
    for k, c in enumerate(candidatas):
        s[0, :, k] = c
    return s


def test_letterbox_conserva_la_proporcion_y_rellena_con_gris() -> None:
    imagen = np.zeros((480, 640, 3), dtype=np.uint8) + 255
    tensor, escala, px, py = op.letterbox(imagen)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert escala == 1.0 and px == 0 and py == 80
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)      # relleno
    assert tensor[0, 0, 320, 320] == pytest.approx(1.0)        # contenido
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_letterbox_de_una_imagen_vertical_rellena_a_los_lados() -> None:
    _, escala, px, py = op.letterbox(np.zeros((1280, 640, 3), dtype=np.uint8))
    assert escala == 0.5 and py == 0 and px == 160


def test_A4_decodificar_devuelve_cajas_en_coordenadas_de_la_imagen_original() -> None:
    # imagen original 480×640 → escala 1, relleno vertical 80; caja en el tensor: centro (320, 400), 100×40
    tensor_forma = (480, 640)
    s = _salida([(320, 400, 100, 40, 0.9)])
    det = op.decodificar(s, escala=1.0, px=0, py=80, forma_original=tensor_forma, conf_min=0.5)
    assert det.shape == (1, 5)
    assert det[0, :4].tolist() == pytest.approx([270, 300, 370, 340])
    assert det[0, 4] == pytest.approx(0.9)


def test_decodificar_descarta_la_confianza_baja_y_recorta_a_la_imagen() -> None:
    s = _salida([(320, 400, 100, 40, 0.3), (10, 100, 100, 40, 0.8)])
    det = op.decodificar(s, 1.0, 0, 0, (480, 640), conf_min=0.5)
    assert len(det) == 1 and det[0, 0] == 0.0                   # la caja se sale por la izquierda y se recorta


def test_nms_deja_la_de_mayor_confianza_entre_cajas_solapadas() -> None:
    s = _salida([(300, 300, 100, 40, 0.6), (302, 301, 100, 40, 0.9), (500, 100, 60, 30, 0.7)])
    det = op.decodificar(s, 1.0, 0, 0, (640, 640), conf_min=0.25)
    assert len(det) == 2
    assert sorted(det[:, 4].tolist()) == pytest.approx([0.7, 0.9])


def test_decodificar_sin_candidatas_devuelve_vacio_y_rechaza_formas_inesperadas() -> None:
    assert op.decodificar(_salida([]), 1.0, 0, 0, (480, 640)).shape == (0, 5)
    with pytest.raises(ValueError):
        op.decodificar(np.zeros((1, 84, 8400), dtype=np.float32), 1.0, 0, 0, (480, 640))


def test_iou() -> None:
    caja = np.array([0.0, 0.0, 10.0, 10.0])
    assert op.iou(caja, np.array([[0.0, 0.0, 10.0, 10.0]]))[0] == pytest.approx(1.0)
    assert op.iou(caja, np.array([[20.0, 20.0, 30.0, 30.0]]))[0] == 0.0
    assert op.iou(caja, np.array([[5.0, 0.0, 15.0, 10.0]]))[0] == pytest.approx(1 / 3)


def _gt() -> list[np.ndarray]:
    return [np.array([[100.0, 100, 200, 160]]), np.array([[50.0, 50, 150, 100]])]


def test_ap50_perfecto_y_sin_detecciones() -> None:
    perfectas = [np.array([[100.0, 100, 200, 160, 0.9]]), np.array([[50.0, 50, 150, 100, 0.8]])]
    assert op.ap50(perfectas, _gt()) == pytest.approx(1.0)
    assert op.ap50([np.zeros((0, 5)), np.zeros((0, 5))], _gt()) == 0.0


def test_ap50_baja_si_una_falsa_alarma_tiene_mas_confianza_que_un_acierto() -> None:
    preds = [np.array([[400.0, 400, 450, 450, 0.95], [100.0, 100, 200, 160, 0.9]]), np.array([[50.0, 50, 150, 100, 0.8]])]
    assert 0.5 < op.ap50(preds, _gt()) < 1.0


def test_ap50_sin_referencia_se_rechaza() -> None:
    with pytest.raises(ValueError):
        op.ap50([np.zeros((0, 5))], [np.zeros((0, 4))])


def test_precision_recall_en_un_punto_de_operacion() -> None:
    preds = [np.array([[100.0, 100, 200, 160, 0.9], [400.0, 400, 450, 450, 0.6]]), np.array([[50.0, 50, 150, 100, 0.3]])]
    r = op.precision_recall(preds, _gt(), conf_min=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert op.precision_recall(preds, _gt(), conf_min=0.95)["precision"] is None


def test_coincidencia_entre_modelos() -> None:
    a = [np.array([[100.0, 100, 200, 160, 0.9]]), np.array([[50.0, 50, 150, 100, 0.8]])]
    b = [np.array([[101.0, 100, 201, 160, 0.85]]), np.zeros((0, 5))]
    r = op.coincidencia(a, b)
    assert r["coinciden"] == 1 and r["solo_el_primero"] == 1 and r["solo_el_segundo"] == 0
    assert r["iou_medio"] > 0.9


def test_voc_a_cajas_da_pixeles_de_la_imagen_original() -> None:
    xml = ("<annotation><size><width>400</width><height>200</height></size>"
           "<object><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>300</xmax><ymax>150</ymax></bndbox></object>"
           "<object><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>")
    assert placas.voc_a_cajas(xml) == [(100.0, 50.0, 300.0, 150.0)]      # la caja degenerada se descarta
