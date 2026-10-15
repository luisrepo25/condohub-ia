import numpy as np

from ml.src.lfw import a_uint8


def test_a_uint8_convierte_floats_entre_0_y_1_a_0_255() -> None:
    entrada = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    assert a_uint8(entrada).tolist() == [[[0, 128, 255]]]


def test_a_uint8_respeta_los_valores_que_ya_vienen_en_0_255() -> None:
    entrada = np.array([[[0.0, 100.0, 255.0]]], dtype=np.float32)
    assert a_uint8(entrada).tolist() == [[[0, 100, 255]]]


def test_a_uint8_da_lo_mismo_con_cualquier_tamano_de_bloque() -> None:
    entrada = np.random.default_rng(0).random((7, 4, 4, 3)).astype(np.float32)
    assert np.array_equal(a_uint8(entrada, bloque=2), a_uint8(entrada, bloque=100))
    assert a_uint8(entrada).dtype == np.uint8 and a_uint8(entrada).shape == entrada.shape
