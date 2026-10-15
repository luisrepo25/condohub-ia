import numpy as np
import pytest

from ml.src import metricas
from ml.src.split import dividir_por_persona


def test_I2_ninguna_persona_aparece_en_dos_particiones(embeddings_sinteticos) -> None:
    _, etiquetas = embeddings_sinteticos
    partes = dividir_por_persona(etiquetas)
    personas = {nombre: set(etiquetas[idx]) for nombre, idx in partes.items()}
    assert not (personas["train"] & personas["val"]) and not (personas["train"] & personas["test"])
    assert not (personas["val"] & personas["test"])
    assert sum(len(idx) for idx in partes.values()) == len(etiquetas)


def test_la_separacion_es_reproducible_con_la_misma_semilla(embeddings_sinteticos) -> None:
    _, etiquetas = embeddings_sinteticos
    a, b = dividir_por_persona(etiquetas, semilla=7), dividir_por_persona(etiquetas, semilla=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_pocas_personas_se_rechazan() -> None:
    with pytest.raises(ValueError):
        dividir_por_persona(np.array([0, 0, 1, 1]))


def test_I2_far_y_frr_de_scores_perfectamente_separados_son_cero() -> None:
    scores = np.array([0.9, 0.95, 0.8, 0.1, 0.2, 0.3])
    y = np.array([1, 1, 1, 0, 0, 0])
    r = metricas.evaluar_en_umbral(scores, y, umbral=0.5)
    assert r["far"] == 0 and r["frr"] == 0 and r["exactitud"] == 1
    assert metricas.eer(metricas.curva_far_frr(scores, y))[0] == 0


def test_I2_umbral_para_far_maximo_respeta_el_limite() -> None:
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(0.7, 0.1, 500), rng.normal(0.3, 0.15, 500)
    scores, y = np.concatenate([pos, neg]), np.array([1] * 500 + [0] * 500)
    curva = metricas.curva_far_frr(scores, y)
    umbral = metricas.umbral_para_far(curva, far_maximo=0.01)
    assert metricas.evaluar_en_umbral(scores, y, umbral)["far"] <= 0.01


def test_los_pares_son_mitad_positivos_mitad_negativos(embeddings_sinteticos) -> None:
    _, etiquetas = embeddings_sinteticos
    i, j, y = metricas.generar_pares(etiquetas, 1000)
    assert y.sum() == 500
    assert np.all(etiquetas[i[y == 1]] == etiquetas[j[y == 1]]) and np.all(
        etiquetas[i[y == 0]] != etiquetas[j[y == 0]]
    )
    assert np.all(i[y == 1] != j[y == 1])  # nunca la misma foto contra sí misma


def test_similitud_coseno_de_vectores_iguales_es_uno() -> None:
    v = np.random.default_rng(0).normal(size=(3, 8))
    assert np.allclose(metricas.similitud_coseno(v, v * 5), 1.0)
