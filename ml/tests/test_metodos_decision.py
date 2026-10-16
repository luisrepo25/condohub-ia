"""Métodos alternativos de decisión y calidad de fotos (ml/src/metodos_decision.py), con embeddings sintéticos: sin red ni GPU."""

import numpy as np
import pytest

from ml.src import metodos_decision as md
from ml.src import registro, zonas


def _equipo(rng: np.random.Generator, dim: int = 32):
    """Tres personas (centros distintos), dos sesiones cada una, y extraños aparte."""
    centros = rng.normal(size=(3, dim))
    emb, personas, sesiones = [], [], []
    for i, nombre in enumerate(("ana", "beto", "carla")):
        for s in (0, 1):
            emb.append(centros[i] + 0.25 * rng.normal(size=(10, dim)))
            personas += [nombre] * 10
            sesiones += [f"s{s}"] * 10
    extranos = rng.normal(size=(60, dim))
    etiquetas = np.repeat(np.arange(12), 5)
    return np.concatenate(emb), np.array(personas), np.array(sesiones), extranos, etiquetas


def test_maximo_coincide_con_el_registro_actual():
    rng = np.random.default_rng(0)
    emb, personas, sesiones, extranos, etiquetas = _equipo(rng)
    nuevo = md.puntajes_equipo_con(md.maximo, emb, personas, sesiones, extranos, etiquetas, semilla=1)
    viejo = zonas.puntajes_equipo(emb, personas, sesiones, extranos, etiquetas, semilla=1)
    np.testing.assert_allclose(nuevo["genuinos"], viejo["genuinos"])
    np.testing.assert_allclose(nuevo["companeros"], viejo["companeros"])
    np.testing.assert_allclose(nuevo["impostores_prueba"], viejo["impostores_prueba"])


@pytest.mark.parametrize("metodo", [md.maximo, md.top_k_promedio(3), md.centroide, md.svm(), md.knn(5)])
def test_cada_metodo_identifica_a_la_persona_correcta(metodo):
    rng = np.random.default_rng(1)
    emb, personas, sesiones, extranos, etiquetas = _equipo(rng)
    p = md.puntajes_equipo_con(metodo, emb, personas, sesiones, extranos, etiquetas)
    assert p["genuinos_correcto"].mean() > 0.95


@pytest.mark.parametrize("metodo", [md.maximo, md.top_k_promedio(3), md.centroide])
def test_I2_los_metodos_por_similitud_separan_extranos(metodo):
    rng = np.random.default_rng(1)
    emb, personas, sesiones, extranos, etiquetas = _equipo(rng)
    p = md.puntajes_equipo_con(metodo, emb, personas, sesiones, extranos, etiquetas)
    assert md.resumen(p)["auc_propios_vs_extranos"] > 0.9


def test_I2_un_clasificador_de_conjunto_cerrado_no_rechaza_extranos():
    """k-NN siempre elige a alguien: su probabilidad no distingue a un extraño, la similitud sí. Por eso I2 usa similitud."""
    rng = np.random.default_rng(1)
    emb, personas, sesiones, extranos, etiquetas = _equipo(rng)
    con_similitud = md.resumen(
        md.puntajes_equipo_con(md.maximo, emb, personas, sesiones, extranos, etiquetas)
    )
    con_knn = md.resumen(md.puntajes_equipo_con(md.knn(5), emb, personas, sesiones, extranos, etiquetas))
    assert con_knn["auc_propios_vs_extranos"] < con_similitud["auc_propios_vs_extranos"]


def test_top_k_promedio_no_supera_al_maximo():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(5, 16))
    reg = {"ana": rng.normal(size=(8, 16))}
    assert (md.top_k_promedio(3)(emb, reg)[1] <= registro.puntuar(emb, reg)[1] + 1e-9).all()


def test_registrar_solo_con_fotos_usables_ignora_las_excluidas():
    rng = np.random.default_rng(3)
    emb, personas, sesiones, extranos, etiquetas = _equipo(rng)
    todas = md.puntajes_equipo_con(md.maximo, emb, personas, sesiones, extranos, etiquetas)
    mitad = np.arange(len(emb)) % 2 == 0
    p = md.puntajes_equipo_con(
        md.maximo, emb, personas, sesiones, extranos, etiquetas, usable_en_registro=mitad
    )
    assert len(p["genuinos"]) == len(todas["genuinos"])  # se prueba con las mismas fotos
    assert (
        p["genuinos"].mean() <= todas["genuinos"].mean() + 1e-9
    )  # con menos fotos registradas no hay más parecido


def test_auc_perfecto_azar_y_empates():
    assert md.auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert md.auc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert md.auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5


def test_nitidez_distingue_nitida_de_borrosa():
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(4)
    nitida = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
    borrosa = np.asarray(Image.fromarray(nitida).filter(ImageFilter.GaussianBlur(6)))
    assert md.nitidez(nitida) > md.nitidez(borrosa)
    assert md.nitidez(np.full((100, 100, 3), 128, dtype=np.uint8)) == 0.0


def test_correlacion_de_rangos():
    x = np.arange(10.0)
    assert md.correlacion_rangos(x, x**3) == pytest.approx(1.0)
    assert md.correlacion_rangos(x, -x) == pytest.approx(-1.0)
