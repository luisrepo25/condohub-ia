import numpy as np
import pytest

from ml.src import cluster, escenario_desconocidos, evaluacion_cluster


def _caras(personas: int = 40, fotos: int = 10, ruido: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Estilo LFW: una nube alrededor del centro de cada persona, normalizada L2 (como los embeddings reales)."""
    rng = np.random.default_rng(0)
    centros = rng.normal(size=(personas, 512))
    etiquetas = np.repeat(np.arange(personas), fotos)
    emb = centros[etiquetas] + rng.normal(scale=ruido, size=(len(etiquetas), 512))
    return emb / np.linalg.norm(emb, axis=1, keepdims=True), etiquetas


def _escenario(semilla: int = 1) -> escenario_desconocidos.Escenario:
    emb, etiquetas = _caras()
    return escenario_desconocidos.construir(
        emb, etiquetas, np.arange(40), recurrentes=6, mismo_dia=3, dos_dias=3, sueltos=10, semilla=semilla
    )


def test_I4_el_escenario_tiene_los_cuatro_tipos_de_persona() -> None:
    e = _escenario()
    reales = evaluacion_cluster.personas_recurrentes(e.personas, e.dias)
    assert len(reales) == 6                               # solo los recurrentes cumplen ≥3 apariciones en ≥3 días
    assert len(set(e.personas.tolist())) == 22
    assert len(e.dias) == len(e.personas) == len(e.embeddings)


def test_el_escenario_es_reproducible_con_la_misma_semilla() -> None:
    a, b = _escenario(5), _escenario(5)
    assert np.array_equal(a.personas, b.personas) and a.dias == b.dias and np.array_equal(a.embeddings, b.embeddings)


def test_el_escenario_rechaza_pedir_mas_personas_de_las_que_hay() -> None:
    emb, etiquetas = _caras(personas=5)
    with pytest.raises(ValueError):
        escenario_desconocidos.construir(emb, etiquetas, np.arange(5), 3, 2, 2, 2)


def test_I4_alertas_perfectas_dan_precision_y_recall_uno() -> None:
    e = _escenario()
    etiquetas = cluster.agrupar(e.embeddings, eps=0.5, min_samples=3)
    r = evaluacion_cluster.evaluar_alertas(etiquetas, e.personas, e.dias)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["falsas_alarmas"] == 0


def test_I4_quien_viene_varias_veces_el_mismo_dia_es_falsa_alarma_si_se_alerta() -> None:
    e = _escenario()
    # Etiquetas «perfectas» (una por persona) pero con los umbrales de días en 1: los de un solo día alertarían
    etiquetas = e.personas.copy()
    r = evaluacion_cluster.evaluar_alertas(etiquetas, e.personas, e.dias, min_apariciones=3, min_dias=1)
    reales_estrictos = evaluacion_cluster.personas_recurrentes(e.personas, e.dias, 3, 3)
    assert r["alertas"] > len(reales_estrictos)           # sin la regla de días distintos sobran alertas


def test_evaluar_sin_alertas_deja_precision_indefinida() -> None:
    e = _escenario()
    r = evaluacion_cluster.evaluar_alertas(np.full(len(e.personas), -1), e.personas, e.dias)
    assert r["alertas"] == 0 and r["precision"] is None and r["recall"] == 0.0


def test_un_cluster_que_mezcla_personas_no_cuenta_como_acierto() -> None:
    personas = np.array([1, 1, 1, 2, 2, 2])
    dias = ["d1", "d2", "d3", "d1", "d2", "d3"]
    etiquetas = np.zeros(6, dtype=int)                    # un solo clúster con dos personas (pureza 0.5)
    r = evaluacion_cluster.evaluar_alertas(etiquetas, personas, dias)
    assert r["alertas"] == 1 and r["falsas_alarmas"] == 1 and r["aciertos"] == 0


def test_barrer_eps_devuelve_una_fila_por_valor_y_un_eps_chico_deja_todo_como_ruido() -> None:
    e = _escenario()
    filas = evaluacion_cluster.barrer_eps(e.embeddings, e.personas, e.dias, [0.001, 0.5])
    assert [f["eps"] for f in filas] == [0.001, 0.5]
    assert filas[0]["clusters"] == 0 and filas[0]["ruido"] == len(e.personas)
    assert filas[1]["recall"] == 1.0


def test_f1() -> None:
    assert evaluacion_cluster.f1({"precision": 1.0, "recall": 0.5}) == pytest.approx(2 / 3)
    assert evaluacion_cluster.f1({"precision": None, "recall": 0.5}) == 0.0
    assert evaluacion_cluster.f1({"precision": 0.0, "recall": 0.0}) == 0.0


def test_el_codo_cae_donde_la_curva_empieza_a_subir_fuerte() -> None:
    plano = np.linspace(0.1, 0.3, 90)
    subida = np.linspace(0.35, 1.0, 10)
    d = np.concatenate([plano, subida])
    assert 0.25 <= evaluacion_cluster.codo(d) <= 0.45
