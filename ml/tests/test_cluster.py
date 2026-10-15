import numpy as np

from ml.src import cluster


def _desconocidos() -> tuple[np.ndarray, list[str]]:
    """Dos visitantes recurrentes (4 y 3 días), uno que vino 3 veces el MISMO día y 5 sueltos."""
    rng = np.random.default_rng(3)
    centros = rng.normal(size=(9, 512))

    def fotos(centro: int, n: int) -> np.ndarray:
        return centros[centro] + rng.normal(scale=0.05, size=(n, 512))

    emb = np.vstack([fotos(0, 4), fotos(1, 3), fotos(2, 3), *[fotos(3 + i, 1) for i in range(5)]])
    dias = ["d1", "d2", "d3", "d4", "d1", "d2", "d3", "d5", "d5", "d5", "d1", "d2", "d3", "d4", "d5"]
    return emb, dias


def test_I4_dbscan_encuentra_los_grupos_y_deja_las_apariciones_sueltas_como_ruido() -> None:
    emb, _ = _desconocidos()
    etiquetas = cluster.agrupar(emb, eps=0.3, min_samples=3)
    assert len(set(etiquetas) - {-1}) == 3
    assert (etiquetas[-5:] == -1).all()


def test_I4_solo_alerta_si_hay_3_apariciones_en_dias_distintos() -> None:
    emb, dias = _desconocidos()
    etiquetas = cluster.agrupar(emb, eps=0.3, min_samples=3)
    recurrentes = cluster.clusters_recurrentes(etiquetas, dias)
    # A (4 días) y B (3 días) alertan; C vino 3 veces el mismo día y NO alerta
    assert len(recurrentes) == 2
    assert etiquetas[8] not in recurrentes


def test_I4_los_umbrales_son_parametros() -> None:
    etiquetas = np.array([0, 0, 0, 0])
    dias = ["d1", "d1", "d2", "d2"]
    assert cluster.clusters_recurrentes(etiquetas, dias) == []
    assert cluster.clusters_recurrentes(etiquetas, dias, min_apariciones=3, min_dias=2) == [0]


def test_la_distancia_k_esta_ordenada_y_sirve_para_elegir_eps() -> None:
    emb, _ = _desconocidos()
    d = cluster.distancias_k(emb, k=3)
    assert len(d) == len(emb) and np.all(np.diff(d) >= 0)


def test_el_resumen_reporta_silhouette_con_dos_o_mas_clusters() -> None:
    emb, _ = _desconocidos()
    r = cluster.resumen(emb, cluster.agrupar(emb, eps=0.3, min_samples=3))
    assert r["clusters"] == 3 and r["ruido"] == 5
    assert r["silhouette"] is not None and r["silhouette"] > 0.5
