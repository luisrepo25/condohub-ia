import json

import numpy as np

from ml.src import linea_base

CFG = {
    "semilla": 42,
    "proporciones": [0.6, 0.2, 0.2],
    "pares_por_parte": 600,
    "far_maximo": 0.01,
    "k_vecinos": 3,
}


def test_I2_el_umbral_se_calibra_en_validacion_y_se_reporta_en_prueba(embeddings_sinteticos) -> None:
    emb, etiquetas = embeddings_sinteticos
    r = linea_base.verificacion(emb, etiquetas, CFG)
    assert 0 <= r["umbral_calibrado_en_validacion"] <= 1
    assert r["prueba"]["umbral"] == r["umbral_calibrado_en_validacion"]  # el de prueba es el de validación
    assert r["prueba"]["exactitud"] > 0.9  # nubes bien separadas


def test_la_identificacion_cerrada_clasifica_bien_nubes_separadas(embeddings_sinteticos) -> None:
    emb, etiquetas = embeddings_sinteticos
    r = linea_base.identificacion(emb, etiquetas, CFG)
    assert r["svm_lineal"]["exactitud"] > 0.95 and r["knn"]["exactitud"] > 0.95


def test_I5_la_corrida_guarda_metrics_json(tmp_path, embeddings_sinteticos) -> None:
    emb, etiquetas = embeddings_sinteticos
    linea_base.ejecutar(None, etiquetas, lambda _: emb, CFG, tmp_path)
    guardado = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert guardado["personas"] == 20 and "verificacion" in guardado and "identificacion" in guardado


def test_la_misma_semilla_da_las_mismas_metricas(embeddings_sinteticos) -> None:
    emb, etiquetas = embeddings_sinteticos
    assert linea_base.verificacion(emb, etiquetas, CFG) == linea_base.verificacion(emb, etiquetas, CFG)
    assert np.isfinite(linea_base.verificacion(emb, etiquetas, CFG)["eer_prueba"])
