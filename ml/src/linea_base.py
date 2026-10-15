"""Línea base facial: embeddings congelados + clasificador. Es la referencia contra la que se compara todo.

`embeber` es una función imágenes -> matriz de embeddings (en Colab: MTCNN + FaceNet). Se inyecta para poder
probar toda esta lógica en CPU con embeddings sintéticos.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ml.src import metricas
from ml.src.split import dividir_por_persona


def verificacion(emb: np.ndarray, etiquetas: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """1:1 con separación por persona: umbral calibrado en VALIDACIÓN, métricas reportadas en PRUEBA (I2)."""
    partes = dividir_por_persona(etiquetas, tuple(cfg["proporciones"]), cfg["semilla"])
    resultados: dict[str, Any] = {}
    scores_por_parte = {}
    for nombre in ("val", "test"):
        idx = partes[nombre]
        i, j, y = metricas.generar_pares(etiquetas[idx], cfg["pares_por_parte"], cfg["semilla"])
        scores_por_parte[nombre] = (metricas.similitud_coseno(emb[idx][i], emb[idx][j]), y)
    curva_val = metricas.curva_far_frr(*scores_por_parte["val"])
    eer_val, umbral_eer = metricas.eer(curva_val)
    umbral = metricas.umbral_para_far(curva_val, cfg["far_maximo"])
    resultados["umbral_calibrado_en_validacion"] = umbral
    resultados["eer_validacion"] = eer_val
    resultados["umbral_en_eer_validacion"] = umbral_eer
    resultados["prueba"] = metricas.evaluar_en_umbral(*scores_por_parte["test"], umbral)
    curva_test = metricas.curva_far_frr(*scores_por_parte["test"])
    resultados["eer_prueba"] = metricas.eer(curva_test)[0]
    return resultados


def identificacion(emb: np.ndarray, etiquetas: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Cerrada: reconocer a quién pertenece un rostro entre las personas conocidas (SVM lineal y k-NN)."""
    rng = np.random.default_rng(cfg["semilla"])
    train, test = [], []
    for persona in np.unique(etiquetas):  # por foto, estratificado: cada persona aporta a train y a test
        idx = rng.permutation(np.flatnonzero(etiquetas == persona))
        corte = max(1, int(len(idx) * 0.7))
        train.append(idx[:corte])
        test.append(idx[corte:])
    train, test = np.concatenate(train), np.concatenate(test)
    modelos = {
        "svm_lineal": SVC(kernel="linear"),
        "knn": KNeighborsClassifier(n_neighbors=cfg["k_vecinos"], metric="cosine"),
    }
    salida: dict[str, Any] = {}
    for nombre, modelo in modelos.items():
        modelo.fit(emb[train], etiquetas[train])
        pred = modelo.predict(emb[test])
        salida[nombre] = {
            "exactitud": float(accuracy_score(etiquetas[test], pred)),
            "matriz_confusion": confusion_matrix(etiquetas[test], pred).tolist(),
        }
    return salida


def ejecutar(
    imagenes: Any,
    etiquetas: np.ndarray,
    embeber: Callable[[Any], np.ndarray],
    cfg: dict[str, Any],
    destino: Path,
) -> dict[str, Any]:
    """Corre la línea base completa y guarda `metrics.json` en la carpeta del experimento."""
    emb = embeber(imagenes)
    metricas_finales = {
        "config": cfg,
        "personas": int(len(np.unique(etiquetas))),
        "imagenes": int(len(etiquetas)),
        "verificacion": verificacion(emb, etiquetas, cfg),
        "identificacion": identificacion(emb, etiquetas, cfg),
    }
    (destino / "metrics.json").write_text(
        json.dumps(metricas_finales, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metricas_finales
