"""Evalúa DBSCAN contra un escenario con respuesta conocida: ¿las alertas (I4) aciertan y no sobran?

Complementa `cluster.py` (que agrupa y aplica la regla I4) con medidas que necesitan saber quién es quién.
"""

import numpy as np

from ml.src import cluster


def personas_recurrentes(
    personas: np.ndarray, dias: list[str], min_apariciones: int = 3, min_dias: int = 3
) -> set[int]:
    """Verdad de terreno de I4: personas con ≥ `min_apariciones` apariciones en ≥ `min_dias` días distintos."""
    if len(personas) != len(dias):
        raise ValueError("Una persona y un día por aparición")
    reales = set()
    for p in set(personas.tolist()):
        propios = [d for q, d in zip(personas, dias, strict=True) if q == p]
        if len(propios) >= min_apariciones and len(set(propios)) >= min_dias:
            reales.add(int(p))
    return reales


def evaluar_alertas(
    etiquetas: np.ndarray,
    personas: np.ndarray,
    dias: list[str],
    min_apariciones: int = 3,
    min_dias: int = 3,
    pureza_minima: float = 0.8,
) -> dict[str, float | int | None]:
    """Compara los clústeres que alertan con las personas que de verdad son recurrentes.

    Una alerta es acierto si su clúster pertenece sobre todo (≥ `pureza_minima`) a una persona realmente recurrente.
    Un clúster que mezcla personas, o que es de alguien que no cumple I4, es una falsa alarma.
    """
    alertas = cluster.clusters_recurrentes(etiquetas, dias, min_apariciones, min_dias)
    reales = personas_recurrentes(personas, dias, min_apariciones, min_dias)
    aciertos, falsas, cubiertas, purezas = 0, 0, set(), []
    for c in alertas:
        miembros = personas[etiquetas == c]
        valores, cuentas = np.unique(miembros, return_counts=True)
        mayoritaria, pureza = int(valores[cuentas.argmax()]), float(cuentas.max() / len(miembros))
        purezas.append(pureza)
        if mayoritaria in reales and pureza >= pureza_minima:
            aciertos += 1
            cubiertas.add(mayoritaria)
        else:
            falsas += 1
    return {
        "alertas": len(alertas),
        "aciertos": aciertos,
        "falsas_alarmas": falsas,
        "recurrentes_reales": len(reales),
        "recurrentes_detectados": len(cubiertas),
        "precision": aciertos / len(alertas) if alertas else None,
        "recall": len(cubiertas) / len(reales) if reales else None,
        "pureza_media": float(np.mean(purezas)) if purezas else None,
    }


def barrer_eps(
    embeddings: np.ndarray,
    personas: np.ndarray,
    dias: list[str],
    valores_eps: list[float],
    min_samples: int = 3,
    min_apariciones: int = 3,
    min_dias: int = 3,
) -> list[dict[str, float | int | None]]:
    """Una fila por `eps`: clústeres, ruido, silhouette y calidad de las alertas."""
    filas = []
    for eps in valores_eps:
        etiquetas = cluster.agrupar(embeddings, eps, min_samples)
        fila: dict[str, float | int | None] = {"eps": float(eps), **cluster.resumen(embeddings, etiquetas)}
        fila.update(evaluar_alertas(etiquetas, personas, dias, min_apariciones, min_dias))
        filas.append(fila)
    return filas


def f1(fila: dict[str, float | int | None]) -> float:
    """Media armónica de precisión y recall; 0 si alguna no está definida o es 0."""
    p, r = fila.get("precision"), fila.get("recall")
    if not p or not r:
        return 0.0
    return float(2 * p * r / (p + r))


def codo(distancias_ordenadas: np.ndarray) -> float:
    """Valor de `eps` sugerido por el codo del gráfico k-distance (punto más alejado de la recta primero-último)."""
    n = len(distancias_ordenadas)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 distancias")
    x = np.linspace(0, 1, n)
    y = (distancias_ordenadas - distancias_ordenadas[0]) / (distancias_ordenadas[-1] - distancias_ordenadas[0] or 1.0)
    return float(distancias_ordenadas[int(np.argmax(x - y))])
