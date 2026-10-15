"""Agrupamiento no supervisado de desconocidos (I3, I4): DBSCAN con métrica coseno."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def distancias_k(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Distancia coseno al k-ésimo vecino de cada punto, ordenada. El «codo» sugiere `eps` (k-distance)."""
    vecinos = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    distancias, _ = vecinos.kneighbors(embeddings)
    return np.sort(distancias[:, k])


def agrupar(embeddings: np.ndarray, eps: float, min_samples: int = 3) -> np.ndarray:
    """Etiquetas de clúster; -1 = ruido (una aparición suelta no es un visitante recurrente)."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings)


def resumen(embeddings: np.ndarray, etiquetas: np.ndarray) -> dict[str, float | int | None]:
    """Número de clústeres, ruido y silhouette (None si no hay al menos 2 clústeres)."""
    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    asignados = etiquetas != -1
    silhouette = None
    if n_clusters >= 2 and asignados.sum() > n_clusters:
        silhouette = float(silhouette_score(embeddings[asignados], etiquetas[asignados], metric="cosine"))
    return {"clusters": n_clusters, "ruido": int((~asignados).sum()), "silhouette": silhouette}


def clusters_recurrentes(
    etiquetas: np.ndarray, dias: list[str], min_apariciones: int = 3, min_dias: int = 3
) -> list[int]:
    """I4: clústeres con al menos `min_apariciones` apariciones en `min_dias` días distintos.

    Los umbrales son parámetros (tabla de parámetros / configuración), no constantes del negocio.
    """
    if len(etiquetas) != len(dias):
        raise ValueError("Una etiqueta por aparición y un día por aparición")
    recurrentes = []
    for cluster in sorted(set(etiquetas) - {-1}):
        propios = [d for e, d in zip(etiquetas, dias, strict=True) if e == cluster]
        if len(propios) >= min_apariciones and len(set(propios)) >= min_dias:
            recurrentes.append(int(cluster))
    return recurrentes
