"""Agrupador DBSCAN con distancia coseno (scikit-learn); el servicio no lo necesita para arrancar."""

import importlib
from collections.abc import Sequence

from app.domain.rostros import Embedding


class AgrupadorDbscan:
    """Puerto `Agrupador`. Con `eps` y `min_samples` elegidos en `ml/` (notebook 05)."""

    def agrupar(self, vectores: Sequence[Embedding], eps: float, min_samples: int) -> list[int]:
        np = importlib.import_module("numpy")
        cluster = importlib.import_module("sklearn.cluster")
        matriz = np.asarray(vectores, dtype=np.float32)
        etiquetas = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(matriz)
        return [int(e) for e in etiquetas]
