"""I4: visitantes recurrentes entre los desconocidos de portería.

DBSCAN (métrica coseno) agrupa a los desconocidos de los últimos 30 días; un grupo con ≥ 3 apariciones
en ≥ 3 días distintos
es un «visitante recurrente no registrado». Dominio puro: el algoritmo de agrupamiento entra por un puerto.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.domain.rostros import Embedding


@dataclass(frozen=True)
class ParametrosAgrupamiento:
    """Parámetros de I4 (configuración, no constantes). `eps` es distancia coseno: 0 < eps <= 2."""

    eps: float = 0.30
    min_samples: int = 3
    min_apariciones: int = 3
    min_dias: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.eps <= 2:
            raise ValueError("eps es una distancia coseno: debe cumplir 0 < eps <= 2")
        if min(self.min_samples, self.min_apariciones, self.min_dias) < 1:
            raise ValueError("min_samples, min_apariciones y min_dias deben ser al menos 1")


@dataclass(frozen=True)
class Aparicion:
    """Un desconocido guardado, listo para agrupar. `cluster_id`: su visitante en una noche anterior."""

    id: str
    vector: Embedding
    dia: str  # yyyy-mm-dd
    capturado_en: datetime
    camara: str | None
    cluster_id: str | None = None


@dataclass(frozen=True)
class GrupoRecurrente:
    miembros: tuple[Aparicion, ...]

    @property
    def apariciones(self) -> int:
        return len(self.miembros)

    @property
    def dias_distintos(self) -> int:
        return len({a.dia for a in self.miembros})

    @property
    def primera_vez(self) -> datetime:
        return min(a.capturado_en for a in self.miembros)

    @property
    def ultima_vez(self) -> datetime:
        return max(a.capturado_en for a in self.miembros)

    @property
    def camara(self) -> str | None:
        """La cámara donde más apareció, o `None` si ninguna aparición la indica."""
        camaras = Counter(a.camara for a in self.miembros if a.camara)
        return camaras.most_common(1)[0][0] if camaras else None

    def cluster_previo(self) -> str | None:
        """Identidad del visitante en noches anteriores: el id que más miembros de este grupo ya tenían."""
        previos = Counter(a.cluster_id for a in self.miembros if a.cluster_id)
        return previos.most_common(1)[0][0] if previos else None


def grupos_recurrentes(
    etiquetas: Sequence[int], apariciones: Sequence[Aparicion], parametros: ParametrosAgrupamiento
) -> list[GrupoRecurrente]:
    """I4: los grupos (etiqueta ≥ 0; -1 es ruido) con suficientes apariciones y días distintos."""
    if len(etiquetas) != len(apariciones):
        raise ValueError("Una etiqueta por aparición")
    por_grupo: dict[int, list[Aparicion]] = {}
    for etiqueta, aparicion in zip(etiquetas, apariciones, strict=True):
        if etiqueta >= 0:
            por_grupo.setdefault(etiqueta, []).append(aparicion)
    grupos = [GrupoRecurrente(tuple(m)) for m in por_grupo.values()]
    return [
        g
        for g in grupos
        if g.apariciones >= parametros.min_apariciones and g.dias_distintos >= parametros.min_dias
    ]
