"""Visitantes recurrentes que ya generaron aviso (I4). Solo lectura: agrupa el job nocturno."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class VisitanteRecurrente:
    cluster_id: str
    apariciones: int
    dias_distintos: int
    primera_vez: datetime
    ultima_vez: datetime
    alertado_en: datetime | None


class LectorVisitantes(Protocol):
    def con_aviso(self, limite: int) -> list[VisitanteRecurrente]:
        """Los visitantes avisados, el más reciente primero."""
        ...


class ListarVisitantesRecurrentes:
    def __init__(self, lector: LectorVisitantes) -> None:
        self._lector = lector

    def ejecutar(self, limite: int = 20) -> list[VisitanteRecurrente]:
        return self._lector.con_aviso(limite)
