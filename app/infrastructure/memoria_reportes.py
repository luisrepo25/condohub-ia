"""Reportes y archivos en memoria (desarrollo y pruebas): se pierden al reiniciar."""

import copy
from datetime import timedelta
from uuid import UUID

from app.application.modelos import AlmacenNoDisponible
from app.domain.reportes import Reporte


class RepositorioReportesMemoria:
    def __init__(self) -> None:
        self._reportes: dict[UUID, Reporte] = {}

    def guardar(self, reporte: Reporte) -> None:
        self._reportes[reporte.id] = copy.deepcopy(reporte)

    def por_id(self, reporte_id: UUID) -> Reporte | None:
        reporte = self._reportes.get(reporte_id)
        return copy.deepcopy(reporte) if reporte else None


class AlmacenReportesMemoria:
    def __init__(self) -> None:
        self.archivos: dict[str, bytes] = {}

    def guardar(self, clave: str, datos: bytes, tipo: str) -> None:
        self.archivos[clave] = datos

    def firmar_descarga(self, clave: str, vigencia: timedelta) -> str:
        if clave not in self.archivos:
            raise AlmacenNoDisponible(f"No existe el archivo {clave}")
        return f"memoria://reportes/{clave}"