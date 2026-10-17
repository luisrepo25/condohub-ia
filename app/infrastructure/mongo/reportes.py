"""Colección `reportes` (G4). Reciben la colección ya abierta; no importan `pymongo`."""

import re
from typing import Any
from uuid import UUID

from app.domain.reportes import EstadoReporte, FormatoReporte, Reporte

# El validador de `db/mongo-init.js` solo admite consultas que empiecen por `query` o `{`:
# una rechazada por G4 (por ejemplo una mutation) no se guarda; el motivo del fallo dice qué se rechazó.
_SOLO_LECTURA = re.compile(r"^\s*(query\b|\{)")


def reporte_a_documento(r: Reporte) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "reporteId": str(r.id),
        "adminSub": r.admin_sub,
        "formato": r.formato.value,
        "estado": r.estado.value,
        "creadoEn": r.creado_en,
    }
    if r.transcripcion is not None:
        doc["transcripcion"] = r.transcripcion
    if r.consulta_graphql is not None and _SOLO_LECTURA.match(r.consulta_graphql):
        doc["consultaGraphql"] = r.consulta_graphql
    if r.archivo_key is not None:
        doc["archivoKey"] = r.archivo_key
    if r.motivo_fallo is not None:
        doc["motivoFallo"] = r.motivo_fallo
    if r.terminado_en is not None:
        doc["terminadoEn"] = r.terminado_en
    return doc


def documento_a_reporte(d: dict[str, Any]) -> Reporte:
    return Reporte(
        id=UUID(d["reporteId"]),
        admin_sub=d["adminSub"],
        formato=FormatoReporte(d["formato"]),
        estado=EstadoReporte(d["estado"]),
        creado_en=d["creadoEn"],
        transcripcion=d.get("transcripcion"),
        consulta_graphql=d.get("consultaGraphql"),
        archivo_key=d.get("archivoKey"),
        motivo_fallo=d.get("motivoFallo"),
        terminado_en=d.get("terminadoEn"),
    )


class RepositorioReportesMongo:
    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def guardar(self, reporte: Reporte) -> None:
        self._c.replace_one({"reporteId": str(reporte.id)}, reporte_a_documento(reporte), upsert=True)

    def por_id(self, reporte_id: UUID) -> Reporte | None:
        doc = self._c.find_one({"reporteId": str(reporte_id)})
        return documento_a_reporte(doc) if doc else None