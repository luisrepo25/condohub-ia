from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.routers.no_implementado import no_implementado
from app.api.schemas import Reporte
from app.api.seguridad import requiere_rol
from app.domain.roles import Rol

router = APIRouter(prefix="/reportes", tags=["Reportes"])


@router.post(
    "/voz",
    status_code=202,
    response_model=Reporte,
    operation_id="crearReportePorVoz",
    dependencies=[Depends(requiere_rol(Rol.ADMIN))],
)
def crear_reporte_por_voz(
    audio: Annotated[UploadFile, File()], formato: Annotated[Literal["EXCEL", "PDF"] | None, Form()] = None
) -> None:
    """Audio → Whisper → consulta GraphQL de solo lectura → Excel/PDF (G4). Solo ADMIN."""
    raise no_implementado("Los reportes por voz")


@router.get(
    "/{id}",
    response_model=Reporte,
    operation_id="obtenerReporte",
    dependencies=[Depends(requiere_rol(Rol.ADMIN))],
)
def obtener_reporte(id: UUID) -> None:  # noqa: A002 - nombre del contrato
    """Estado y archivo de un reporte. Solo ADMIN."""
    raise no_implementado("La consulta de reportes")
