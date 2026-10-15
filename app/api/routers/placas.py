from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.routers.no_implementado import no_implementado
from app.api.schemas import LecturaPlaca, ModeloPlacas
from app.api.seguridad import requiere_rol
from app.domain.roles import Rol

router = APIRouter(tags=["Placas"])


@router.post(
    "/placas/leer",
    response_model=LecturaPlaca,
    operation_id="leerPlaca",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def leer_placa(imagen: Annotated[UploadFile, File()]) -> None:
    """Lee la placa. El umbral de 0.85 (A4) lo aplica Acceso, no este servicio."""
    raise no_implementado("La lectura de placas")


@router.get(
    "/modelos/placas/actual",
    response_model=ModeloPlacas,
    operation_id="modeloPlacasActual",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def modelo_placas_actual() -> None:
    """URL firmada del ONNX cuantizado que la PWA de guardia ejecuta en el dispositivo."""
    raise no_implementado("La descarga del modelo de placas")
