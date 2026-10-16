from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencias import caso_obtener_modelo_placas
from app.api.problemas import ProblemaError
from app.api.routers.no_implementado import no_implementado
from app.api.schemas import LecturaPlaca, ModeloPlacas
from app.api.seguridad import requiere_rol
from app.application.modelos import AlmacenNoDisponible, ModeloNoDisponible, ObtenerModeloPlacas
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
def modelo_placas_actual(
    caso: Annotated[ObtenerModeloPlacas, Depends(caso_obtener_modelo_placas)],
) -> ModeloPlacas:
    """URL firmada del ONNX cuantizado que la PWA de guardia ejecuta en el dispositivo (A4, O6)."""
    try:
        m = caso.ejecutar()
    except ModeloNoDisponible as e:
        raise ProblemaError(404, "No encontrado", str(e)) from e
    except AlmacenNoDisponible as e:
        raise ProblemaError(503, "Almacén de modelos no disponible", str(e)) from e
    return ModeloPlacas(
        modelo=m.modelo,
        version=m.version,
        formato=m.formato,
        url=m.url,
        sha256=m.sha256,
        expiraEn=m.expira_en,
    )
