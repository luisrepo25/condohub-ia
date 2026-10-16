from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.api.problemas import ProblemaError
from app.api.seguridad import requiere_rol
from app.application.modelo_lectura import ObtenerModeloLectura
from app.application.modelos import AlmacenNoDisponible, ModeloNoDisponible
from app.domain.roles import Rol

router = APIRouter(tags=["Placas"])


class ModeloLecturaPlacas(BaseModel):
    modelo: str
    version: str
    formato: str
    url: str
    sha256: str
    fichaUrl: str  # noqa: N815
    expiraEn: datetime  # noqa: N815


def caso_obtener_modelo_lectura(request: Request) -> ObtenerModeloLectura:
    caso: ObtenerModeloLectura = request.app.state.obtener_modelo_lectura
    return caso


@router.get(
    "/modelos/lectura-placas/actual",
    response_model=ModeloLecturaPlacas,
    operation_id="modeloLecturaPlacasActual",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def modelo_lectura_placas_actual(
    caso: Annotated[ObtenerModeloLectura, Depends(caso_obtener_modelo_lectura)],
) -> ModeloLecturaPlacas:
    """URLs firmadas del reconocedor ONNX y de su ficha, para leer la placa en el dispositivo (A4, O6)."""
    try:
        m = caso.ejecutar()
    except ModeloNoDisponible as e:
        raise ProblemaError(404, "No encontrado", str(e)) from e
    except AlmacenNoDisponible as e:
        raise ProblemaError(503, "Almacén de modelos no disponible", str(e)) from e
    return ModeloLecturaPlacas(
        modelo=m.modelo,
        version=m.version,
        formato=m.formato,
        url=m.url,
        sha256=m.sha256,
        fichaUrl=m.ficha_url,
        expiraEn=m.expira_en,
    )
