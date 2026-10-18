from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencias import caso_leer_placa, caso_obtener_modelo_placas
from app.api.problemas import ProblemaError
from app.api.schemas import LecturaPlaca, ModeloPlacas, ResultadoPlaca
from app.api.seguridad import requiere_rol
from app.api.traduccion import traduciendo_errores
from app.application.leer_placa import LeerPlaca
from app.application.modelos import AlmacenNoDisponible, ModeloNoDisponible, ObtenerModeloPlacas
from app.domain.roles import Rol

router = APIRouter(tags=["Placas"])


@router.post(
    "/placas/leer",
    response_model=LecturaPlaca,
    operation_id="leerPlaca",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def leer_placa(
    caso: Annotated[LeerPlaca, Depends(caso_leer_placa)],
    imagen: Annotated[UploadFile, File()],
    camara: Annotated[str | None, Form()] = None,
) -> LecturaPlaca:
    """Lee la placa (A4, I5). El umbral de 0.85 y la comparación con las registradas son de Acceso."""
    with traduciendo_errores():
        r = caso.ejecutar(imagen.file.read(), camara)
    return LecturaPlaca(
        resultado=ResultadoPlaca(r.resultado),
        placa=r.placa,
        confianza=r.confianza,
        modelo=r.modelo,
        version=r.version,
        prediccionId=r.prediccion_id,
    )


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
