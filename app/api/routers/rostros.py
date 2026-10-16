from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencias import casos_rostros
from app.api.schemas import IdentificacionRostro, PerfilRegistrado, ResultadoRostro, RetiroAceptado
from app.api.seguridad import requiere_rol
from app.api.traduccion import traduciendo_errores
from app.application.rostros import CasosRostros
from app.domain.roles import Rol, Solicitante

router = APIRouter(prefix="/rostros", tags=["Rostros"])


@router.post(
    "/identificar",
    response_model=IdentificacionRostro,
    operation_id="identificarRostro",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def identificar_rostro(
    casos: Annotated[CasosRostros, Depends(casos_rostros)],
    imagen: Annotated[UploadFile, File()],
    camara: Annotated[str | None, Form()] = None,
) -> IdentificacionRostro:
    """Identifica el rostro (I2, I3, I5). Asiste al guardia; nunca abre por sí solo (A5)."""
    with traduciendo_errores():
        r = casos.identificar.ejecutar(imagen.file.read(), camara)
    return IdentificacionRostro(
        resultado=ResultadoRostro(r.resultado),
        residenteId=r.residente_id,
        confianza=r.confianza,
        modelo=r.modelo,
        version=r.version,
        prediccionId=r.prediccion_id,
    )


@router.post(
    "/registrar",
    status_code=201,
    response_model=PerfilRegistrado,
    operation_id="registrarRostro",
    dependencies=[Depends(requiere_rol(Rol.ADMIN))],
)
def registrar_rostro(
    casos: Annotated[CasosRostros, Depends(casos_rostros)],
    residenteId: Annotated[UUID, Form()],  # noqa: N803 - nombre del contrato
    residenteSub: Annotated[str, Form(min_length=1)],  # noqa: N803
    unidadId: Annotated[UUID, Form()],  # noqa: N803
    fotos: Annotated[list[UploadFile], File()],
    consentimientoFecha: Annotated[datetime | None, Form()] = None,  # noqa: N803
    consentimientoVersion: Annotated[str | None, Form()] = None,  # noqa: N803
) -> PerfilRegistrado:
    """Solo con consentimiento explícito (I1): fecha con zona horaria y versión del texto aceptado.

    Si falta cualquiera de los dos, la respuesta es 422 con `code: I1` (como dice el contrato),
    no un 400 de formato.
    """
    with traduciendo_errores():
        r = casos.registrar.ejecutar(
            residenteId,
            residenteSub,
            unidadId,
            consentimientoFecha,
            consentimientoVersion,
            [f.file.read() for f in fotos],
        )
    return PerfilRegistrado(
        residenteId=r.residente_id, embeddings=r.embeddings, modelo=r.modelo, version=r.version
    )


@router.delete(
    "/{residenteId}",
    status_code=202,
    response_model=RetiroAceptado,
    operation_id="retirarConsentimiento",
)
def retirar_consentimiento(
    casos: Annotated[CasosRostros, Depends(casos_rostros)],
    solicitante: Annotated[Solicitante, Depends(requiere_rol(Rol.ADMIN, Rol.RESIDENTE))],
    residenteId: UUID,  # noqa: N803
) -> RetiroAceptado:
    """Retiro de consentimiento: borra fotos y embeddings (I1). Un RESIDENTE solo retira el suyo."""
    with traduciendo_errores():
        r = casos.retirar.ejecutar(solicitante, residenteId)
    return RetiroAceptado(residenteId=r.residente_id, borradoAntesDe=r.borrado_antes_de)
