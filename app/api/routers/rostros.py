from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.routers.no_implementado import no_implementado
from app.api.schemas import IdentificacionRostro, PerfilRegistrado, RetiroAceptado
from app.api.seguridad import requiere_rol
from app.domain.roles import Rol

router = APIRouter(prefix="/rostros", tags=["Rostros"])


@router.post(
    "/identificar",
    response_model=IdentificacionRostro,
    operation_id="identificarRostro",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def identificar_rostro(
    imagen: Annotated[UploadFile, File()], camara: Annotated[str | None, Form()] = None
) -> None:
    """Identifica el rostro (I2, I3, I5). Asiste al guardia; nunca abre por sí solo (A5)."""
    raise no_implementado("La identificación de rostros")


@router.post(
    "/registrar",
    status_code=201,
    response_model=PerfilRegistrado,
    operation_id="registrarRostro",
    dependencies=[Depends(requiere_rol(Rol.ADMIN))],
)
def registrar_rostro(
    residenteId: Annotated[UUID, Form()],  # noqa: N803 - nombre del contrato
    unidadId: Annotated[UUID, Form()],  # noqa: N803
    consentimientoFecha: Annotated[datetime, Form()],  # noqa: N803
    consentimientoVersion: Annotated[str, Form()],  # noqa: N803
    fotos: Annotated[list[UploadFile], File()],
) -> None:
    """Solo con consentimiento explícito (I1)."""
    raise no_implementado("El registro de perfiles faciales")


# TODO(regla): el RESIDENTE solo debe poder retirar SU consentimiento. El token trae el `sub` de
# Keycloak y la ruta trae `residenteId` (id de la persona en Gestión): falta definir cómo se
# enlazan antes de aplicar esa restricción.
@router.delete(
    "/{residenteId}",
    status_code=202,
    response_model=RetiroAceptado,
    operation_id="retirarConsentimiento",
    dependencies=[Depends(requiere_rol(Rol.ADMIN, Rol.RESIDENTE))],
)
def retirar_consentimiento(residenteId: UUID) -> None:  # noqa: N803
    """Retiro de consentimiento: borra fotos y embeddings en menos de 24 h (I1)."""
    raise no_implementado("El retiro de consentimiento")
