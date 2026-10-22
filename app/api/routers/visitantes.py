from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from app.api.seguridad import requiere_rol
from app.application.visitantes import ListarVisitantesRecurrentes
from app.domain.roles import Rol

router = APIRouter(prefix="/rostros", tags=["Rostros"])


class VisitanteRecurrenteSalida(BaseModel):
    clusterId: str  # noqa: N815 - nombre del contrato
    apariciones: int = Field(ge=1)
    diasDistintos: int = Field(ge=1)  # noqa: N815
    primeraVez: datetime  # noqa: N815
    ultimaVez: datetime  # noqa: N815
    alertadoEn: datetime | None = None  # noqa: N815


def _caso(request: Request) -> ListarVisitantesRecurrentes:
    caso: ListarVisitantesRecurrentes = request.app.state.listar_visitantes
    return caso


@router.get(
    "/visitantes-recurrentes",
    response_model=list[VisitanteRecurrenteSalida],
    operation_id="listarVisitantesRecurrentes",
    dependencies=[Depends(requiere_rol(Rol.GUARDIA, Rol.ADMIN))],
)
def listar_visitantes_recurrentes(
    caso: Annotated[ListarVisitantesRecurrentes, Depends(_caso)],
    limite: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[VisitanteRecurrenteSalida]:
    """Visitantes recurrentes no registrados con aviso (I4), el más reciente primero."""
    return [
        VisitanteRecurrenteSalida(
            clusterId=v.cluster_id,
            apariciones=v.apariciones,
            diasDistintos=v.dias_distintos,
            primeraVez=v.primera_vez,
            ultimaVez=v.ultima_vez,
            alertadoEn=v.alertado_en,
        )
        for v in caso.ejecutar(limite)
    ]
