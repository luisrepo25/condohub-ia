from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.api.dependencias_reportes import casos_reportes, token_bruto
from app.api.problemas import ProblemaError
from app.api.schemas import EstadoReporte as EstadoApi
from app.api.schemas import Reporte as ReporteApi
from app.api.seguridad import requiere_rol
from app.application.modelos import AlmacenNoDisponible
from app.application.reportes import (
    AudioInvalido,
    CasosReportes,
    ConstructorNoDisponible,
    ConsultaInvalida,
    GestionNoDisponible,
    ReporteConArchivo,
    ReporteNoEncontrado,
    TranscripcionNoDisponible,
)
from app.domain.errores import ReglaViolada
from app.domain.reportes import FormatoReporte
from app.domain.roles import Rol, Solicitante

router = APIRouter(prefix="/reportes", tags=["Reportes"])


@contextmanager
def _traduciendo_errores() -> Iterator[None]:
    """`SinPermiso` y `TokenInvalido` no se traducen aquí: ya los atiende el manejador global."""
    try:
        yield
    except ReglaViolada as e:
        raise ProblemaError(422, "Regla de negocio violada", e.mensaje, code=e.codigo) from e
    except ConsultaInvalida as e:
        raise ProblemaError(422, "Consulta no válida", str(e), code="G4_CONSULTA") from e
    except AudioInvalido as e:
        raise ProblemaError(400, "Solicitud inválida", str(e)) from e
    except ReporteNoEncontrado as e:
        raise ProblemaError(404, "No encontrado", str(e)) from e
    except (
        TranscripcionNoDisponible,
        ConstructorNoDisponible,
        GestionNoDisponible,
        AlmacenNoDisponible,
    ) as e:
        raise ProblemaError(503, "Servicio no disponible", str(e)) from e


def _a_api(vista: ReporteConArchivo) -> ReporteApi:
    r = vista.reporte
    return ReporteApi(
        id=r.id,
        estado=EstadoApi(r.estado.value),
        formato=r.formato.value,
        transcripcion=r.transcripcion,
        consultaGraphql=r.consulta_graphql,
        archivoUrl=vista.archivo_url,
        motivoFallo=r.motivo_fallo,
    )


@router.post("/voz", status_code=202, response_model=ReporteApi, operation_id="crearReportePorVoz")
def crear_reporte_por_voz(
    segundo_plano: BackgroundTasks,
    audio: Annotated[UploadFile, File()],
    admin: Annotated[Solicitante, Depends(requiere_rol(Rol.ADMIN))],
    token: Annotated[str, Depends(token_bruto)],
    casos: Annotated[CasosReportes, Depends(casos_reportes)],
    formato: Annotated[Literal["EXCEL", "PDF"] | None, Form()] = None,
) -> ReporteApi:
    """Audio → Whisper → consulta GraphQL de solo lectura → Excel/PDF (G4). Solo ADMIN.

    La transcripción, la consulta y la comprobación de solo lectura son síncronas (una mutation da 422 `G4`);
    el archivo se genera en segundo plano y se consulta con `GET /reportes/{id}`.
    """
    with _traduciendo_errores():
        elegido = FormatoReporte(formato) if formato else None
        reporte = casos.iniciar.iniciar(admin, audio.file.read(), token, elegido)
    segundo_plano.add_task(casos.generar.generar, reporte.id, token)
    return _a_api(ReporteConArchivo(reporte, None))


@router.get(
    "/{id}",
    response_model=ReporteApi,
    operation_id="obtenerReporte",
    dependencies=[Depends(requiere_rol(Rol.ADMIN))],
)
def obtener_reporte(id: UUID, casos: Annotated[CasosReportes, Depends(casos_reportes)]) -> ReporteApi:  # noqa: A002 - nombre del contrato
    """Estado y archivo de un reporte. Solo ADMIN. `archivoUrl` es una URL firmada y temporal."""
    with _traduciendo_errores():
        return _a_api(casos.obtener.obtener(id))
