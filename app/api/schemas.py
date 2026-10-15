"""Esquemas de respuesta, espejo de `contracts/openapi/ia.yaml`."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class Salud(BaseModel):
    servicio: str
    estado: str
    modelosCargados: bool  # noqa: N815 - nombre del contrato


class ResultadoRostro(StrEnum):
    RECONOCIDO = "RECONOCIDO"
    DESCONOCIDO = "DESCONOCIDO"
    SIN_ROSTRO = "SIN_ROSTRO"


class IdentificacionRostro(BaseModel):
    resultado: ResultadoRostro
    residenteId: UUID | None = None  # noqa: N815
    confianza: float = Field(ge=0, le=1)
    modelo: str
    version: str
    prediccionId: str  # noqa: N815


class PerfilRegistrado(BaseModel):
    residenteId: UUID  # noqa: N815
    embeddings: int = Field(ge=1)
    modelo: str
    version: str


class RetiroAceptado(BaseModel):
    residenteId: UUID  # noqa: N815
    borradoAntesDe: datetime  # noqa: N815


class ResultadoPlaca(StrEnum):
    PLACA_LEIDA = "PLACA_LEIDA"
    SIN_PLACA = "SIN_PLACA"


class LecturaPlaca(BaseModel):
    resultado: ResultadoPlaca
    placa: str | None = None
    confianza: float = Field(ge=0, le=1)
    modelo: str
    version: str
    prediccionId: str  # noqa: N815


class ModeloPlacas(BaseModel):
    modelo: str
    version: str
    formato: str
    url: str
    sha256: str
    expiraEn: datetime  # noqa: N815


class EstadoReporte(StrEnum):
    PROCESANDO = "PROCESANDO"
    LISTO = "LISTO"
    FALLIDO = "FALLIDO"


class Reporte(BaseModel):
    id: UUID
    estado: EstadoReporte
    formato: str | None = None
    transcripcion: str | None = None
    consultaGraphql: str | None = None  # noqa: N815
    archivoUrl: str | None = None  # noqa: N815
    motivoFallo: str | None = None  # noqa: N815
