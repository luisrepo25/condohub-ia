"""Casos de uso de los reportes por voz (G4): audio → Whisper → LLM → GraphQL de solo lectura → Excel o PDF.

La aplicación orquesta el dominio y los puertos; no conoce FastAPI, Whisper, Gemini, openpyxl ni MongoDB.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

from app.application.modelos import Reloj
from app.domain.errores import ReglaViolada
from app.domain.reportes import (
    Fila,
    FormatoReporte,
    Reporte,
    exigir_solo_lectura,
    filas_desde_respuesta,
)
from app.domain.roles import Solicitante


class AudioInvalido(Exception):
    """El audio está vacío o no se puede decodificar."""


class TranscripcionNoDisponible(Exception):
    """El modelo de voz no está instalado o configurado en este servicio."""


class ConstructorNoDisponible(Exception):
    """El LLM no está configurado o no respondió."""


class GestionNoDisponible(Exception):
    """Gestión no respondió o rechazó la consulta."""


class ConsultaInvalida(Exception):
    """El LLM no devolvió una consulta GraphQL válida para el esquema de Gestión."""


class ReporteNoEncontrado(Exception):
    """No existe un reporte con ese id."""


# ------------------------------------------------------------------ puertos


class Transcriptor(Protocol):
    def transcribir(self, audio: bytes) -> str:
        """Texto de la instrucción hablada. Lanza `AudioInvalido` o `TranscripcionNoDisponible`."""
        ...


class ConstructorConsulta(Protocol):
    def construir(self, instruccion: str, esquema: str) -> str:
        """Consulta GraphQL para la instrucción, dado el esquema (SDL). Lanza `ConstructorNoDisponible`."""
        ...


class AnalizadorGraphql(Protocol):
    def operaciones(self, consulta: str) -> list[str]:
        """Tipo de cada operación (`query`, `mutation`, `subscription`). Lanza `ConsultaInvalida`."""
        ...


class Gestion(Protocol):
    def esquema(self, token: str) -> str:
        """Esquema GraphQL de Gestión en SDL. Lanza `GestionNoDisponible`."""
        ...

    def ejecutar(self, consulta: str, token: str) -> dict[str, Any]:
        """`data` de la respuesta. Lanza `GestionNoDisponible` si hay errores o no responde."""
        ...


class GeneradorArchivo(Protocol):
    extension: str
    tipo: str

    def generar(self, titulo: str, filas: list[Fila]) -> bytes: ...


class AlmacenReportes(Protocol):
    def guardar(self, clave: str, datos: bytes, tipo: str) -> None: ...

    def firmar_descarga(self, clave: str, vigencia: timedelta) -> str:
        """URL temporal de lectura del archivo."""
        ...


class RepositorioReportes(Protocol):
    def guardar(self, reporte: Reporte) -> None: ...

    def por_id(self, reporte_id: UUID) -> Reporte | None: ...


# ------------------------------------------------------------------ casos de uso


class IniciarReporte:
    """Pasos síncronos: transcribir, armar la consulta y comprobar que es de solo lectura (G4).

    Todo reporte iniciado queda guardado; si viola G4 queda `FALLIDO` con el motivo, como rastro de auditoría.
    """

    def __init__(
        self,
        repositorio: RepositorioReportes,
        transcriptor: Transcriptor,
        constructor: ConstructorConsulta,
        analizador: AnalizadorGraphql,
        gestion: Gestion,
        reloj: Reloj,
        nuevo_id: Callable[[], UUID] = uuid4,
    ) -> None:
        self._repositorio = repositorio
        self._transcriptor = transcriptor
        self._constructor = constructor
        self._analizador = analizador
        self._gestion = gestion
        self._reloj = reloj
        self._nuevo_id = nuevo_id

    def iniciar(
        self, admin: Solicitante, audio: bytes, token: str, formato_elegido: FormatoReporte | None
    ) -> Reporte:
        if not audio:
            raise AudioInvalido("El audio está vacío")
        instruccion = self._transcriptor.transcribir(audio)
        reporte = Reporte(
            id=self._nuevo_id(),
            admin_sub=admin.sub,
            formato=formato_elegido or FormatoReporte.EXCEL,
            creado_en=self._reloj.ahora(),
        )
        reporte.registrar_instruccion(instruccion, formato_elegido)
        try:
            consulta = self._constructor.construir(instruccion, self._gestion.esquema(token))
            reporte.registrar_consulta(consulta)
            exigir_solo_lectura(self._analizador.operaciones(consulta))
        except (ReglaViolada, ConsultaInvalida) as e:
            reporte.fallar(e.mensaje if isinstance(e, ReglaViolada) else str(e), self._reloj.ahora())
            self._repositorio.guardar(reporte)
            raise
        self._repositorio.guardar(reporte)
        return reporte


class GenerarArchivoReporte:
    """Pasos que van en segundo plano: ejecutar la consulta en Gestión y producir el archivo. Nunca lanza."""

    def __init__(
        self,
        repositorio: RepositorioReportes,
        gestion: Gestion,
        generadores: Mapping[FormatoReporte, GeneradorArchivo],
        almacen: AlmacenReportes,
        reloj: Reloj,
    ) -> None:
        self._repositorio = repositorio
        self._gestion = gestion
        self._generadores = generadores
        self._almacen = almacen
        self._reloj = reloj

    def generar(self, reporte_id: UUID, token: str) -> None:
        reporte = self._repositorio.por_id(reporte_id)
        if reporte is None or reporte.consulta_graphql is None:
            return
        try:
            datos = self._gestion.ejecutar(reporte.consulta_graphql, token)
            generador = self._generadores[reporte.formato]
            titulo = reporte.transcripcion or "Reporte"
            archivo = generador.generar(titulo, filas_desde_respuesta(datos))
            clave = f"{reporte.id}.{generador.extension}"
            self._almacen.guardar(clave, archivo, generador.tipo)
            reporte.completar(clave, self._reloj.ahora())
        except Exception as e:  # noqa: BLE001 - el reporte debe quedar FALLIDO con su motivo, cualquiera sea la causa
            reporte.fallar(f"{type(e).__name__}: {e}"[:300], self._reloj.ahora())
        self._repositorio.guardar(reporte)


@dataclass(frozen=True)
class ReporteConArchivo:
    reporte: Reporte
    archivo_url: str | None


class ObtenerReporte:
    def __init__(
        self, repositorio: RepositorioReportes, almacen: AlmacenReportes, vigencia: timedelta
    ) -> None:
        self._repositorio = repositorio
        self._almacen = almacen
        self._vigencia = vigencia

    def obtener(self, reporte_id: UUID) -> ReporteConArchivo:
        reporte = self._repositorio.por_id(reporte_id)
        if reporte is None:
            raise ReporteNoEncontrado(f"No existe el reporte {reporte_id}")
        url = None
        if reporte.archivo_key is not None:
            url = self._almacen.firmar_descarga(reporte.archivo_key, self._vigencia)
        return ReporteConArchivo(reporte, url)


@dataclass(frozen=True)
class CasosReportes:
    iniciar: IniciarReporte
    generar: GenerarArchivoReporte
    obtener: ObtenerReporte
