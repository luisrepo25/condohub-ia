"""Reportes por voz (G4). Dominio puro: sin FastAPI, GraphQL, Whisper ni librerías de archivos."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from app.domain.errores import ReglaViolada

Fila = dict[str, str | int | float | bool | None]


class FormatoReporte(StrEnum):
    EXCEL = "EXCEL"
    PDF = "PDF"


class EstadoReporte(StrEnum):
    PROCESANDO = "PROCESANDO"
    LISTO = "LISTO"
    FALLIDO = "FALLIDO"


def formato_desde_instruccion(texto: str, elegido: FormatoReporte | None) -> FormatoReporte:
    """El formato pedido manda; si no, el de la instrucción («en PDF»); por defecto Excel."""
    if elegido is not None:
        return elegido
    return FormatoReporte.PDF if "pdf" in texto.lower() else FormatoReporte.EXCEL


def exigir_solo_lectura(operaciones: Sequence[str]) -> None:
    """G4: el LLM nunca recibe permisos de escritura. Solo se admiten operaciones `query`."""
    if not operaciones:
        raise ReglaViolada("G4", "La consulta no contiene ninguna operación")
    distintas = sorted({op for op in operaciones if op != "query"})
    if distintas:
        raise ReglaViolada("G4", f"Los reportes solo pueden leer datos; se rechazó: {', '.join(distintas)}")


def _aplanar(valor: Any, prefijo: str, destino: Fila) -> None:
    if isinstance(valor, dict):
        for clave, interno in valor.items():
            _aplanar(interno, f"{prefijo}.{clave}" if prefijo else str(clave), destino)
    elif isinstance(valor, list):
        destino[prefijo] = ", ".join(str(v) for v in valor)
    else:
        destino[prefijo] = valor


def filas_desde_respuesta(datos: dict[str, Any]) -> list[Fila]:
    """Convierte `data` de GraphQL en filas: la primera lista son las filas; si no hay, una sola fila."""

    def primera_lista(nodo: Any) -> list[Any] | None:
        if isinstance(nodo, list):
            return nodo
        if isinstance(nodo, dict):
            for hijo in nodo.values():
                encontrada = primera_lista(hijo)
                if encontrada is not None:
                    return encontrada
        return None

    registros = primera_lista(datos)
    if registros is None:
        registros = [datos]
    filas: list[Fila] = []
    for registro in registros:
        fila: Fila = {}
        _aplanar(registro, "", fila)
        filas.append(fila)
    return filas


LIMITE_VISTA_PREVIA = 50


@dataclass(frozen=True)
class VistaPrevia:
    """Primeras filas del resultado para verlas en pantalla (G4): alineadas con las columnas, en texto."""

    columnas: list[str]
    filas: list[list[str | None]]
    total: int


def vista_previa_desde(filas: list[Fila], limite: int = LIMITE_VISTA_PREVIA) -> VistaPrevia:
    """Las claves aplanadas llevan puntos (`titular.nombre`): se guardan como columnas, no como campos."""
    columnas: list[str] = []
    for fila in filas:
        columnas.extend(c for c in fila if c not in columnas)
    muestra = [[None if f.get(c) is None else str(f[c]) for c in columnas] for f in filas[:limite]]
    return VistaPrevia(columnas=columnas, filas=muestra, total=len(filas))


@dataclass
class Reporte:
    """Una instrucción de voz y todo lo que produjo: texto, consulta y archivo (G4)."""

    id: UUID
    admin_sub: str
    formato: FormatoReporte
    creado_en: datetime
    estado: EstadoReporte = EstadoReporte.PROCESANDO
    transcripcion: str | None = None
    consulta_graphql: str | None = None
    archivo_key: str | None = None
    motivo_fallo: str | None = None
    terminado_en: datetime | None = None
    vista_previa: VistaPrevia | None = None

    def registrar_instruccion(self, transcripcion: str, formato_elegido: FormatoReporte | None) -> None:
        self.transcripcion = transcripcion
        self.formato = formato_desde_instruccion(transcripcion, formato_elegido)

    def registrar_consulta(self, consulta: str) -> None:
        self.consulta_graphql = consulta

    def registrar_vista_previa(self, vista: VistaPrevia) -> None:
        self.vista_previa = vista

    def completar(self, archivo_key: str, ahora: datetime) -> None:
        self.archivo_key = archivo_key
        self.estado = EstadoReporte.LISTO
        self.terminado_en = ahora

    def fallar(self, motivo: str, ahora: datetime) -> None:
        self.motivo_fallo = motivo
        self.estado = EstadoReporte.FALLIDO
        self.terminado_en = ahora
