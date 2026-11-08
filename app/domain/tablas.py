"""Tablas de un reporte (G4): una por lista de la consulta, con los criterios aplicados. Dominio puro."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.domain.criterios import (
    Criterio,
    Especificacion,
    agrupar,
    filtrar,
    ocultar,
    ordenar,
)
from app.domain.errores import ReglaViolada
from app.domain.reportes import Fila, filas_desde_respuesta

NOMBRE_POR_DEFECTO = "Reporte"


@dataclass(frozen=True)
class Tabla:
    nombre: str
    filas: list[Fila]

    @property
    def columnas(self) -> list[str]:
        columnas: list[str] = []
        for fila in self.filas:
            columnas.extend(c for c in fila if c not in columnas)
        return columnas


def _tiene_lista(nodo: Any) -> bool:
    if isinstance(nodo, list):
        return True
    return isinstance(nodo, dict) and any(_tiene_lista(hijo) for hijo in nodo.values())


def tablas_desde_respuesta(datos: dict[str, Any]) -> list[Tabla]:
    """Una tabla por cada campo que devuelve una lista; sin listas, una sola fila (como antes)."""
    tablas = [Tabla(str(c), filas_desde_respuesta({c: v})) for c, v in datos.items() if _tiene_lista(v)]
    return tablas or [Tabla(NOMBRE_POR_DEFECTO, filas_desde_respuesta(datos))]


def _error(mensaje: str) -> ReglaViolada:
    return ReglaViolada("G4", f"Los criterios del reporte no son válidos: {mensaje}")


def _exigir_tabla(tablas: list[Tabla], tabla: str | None) -> None:
    if tabla is not None and not any(t.nombre == tabla for t in tablas):
        existentes = ", ".join(t.nombre for t in tablas)
        raise _error(f"la tabla «{tabla[:40]}» no existe en el resultado ({existentes})")


def _en_alcance(
    tablas: list[Tabla], tabla: str | None, campos: list[str], operacion: Callable[[Tabla], Tabla]
) -> list[Tabla]:
    """Aplica `operacion` a las tablas que corresponden. Con tabla explícita, el campo debe existir en ella;
    sin tabla, va a las que lo tienen y, si ninguna lo tiene, se rechaza (no se ignora un criterio)."""
    _exigir_tabla(tablas, tabla)
    candidatas = [t for t in tablas if tabla is None or t.nombre == tabla]
    con_campos = [t for t in candidatas if not t.filas or all(c in t.columnas for c in campos)]
    if not con_campos or (tabla is not None and len(con_campos) < len(candidatas)):
        faltan = [c for c in campos if not any(c in t.columnas for t in candidatas)] or campos
        raise _error(f"el campo «{faltan[0][:40]}» no está en el resultado; pídalo en la consulta")
    elegidas = {id(t) for t in con_campos}  # por identidad: dos tablas pueden tener las mismas filas
    return [operacion(t) if id(t) in elegidas else t for t in tablas]


def _filtro(criterio: Criterio) -> Callable[[Tabla], Tabla]:
    return lambda t: Tabla(t.nombre, filtrar(t.filas, (criterio,)))


def aplicar_especificacion(tablas: list[Tabla], especificacion: Especificacion) -> list[Tabla]:
    """Orden de aplicación: filtros → agrupación → orden → límite → ocultar columnas auxiliares."""
    if especificacion.vacia:
        return tablas
    _exigir_tabla(tablas, especificacion.tabla)
    for criterio in especificacion.filtros:
        tablas = _en_alcance(
            tablas, criterio.tabla or especificacion.tabla, [criterio.campo], _filtro(criterio)
        )
    alcance = especificacion.tabla
    if especificacion.agrupacion:
        g = especificacion.agrupacion
        campos = [*g.por, *(c.campo for c in g.calculos if c.campo)]
        tablas = _en_alcance(tablas, alcance, campos, lambda t: Tabla(t.nombre, agrupar(t.filas, g)))
    if especificacion.orden:
        o = especificacion.orden
        tablas = _en_alcance(
            tablas, alcance, [x.campo for x in o], lambda t: Tabla(t.nombre, ordenar(t.filas, o))
        )
    if especificacion.limite:
        n = especificacion.limite
        tablas = [Tabla(t.nombre, t.filas[:n]) if alcance in (None, t.nombre) else t for t in tablas]
    if especificacion.ocultar:
        campos_ocultos = especificacion.ocultar
        tablas = [
            Tabla(t.nombre, ocultar(t.filas, campos_ocultos)) if alcance in (None, t.nombre) else t
            for t in tablas
        ]
    return tablas
