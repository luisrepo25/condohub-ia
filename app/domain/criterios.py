"""Criterios de un reporte (G4): qué filas entran, cómo se agrupan, en qué orden y cuántas. Dominio puro.

El LLM declara la especificación en un comentario GraphQL al final de la consulta (`# ESPECIFICACION: {...}`);
el servicio la lee, la valida contra una lista blanca y la aplica sobre el resultado. Es entrada NO confiable
(viene de una instrucción hablada): nada se ejecuta ni se evalúa, y una especificación mal formada se rechaza
con `G4` en lugar de ignorarse en silencio.
"""

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from app.domain.errores import ReglaViolada
from app.domain.reportes import Fila

OPERADORES = frozenset({"=", "!=", ">", ">=", "<", "<=", "contiene"})
FUNCIONES = frozenset({"contar", "suma", "promedio", "minimo", "maximo"})
MARCA = "ESPECIFICACION:"
MAX_CARACTERES = 4000
MAX_CRITERIOS = 20
MAX_LIMITE = 10_000
_LINEA = re.compile(r"^[ \t]*#[ \t]*" + MARCA + r"[ \t]*(.*)$", re.MULTILINE)
_CAMPO = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,79}$")
_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]{0,39}$")
_CLAVES = frozenset({"tabla", "filtros", "agrupar", "orden", "limite", "ocultar"})


def _error(mensaje: str) -> ReglaViolada:
    return ReglaViolada("G4", f"Los criterios del reporte no son válidos: {mensaje}")


@dataclass(frozen=True)
class Criterio:
    campo: str
    operador: str
    valor: str | int | float | bool
    tabla: str | None = None


@dataclass(frozen=True)
class Calculo:
    funcion: str
    como: str
    campo: str | None = None


@dataclass(frozen=True)
class Agrupacion:
    por: tuple[str, ...]
    calculos: tuple[Calculo, ...]


@dataclass(frozen=True)
class Orden:
    campo: str
    descendente: bool = False


@dataclass(frozen=True)
class Especificacion:
    """Todo opcional: sin especificación el reporte sale como lo devuelve Gestión."""

    tabla: str | None = None
    filtros: tuple[Criterio, ...] = ()
    agrupacion: Agrupacion | None = None
    orden: tuple[Orden, ...] = ()
    limite: int | None = None
    ocultar: tuple[str, ...] = ()

    @property
    def vacia(self) -> bool:
        return not (self.filtros or self.agrupacion or self.orden or self.limite or self.ocultar)


def normalizar_consulta(consulta: str) -> str:
    """La especificación va al final: Mongo exige que la consulta EMPIECE por `query` o `{`."""
    lineas = _LINEA.findall(consulta)
    if not lineas:
        return consulta
    cuerpo = _LINEA.sub("", consulta).rstrip()
    return f"{cuerpo}\n# {MARCA} {lineas[0].strip()}"


def _texto(valor: Any, que: str, largo: int = 100) -> str:
    if not isinstance(valor, str) or not valor or len(valor) > largo:
        raise _error(f"{que} debe ser un texto de 1 a {largo} caracteres")
    return valor


def _campo(valor: Any, que: str = "campo") -> str:
    if not isinstance(valor, str) or not _CAMPO.match(valor):
        raise _error(f"{que} «{str(valor)[:40]}» no es un nombre de campo válido")
    return valor


def _alias(valor: Any) -> str:
    if not isinstance(valor, str) or not _ALIAS.match(valor):
        raise _error(f"el nombre de columna «{str(valor)[:40]}» no es válido")
    return valor


def _criterio(crudo: Any) -> Criterio:
    if (
        not isinstance(crudo, dict)
        or not {"campo", "op", "valor"} <= crudo.keys()
        or not set(crudo) <= {"campo", "op", "valor", "tabla"}
    ):
        raise _error("cada filtro lleva campo, op y valor")
    if crudo["op"] not in OPERADORES:
        raise _error(f"el operador «{str(crudo['op'])[:20]}» no existe (use {', '.join(sorted(OPERADORES))})")
    valor = crudo["valor"]
    if isinstance(valor, str):
        valor = _texto(valor, "el valor del filtro")
    elif not isinstance(valor, int | float | bool):
        raise _error("el valor de un filtro es un texto, un número o verdadero/falso")
    tabla = None if crudo.get("tabla") is None else _texto(crudo["tabla"], "la tabla del filtro", 80)
    return Criterio(_campo(crudo["campo"]), crudo["op"], valor, tabla)


def _calculo(crudo: Any) -> Calculo:
    if (
        not isinstance(crudo, dict)
        or crudo.get("fn") not in FUNCIONES
        or not set(crudo) <= {"fn", "campo", "como"}
    ):
        raise _error(f"cada cálculo lleva fn ({', '.join(sorted(FUNCIONES))}) y como")
    funcion = crudo["fn"]
    if funcion != "contar" and crudo.get("campo") is None:
        raise _error(f"el cálculo «{funcion}» necesita un campo")
    campo = None if crudo.get("campo") is None else _campo(crudo["campo"])
    return Calculo(funcion, _alias(crudo.get("como") or funcion), campo)


def _agrupacion(crudo: Any) -> Agrupacion:
    if not isinstance(crudo, dict) or not set(crudo) <= {"por", "calcular"}:
        raise _error("agrupar lleva por y calcular")
    por, calcular = crudo.get("por", []), crudo.get("calcular", [])
    if (
        not isinstance(por, list)
        or not isinstance(calcular, list)
        or len(por) > 5
        or len(calcular) > 8
        or not (por or calcular)
    ):
        raise _error("agrupar necesita campos en por o cálculos en calcular (hasta 5 y 8)")
    return Agrupacion(tuple(_campo(c) for c in por), tuple(_calculo(c) for c in calcular))


def _orden(crudo: Any) -> Orden:
    if not isinstance(crudo, dict) or not set(crudo) <= {"campo", "sentido"} or "campo" not in crudo:
        raise _error("cada orden lleva campo y sentido")
    if crudo.get("sentido", "asc") not in {"asc", "desc"}:
        raise _error("el sentido del orden es asc o desc")
    return Orden(_campo(crudo["campo"]), crudo.get("sentido", "asc") == "desc")


def _lista(crudo: dict[str, Any], clave: str, maximo: int) -> list[Any]:
    valor = crudo.get(clave, [])
    if not isinstance(valor, list) or len(valor) > maximo:
        raise _error(f"{clave} es una lista de hasta {maximo} elementos")
    return valor


def leer_especificacion(consulta: str) -> Especificacion:
    """Especificación de la consulta (vacía sin línea `# ESPECIFICACION:`). Lanza `ReglaViolada` (G4)."""
    encontradas = _LINEA.findall(consulta)
    if not encontradas:
        return Especificacion()
    if len(encontradas) > 1:
        raise _error("hay más de una línea ESPECIFICACION")
    texto = encontradas[0].strip()
    if len(texto) > MAX_CARACTERES:
        raise _error(f"la especificación supera {MAX_CARACTERES} caracteres")
    try:
        crudo = json.loads(texto)
    except ValueError as e:
        raise _error("no es JSON válido") from e
    if not isinstance(crudo, dict) or not set(crudo) <= _CLAVES:
        raise _error(f"solo admite las claves {', '.join(sorted(_CLAVES))}")
    limite = crudo.get("limite")
    if limite is not None and (
        not isinstance(limite, int) or isinstance(limite, bool) or not 1 <= limite <= MAX_LIMITE
    ):
        raise _error(f"limite es un entero de 1 a {MAX_LIMITE}")
    return Especificacion(
        tabla=None if crudo.get("tabla") is None else _texto(crudo["tabla"], "tabla", 80),
        filtros=tuple(_criterio(c) for c in _lista(crudo, "filtros", MAX_CRITERIOS)),
        agrupacion=None if crudo.get("agrupar") is None else _agrupacion(crudo["agrupar"]),
        orden=tuple(_orden(o) for o in _lista(crudo, "orden", 5)),
        limite=limite,
        ocultar=tuple(_campo(c, "campo a ocultar") for c in _lista(crudo, "ocultar", 20)),
    )


# ------------------------------------------------------------------ aplicación sobre filas


def _numero(valor: Any) -> Decimal | None:
    if isinstance(valor, bool) or valor is None:
        return None
    try:
        numero = Decimal(str(valor).strip())
    except InvalidOperation:
        return None
    return numero if numero.is_finite() else None


def _comparar(izquierda: Any, operador: str, derecha: Any) -> bool:
    return bool({
        "=": izquierda == derecha,
        "!=": izquierda != derecha,
        ">": izquierda > derecha,
        ">=": izquierda >= derecha,
        "<": izquierda < derecha,
        "<=": izquierda <= derecha,
    }[operador])


def _coincide(dato: Any, operador: str, esperado: Any) -> bool:
    if dato is None:
        return operador == "!="
    if operador == "contiene":
        return str(esperado).casefold() in str(dato).casefold()
    a, b = _numero(dato), _numero(esperado)
    if a is not None and b is not None:
        return _comparar(a, operador, b)
    return _comparar(str(dato).casefold(), operador, str(esperado).casefold())


def filtrar(filas: list[Fila], criterios: tuple[Criterio, ...]) -> list[Fila]:
    return [f for f in filas if all(_coincide(f.get(c.campo), c.operador, c.valor) for c in criterios)]


def _decimal(valor: Any, campo: str) -> Decimal:
    numero = _numero(valor)
    if numero is None:
        raise _error(f"el campo «{campo}» no es numérico y no se puede sumar ni promediar")
    return numero


def _calcular(calculo: Calculo, grupo: list[Fila]) -> str | int | None:
    if calculo.funcion == "contar":
        return len(grupo)
    campo = calculo.campo or ""
    valores = [_decimal(f[campo], campo) for f in grupo if f.get(campo) is not None]
    if not valores:
        return None
    resultado = {
        "suma": sum(valores, Decimal(0)),
        "promedio": sum(valores, Decimal(0)) / len(valores),
        "minimo": min(valores),
        "maximo": max(valores),
    }[calculo.funcion]
    return format(
        resultado.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"
    )  # dinero como texto decimal (F10)


def agrupar(filas: list[Fila], agrupacion: Agrupacion) -> list[Fila]:
    grupos: dict[tuple[Any, ...], list[Fila]] = {}
    for fila in filas:
        grupos.setdefault(tuple(fila.get(c) for c in agrupacion.por), []).append(fila)
    if not agrupacion.por:  # solo cálculos: una fila con los totales
        grupos = {(): filas}
    salida: list[Fila] = []
    for clave, grupo in grupos.items():
        resultado: Fila = dict(zip(agrupacion.por, clave, strict=True))
        for calculo in agrupacion.calculos:
            resultado[calculo.como] = _calcular(calculo, grupo)
        salida.append(resultado)
    return salida


def ordenar(filas: list[Fila], orden: tuple[Orden, ...]) -> list[Fila]:
    resultado = list(filas)
    for criterio in reversed(orden):  # orden estable: el último criterio se aplica primero
        con_dato = [f for f in resultado if f.get(criterio.campo) is not None]
        sin_dato = [f for f in resultado if f.get(criterio.campo) is None]

        def clave(f: Fila, campo: str = criterio.campo) -> tuple[int, Decimal, str]:
            numero = _numero(f[campo])
            return (0, numero, "") if numero is not None else (1, Decimal(0), str(f[campo]).casefold())

        resultado = sorted(con_dato, key=clave, reverse=criterio.descendente) + sin_dato
    return resultado


def ocultar(filas: list[Fila], campos: tuple[str, ...]) -> list[Fila]:
    return [{k: v for k, v in f.items() if k not in campos} for f in filas]
