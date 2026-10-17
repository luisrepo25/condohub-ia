"""Dependencias de FastAPI de los reportes por voz."""

from fastapi import Request

from app.application.errores import TokenInvalido
from app.application.reportes import CasosReportes


def casos_reportes(request: Request) -> CasosReportes:
    casos: CasosReportes = request.app.state.casos_reportes
    return casos


def token_bruto(request: Request) -> str:
    """El access token del ADMIN, ya validado; se reenvía a Gestión, que lo valida otra vez."""
    cabecera = request.headers.get("Authorization", "")
    if not cabecera.startswith("Bearer "):
        raise TokenInvalido("Falta el header Authorization: Bearer")
    return cabecera.removeprefix("Bearer ")