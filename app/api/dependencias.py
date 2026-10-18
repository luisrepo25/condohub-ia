"""Casos de uso disponibles para los routers (se arman en `create_app`)."""

from fastapi import Request

from app.application.leer_placa import LeerPlaca
from app.application.modelos import ObtenerModeloPlacas
from app.application.rostros import CasosRostros


def caso_obtener_modelo_placas(request: Request) -> ObtenerModeloPlacas:
    caso: ObtenerModeloPlacas = request.app.state.obtener_modelo_placas
    return caso


def casos_rostros(request: Request) -> CasosRostros:
    casos: CasosRostros = request.app.state.casos_rostros
    return casos


def caso_leer_placa(request: Request) -> LeerPlaca:
    caso: LeerPlaca = request.app.state.leer_placa
    return caso
