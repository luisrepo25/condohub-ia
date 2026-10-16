"""Casos de uso disponibles para los routers (se arman en `create_app`)."""

from fastapi import Request

from app.application.modelos import ObtenerModeloPlacas


def caso_obtener_modelo_placas(request: Request) -> ObtenerModeloPlacas:
    caso: ObtenerModeloPlacas = request.app.state.obtener_modelo_placas
    return caso
