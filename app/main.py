from fastapi import FastAPI

from app.api.problemas import registrar_manejadores
from app.api.routers import placas, reportes, rostros, salud
from app.application.modelos import ObtenerModeloPlacas
from app.application.puertos import VerificadorToken
from app.config import Config
from app.infrastructure.composicion import construir_obtener_modelo_placas
from app.infrastructure.keycloak import JwtVerificador


def create_app(
    verificador: VerificadorToken | None = None,
    obtener_modelo_placas: ObtenerModeloPlacas | None = None,
) -> FastAPI:
    """Fábrica de la aplicación. Los tests inyectan un verificador con claves propias y dobles."""
    app = FastAPI(title="CondoHub · Servicio IA", version="0.1.0")
    if verificador is None:
        config = Config.desde_entorno()
        verificador = JwtVerificador.desde_jwks(config.keycloak_issuer, config.keycloak_jwks)
    app.state.verificador = verificador
    if obtener_modelo_placas is None:
        obtener_modelo_placas = construir_obtener_modelo_placas(Config.desde_entorno())
    app.state.obtener_modelo_placas = obtener_modelo_placas
    registrar_manejadores(app)
    for router in (salud.router, rostros.router, placas.router, reportes.router):
        app.include_router(router)
    return app


app = create_app()
