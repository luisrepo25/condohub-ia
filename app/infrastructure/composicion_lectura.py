"""Arma el caso de uso del reconocedor de placas con sus adaptadores reales a partir de la configuración."""

from datetime import timedelta

from app.application.modelo_lectura import ObtenerModeloLectura
from app.config import Config
from app.infrastructure.catalogo_ficha_lectura import CatalogoLecturaFichaJson
from app.infrastructure.composicion import RelojSistema
from app.infrastructure.s3_firmador import FirmadorS3


def construir_obtener_modelo_lectura(config: Config) -> ObtenerModeloLectura:
    return ObtenerModeloLectura(
        catalogo=CatalogoLecturaFichaJson(config.modelo_lectura_ficha),
        firmador=FirmadorS3(
            config.s3_endpoint,
            config.s3_bucket_modelos,
            config.s3_acceso,
            config.s3_secreto,
            config.s3_region,
        ),
        reloj=RelojSistema(),
        vigencia=timedelta(seconds=config.url_firmada_segundos),
    )
