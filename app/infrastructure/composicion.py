"""Arma los casos de uso con sus adaptadores reales a partir de la configuración."""

from datetime import UTC, datetime, timedelta

from app.application.modelos import ObtenerModeloPlacas
from app.config import Config
from app.infrastructure.catalogo_ficha import CatalogoFichaJson
from app.infrastructure.s3_firmador import FirmadorS3


class RelojSistema:
    def ahora(self) -> datetime:
        return datetime.now(UTC)


def construir_obtener_modelo_placas(config: Config) -> ObtenerModeloPlacas:
    return ObtenerModeloPlacas(
        catalogo=CatalogoFichaJson(config.modelo_placas_ficha),
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
