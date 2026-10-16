"""Arma los casos de uso con sus adaptadores reales a partir de la configuración."""

from datetime import UTC, datetime, timedelta

from app.application.modelos import ObtenerModeloPlacas
from app.application.rostros import (
    CasosRostros,
    IdentificarRostro,
    ModeloFacial,
    ModeloFacialNoDisponible,
    RegistrarRostro,
    RetirarConsentimiento,
)
from app.config import Config
from app.domain.rostros import Embedding, ReconocedorFacial, Umbrales
from app.infrastructure.catalogo_ficha import CatalogoFichaJson
from app.infrastructure.memoria import (
    AlmacenFotosMemoria,
    RepositorioDesconocidosMemoria,
    RepositorioPerfilesMemoria,
    RepositorioPrediccionesMemoria,
)
from app.infrastructure.s3_firmador import FirmadorS3


class RelojSistema:
    def ahora(self) -> datetime:
        return datetime.now(UTC)


class ModeloFacialNoConfigurado:
    """Marcador hasta que se conecte el modelo real (MTCNN + FaceNet): cualquier uso responde 503."""

    nombre = "no-configurado"
    version = "0"

    def embeber(self, imagen: bytes) -> Embedding | None:
        raise ModeloFacialNoDisponible("El modelo facial no está configurado en este servicio")


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


def construir_casos_rostros(config: Config, modelo: ModeloFacial | None = None) -> CasosRostros:
    """Rostros con repositorios en memoria: sin persistencia hasta tener los adaptadores de MongoDB."""
    modelo = modelo or ModeloFacialNoConfigurado()
    perfiles = RepositorioPerfilesMemoria()
    fotos = AlmacenFotosMemoria()
    reloj = RelojSistema()
    return CasosRostros(
        registrar=RegistrarRostro(modelo, perfiles, fotos, reloj),
        identificar=IdentificarRostro(
            modelo,
            ReconocedorFacial(Umbrales(config.umbral_reconocido, config.umbral_dudoso)),
            perfiles,
            RepositorioDesconocidosMemoria(),
            RepositorioPrediccionesMemoria(),
            reloj,
            retencion_desconocidos=timedelta(days=config.desconocidos_retencion_dias),
        ),
        retirar=RetirarConsentimiento(
            perfiles, fotos, reloj, plazo=timedelta(hours=config.retiro_plazo_horas)
        ),
    )
