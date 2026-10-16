"""Arma los casos de uso con sus adaptadores reales a partir de la configuración."""

import importlib
from datetime import UTC, datetime, timedelta

from app.application.modelos import ObtenerModeloPlacas
from app.application.rostros import (
    AlmacenFotos,
    CasosRostros,
    IdentificarRostro,
    ModeloFacial,
    ModeloFacialNoDisponible,
    RegistrarRostro,
    RepositorioDesconocidos,
    RepositorioPerfiles,
    RepositorioPredicciones,
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
from app.infrastructure.mongo.repositorios import (
    RepositorioDesconocidosMongo,
    RepositorioPerfilesMongo,
    RepositorioPrediccionesMongo,
)
from app.infrastructure.s3_cliente import AlmacenFotosS3, ClienteS3
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


def construir_modelo_facial(config: Config) -> ModeloFacial:
    """`MODELO_FACIAL=facenet` conecta el modelo real (carga al primer uso); vacío deja el marcador (503)."""
    if config.modelo_facial == "":
        return ModeloFacialNoConfigurado()
    if config.modelo_facial == "facenet":
        from app.infrastructure.facenet import ModeloFacialFacenet

        return ModeloFacialFacenet(
            config.modelo_facial_dispositivo or None, version=config.modelo_facial_version
        )
    raise ValueError(f"MODELO_FACIAL desconocido: {config.modelo_facial!r} (use «facenet» o déjelo vacío)")


def construir_repositorios(
    config: Config,
) -> tuple[RepositorioPerfiles, RepositorioDesconocidos, RepositorioPredicciones]:
    """`PERSISTENCIA=memoria` (se pierde al reiniciar) o `mongo` (colecciones de `db/mongo-init.js`)."""
    if config.persistencia == "memoria":
        return (
            RepositorioPerfilesMemoria(),
            RepositorioDesconocidosMemoria(),
            RepositorioPrediccionesMemoria(),
        )
    if config.persistencia == "mongo":
        # El cliente se abre con zona horaria (UTC) y conecta al primer uso, no al arrancar el servicio
        cliente = importlib.import_module("pymongo").MongoClient(config.mongo_uri, tz_aware=True)
        base = cliente[config.mongo_db]
        return (
            RepositorioPerfilesMongo(base["perfiles_faciales"]),
            RepositorioDesconocidosMongo(base["desconocidos"]),
            RepositorioPrediccionesMongo(base["predicciones"]),
        )
    raise ValueError(f"PERSISTENCIA desconocida: {config.persistencia!r} (use «memoria» o «mongo»)")


def construir_almacen_fotos(config: Config) -> AlmacenFotos:
    """`ALMACEN_FOTOS=memoria` (se pierden al reiniciar) o `s3` (bucket privado del almacén de objetos)."""
    if config.almacen_fotos == "memoria":
        return AlmacenFotosMemoria()
    if config.almacen_fotos == "s3":
        cliente = ClienteS3(config.s3_endpoint, config.s3_acceso, config.s3_secreto, config.s3_region)
        return AlmacenFotosS3(cliente, config.s3_bucket_rostros)
    raise ValueError(f"ALMACEN_FOTOS desconocido: {config.almacen_fotos!r} (use «memoria» o «s3»)")


def construir_casos_rostros(config: Config, modelo: ModeloFacial | None = None) -> CasosRostros:
    """Rostros con el modelo, los repositorios y el almacén de fotos que indique la configuración."""
    modelo = modelo or construir_modelo_facial(config)
    perfiles, desconocidos, predicciones = construir_repositorios(config)
    fotos = construir_almacen_fotos(config)
    reloj = RelojSistema()
    return CasosRostros(
        registrar=RegistrarRostro(modelo, perfiles, fotos, reloj),
        identificar=IdentificarRostro(
            modelo,
            ReconocedorFacial(Umbrales(config.umbral_reconocido, config.umbral_dudoso)),
            perfiles,
            desconocidos,
            predicciones,
            reloj,
            retencion_desconocidos=timedelta(days=config.desconocidos_retencion_dias),
        ),
        retirar=RetirarConsentimiento(
            perfiles, fotos, reloj, plazo=timedelta(hours=config.retiro_plazo_horas)
        ),
    )
