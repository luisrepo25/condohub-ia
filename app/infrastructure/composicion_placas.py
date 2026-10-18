"""Arma el caso de uso de lectura de placas con el lector y el repositorio que indique la configuración."""

import importlib
from datetime import UTC, datetime

from app.application.leer_placa import (
    LectorPlacas,
    LectorPlacasNoDisponible,
    LecturaCruda,
    LeerPlaca,
    RepositorioPrediccionesPlacas,
)
from app.config import Config
from app.infrastructure.predicciones_placas import (
    RepositorioPrediccionesPlacasMemoria,
    RepositorioPrediccionesPlacasMongo,
)


class _Reloj:
    def ahora(self) -> datetime:
        return datetime.now(UTC)


class LectorPlacasNoConfigurado:
    """Marcador hasta configurar `LECTOR_PLACAS=onnx`: cualquier uso responde 503."""

    nombre = "no-configurado"
    version = "0"

    def leer(self, imagen: bytes) -> LecturaCruda | None:
        raise LectorPlacasNoDisponible("El lector de placas no está configurado (LECTOR_PLACAS=onnx)")


def construir_lector_placas(config: Config) -> LectorPlacas:
    if config.lector_placas == "":
        return LectorPlacasNoConfigurado()
    if config.lector_placas == "onnx":
        from app.infrastructure.lector_placas_onnx import LectorPlacasOnnx

        return LectorPlacasOnnx(
            config.lector_placas_detector, config.lector_placas_modelo, config.lector_placas_version
        )
    raise ValueError(f"LECTOR_PLACAS desconocido: {config.lector_placas!r} (use «onnx» o déjelo vacío)")


def construir_repositorio_predicciones_placas(config: Config) -> RepositorioPrediccionesPlacas:
    if config.persistencia == "memoria":
        return RepositorioPrediccionesPlacasMemoria()
    if config.persistencia == "mongo":
        cliente = importlib.import_module("pymongo").MongoClient(config.mongo_uri, tz_aware=True)
        return RepositorioPrediccionesPlacasMongo(cliente[config.mongo_db]["predicciones"])
    raise ValueError(f"PERSISTENCIA desconocida: {config.persistencia!r} (use «memoria» o «mongo»)")


def construir_leer_placa(config: Config, lector: LectorPlacas | None = None) -> LeerPlaca:
    return LeerPlaca(
        lector or construir_lector_placas(config), construir_repositorio_predicciones_placas(config), _Reloj()
    )
