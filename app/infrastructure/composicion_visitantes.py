"""Arma la consulta de visitantes recurrentes según la configuración."""

import importlib

from app.application.visitantes import ListarVisitantesRecurrentes
from app.config import Config
from app.infrastructure.mongo.visitantes import LectorVisitantesMongo, LectorVisitantesVacio


def construir_listar_visitantes(config: Config) -> ListarVisitantesRecurrentes:
    if config.persistencia == "mongo":
        cliente = importlib.import_module("pymongo").MongoClient(config.mongo_uri, tz_aware=True)
        coleccion = cliente[config.mongo_db]["clusters_desconocidos"]
        return ListarVisitantesRecurrentes(LectorVisitantesMongo(coleccion))
    return ListarVisitantesRecurrentes(LectorVisitantesVacio())
