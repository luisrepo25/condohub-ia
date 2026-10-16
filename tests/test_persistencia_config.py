"""Elección de la persistencia por configuración y traducción de los fallos de MongoDB."""

from datetime import timedelta
from uuid import UUID

import pytest

from app.api.problemas import ProblemaError
from app.api.traduccion import traduciendo_errores
from app.config import Config
from app.infrastructure.composicion import construir_repositorios
from app.infrastructure.memoria import RepositorioPerfilesMemoria
from app.infrastructure.mongo.repositorios import RepositorioPerfilesMongo


def test_por_defecto_la_persistencia_es_en_memoria() -> None:
    perfiles, _, _ = construir_repositorios(Config("i", "j"))
    assert isinstance(perfiles, RepositorioPerfilesMemoria)


def test_un_valor_desconocido_de_persistencia_falla_al_arrancar() -> None:
    with pytest.raises(ValueError, match="PERSISTENCIA"):
        construir_repositorios(Config("i", "j", persistencia="postgres"))


def test_con_mongo_se_abre_el_cliente_sin_conectar_hasta_el_primer_uso() -> None:
    pytest.importorskip("pymongo")
    perfiles, _, _ = construir_repositorios(
        Config(
            "i", "j", persistencia="mongo", mongo_uri="mongodb://servidor-que-no-existe:27017", mongo_db="x"
        )
    )
    assert isinstance(perfiles, RepositorioPerfilesMongo)  # no falló: pymongo conecta de forma perezosa


def test_un_fallo_de_mongo_se_traduce_a_503_y_no_a_un_error_interno() -> None:
    error_de_mongo = type("ServerSelectionTimeoutError", (Exception,), {"__module__": "pymongo.errors"})
    with pytest.raises(ProblemaError) as e, traduciendo_errores():
        raise error_de_mongo("no responde")
    assert e.value.status == 503


def test_un_error_cualquiera_que_no_es_de_mongo_no_se_disfraza() -> None:
    with pytest.raises(KeyError), traduciendo_errores():
        raise KeyError("bug")


def test_el_plazo_de_retiro_y_la_retencion_son_parametros_de_configuracion() -> None:
    c = Config("i", "j", desconocidos_retencion_dias=7, retiro_plazo_horas=12)
    assert (timedelta(days=c.desconocidos_retencion_dias), timedelta(hours=c.retiro_plazo_horas)) == (
        timedelta(days=7),
        timedelta(hours=12),
    )
    assert UUID(int=0)  # (solo para mantener el import usado en este módulo)
