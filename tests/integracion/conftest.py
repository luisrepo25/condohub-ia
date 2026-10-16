"""Fixtures compartidos de las pruebas contra un MongoDB real (se omiten solas si no hay servidor)."""

import os
from collections.abc import Iterator
from typing import Any

import pytest

COLECCIONES = ("perfiles_faciales", "desconocidos", "predicciones")
URI = os.environ.get("MONGO_URI_PRUEBAS", "mongodb://localhost:27017")
BASE = os.environ.get("MONGO_DB_PRUEBAS", "ia_pruebas")


@pytest.fixture(scope="module")
def base() -> Iterator[Any]:
    pymongo = pytest.importorskip("pymongo")
    cliente = pymongo.MongoClient(URI, tz_aware=True, serverSelectionTimeoutMS=1500)
    try:
        cliente.admin.command("ping")
    except pymongo.errors.PyMongoError:
        pytest.skip("No hay un MongoDB disponible (docker compose up -d mongo)")
    db = cliente[BASE]
    for nombre in COLECCIONES:
        opciones = next(iter(db.list_collections(filter={"name": nombre})), None)
        if opciones is None or "validator" not in opciones.get("options", {}):
            pytest.skip(
                f"La base de pruebas no tiene el validador de «{nombre}»: ejecuta db/mongo-init.js con IA_DB"
            )
    yield db
    cliente.close()


@pytest.fixture(autouse=True)
def limpia(base: Any) -> Iterator[None]:
    for nombre in COLECCIONES:
        base[nombre].delete_many({})
    yield
