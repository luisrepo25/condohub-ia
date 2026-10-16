"""Fixtures compartidos de las pruebas contra un MongoDB real (se omiten solas si no hay servidor)."""

import os
from collections.abc import Iterator
from typing import Any

import pytest

COLECCIONES = ("perfiles_faciales", "desconocidos", "predicciones", "clusters_desconocidos", "outbox")
RABBIT_URL = os.environ.get("RABBITMQ_URL_PRUEBAS", "amqp://condohub:condohub@localhost:5672/%2F")
URI = os.environ.get("MONGO_URI_PRUEBAS", "mongodb://localhost:27017")
BASE = os.environ.get("MONGO_DB_PRUEBAS", "ia_pruebas")
# Almacén de objetos del compose (SeaweedFS); credenciales de desarrollo de infra/seaweedfs/s3.json
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_PRUEBAS", "http://localhost:9000")
S3_ACCESO = os.environ.get("S3_ACCESS_KEY_PRUEBAS", "condohub")
S3_SECRETO = os.environ.get("S3_SECRET_KEY_PRUEBAS", "condohub123")
BUCKET_PRUEBAS = "pruebas-rostros"


@pytest.fixture(scope="module")
def s3() -> Any:
    from app.application.modelos import AlmacenNoDisponible
    from app.infrastructure.s3_cliente import ClienteS3

    cliente = ClienteS3(S3_ENDPOINT, S3_ACCESO, S3_SECRETO)
    try:
        cliente.crear_bucket(BUCKET_PRUEBAS)
    except AlmacenNoDisponible:
        pytest.skip("No hay un almacén S3 (docker compose -f infra/docker-compose.yml up -d minio)")
    return cliente


@pytest.fixture
def cola_de_eventos() -> Iterator[Any]:
    """Una cola temporal enlazada al exchange `condohub.events` para ver lo que publica el servicio."""
    pika = pytest.importorskip("pika")
    try:
        conexion = pika.BlockingConnection(
            pika.URLParameters(f"{RABBIT_URL}?connection_attempts=1&socket_timeout=2")
        )
        canal = conexion.channel()
        cola = canal.queue_declare(queue="", exclusive=True).method.queue
        canal.queue_bind(
            exchange="condohub.events", queue=cola, routing_key="ia.VisitanteRecurrenteDetectado"
        )
    except Exception as e:  # noqa: BLE001 - cualquier fallo de conexión o de topología significa «no hay RabbitMQ listo»
        pytest.skip(f"No hay un RabbitMQ con el exchange condohub.events ({type(e).__name__})")

    def recibir(espera_s: float = 3.0) -> dict[str, Any] | None:
        import json
        import time

        limite = time.time() + espera_s
        while time.time() < limite:
            metodo, _, cuerpo = canal.basic_get(queue=cola, auto_ack=True)
            if metodo is not None:
                return json.loads(cuerpo)  # type: ignore[no-any-return]
            time.sleep(0.1)
        return None

    yield recibir
    conexion.close()


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
