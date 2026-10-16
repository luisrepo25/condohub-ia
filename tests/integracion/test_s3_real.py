"""El almacén de objetos REAL (SeaweedFS del compose): fotos, URLs firmadas y el ciclo completo de I1."""

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.request import urlopen
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

pymongo = pytest.importorskip("pymongo")

from app.config import Config  # noqa: E402
from app.infrastructure.composicion import construir_casos_rostros  # noqa: E402
from app.infrastructure.keycloak import JwtVerificador  # noqa: E402
from app.infrastructure.s3_cliente import AlmacenFotosS3  # noqa: E402
from app.infrastructure.s3_firmador import FirmadorS3  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.conftest import ISSUER, CrearToken, _clave  # noqa: E402
from tests.dobles import ANA, UNIDAD, VECTORES, ModeloFalso  # noqa: E402
from tests.integracion.conftest import (  # noqa: E402
    BASE,
    BUCKET_PRUEBAS,
    S3_ACCESO,
    S3_ENDPOINT,
    S3_SECRETO,
    URI,
)

pytestmark = pytest.mark.integracion


def test_una_foto_subida_se_descarga_igual_y_despues_de_borrarla_ya_no_existe(s3: Any) -> None:
    s3.subir(BUCKET_PRUEBAS, "prueba/a.jpg", b"foto-de-prueba", "image/jpeg")
    assert s3.descargar(BUCKET_PRUEBAS, "prueba/a.jpg") == b"foto-de-prueba"
    s3.borrar(BUCKET_PRUEBAS, "prueba/a.jpg")
    assert s3.descargar(BUCKET_PRUEBAS, "prueba/a.jpg") is None
    s3.borrar(BUCKET_PRUEBAS, "prueba/a.jpg")  # borrar de nuevo no falla


def test_las_credenciales_falsas_se_rechazan(s3: Any) -> None:
    from app.application.modelos import AlmacenNoDisponible
    from app.infrastructure.s3_cliente import ClienteS3

    intruso = ClienteS3(S3_ENDPOINT, S3_ACCESO, "clave-inventada")
    with pytest.raises(AlmacenNoDisponible, match="403"):
        intruso.subir(BUCKET_PRUEBAS, "prueba/intruso.jpg", b"x")


def test_O6_una_url_firmada_de_descarga_funciona_y_vence_por_su_vigencia(s3: Any) -> None:
    """Cierra el pendiente del ADR 0002: el almacén acepta las URLs prefirmadas SigV4 del servicio."""
    s3.subir(BUCKET_PRUEBAS, "prueba/firmada.bin", b"contenido-privado")
    firmador = FirmadorS3(S3_ENDPOINT, BUCKET_PRUEBAS, S3_ACCESO, S3_SECRETO, "us-east-1")
    url = firmador.firmar_descarga("prueba/firmada.bin", timedelta(minutes=5), datetime.now(UTC))
    with urlopen(url, timeout=10) as r:  # noqa: S310
        assert r.read() == b"contenido-privado"
    caducada = firmador.firmar_descarga(
        "prueba/firmada.bin", timedelta(seconds=1), datetime.now(UTC) - timedelta(hours=1)
    )
    with pytest.raises(Exception) as e:
        urlopen(caducada, timeout=10)  # noqa: S310
    assert "403" in str(e.value)
    s3.borrar(BUCKET_PRUEBAS, "prueba/firmada.bin")


def test_I1_las_fotos_de_un_residente_se_guardan_y_se_borran_del_almacen_real(s3: Any) -> None:
    almacen = AlmacenFotosS3(s3, BUCKET_PRUEBAS)
    claves = almacen.guardar(UUID(int=5), [b"uno", b"dos"])
    assert [s3.descargar(BUCKET_PRUEBAS, c) for c in claves] == [b"uno", b"dos"]
    almacen.borrar(claves)
    assert [s3.descargar(BUCKET_PRUEBAS, c) for c in claves] == [None, None]


def test_I1_retirar_el_consentimiento_por_la_api_borra_las_fotos_del_almacen_y_los_embeddings_de_mongo(
    s3: Any, base: Any, crear_token: CrearToken
) -> None:
    config = Config(
        "i",
        "j",
        persistencia="mongo",
        mongo_uri=URI,
        mongo_db=BASE,
        almacen_fotos="s3",
        s3_endpoint=S3_ENDPOINT,
        s3_acceso=S3_ACCESO,
        s3_secreto=S3_SECRETO,
        s3_bucket_rostros=BUCKET_PRUEBAS,
    )
    casos = construir_casos_rostros(config, modelo=ModeloFalso(dict(VECTORES)))
    api = TestClient(create_app(JwtVerificador(ISSUER, lambda _t: _clave.public_key()), casos_rostros=casos))
    admin = {"Authorization": f"Bearer {crear_token('ADMIN')}"}
    consentimiento = (datetime.now(UTC) - timedelta(days=1)).isoformat().replace("+00:00", "Z")

    r = api.post(
        "/rostros/registrar",
        headers=admin,
        data={
            "residenteId": str(ANA),
            "residenteSub": "sub-ana",
            "unidadId": str(UNIDAD),
            "consentimientoFecha": consentimiento,
            "consentimientoVersion": "v1",
        },
        files=[("fotos", (f"f{i}.jpg", b"ana-1", "image/jpeg")) for i in range(2)],
    )
    assert r.status_code == 201, r.text
    claves = [
        e["fotoKey"] for e in base["perfiles_faciales"].find_one({"residenteId": str(ANA)})["embeddings"]
    ]
    assert len(claves) == 2 and all(
        s3.descargar(BUCKET_PRUEBAS, c) == b"ana-1" for c in claves
    )  # fotos en el almacén

    retiro = api.delete(
        f"/rostros/{ANA}", headers={"Authorization": f"Bearer {crear_token('RESIDENTE', sub='sub-ana')}"}
    )
    assert retiro.status_code == 202
    assert all(s3.descargar(BUCKET_PRUEBAS, c) is None for c in claves)  # fotos borradas
    assert (
        base["perfiles_faciales"].find_one({"residenteId": str(ANA)})["embeddings"] == []
    )  # embeddings borrados
