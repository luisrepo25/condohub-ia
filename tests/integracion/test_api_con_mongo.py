"""El servicio por HTTP con `PERSISTENCIA=mongo`: lo que hace la API queda (o se va) de MongoDB."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

pymongo = pytest.importorskip("pymongo")

from app.config import Config  # noqa: E402
from app.infrastructure.composicion import construir_casos_rostros  # noqa: E402
from app.infrastructure.keycloak import JwtVerificador  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.conftest import ISSUER, CrearToken, _clave  # noqa: E402
from tests.dobles import ANA, UNIDAD, VECTORES, ModeloFalso  # noqa: E402
from tests.integracion.conftest import BASE, URI  # noqa: E402

pytestmark = pytest.mark.integracion
FORM = {
    "residenteId": str(ANA),
    "residenteSub": "sub-ana",
    "unidadId": str(UNIDAD),
    # El servicio usa el reloj real, así que el consentimiento se fecha respecto a hoy
    "consentimientoFecha": (datetime.now(UTC) - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    "consentimientoVersion": "v1",
}


@pytest.fixture
def api(base: Any) -> TestClient:
    config = Config("i", "j", persistencia="mongo", mongo_uri=URI, mongo_db=BASE)
    casos = construir_casos_rostros(config, modelo=ModeloFalso(dict(VECTORES)))
    return TestClient(create_app(JwtVerificador(ISSUER, lambda _t: _clave.public_key()), casos_rostros=casos))


def _auth(crear_token: CrearToken, rol: str, sub: str = "sub-x") -> dict[str, str]:
    return {"Authorization": f"Bearer {crear_token(rol, sub=sub)}"}


def _registrar(api: TestClient, crear_token: CrearToken) -> None:
    fotos = [("fotos", (f"f{i}.jpg", b"ana-1", "image/jpeg")) for i in range(2)]
    r = api.post("/rostros/registrar", headers=_auth(crear_token, "ADMIN"), data=FORM, files=fotos)
    assert r.status_code == 201, r.text


def test_registrar_por_la_api_deja_el_perfil_con_su_consentimiento_en_mongo(
    api: TestClient, crear_token: CrearToken, base: Any
) -> None:
    _registrar(api, crear_token)
    doc = base["perfiles_faciales"].find_one({"residenteId": str(ANA)})
    assert doc is not None and doc["residenteSub"] == "sub-ana" and len(doc["embeddings"]) == 2
    assert doc["consentimiento"]["versionTexto"] == "v1" and doc["consentimiento"]["retiradoEn"] is None
    assert len(doc["embeddings"][0]["vector"]) == 512


def test_identificar_por_la_api_registra_la_prediccion_y_guarda_al_desconocido(
    api: TestClient, crear_token: CrearToken, base: Any
) -> None:
    _registrar(api, crear_token)
    imagen = {"imagen": ("a.jpg", b"extrano", "image/jpeg")}
    r = api.post(
        "/rostros/identificar",
        headers=_auth(crear_token, "GUARDIA"),
        files=imagen,
        data={"camara": "porteria"},
    )
    assert r.status_code == 200 and r.json()["resultado"] == "DESCONOCIDO"
    prediccion = base["predicciones"].find_one({})
    assert str(prediccion["_id"]) == r.json()["prediccionId"]
    assert prediccion["resultado"] == {"zona": "DESCONOCIDO", "residenteId": None}
    assert prediccion["modelo"] == {"nombre": "facenet-falso", "version": "0.0.1"}
    desconocido = base["desconocidos"].find_one({})
    assert desconocido["camara"] == "porteria" and desconocido["expiraEn"] > desconocido["capturadoEn"]
    assert (desconocido["expiraEn"] - desconocido["capturadoEn"]).days == 30


def test_un_residente_reconocido_no_se_guarda_como_desconocido(
    api: TestClient, crear_token: CrearToken, base: Any
) -> None:
    _registrar(api, crear_token)
    r = api.post(
        "/rostros/identificar",
        headers=_auth(crear_token, "GUARDIA"),
        files={"imagen": ("a.jpg", b"ana-1", "image/jpeg")},
    )
    assert r.json()["resultado"] == "RECONOCIDO"
    assert base["desconocidos"].count_documents({}) == 0 and base["predicciones"].count_documents({}) == 1


def test_I1_retirar_el_consentimiento_por_la_api_vacia_los_embeddings_en_mongo(
    api: TestClient, crear_token: CrearToken, base: Any
) -> None:
    _registrar(api, crear_token)
    r = api.delete(f"/rostros/{ANA}", headers=_auth(crear_token, "RESIDENTE", sub="sub-ana"))
    assert r.status_code == 202
    doc = base["perfiles_faciales"].find_one({"residenteId": str(ANA)})
    assert doc["embeddings"] == [] and doc["consentimiento"]["retiradoEn"] is not None
    despues = api.post(
        "/rostros/identificar",
        headers=_auth(crear_token, "GUARDIA"),
        files={"imagen": ("a.jpg", b"ana-1", "image/jpeg")},
    )
    assert despues.json()["resultado"] == "DESCONOCIDO"


def test_el_perfil_sobrevive_a_un_reinicio_del_servicio(
    api: TestClient, crear_token: CrearToken, base: Any
) -> None:
    _registrar(api, crear_token)
    config = Config("i", "j", persistencia="mongo", mongo_uri=URI, mongo_db=BASE)
    reiniciado = TestClient(
        create_app(
            JwtVerificador(ISSUER, lambda _t: _clave.public_key()),
            casos_rostros=construir_casos_rostros(config, modelo=ModeloFalso(dict(VECTORES))),
        )
    )
    r = reiniciado.post(
        "/rostros/identificar",
        headers=_auth(crear_token, "GUARDIA"),
        files={"imagen": ("a.jpg", b"ana-1", "image/jpeg")},
    )
    assert r.json()["resultado"] == "RECONOCIDO"  # lo que con la memoria se perdía al reiniciar
