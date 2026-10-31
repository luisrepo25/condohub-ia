"""I2, A5: al reconocer a un residente el guardia ve su nombre y su unidad, no un identificador interno."""

from datetime import timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.domain.rostros import Consentimiento, EmbeddingRegistrado, PerfilFacial
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.mongo.documentos import documento_a_perfil, perfil_a_documento
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave
from tests.dobles import AHORA, ANA, FECHA, UNIDAD, Mundo, eje


@pytest.fixture
def mundo() -> Mundo:
    return Mundo()


def _registrar_con_datos(mundo: Mundo) -> None:
    mundo.registrar.ejecutar(
        ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"ana-1", b"ana-2"], nombre="Ana Pérez", unidad_codigo="B-101"
    )


def test_I2_el_rostro_reconocido_trae_el_nombre_y_la_unidad_del_residente(mundo: Mundo) -> None:
    _registrar_con_datos(mundo)
    r = mundo.identificar.ejecutar(b"ana-otra")
    assert r.resultado == "RECONOCIDO" and r.residente_id == ANA
    assert (r.nombre, r.unidad_codigo) == ("Ana Pérez", "B-101")


def test_I2_el_candidato_dudoso_tambien_trae_sus_datos_para_que_el_guardia_decida(mundo: Mundo) -> None:
    _registrar_con_datos(mundo)
    r = mundo.identificar.ejecutar(b"parecido")
    assert r.resultado == "DUDOSO" and r.nombre == "Ana Pérez"


def test_I2_un_desconocido_no_trae_datos_de_nadie(mundo: Mundo) -> None:
    _registrar_con_datos(mundo)
    r = mundo.identificar.ejecutar(b"extrano")
    assert r.resultado == "DESCONOCIDO" and (r.nombre, r.unidad_codigo) == (None, None)


def test_un_perfil_registrado_sin_nombre_se_reconoce_igual_pero_sin_datos(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.identificar.ejecutar(b"ana-otra")
    assert r.resultado == "RECONOCIDO" and (r.nombre, r.unidad_codigo) == (None, None)


def test_el_nombre_en_blanco_se_trata_como_ausente(mundo: Mundo) -> None:
    mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"ana-1"], nombre="   ", unidad_codigo="")
    assert mundo.identificar.ejecutar(b"ana-otra").nombre is None


def _perfil(**datos: str) -> PerfilFacial:
    return PerfilFacial.registrar(
        ANA, "sub-ana", UNIDAD, Consentimiento(AHORA - timedelta(days=1), "v1"),
        [EmbeddingRegistrado(eje(0), "facenet-vggface2", "1.0.0", "rostros/ana/a.jpg", AHORA)],
        AHORA,
        **datos,
    )


def test_el_nombre_y_el_codigo_de_unidad_viajan_por_mongo() -> None:
    doc = perfil_a_documento(_perfil(nombre="Ana Pérez", unidad_codigo="B-101"))
    assert doc["nombre"] == "Ana Pérez" and doc["unidadCodigo"] == "B-101"
    leido = documento_a_perfil(doc)
    assert (leido.nombre, leido.unidad_codigo) == ("Ana Pérez", "B-101")


def test_sin_datos_el_documento_omite_los_campos_porque_el_validador_exige_texto() -> None:
    doc = perfil_a_documento(_perfil())
    assert "nombre" not in doc and "unidadCodigo" not in doc
    assert documento_a_perfil(doc).nombre is None


# ------------------------------------------------------------------ HTTP


@pytest.fixture
def api(mundo: Mundo) -> TestClient:
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    return TestClient(create_app(verificador, casos_rostros=mundo.casos))


def test_el_registro_acepta_nombre_y_unidad_y_la_identificacion_los_devuelve(
    api: TestClient, mundo: Mundo, crear_token: CrearToken
) -> None:
    form = {
        "residenteId": str(ANA), "residenteSub": "sub-ana", "unidadId": str(UNIDAD),
        "consentimientoFecha": (AHORA - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "consentimientoVersion": "v1", "nombre": "Ana Pérez", "unidadCodigo": "B-101",
    }
    registro = api.post(
        "/rostros/registrar", headers={"Authorization": f"Bearer {crear_token('ADMIN')}"}, data=form,
        files=[("fotos", ("f.jpg", b"ana-1", "image/jpeg"))],
    )
    assert registro.status_code == 201
    r = api.post(
        "/rostros/identificar", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"},
        files={"imagen": ("a.jpg", b"ana-otra", "image/jpeg")},
    )
    cuerpo = r.json()
    assert UUID(cuerpo["residenteId"]) == ANA
    assert (cuerpo["nombre"], cuerpo["unidadCodigo"]) == ("Ana Pérez", "B-101")


def test_sin_datos_la_respuesta_conserva_exactamente_los_campos_del_contrato_anterior(
    api: TestClient, mundo: Mundo, crear_token: CrearToken
) -> None:
    mundo.registrar_a_ana()
    r = api.post(
        "/rostros/identificar", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"},
        files={"imagen": ("a.jpg", b"ana-otra", "image/jpeg")},
    )
    assert set(r.json()) == {"resultado", "residenteId", "confianza", "modelo", "version", "prediccionId"}
