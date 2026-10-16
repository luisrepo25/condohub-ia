"""Endpoints de rostros por HTTP: formato del contrato, errores en problem+json y reglas I1, I2, I5."""

from datetime import timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.infrastructure.keycloak import JwtVerificador
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave
from tests.dobles import AHORA, ANA, UNIDAD, Mundo

SUB_ANA = "sub-ana"
FORM = {
    "residenteId": str(ANA),
    "residenteSub": SUB_ANA,
    "unidadId": str(UNIDAD),
    "consentimientoFecha": (AHORA - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    "consentimientoVersion": "v1",
}


@pytest.fixture
def mundo() -> Mundo:
    return Mundo()


@pytest.fixture
def api(mundo: Mundo) -> TestClient:
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    return TestClient(create_app(verificador, casos_rostros=mundo.casos))


def _auth(
    crear_token: CrearToken, rol: str, sub: str = "00000000-0000-0000-0000-00000000a001"
) -> dict[str, str]:
    return {"Authorization": f"Bearer {crear_token(rol, sub=sub)}"}


def _fotos(*nombres: bytes) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("fotos", (f"f{i}.jpg", n, "image/jpeg")) for i, n in enumerate(nombres)]


# ------------------------------------------------------------------ POST /rostros/registrar


def test_I1_el_admin_registra_un_perfil_y_recibe_lo_que_dice_el_contrato(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    r = api.post(
        "/rostros/registrar", headers=_auth(crear_token, "ADMIN"), data=FORM, files=_fotos(b"ana-1", b"ana-2")
    )
    assert r.status_code == 201
    assert r.json() == {
        "residenteId": str(ANA),
        "embeddings": 2,
        "modelo": "facenet-falso",
        "version": "0.0.1",
    }
    perfil = mundo.perfiles.por_residente(ANA)
    assert perfil is not None and perfil.residente_sub == SUB_ANA and perfil.unidad_id == UNIDAD


@pytest.mark.parametrize("campo", ["consentimientoVersion", "consentimientoFecha"])
@pytest.mark.parametrize("vacio", [True, False])
def test_I1_sin_consentimiento_responde_422_con_code_I1_ya_sea_vacio_o_ausente(
    api: TestClient, crear_token: CrearToken, mundo: Mundo, campo: str, vacio: bool
) -> None:
    datos = {**FORM, campo: ""} if vacio else {k: v for k, v in FORM.items() if k != campo}
    r = api.post(
        "/rostros/registrar", headers=_auth(crear_token, "ADMIN"), data=datos, files=_fotos(b"ana-1")
    )
    assert r.status_code == 422 and r.json()["code"] == "I1"
    assert r.headers["content-type"].startswith("application/problem+json")
    assert mundo.perfiles.por_residente(ANA) is None and mundo.modelo.llamadas == 0


def test_I1_un_consentimiento_sin_zona_horaria_responde_422_con_code_I1(
    api: TestClient, crear_token: CrearToken
) -> None:
    r = api.post(
        "/rostros/registrar",
        headers=_auth(crear_token, "ADMIN"),
        data={**FORM, "consentimientoFecha": "2026-10-19T10:00:00"},
        files=_fotos(b"ana-1"),
    )
    assert r.status_code == 422 and r.json()["code"] == "I1"


def test_registrar_sin_el_residente_sub_es_una_solicitud_invalida(
    api: TestClient, crear_token: CrearToken
) -> None:
    datos = {k: v for k, v in FORM.items() if k != "residenteSub"}
    r = api.post(
        "/rostros/registrar", headers=_auth(crear_token, "ADMIN"), data=datos, files=_fotos(b"ana-1")
    )
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/problem+json")


def test_registrar_con_fotos_sin_rostro_responde_422_sin_rostro(
    api: TestClient, crear_token: CrearToken
) -> None:
    r = api.post(
        "/rostros/registrar", headers=_auth(crear_token, "ADMIN"), data=FORM, files=_fotos(b"sin-rostro")
    )
    assert r.status_code == 422 and r.json()["code"] == "SIN_ROSTRO"


def test_registrar_con_una_foto_ilegible_responde_400(api: TestClient, crear_token: CrearToken) -> None:
    r = api.post(
        "/rostros/registrar", headers=_auth(crear_token, "ADMIN"), data=FORM, files=_fotos(b"basura")
    )
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/problem+json")


# ------------------------------------------------------------------ POST /rostros/identificar


def _identificar(
    api: TestClient, crear_token: CrearToken, imagen: bytes, rol: str = "GUARDIA"
) -> tuple[int, dict]:  # type: ignore[type-arg]
    r = api.post(
        "/rostros/identificar",
        headers=_auth(crear_token, rol),
        files={"imagen": ("a.jpg", imagen, "image/jpeg")},
    )
    return r.status_code, r.json()


def test_A5_el_guardia_recibe_el_residente_reconocido_con_su_confianza(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    mundo.registrar_a_ana()
    estado, cuerpo = _identificar(api, crear_token, b"ana-otra")
    assert estado == 200
    assert set(cuerpo) == {"resultado", "residenteId", "confianza", "modelo", "version", "prediccionId"}
    assert cuerpo["resultado"] == "RECONOCIDO" and UUID(cuerpo["residenteId"]) == ANA
    assert cuerpo["confianza"] == pytest.approx(0.95)


def test_I2_la_zona_dudosa_llega_al_guardia_con_el_candidato(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    mundo.registrar_a_ana()
    estado, cuerpo = _identificar(api, crear_token, b"parecido")
    assert estado == 200 and cuerpo["resultado"] == "DUDOSO" and UUID(cuerpo["residenteId"]) == ANA


def test_I2_un_desconocido_no_trae_residente(api: TestClient, crear_token: CrearToken, mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    estado, cuerpo = _identificar(api, crear_token, b"extrano", rol="ADMIN")
    assert estado == 200 and cuerpo["resultado"] == "DESCONOCIDO" and cuerpo["residenteId"] is None


def test_sin_rostro_en_la_imagen_responde_200_sin_rostro(api: TestClient, crear_token: CrearToken) -> None:
    estado, cuerpo = _identificar(api, crear_token, b"sin-rostro")
    assert estado == 200 and cuerpo["resultado"] == "SIN_ROSTRO" and cuerpo["confianza"] == 0.0


def test_I5_la_identificacion_deja_su_prediccion_registrada(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    _, cuerpo = _identificar(api, crear_token, b"sin-rostro")
    assert [i for i, _ in mundo.predicciones.guardadas] == [cuerpo["prediccionId"]]


def test_identificar_una_imagen_ilegible_responde_400(api: TestClient, crear_token: CrearToken) -> None:
    estado, _ = _identificar(api, crear_token, b"basura")
    assert estado == 400


def test_sin_modelo_facial_configurado_el_servicio_responde_503(
    client: TestClient, crear_token: CrearToken
) -> None:
    """Sin modelo facial configurado el servicio responde 503, no un error confuso."""
    r = client.post(
        "/rostros/identificar",
        headers=_auth(crear_token, "GUARDIA"),
        files={"imagen": ("a.jpg", b"x", "image/jpeg")},
    )
    assert r.status_code == 503 and r.headers["content-type"].startswith("application/problem+json")


# ------------------------------------------------------------------ DELETE /rostros/{residenteId}


def test_I1_el_residente_retira_su_consentimiento_y_recibe_el_plazo(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    mundo.registrar_a_ana()
    r = api.delete(f"/rostros/{ANA}", headers=_auth(crear_token, "RESIDENTE", sub=SUB_ANA))
    assert r.status_code == 202
    assert r.json() == {"residenteId": str(ANA), "borradoAntesDe": "2026-10-21T12:00:00Z"}
    assert mundo.fotos.objetos == {}


def test_I1_retirarlo_otra_vez_devuelve_exactamente_lo_mismo(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    mundo.registrar_a_ana()
    encabezados = _auth(crear_token, "RESIDENTE", sub=SUB_ANA)
    primera = api.delete(f"/rostros/{ANA}", headers=encabezados)
    mundo.reloj.avanzar(timedelta(hours=3))
    segunda = api.delete(f"/rostros/{ANA}", headers=encabezados)
    assert segunda.status_code == 202 and segunda.json() == primera.json()


def test_I1_un_residente_no_puede_retirar_el_consentimiento_de_otro_responde_403(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    mundo.registrar_a_ana()
    r = api.delete(f"/rostros/{ANA}", headers=_auth(crear_token, "RESIDENTE", sub="sub-bea"))
    assert r.status_code == 403 and r.headers["content-type"].startswith("application/problem+json")
    assert len(mundo.fotos.objetos) == 2


def test_I1_el_admin_puede_retirar_el_consentimiento_de_cualquiera(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    mundo.registrar_a_ana()
    assert api.delete(f"/rostros/{ANA}", headers=_auth(crear_token, "ADMIN")).status_code == 202


def test_retirar_el_consentimiento_de_quien_no_tiene_perfil_responde_404(
    api: TestClient, crear_token: CrearToken
) -> None:
    r = api.delete(f"/rostros/{ANA}", headers=_auth(crear_token, "ADMIN"))
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")
