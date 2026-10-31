"""Reportes por texto (G4): la orden también puede escribirse, no solo hablarse."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.application.reportes import CasosReportes, IniciarReporte
from app.application.reportes_texto import TranscriptorDeTexto
from app.infrastructure.graphql_analizador import AnalizadorGraphqlCore
from app.infrastructure.keycloak import JwtVerificador
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave
from tests.test_reportes import CONSULTA, MUTATION, ConstructorFalso, Mundo, RelojFijo


def _con_texto(mundo: Mundo) -> CasosReportes:
    """Los mismos casos de uso del mundo de pruebas, más el que recibe la orden ya escrita."""
    escrito = IniciarReporte(
        mundo.repositorio,
        TranscriptorDeTexto(),
        mundo.constructor,
        AnalizadorGraphqlCore(),
        mundo.gestion,
        RelojFijo(),
    )
    return CasosReportes(mundo.casos.iniciar, mundo.casos.generar, mundo.casos.obtener, escrito)


def _api(mundo: Mundo) -> TestClient:
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    return TestClient(create_app(verificador, casos_reportes=_con_texto(mundo)))


def _auth(crear_token: CrearToken, rol: str = "ADMIN") -> dict[str, str]:
    return {"Authorization": f"Bearer {crear_token(rol)}"}


def test_G4_el_transcriptor_de_texto_devuelve_la_orden_tal_cual_sin_espacios_sobrantes() -> None:
    assert (
        TranscriptorDeTexto().transcribir(b"  morosos del mes en pdf \n") == "morosos del mes en pdf"
    )


def test_G4_una_orden_escrita_arma_el_reporte_sin_pasar_por_Whisper(crear_token: CrearToken) -> None:
    mundo = Mundo()
    api = _api(mundo)

    r = api.post(
        "/reportes/voz", headers=_auth(crear_token), data={"instruccion": "Dame los morosos en Excel"}
    )

    assert r.status_code == 202
    cuerpo: dict[str, Any] = r.json()
    assert (
        cuerpo["transcripcion"] == "Dame los morosos en Excel"
    )  # la orden escrita, no la del transcriptor falso
    assert cuerpo["consultaGraphql"] == CONSULTA
    listo = api.get(f"/reportes/{cuerpo['id']}", headers=_auth(crear_token)).json()
    assert listo["estado"] == "LISTO" and listo["vistaPrevia"]["total"] == 2


def test_G4_la_orden_escrita_respeta_el_formato_pedido(crear_token: CrearToken) -> None:
    api = _api(Mundo())

    r = api.post(
        "/reportes/voz", headers=_auth(crear_token), data={"instruccion": "morosos", "formato": "PDF"}
    )

    assert r.status_code == 202 and r.json()["formato"] == "PDF"


@pytest.mark.parametrize("instruccion", ["", "   ", "x" * 1001])
def test_G4_una_orden_vacia_o_demasiado_larga_es_400(crear_token: CrearToken, instruccion: str) -> None:
    r = _api(Mundo()).post("/reportes/voz", headers=_auth(crear_token), data={"instruccion": instruccion})

    assert r.status_code == 400


def test_G4_hay_que_enviar_audio_o_instruccion_pero_no_ambos_ni_ninguno(crear_token: CrearToken) -> None:
    api = _api(Mundo())
    audio = {"audio": ("a.wav", b"audio", "audio/wav")}

    ninguno = api.post("/reportes/voz", headers=_auth(crear_token))
    ambos = api.post(
        "/reportes/voz", headers=_auth(crear_token), files=audio, data={"instruccion": "morosos"}
    )

    assert ninguno.status_code == 400 and ambos.status_code == 400


def test_G4_el_audio_sigue_funcionando_como_antes(crear_token: CrearToken) -> None:
    r = _api(Mundo()).post(
        "/reportes/voz", headers=_auth(crear_token), files={"audio": ("a.wav", b"audio", "audio/wav")}
    )

    assert r.status_code == 202
    assert r.json()["transcripcion"] == "Dame los morosos del mes"  # lo que dijo el transcriptor


def test_G4_una_orden_escrita_que_arma_una_mutation_se_rechaza_igual_que_la_hablada(
    crear_token: CrearToken,
) -> None:
    mundo = Mundo(constructor=ConstructorFalso(MUTATION))

    r = _api(mundo).post(
        "/reportes/voz", headers=_auth(crear_token), data={"instruccion": "borra todas las unidades"}
    )

    assert r.status_code == 422 and r.json()["code"] == "G4"
    assert mundo.gestion.ejecutadas == []


def test_G4_solo_el_ADMIN_puede_dar_ordenes_escritas(crear_token: CrearToken) -> None:
    api = _api(Mundo())

    for rol in ("RESIDENTE", "GUARDIA"):
        r = api.post("/reportes/voz", headers=_auth(crear_token, rol), data={"instruccion": "morosos"})
        assert r.status_code == 403
