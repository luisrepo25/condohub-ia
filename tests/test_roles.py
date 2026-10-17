"""Matriz rol × endpoint. Permitido ⇒ llega a la lógica (hoy 501); no permitido ⇒ 403; sin token ⇒ 401."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import CrearToken

UUID_X = "11111111-1111-4111-8111-111111111111"
IMAGEN = {"imagen": ("a.jpg", b"x", "image/jpeg")}


@dataclass(frozen=True)
class Endpoint:
    nombre: str
    metodo: str
    ruta: str
    permitidos: frozenset[str]
    kwargs: dict[str, Any] = field(default_factory=dict)
    implementado: bool = False   # los implementados ya no responden 501; sigue valiendo 401/403 por rol


ENDPOINTS = [
    Endpoint(
        "A5_identificar_rostro",
        "POST",
        "/rostros/identificar",
        frozenset({"GUARDIA", "ADMIN"}),
        {"files": IMAGEN},
        implementado=True,
    ),
    Endpoint(
        "I1_registrar_rostro",
        "POST",
        "/rostros/registrar",
        frozenset({"ADMIN"}),
        {
            "data": {
                "residenteId": UUID_X,
                "unidadId": UUID_X,
                "consentimientoFecha": "2026-10-20T10:00:00Z",
                "consentimientoVersion": "v1",
            },
            "files": [("fotos", ("a.jpg", b"x", "image/jpeg"))],
        },
        implementado=True,
    ),
    Endpoint(
        "I1_retirar_consentimiento",
        "DELETE",
        f"/rostros/{UUID_X}",
        frozenset({"ADMIN", "RESIDENTE"}),
        implementado=True,
    ),
    Endpoint("A4_leer_placa", "POST", "/placas/leer", frozenset({"GUARDIA", "ADMIN"}), {"files": IMAGEN}),
    Endpoint(
        "A4_modelo_placas",
        "GET",
        "/modelos/placas/actual",
        frozenset({"GUARDIA", "ADMIN"}),
        implementado=True,
    ),
    Endpoint(
        "G4_crear_reporte_por_voz",
        "POST",
        "/reportes/voz",
        frozenset({"ADMIN"}),
        {"files": {"audio": ("a.wav", b"", "audio/wav")}},   # audio vacío: da 400 antes de cargar Whisper
        implementado=True,
    ),
    Endpoint("G4_obtener_reporte", "GET", f"/reportes/{UUID_X}", frozenset({"ADMIN"}), implementado=True),
]
ROLES = ("ADMIN", "RESIDENTE", "GUARDIA")


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=lambda e: e.nombre)
def test_sin_token_es_401(client: TestClient, endpoint: Endpoint) -> None:
    r = client.request(endpoint.metodo, endpoint.ruta, **endpoint.kwargs)
    assert r.status_code == 401


@pytest.mark.parametrize("rol", ROLES)
@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=lambda e: e.nombre)
def test_cada_rol_solo_usa_lo_que_le_toca(
    client: TestClient, crear_token: CrearToken, endpoint: Endpoint, rol: str
) -> None:
    r = client.request(
        endpoint.metodo,
        endpoint.ruta,
        headers={"Authorization": f"Bearer {crear_token(rol)}"},
        **endpoint.kwargs,
    )
    if rol in endpoint.permitidos and endpoint.implementado:
        assert r.status_code not in (401, 403), f"{rol} en {endpoint.nombre}"
    else:
        esperado = 501 if rol in endpoint.permitidos else 403
        assert r.status_code == esperado, f"{rol} en {endpoint.nombre}"
    if r.status_code >= 400:
        assert r.headers["content-type"].startswith("application/problem+json")


def test_los_501_se_identifican_como_no_implementado(client: TestClient, crear_token: CrearToken) -> None:
    cabecera = {"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    r = client.post("/placas/leer", headers=cabecera, files=IMAGEN)
    assert r.json()["code"] == "NO_IMPLEMENTADO"


def test_solicitud_mal_formada_es_400_para_quien_tiene_permiso(
    client: TestClient, crear_token: CrearToken
) -> None:
    r = client.post("/placas/leer", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/problem+json")
