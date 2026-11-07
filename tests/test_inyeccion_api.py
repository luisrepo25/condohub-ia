"""Evidencia repetible contra inyección (SQL y de operadores de MongoDB) por la API real del servicio IA.

Esperado: nunca un 5xx; un identificador con un ataque se rechaza por su formato (UUID, 4xx) antes de llegar
a la base; una instrucción o un formato con operadores se rechaza.
Los archivos Excel y PDF de los reportes se cubren en `test_reportes_archivos_seguridad.py`.
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import CrearToken

ATAQUES = [
    "' OR '1'='1",
    "'; DROP TABLE rostros; --",
    "x' UNION SELECT usename FROM pg_user --",
    '{"$gt": ""}',
    '" OR 1=1 --',
]


def _admin(crear_token: CrearToken) -> dict[str, str]:
    return {"Authorization": f"Bearer {crear_token('ADMIN')}"}


@pytest.mark.parametrize("ataque", ATAQUES)
def test_un_ataque_en_el_identificador_de_un_reporte_se_rechaza_por_su_formato(
    client: TestClient, crear_token: CrearToken, ataque: str
) -> None:
    r = client.get(f"/reportes/{ataque}", headers=_admin(crear_token))

    assert 400 <= r.status_code < 500


@pytest.mark.parametrize("ataque", ATAQUES)
def test_un_ataque_en_el_identificador_de_un_residente_se_rechaza_o_no_encuentra_nada(
    client: TestClient, crear_token: CrearToken, ataque: str
) -> None:
    r = client.delete(f"/rostros/{ataque}", headers=_admin(crear_token))

    assert 400 <= r.status_code < 500


def test_un_objeto_con_operadores_en_la_instruccion_o_el_formato_se_rechaza(
    client: TestClient, crear_token: CrearToken
) -> None:
    h = _admin(crear_token)

    for datos in (
        {"instruccion": '{"$gt": ""}', "formato": "'; DROP TABLE reportes; --"},
        {"instruccion": "morosos", "formato": '{"$ne": null}'},
    ):
        r = client.post("/reportes/voz", headers=h, data=datos)
        assert 400 <= r.status_code < 500
