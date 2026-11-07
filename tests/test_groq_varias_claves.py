"""G4: varias claves de Groq (GROQ_API_KEY=k1,k2): si una se agota (429) o es rechazada (401), se usa otra."""

import httpx
import pytest

from app.application.reportes import ConstructorNoDisponible
from app.infrastructure.groq import ConstructorConsultaGroq

OK = httpx.Response(200, json={"choices": [{"message": {"content": "{ morosos { id } }"}}]})


def _constructor(clave: str, respuestas: dict[str, int], usadas: list[str]) -> ConstructorConsultaGroq:
    def manejador(p: httpx.Request) -> httpx.Response:
        usada = p.headers["Authorization"].removeprefix("Bearer ")
        usadas.append(usada)
        codigo = respuestas.get(usada, 200)
        return OK if codigo == 200 else httpx.Response(codigo)

    return ConstructorConsultaGroq(clave, "m", cliente=httpx.Client(transport=httpx.MockTransport(manejador)))


def test_G4_si_la_primera_clave_agota_su_limite_se_usa_la_segunda() -> None:
    usadas: list[str] = []

    consulta = _constructor("k1,k2", {"k1": 429}, usadas).construir("morosos", "esquema")

    assert "morosos" in consulta
    assert usadas == ["k1", "k2"]


def test_G4_la_clave_que_funciona_se_recuerda_para_las_llamadas_siguientes() -> None:
    usadas: list[str] = []
    constructor = _constructor("k1,k2", {"k1": 429}, usadas)

    constructor.construir("a", "e")
    constructor.construir("b", "e")

    assert usadas == ["k1", "k2", "k2"]


def test_G4_una_clave_rechazada_con_401_tambien_pasa_a_la_siguiente() -> None:
    usadas: list[str] = []

    _constructor("k1, k2", {"k1": 401}, usadas).construir("a", "e")

    assert usadas == ["k1", "k2"]


def test_G4_si_todas_fallan_el_constructor_no_esta_disponible() -> None:
    usadas: list[str] = []

    with pytest.raises(ConstructorNoDisponible):
        _constructor("k1,k2", {"k1": 429, "k2": 429}, usadas).construir("a", "e")

    assert usadas == ["k1", "k2"]


def test_G4_con_una_sola_clave_o_sin_ella_todo_sigue_igual() -> None:
    usadas: list[str] = []
    _constructor("k1", {}, usadas).construir("a", "e")
    assert usadas == ["k1"]

    with pytest.raises(ConstructorNoDisponible):
        _constructor("", {}, []).construir("a", "e")
