from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.application.visitantes import ListarVisitantesRecurrentes, VisitanteRecurrente
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.mongo.visitantes import LectorVisitantesMongo, LectorVisitantesVacio
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave

AHORA = datetime(2026, 10, 22, 1, 50, tzinfo=UTC)


def _visitante(cluster_id: str, apariciones: int = 5) -> VisitanteRecurrente:
    return VisitanteRecurrente(cluster_id, apariciones, 1, AHORA, AHORA, AHORA)


class LectorFalso:
    def __init__(self, visitantes: list[VisitanteRecurrente]) -> None:
        self.visitantes = visitantes
        self.limite_pedido = 0

    def con_aviso(self, limite: int) -> list[VisitanteRecurrente]:
        self.limite_pedido = limite
        return self.visitantes


class ColeccionFalsa:
    """Imita `find().sort().limit()` de PyMongo."""

    def __init__(self, documentos: list[dict[str, object]]) -> None:
        self.documentos = documentos
        self.filtro: dict[str, object] = {}

    def find(self, filtro: dict[str, object]) -> "ColeccionFalsa":
        self.filtro = filtro
        return self

    def sort(self, _campo: str, _orden: int) -> "ColeccionFalsa":
        return self

    def limit(self, n: int) -> list[dict[str, object]]:
        return self.documentos[:n]


def _cliente(lector: LectorFalso) -> TestClient:
    verificador = JwtVerificador(ISSUER, lambda _token: _clave.public_key())
    return TestClient(create_app(verificador, listar_visitantes=ListarVisitantesRecurrentes(lector)))


def test_I4_el_caso_de_uso_pide_los_visitantes_con_aviso() -> None:
    lector = LectorFalso([_visitante("v-1")])
    assert ListarVisitantesRecurrentes(lector).ejecutar(7) == [_visitante("v-1")]
    assert lector.limite_pedido == 7


def test_I4_guardia_y_admin_ven_los_visitantes_sin_embeddings(crear_token: CrearToken) -> None:
    cliente = _cliente(LectorFalso([_visitante("v-1", 10)]))
    for rol in ("GUARDIA", "ADMIN"):
        cabeceras = {"Authorization": f"Bearer {crear_token(rol)}"}
        r = cliente.get("/rostros/visitantes-recurrentes", headers=cabeceras)
        assert r.status_code == 200
        assert r.json() == [
            {
                "clusterId": "v-1",
                "apariciones": 10,
                "diasDistintos": 1,
                "primeraVez": "2026-10-22T01:50:00Z",
                "ultimaVez": "2026-10-22T01:50:00Z",
                "alertadoEn": "2026-10-22T01:50:00Z",
            }
        ]


def test_I4_un_residente_no_ve_los_visitantes_y_sin_token_es_401(crear_token: CrearToken) -> None:
    cliente = _cliente(LectorFalso([]))
    assert cliente.get("/rostros/visitantes-recurrentes").status_code == 401
    cabeceras = {"Authorization": f"Bearer {crear_token('RESIDENTE')}"}
    r = cliente.get("/rostros/visitantes-recurrentes", headers=cabeceras)
    assert r.status_code == 403


def test_I4_el_limite_se_valida(crear_token: CrearToken) -> None:
    cliente = _cliente(LectorFalso([]))
    cabeceras = {"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    r = cliente.get("/rostros/visitantes-recurrentes?limite=0", headers=cabeceras)
    assert r.status_code == 400


def test_I4_mongo_solo_lee_los_visitantes_ya_avisados() -> None:
    coleccion = ColeccionFalsa(
        [
            {
                "clusterId": "v-2",
                "apariciones": 4,
                "diasDistintos": 3,
                "primeraVez": AHORA,
                "ultimaVez": AHORA,
                "alertadoEn": AHORA,
            }
        ]
    )
    visitantes = LectorVisitantesMongo(coleccion).con_aviso(20)
    assert coleccion.filtro == {"alertado": True}
    assert visitantes == [VisitanteRecurrente("v-2", 4, 3, AHORA, AHORA, AHORA)]


def test_sin_persistencia_no_hay_visitantes() -> None:
    assert LectorVisitantesVacio().con_aviso(20) == []
