"""Repositorios de MongoDB con una colección falsa (el servidor real se prueba en integracion/)."""

from copy import deepcopy
from datetime import timedelta
from typing import Any
from uuid import UUID

from app.domain.rostros import (
    Consentimiento,
    Desconocido,
    EmbeddingRegistrado,
    PerfilFacial,
    Prediccion,
)
from app.infrastructure.mongo.repositorios import (
    RepositorioDesconocidosMongo,
    RepositorioPerfilesMongo,
    RepositorioPrediccionesMongo,
)
from tests.dobles import AHORA, ANA, BEA, UNIDAD, eje


class _Resultado:
    def __init__(self, inserted_id: object) -> None:
        self.inserted_id = inserted_id


class ColeccionFalsa:
    """Imita `replace_one`, `find_one`, `find` e `insert_one` con filtros de igualdad (y campos con punto)."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self._siguiente = 1

    @staticmethod
    def _valor(doc: dict[str, Any], ruta: str) -> Any:
        actual: Any = doc
        for parte in ruta.split("."):
            actual = actual.get(parte) if isinstance(actual, dict) else None
        return actual

    def _coincide(self, doc: dict[str, Any], filtro: dict[str, Any]) -> bool:
        return all(self._valor(doc, ruta) == esperado for ruta, esperado in filtro.items())

    def replace_one(self, filtro: dict[str, Any], doc: dict[str, Any], upsert: bool = False) -> None:
        for i, existente in enumerate(self.docs):
            if self._coincide(existente, filtro):
                self.docs[i] = deepcopy(doc)
                return
        if upsert:
            self.docs.append(deepcopy(doc))

    def find_one(self, filtro: dict[str, Any]) -> dict[str, Any] | None:
        return next((deepcopy(d) for d in self.docs if self._coincide(d, filtro)), None)

    def find(self, filtro: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [deepcopy(d) for d in self.docs if self._coincide(d, filtro or {})]

    def insert_one(self, doc: dict[str, Any]) -> _Resultado:
        self.docs.append(deepcopy(doc))
        self._siguiente += 1
        return _Resultado(f"id-{self._siguiente - 1}")


def _perfil(residente: UUID = ANA, sub: str = "sub-ana") -> PerfilFacial:
    return PerfilFacial.registrar(
        residente,
        sub,
        UNIDAD,
        Consentimiento(AHORA - timedelta(days=1), "v1"),
        [EmbeddingRegistrado(eje(0), "facenet-vggface2", "1.0.0", "rostros/f.jpg", AHORA)],
        AHORA,
    )


def test_el_perfil_guardado_se_recupera_por_residente() -> None:
    repo = RepositorioPerfilesMongo(ColeccionFalsa())
    repo.guardar(_perfil())
    p = repo.por_residente(ANA)
    assert p is not None and p.residente_sub == "sub-ana" and len(p.embeddings) == 1


def test_un_residente_sin_perfil_no_se_encuentra() -> None:
    assert RepositorioPerfilesMongo(ColeccionFalsa()).por_residente(ANA) is None


def test_guardar_dos_veces_el_mismo_residente_reemplaza_su_perfil_y_no_lo_duplica() -> None:
    coleccion = ColeccionFalsa()
    repo = RepositorioPerfilesMongo(coleccion)
    perfil = _perfil()
    repo.guardar(perfil)
    perfil.retirar_consentimiento(AHORA)
    repo.guardar(perfil)
    assert len(coleccion.docs) == 1
    recuperado = repo.por_residente(ANA)
    assert recuperado is not None and not recuperado.vigente and recuperado.embeddings == []


def test_I1_solo_los_perfiles_con_consentimiento_vigente_se_usan_para_reconocer() -> None:
    repo = RepositorioPerfilesMongo(ColeccionFalsa())
    vigente, retirado = _perfil(ANA, "sub-ana"), _perfil(BEA, "sub-bea")
    retirado.retirar_consentimiento(AHORA)
    repo.guardar(vigente)
    repo.guardar(retirado)
    assert [p.residente_id for p in repo.con_consentimiento_vigente()] == [ANA]


def test_I3_el_desconocido_se_inserta_con_su_vencimiento() -> None:
    coleccion = ColeccionFalsa()
    RepositorioDesconocidosMongo(coleccion).guardar(
        Desconocido.capturar(eje(2), "facenet-vggface2", "1.0.0", "porteria", AHORA, timedelta(days=30))
    )
    assert len(coleccion.docs) == 1 and coleccion.docs[0]["expiraEn"] == AHORA + timedelta(days=30)


def test_I5_la_prediccion_devuelve_el_id_que_asigno_mongo_como_texto() -> None:
    coleccion = ColeccionFalsa()
    repo = RepositorioPrediccionesMongo(coleccion)
    p = Prediccion("ROSTRO", "RECONOCIDO", ANA, 0.9, 0.7, 0.5, "facenet-vggface2", "1.0.0", None, 12.0, AHORA)
    assert repo.guardar(p) == "id-1" and repo.guardar(p) == "id-2"
    assert coleccion.docs[0]["resultado"] == {"zona": "RECONOCIDO", "residenteId": str(ANA)}
