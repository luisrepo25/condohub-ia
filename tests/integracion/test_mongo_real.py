"""Repositorios contra un MongoDB REAL, con el validador $jsonSchema y los índices de `db/mongo-init.js`.

Se omiten solas si no hay servidor o si la base de pruebas no se inicializó. Para correrlas:

    docker compose -f infra/docker-compose.yml up -d mongo
    docker compose -f infra/docker-compose.yml exec -T -e IA_DB=ia_pruebas mongo mongosh --quiet \
        /docker-entrypoint-initdb.d/01-ia.js        # el script montado por el compose, sin instalar mongosh
    .venv\\Scripts\\python -m pytest tests/integracion -v

Variables opcionales: `MONGO_URI_PRUEBAS` (mongodb://localhost:27017) y `MONGO_DB_PRUEBAS` (ia_pruebas).
"""

from datetime import timedelta
from typing import Any

import pytest

pymongo = pytest.importorskip("pymongo")

from app.domain.rostros import (  # noqa: E402
    Consentimiento,
    Desconocido,
    EmbeddingRegistrado,
    PerfilFacial,
    Prediccion,
)
from app.infrastructure.mongo.documentos import perfil_a_documento  # noqa: E402
from app.infrastructure.mongo.repositorios import (  # noqa: E402
    RepositorioDesconocidosMongo,
    RepositorioPerfilesMongo,
    RepositorioPrediccionesMongo,
)
from tests.dobles import AHORA, ANA, BEA, UNIDAD, eje  # noqa: E402

pytestmark = pytest.mark.integracion


def _perfil(residente=ANA, sub: str = "sub-ana") -> PerfilFacial:  # type: ignore[no-untyped-def]
    return PerfilFacial.registrar(
        residente,
        sub,
        UNIDAD,
        Consentimiento(AHORA - timedelta(days=1), "v1"),
        [EmbeddingRegistrado(eje(0), "facenet-vggface2", "1.0.0", "rostros/f.jpg", AHORA)],
        AHORA,
    )


def test_el_validador_acepta_un_perfil_vigente_y_uno_retirado(base: Any) -> None:
    repo = RepositorioPerfilesMongo(base["perfiles_faciales"])
    perfil = _perfil()
    repo.guardar(perfil)
    perfil.retirar_consentimiento(AHORA)
    repo.guardar(perfil)
    recuperado = repo.por_residente(ANA)
    assert recuperado is not None and not recuperado.vigente and recuperado.embeddings == []
    assert base["perfiles_faciales"].count_documents({}) == 1


def test_I1_el_validador_rechaza_un_perfil_retirado_que_conserva_embeddings(base: Any) -> None:
    perfil = _perfil()
    doc = perfil_a_documento(perfil)
    doc["consentimiento"]["retiradoEn"] = AHORA  # retirado pero con embeddings: viola I1
    with pytest.raises(pymongo.errors.WriteError):
        base["perfiles_faciales"].insert_one(doc)


def test_el_indice_unico_impide_dos_perfiles_del_mismo_residente(base: Any) -> None:
    base["perfiles_faciales"].insert_one(perfil_a_documento(_perfil()))
    with pytest.raises(pymongo.errors.DuplicateKeyError):
        base["perfiles_faciales"].insert_one(perfil_a_documento(_perfil(ANA, "otro-sub")))


def test_I1_solo_se_leen_los_perfiles_con_consentimiento_vigente(base: Any) -> None:
    repo = RepositorioPerfilesMongo(base["perfiles_faciales"])
    retirado = _perfil(BEA, "sub-bea")
    retirado.retirar_consentimiento(AHORA)
    repo.guardar(_perfil())
    repo.guardar(retirado)
    assert [p.residente_id for p in repo.con_consentimiento_vigente()] == [ANA]


def test_I3_el_desconocido_se_acepta_con_o_sin_camara_y_el_indice_ttl_existe(base: Any) -> None:
    repo = RepositorioDesconocidosMongo(base["desconocidos"])
    repo.guardar(
        Desconocido.capturar(eje(1), "facenet-vggface2", "1.0.0", "porteria", AHORA, timedelta(days=30))
    )
    repo.guardar(Desconocido.capturar(eje(2), "facenet-vggface2", "1.0.0", None, AHORA, timedelta(days=30)))
    assert base["desconocidos"].count_documents({}) == 2
    ttl = [i for i in base["desconocidos"].index_information().values() if i.get("expireAfterSeconds") == 0]
    assert ttl and ttl[0]["key"] == [("expiraEn", 1)]


def test_I5_la_prediccion_se_acepta_y_devuelve_su_id(base: Any) -> None:
    repo = RepositorioPrediccionesMongo(base["predicciones"])
    prediccion_id = repo.guardar(
        Prediccion(
            "ROSTRO", "DUDOSO", ANA, 0.6, 0.7, 0.5, "facenet-vggface2", "1.0.0", "porteria", 81.0, AHORA
        )
    )
    assert base["predicciones"].count_documents({}) == 1 and len(prediccion_id) == 24  # ObjectId en texto
