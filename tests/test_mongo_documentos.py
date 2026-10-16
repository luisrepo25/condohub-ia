"""Traducción dominio <-> documentos de MongoDB. Los campos salen del validador de `db/mongo-init.js`."""

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from app.domain.rostros import (
    Consentimiento,
    Desconocido,
    EmbeddingRegistrado,
    PerfilFacial,
    Prediccion,
)
from app.infrastructure.mongo.documentos import (
    desconocido_a_documento,
    documento_a_perfil,
    perfil_a_documento,
    prediccion_a_documento,
)
from tests.dobles import AHORA, ANA, UNIDAD, con_similitud, eje

ESQUEMA_JS = Path(__file__).resolve().parents[1] / "db" / "mongo-init.js"


def _perfil(retirado: bool = False) -> PerfilFacial:
    perfil = PerfilFacial.registrar(
        ANA,
        "sub-ana",
        UNIDAD,
        Consentimiento(AHORA - timedelta(days=1), "v1"),
        [
            EmbeddingRegistrado(eje(0), "facenet-vggface2", "1.0.0", "rostros/ana/a.jpg", AHORA),
            EmbeddingRegistrado(con_similitud(0.9), "facenet-vggface2", "1.0.0", "rostros/ana/b.jpg", AHORA),
        ],
        AHORA,
    )
    if retirado:
        perfil.retirar_consentimiento(AHORA + timedelta(hours=1))
    return perfil


def _claves_en_el_validador(documento: dict, texto_js: str) -> list[str]:
    """Claves de primer nivel del documento que el validador no menciona (señal de que se desalinearon)."""
    return [k for k in documento if k != "_id" and not re.search(rf"\b{re.escape(k)}\b", texto_js)]


# ------------------------------------------------------------------ perfiles


def test_I1_un_perfil_vigente_se_guarda_con_los_campos_del_esquema() -> None:
    d = perfil_a_documento(_perfil())
    assert d["residenteId"] == str(ANA) and d["residenteSub"] == "sub-ana" and d["unidadId"] == str(UNIDAD)
    assert d["consentimiento"] == {
        "otorgado": True,
        "fecha": AHORA - timedelta(days=1),
        "versionTexto": "v1",
        "retiradoEn": None,  # el validador exige null explícito mientras el consentimiento está vigente
    }
    assert len(d["embeddings"]) == 2
    e = d["embeddings"][0]
    assert e["modelo"] == {"nombre": "facenet-vggface2", "version": "1.0.0"}
    assert e["fotoKey"] == "rostros/ana/a.jpg" and e["creadoEn"] == AHORA and len(e["vector"]) == 512
    assert d["creadoEn"] == AHORA


def test_I1_un_perfil_retirado_se_guarda_sin_embeddings_y_con_la_fecha_de_retiro() -> None:
    d = perfil_a_documento(_perfil(retirado=True))
    assert d["embeddings"] == []
    assert d["consentimiento"]["retiradoEn"] == AHORA + timedelta(hours=1)


@pytest.mark.parametrize("retirado", [False, True])
def test_el_perfil_sobrevive_al_viaje_de_ida_y_vuelta(retirado: bool) -> None:
    original = _perfil(retirado)
    vuelto = documento_a_perfil(perfil_a_documento(original))
    assert (vuelto.residente_id, vuelto.residente_sub, vuelto.unidad_id) == (ANA, "sub-ana", UNIDAD)
    assert vuelto.consentimiento == original.consentimiento
    assert vuelto.embeddings == original.embeddings
    assert vuelto.retirado_en == original.retirado_en and vuelto.vigente == original.vigente
    assert vuelto.creado_en == original.creado_en


def test_los_vectores_vuelven_como_tuplas_de_floats() -> None:
    vuelto = documento_a_perfil(perfil_a_documento(_perfil()))
    assert isinstance(vuelto.embeddings[0].vector, tuple)
    assert all(isinstance(x, float) for x in vuelto.embeddings[0].vector)


def test_un_documento_sin_retiro_en_se_lee_como_vigente() -> None:
    d = perfil_a_documento(_perfil())
    d["consentimiento"].pop("retiradoEn")  # documentos viejos o escritos a mano
    assert documento_a_perfil(d).vigente


# ------------------------------------------------------------------ desconocidos


def test_I3_el_desconocido_lleva_su_vencimiento_para_el_indice_ttl() -> None:
    d = Desconocido.capturar(eje(3), "facenet-vggface2", "1.0.0", "porteria", AHORA, timedelta(days=30))
    doc = desconocido_a_documento(d)
    assert doc["expiraEn"] == AHORA + timedelta(days=30) and doc["capturadoEn"] == AHORA
    assert doc["dia"] == "2026-10-20" and doc["camara"] == "porteria"
    assert doc["modelo"] == {"nombre": "facenet-vggface2", "version": "1.0.0"} and len(doc["vector"]) == 512


def test_I3_sin_camara_el_campo_se_omite_en_lugar_de_guardar_null() -> None:
    d = Desconocido.capturar(eje(3), "facenet-vggface2", "1.0.0", None, AHORA, timedelta(days=30))
    assert "camara" not in desconocido_a_documento(d)


# ------------------------------------------------------------------ predicciones


def _prediccion(**cambios: object) -> Prediccion:
    datos: dict[str, object] = dict(
        tipo="ROSTRO",
        resultado="DUDOSO",
        residente_id=ANA,
        confianza=0.6,
        umbral_alto=0.7,
        umbral_bajo=0.5,
        modelo="facenet-vggface2",
        version_modelo="1.0.0",
        camara="porteria",
        latencia_ms=81.5,
        fecha=AHORA,
    )
    return Prediccion(**{**datos, **cambios})  # type: ignore[arg-type]


def test_I5_la_prediccion_guarda_modelo_version_confianza_y_fecha() -> None:
    d = prediccion_a_documento(_prediccion())
    assert d["tipo"] == "ROSTRO" and d["confianza"] == 0.6 and d["fecha"] == AHORA
    assert d["modelo"] == {"nombre": "facenet-vggface2", "version": "1.0.0"}
    assert d["resultado"] == {"zona": "DUDOSO", "residenteId": str(ANA)}
    assert (d["umbral"], d["umbralBajo"], d["latenciaMs"], d["camara"]) == (0.7, 0.5, 81.5, "porteria")


def test_I5_un_desconocido_o_sin_rostro_guarda_el_residente_como_null() -> None:
    d = prediccion_a_documento(_prediccion(resultado="DESCONOCIDO", residente_id=None, camara=None))
    assert d["resultado"] == {"zona": "DESCONOCIDO", "residenteId": None} and "camara" not in d


# ------------------------------------------------------------------ alineación con db/mongo-init.js


@pytest.mark.skipif(not ESQUEMA_JS.exists(), reason="repo abierto sin db/mongo-init.js")
def test_los_campos_que_se_escriben_existen_en_el_validador_de_mongo() -> None:
    js = ESQUEMA_JS.read_text(encoding="utf-8")
    documentos = [
        perfil_a_documento(_perfil()),
        desconocido_a_documento(Desconocido.capturar(eje(0), "m", "1", "c", AHORA, timedelta(days=30))),
        prediccion_a_documento(_prediccion()),
    ]
    for doc in documentos:
        assert _claves_en_el_validador(doc, js) == [], doc.keys()
        anidados = [v for v in doc.values() if isinstance(v, dict)]
        for sub in anidados:
            assert _claves_en_el_validador(sub, js) == []


def test_las_fechas_se_guardan_en_utc() -> None:
    d = perfil_a_documento(_perfil())
    assert d["creadoEn"].tzinfo is not None and d["creadoEn"].utcoffset() == timedelta(0)
    assert (
        isinstance(d["creadoEn"], datetime) and d["creadoEn"].tzinfo == UTC and UUID(d["residenteId"]) == ANA
    )
