"""Traducción entre el dominio y los documentos de MongoDB.

Los nombres de campo son los del validador `$jsonSchema` de `db/mongo-init.js`.
Las fechas son UTC con zona horaria
(el cliente se abre con `tz_aware=True`).
"""

from typing import Any
from uuid import UUID

from app.domain.rostros import (
    Consentimiento,
    Desconocido,
    EmbeddingRegistrado,
    PerfilFacial,
    Prediccion,
)


def _modelo(nombre: str, version: str) -> dict[str, str]:
    return {"nombre": nombre, "version": version}


def perfil_a_documento(perfil: PerfilFacial) -> dict[str, Any]:
    return {
        "residenteId": str(perfil.residente_id),
        "residenteSub": perfil.residente_sub,
        "unidadId": str(perfil.unidad_id),
        "consentimiento": {
            "otorgado": True,
            "fecha": perfil.consentimiento.otorgado_en,
            "versionTexto": perfil.consentimiento.version,
            "retiradoEn": perfil.retirado_en,    # null mientras esté vigente: lo exige el validador (I1)
        },
        "embeddings": [
            {
                "vector": list(e.vector),
                "modelo": _modelo(e.modelo, e.version_modelo),
                "fotoKey": e.foto_key,
                "creadoEn": e.creado_en,
            }
            for e in perfil.embeddings
        ],
        "creadoEn": perfil.creado_en,
    }


def documento_a_perfil(doc: dict[str, Any]) -> PerfilFacial:
    c = doc["consentimiento"]
    return PerfilFacial(
        residente_id=UUID(doc["residenteId"]),
        residente_sub=doc["residenteSub"],
        unidad_id=UUID(doc["unidadId"]),
        consentimiento=Consentimiento(c["fecha"], c["versionTexto"]),
        embeddings=[
            EmbeddingRegistrado(
                tuple(float(x) for x in e["vector"]),
                e["modelo"]["nombre"],
                e["modelo"]["version"],
                e["fotoKey"],
                e["creadoEn"],
            )
            for e in doc["embeddings"]
        ],
        creado_en=doc["creadoEn"],
        retirado_en=c.get("retiradoEn"),
    )


def desconocido_a_documento(d: Desconocido) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "vector": list(d.vector),
        "modelo": _modelo(d.modelo, d.version_modelo),
        "capturadoEn": d.capturado_en,
        "dia": d.dia,                  # para contar días distintos en I4
        "expiraEn": d.expira_en,       # el índice TTL de MongoDB lo borra a los 30 días (I3)
    }
    if d.camara is not None:
        doc["camara"] = d.camara
    return doc


def prediccion_a_documento(p: Prediccion) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "tipo": p.tipo,
        "resultado": {"zona": p.resultado, "residenteId": str(p.residente_id) if p.residente_id else None},
        "confianza": p.confianza,
        "modelo": _modelo(p.modelo, p.version_modelo),
        "latenciaMs": p.latencia_ms,
        "fecha": p.fecha,
    }
    if p.umbral_alto is not None:
        doc["umbral"] = p.umbral_alto
    if p.umbral_bajo is not None:
        doc["umbralBajo"] = p.umbral_bajo
    if p.camara is not None:
        doc["camara"] = p.camara
    return doc
