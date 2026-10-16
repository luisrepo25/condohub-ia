"""Repositorios de MongoDB (PyMongo síncrono). Reciben la colección ya abierta; no importan `pymongo`."""

from typing import Any
from uuid import UUID

from app.domain.rostros import Desconocido, PerfilFacial, Prediccion
from app.infrastructure.mongo.documentos import (
    desconocido_a_documento,
    documento_a_perfil,
    perfil_a_documento,
    prediccion_a_documento,
)


class RepositorioPerfilesMongo:
    """Colección `perfiles_faciales`: un documento por residente (índice único sobre `residenteId`)."""

    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def guardar(self, perfil: PerfilFacial) -> None:
        filtro = {"residenteId": str(perfil.residente_id)}
        self._c.replace_one(filtro, perfil_a_documento(perfil), upsert=True)

    def por_residente(self, residente_id: UUID) -> PerfilFacial | None:
        doc = self._c.find_one({"residenteId": str(residente_id)})
        return documento_a_perfil(doc) if doc else None

    def con_consentimiento_vigente(self) -> list[PerfilFacial]:
        return [documento_a_perfil(d) for d in self._c.find({"consentimiento.retiradoEn": None})]


class RepositorioDesconocidosMongo:
    """Colección `desconocidos`: con índice TTL sobre `expiraEn` (I3)."""

    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def guardar(self, desconocido: Desconocido) -> None:
        self._c.insert_one(desconocido_a_documento(desconocido))


class RepositorioPrediccionesMongo:
    """Colección `predicciones` (I5): devuelve el `_id` que asignó MongoDB como `prediccionId`."""

    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def guardar(self, prediccion: Prediccion) -> str:
        return str(self._c.insert_one(prediccion_a_documento(prediccion)).inserted_id)
