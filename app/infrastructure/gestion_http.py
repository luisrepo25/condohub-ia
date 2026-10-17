"""Cliente de Gestión (GraphQL por HTTP).

Reenvía el token del ADMIN que pidió el reporte; Gestión lo valida de nuevo."""

from typing import Any

import httpx
from graphql import build_client_schema, get_introspection_query, print_schema

from app.application.reportes import GestionNoDisponible


class GestionHttp:
    def __init__(self, url: str, cliente: httpx.Client | None = None, tiempo_s: float = 20.0) -> None:
        self._url = url
        self._cliente = cliente or httpx.Client(timeout=tiempo_s)

    def _post(self, consulta: str, token: str) -> dict[str, Any]:
        try:
            r = self._cliente.post(
                self._url, json={"query": consulta}, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as e:
            raise GestionNoDisponible("No se pudo conectar con Gestión") from e
        if r.status_code != 200:
            raise GestionNoDisponible(f"Gestión respondió HTTP {r.status_code}")
        cuerpo: dict[str, Any] = r.json()
        if cuerpo.get("errors"):
            raise GestionNoDisponible(str(cuerpo["errors"][0].get("message", "Error de Gestión")))
        datos = cuerpo.get("data")
        if not isinstance(datos, dict):
            raise GestionNoDisponible("Gestión no devolvió datos")
        return datos

    def esquema(self, token: str) -> str:
        return print_schema(build_client_schema(self._post(get_introspection_query(), token)))  # type: ignore[arg-type]

    def ejecutar(self, consulta: str, token: str) -> dict[str, Any]:
        return self._post(consulta, token)