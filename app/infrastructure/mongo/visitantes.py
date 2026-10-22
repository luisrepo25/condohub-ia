"""Lectura de `clusters_desconocidos`: los visitantes que ya tienen aviso (I4)."""

from typing import Any

from app.application.visitantes import VisitanteRecurrente


class LectorVisitantesMongo:
    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def con_aviso(self, limite: int) -> list[VisitanteRecurrente]:
        documentos = self._c.find({"alertado": True}).sort("alertadoEn", -1).limit(limite)
        return [
            VisitanteRecurrente(
                cluster_id=d["clusterId"],
                apariciones=d["apariciones"],
                dias_distintos=d["diasDistintos"],
                primera_vez=d["primeraVez"],
                ultima_vez=d["ultimaVez"],
                alertado_en=d.get("alertadoEn"),
            )
            for d in documentos
        ]


class LectorVisitantesVacio:
    """Con `PERSISTENCIA=memoria` no hay agrupamiento nocturno: no hay visitantes que mostrar."""

    def con_aviso(self, limite: int) -> list[VisitanteRecurrente]:
        return []
