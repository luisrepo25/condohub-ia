"""Predicciones de lectura de placas (I5): repositorios en memoria y en MongoDB (colección `predicciones`)."""

from typing import Any
from uuid import uuid4

from app.application.leer_placa import PrediccionPlaca


class RepositorioPrediccionesPlacasMemoria:
    def __init__(self) -> None:
        self.guardadas: list[tuple[str, PrediccionPlaca]] = []

    def guardar(self, prediccion: PrediccionPlaca) -> str:
        prediccion_id = str(uuid4())
        self.guardadas.append((prediccion_id, prediccion))
        return prediccion_id


def prediccion_placa_a_documento(p: PrediccionPlaca) -> dict[str, Any]:
    """Nombres del validador de `db/mongo-init.js`: tipo PLACA, resultado `{texto, bbox}`."""
    resultado: dict[str, Any] = {
        "texto": p.texto,
        "bbox": list(p.caja) if p.caja else None,
        "estado": p.resultado,
    }
    doc: dict[str, Any] = {
        "tipo": "PLACA",
        "resultado": resultado,
        "confianza": p.confianza,
        "modelo": {"nombre": p.modelo, "version": p.version_modelo},
        "latenciaMs": p.latencia_ms,
        "fecha": p.fecha,
    }
    if p.camara is not None:
        doc["camara"] = p.camara
    return doc


class RepositorioPrediccionesPlacasMongo:
    def __init__(self, coleccion: Any) -> None:
        self._c = coleccion

    def guardar(self, prediccion: PrediccionPlaca) -> str:
        return str(self._c.insert_one(prediccion_placa_a_documento(prediccion)).inserted_id)
