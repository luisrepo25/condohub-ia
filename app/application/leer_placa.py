"""Caso de uso de lectura de placas (A4, I5): lee el texto de la placa de una imagen y guarda la predicción.

No decide el ingreso: el umbral de 0.85 y la comparación con las placas registradas los aplica Acceso.
La aplicación orquesta puertos; no conoce FastAPI, MongoDB ni ONNX.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.application.modelos import Reloj

PLACA_LEIDA = "PLACA_LEIDA"
SIN_PLACA = "SIN_PLACA"
LARGO_MINIMO = 4   # menos caracteres no es una placa (el formato A4 tiene 7; el lector global admite 3 a 10)


class LectorPlacasNoDisponible(Exception):
    """Los modelos de lectura de placas no están configurados o no se pudieron cargar."""


# ------------------------------------------------------------------ puertos

@dataclass(frozen=True)
class LecturaCruda:
    texto: str
    confianza: float   # 0..1
    caja: tuple[float, float, float, float]   # x1, y1, x2, y2 en píxeles de la imagen original


class LectorPlacas(Protocol):
    nombre: str
    version: str

    def leer(self, imagen: bytes) -> LecturaCruda | None:
        """Texto de la placa más clara de la imagen, o `None` si no hay ninguna.

        Lanza `ImagenInvalida` si la imagen no se abre y `LectorPlacasNoDisponible` si faltan los modelos.
        """
        ...


@dataclass(frozen=True)
class PrediccionPlaca:
    """I5: toda predicción guarda modelo, versión, confianza y fecha."""

    resultado: str
    texto: str | None
    caja: tuple[float, float, float, float] | None
    confianza: float
    modelo: str
    version_modelo: str
    camara: str | None
    latencia_ms: float
    fecha: datetime


class RepositorioPrediccionesPlacas(Protocol):
    def guardar(self, prediccion: PrediccionPlaca) -> str:
        """Guarda la predicción (I5) y devuelve su id."""
        ...


# ------------------------------------------------------------------ resultado

@dataclass(frozen=True)
class ResultadoLecturaPlaca:
    resultado: str          # PLACA_LEIDA o SIN_PLACA
    placa: str | None
    confianza: float        # 0..1 (I5)
    modelo: str
    version: str
    prediccion_id: str


# ------------------------------------------------------------------ caso de uso

class LeerPlaca:
    def __init__(
        self, lector: LectorPlacas, predicciones: RepositorioPrediccionesPlacas, reloj: Reloj
    ) -> None:
        self._lector, self._predicciones, self._reloj = lector, predicciones, reloj

    def ejecutar(self, imagen: bytes, camara: str | None = None) -> ResultadoLecturaPlaca:
        inicio = time.perf_counter()
        lectura = self._lector.leer(imagen)
        if lectura is None or len(lectura.texto) < LARGO_MINIMO:
            resultado, texto, caja, confianza = SIN_PLACA, None, None, 0.0
        else:
            resultado, texto, caja = PLACA_LEIDA, lectura.texto, lectura.caja
            confianza = min(1.0, max(0.0, lectura.confianza))
        prediccion_id = self._predicciones.guardar(
            PrediccionPlaca(
                resultado=resultado,
                texto=texto,
                caja=caja,
                confianza=confianza,
                modelo=self._lector.nombre,
                version_modelo=self._lector.version,
                camara=camara,
                latencia_ms=(time.perf_counter() - inicio) * 1000,
                fecha=self._reloj.ahora(),
            )
        )
        return ResultadoLecturaPlaca(
            resultado, texto, confianza, self._lector.nombre, self._lector.version, prediccion_id
        )
