"""Ayudas de prueba para rostros: vectores con similitud exacta y un modelo facial falso (sin PyTorch)."""

import math
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.application.rostros import (
    CasosRostros,
    IdentificarRostro,
    ImagenInvalida,
    RegistrarRostro,
    RetirarConsentimiento,
)
from app.domain.rostros import DIMENSION, Embedding, ReconocedorFacial, Umbrales
from app.infrastructure.memoria import (
    AlmacenFotosMemoria,
    RepositorioDesconocidosMemoria,
    RepositorioPerfilesMemoria,
    RepositorioPrediccionesMemoria,
)

AHORA = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
FECHA = AHORA - timedelta(days=1)
ANA, BEA, UNIDAD = UUID(int=1), UUID(int=2), UUID(int=99)


class RelojFijo:
    """Reloj controlable: el tiempo solo avanza cuando el test lo pide."""

    def __init__(self, ahora: datetime = AHORA) -> None:
        self._ahora = ahora

    def ahora(self) -> datetime:
        return self._ahora

    def avanzar(self, delta: timedelta) -> None:
        self._ahora += delta


def eje(i: int) -> Embedding:
    """Vector unitario sobre el eje `i`: dos ejes distintos tienen similitud 0."""
    return tuple(1.0 if k == i else 0.0 for k in range(DIMENSION))


def con_similitud(s: float, base: int = 0, otro: int = 1) -> Embedding:
    """Vector unitario cuya similitud coseno con `eje(base)` es exactamente `s`."""
    v = [0.0] * DIMENSION
    v[base], v[otro] = s, math.sqrt(1 - s * s)
    return tuple(v)


class ModeloFalso:
    """Vector asociado a los bytes de la imagen; `None` = sin rostro; bytes desconocidos = ilegible."""

    nombre = "facenet-falso"
    version = "0.0.1"

    def __init__(self, vectores: dict[bytes, Embedding | None]) -> None:
        self._vectores = vectores
        self.llamadas = 0

    def embeber(self, imagen: bytes) -> Embedding | None:
        self.llamadas += 1
        if imagen not in self._vectores:
            raise ImagenInvalida("No se pudo leer la imagen")
        return self._vectores[imagen]


VECTORES: dict[bytes, Embedding | None] = {
    b"ana-1": eje(0),
    b"ana-2": eje(0),
    b"ana-otra": con_similitud(0.95),  # muy parecida a Ana
    b"parecido": con_similitud(0.60),  # entre 0.50 y 0.70: DUDOSO
    b"extrano": eje(9),  # sin parecido
    b"opuesto": tuple(-x for x in eje(0)),
    b"sin-rostro": None,
}


class Mundo:
    """Los casos de uso de rostros conectados a dobles en memoria, con umbrales 0.70 y 0.50 (I2)."""

    def __init__(self) -> None:
        self.modelo = ModeloFalso(dict(VECTORES))
        self.perfiles = RepositorioPerfilesMemoria()
        self.desconocidos = RepositorioDesconocidosMemoria()
        self.predicciones = RepositorioPrediccionesMemoria()
        self.fotos = AlmacenFotosMemoria()
        self.reloj = RelojFijo()
        self.registrar = RegistrarRostro(self.modelo, self.perfiles, self.fotos, self.reloj)
        self.identificar = IdentificarRostro(
            self.modelo,
            ReconocedorFacial(Umbrales(0.70, 0.50)),
            self.perfiles,
            self.desconocidos,
            self.predicciones,
            self.reloj,
            retencion_desconocidos=timedelta(days=30),
        )
        self.retirar = RetirarConsentimiento(self.perfiles, self.fotos, self.reloj, plazo=timedelta(hours=24))
        self.casos = CasosRostros(self.registrar, self.identificar, self.retirar)

    def registrar_a_ana(self, fotos: list[bytes] | None = None, version: str = "v1") -> None:
        self.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, version, fotos or [b"ana-1", b"ana-2"])
