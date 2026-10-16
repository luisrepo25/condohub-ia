"""Reconocimiento facial: consentimiento (I1), tres zonas (I2), desconocidos (I3) y predicciones (I5).

Dominio puro: sin FastAPI, MongoDB ni librerías de ML. Los vectores son tuplas de floats normalizadas (L2),
de modo que el producto punto es la similitud coseno.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from operator import mul
from uuid import UUID

from app.domain.errores import ReglaViolada

DIMENSION = 512
Embedding = tuple[float, ...]
_TOLERANCIA_RELOJ = timedelta(minutes=5)


def normalizar(vector: Sequence[float]) -> Embedding:
    """Vector de 512 dimensiones con norma 1."""
    if len(vector) != DIMENSION:
        raise ValueError(f"El embedding debe tener {DIMENSION} dimensiones")
    norma = math.sqrt(sum(x * x for x in vector))
    if norma == 0:
        raise ValueError("El embedding no puede ser nulo")
    return tuple(x / norma for x in vector)


def similitud(a: Embedding, b: Embedding) -> float:
    """Similitud coseno de dos vectores ya normalizados."""
    return float(sum(map(mul, a, b)))


class Zona(StrEnum):
    """I2: a dónde cae un rostro según su similitud con el perfil más parecido."""

    RECONOCIDO = "RECONOCIDO"
    DUDOSO = "DUDOSO"          # se muestra el candidato y decide el guardia; no se guarda como desconocido
    DESCONOCIDO = "DESCONOCIDO"


@dataclass(frozen=True)
class Umbrales:
    """Umbrales de I2 (parámetros de configuración, no constantes del negocio)."""

    alto: float
    bajo: float

    def __post_init__(self) -> None:
        if not 0 < self.bajo <= self.alto <= 1:
            raise ValueError("Se requiere 0 < bajo <= alto <= 1")

    def clasificar(self, similitud_: float) -> Zona:
        if similitud_ >= self.alto:
            return Zona.RECONOCIDO
        if similitud_ >= self.bajo:
            return Zona.DUDOSO
        return Zona.DESCONOCIDO


@dataclass(frozen=True)
class Consentimiento:
    """I1: fecha (UTC) y versión del texto que la persona aceptó."""

    otorgado_en: datetime
    version: str

    def __post_init__(self) -> None:
        if self.otorgado_en.tzinfo is None:
            raise ReglaViolada("I1", "La fecha del consentimiento debe llevar zona horaria (UTC)")
        if not self.version.strip():
            raise ReglaViolada("I1", "El consentimiento debe indicar la versión del texto aceptado")


@dataclass(frozen=True)
class EmbeddingRegistrado:
    vector: Embedding
    modelo: str
    version_modelo: str
    foto_key: str            # objeto en el almacén S3; la foto nunca vive en la base
    creado_en: datetime


class PerfilFacial:
    """Agregado: el perfil facial de un residente. Sin consentimiento vigente no hay embeddings (I1)."""

    def __init__(
        self,
        residente_id: UUID,
        residente_sub: str,
        unidad_id: UUID,
        consentimiento: Consentimiento,
        embeddings: list[EmbeddingRegistrado],
        creado_en: datetime,
        retirado_en: datetime | None = None,
    ) -> None:
        self.residente_id = residente_id
        self.residente_sub = residente_sub
        self.unidad_id = unidad_id
        self.consentimiento = consentimiento
        self.creado_en = creado_en
        self._embeddings = list(embeddings)
        self._retirado_en = retirado_en

    @classmethod
    def registrar(
        cls,
        residente_id: UUID,
        residente_sub: str,
        unidad_id: UUID,
        consentimiento: Consentimiento,
        embeddings: list[EmbeddingRegistrado],
        ahora: datetime,
    ) -> "PerfilFacial":
        if consentimiento.otorgado_en > ahora + _TOLERANCIA_RELOJ:
            raise ReglaViolada("I1", "La fecha del consentimiento no puede ser futura")
        if not residente_sub.strip():
            raise ValueError("El perfil necesita el sub de Keycloak del residente")
        if not embeddings:
            raise ValueError("Un perfil nace con al menos un embedding")
        return cls(residente_id, residente_sub, unidad_id, consentimiento, embeddings, ahora)

    @property
    def embeddings(self) -> list[EmbeddingRegistrado]:
        return list(self._embeddings)

    @property
    def retirado_en(self) -> datetime | None:
        return self._retirado_en

    @property
    def vigente(self) -> bool:
        return self._retirado_en is None

    def claves_fotos(self) -> list[str]:
        return [e.foto_key for e in self._embeddings]

    def agregar(self, nuevos: list[EmbeddingRegistrado]) -> None:
        if not self.vigente:
            raise ReglaViolada("I1", "El consentimiento fue retirado: hay que registrarlo de nuevo")
        self._embeddings.extend(nuevos)

    def retirar_consentimiento(self, ahora: datetime) -> None:
        """Borra los embeddings. Repetirlo no cambia nada (la primera fecha de retiro es la que vale)."""
        if self._retirado_en is None:
            self._retirado_en = ahora
            self._embeddings = []

    def similitud_con(self, vector: Embedding) -> float:
        """Similitud con la foto registrada más parecida; -1 si el perfil no tiene embeddings."""
        return max((similitud(e.vector, vector) for e in self._embeddings), default=-1.0)


@dataclass(frozen=True)
class Reconocimiento:
    zona: Zona
    residente_id: UUID | None     # solo en RECONOCIDO y DUDOSO
    similitud: float


class ReconocedorFacial:
    """Servicio de dominio de I2: decide la zona del rostro frente a los perfiles con consentimiento."""

    def __init__(self, umbrales: Umbrales) -> None:
        self.umbrales = umbrales

    def identificar(self, vector: Embedding, perfiles: Iterable[PerfilFacial]) -> Reconocimiento:
        mejor_id: UUID | None = None
        mejor = -1.0
        for perfil in perfiles:
            if not perfil.vigente or not perfil.embeddings:
                continue
            s = perfil.similitud_con(vector)
            if s > mejor:
                mejor_id, mejor = perfil.residente_id, s
        if mejor_id is None:
            return Reconocimiento(Zona.DESCONOCIDO, None, 0.0)
        zona = self.umbrales.clasificar(mejor)
        return Reconocimiento(zona, None if zona is Zona.DESCONOCIDO else mejor_id, mejor)


@dataclass(frozen=True)
class Desconocido:
    """I3: rostro sin coincidencia; se guarda para el agrupamiento nocturno y vence a los 30 días."""

    vector: Embedding
    modelo: str
    version_modelo: str
    camara: str | None
    capturado_en: datetime
    expira_en: datetime

    @classmethod
    def capturar(
        cls,
        vector: Embedding,
        modelo: str,
        version_modelo: str,
        camara: str | None,
        ahora: datetime,
        retencion: timedelta,
    ) -> "Desconocido":
        if retencion <= timedelta(0):
            raise ValueError("La retención debe ser positiva")
        return cls(vector, modelo, version_modelo, camara, ahora, ahora + retencion)

    @property
    def dia(self) -> str:
        """yyyy-mm-dd, para contar días distintos en I4."""
        return self.capturado_en.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Prediccion:
    """I5: toda predicción guarda modelo, versión, confianza y fecha."""

    tipo: str
    resultado: str
    residente_id: UUID | None
    confianza: float
    umbral_alto: float | None
    umbral_bajo: float | None
    modelo: str
    version_modelo: str
    camara: str | None
    latencia_ms: float
    fecha: datetime
