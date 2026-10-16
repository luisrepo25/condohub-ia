"""Modelos de IA publicados para correr fuera del servicio.

Dominio puro: sin FastAPI, S3 ni librerías de ML.
"""

import re
from dataclasses import dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}")
FORMATOS_PLACAS = frozenset({"onnx-int8"})   # lo único que declara el contrato `ModeloPlacas`


class ModeloInvalido(ValueError):
    """La ficha del modelo no cumple lo que la PWA necesita para descargarlo y verificarlo."""


@dataclass(frozen=True)
class ModeloDePlacas:
    """Detector de placas listo para el dispositivo del guardia (O6). Valida al construirse."""

    nombre: str
    version: str
    formato: str
    archivo: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.nombre or not self.version or not self.archivo:
            raise ModeloInvalido("El modelo necesita nombre, versión y archivo")
        if self.formato not in FORMATOS_PLACAS:
            raise ModeloInvalido(f"Formato no soportado por el contrato: {self.formato}")
        if not _SHA256.fullmatch(self.sha256):
            raise ModeloInvalido("El hash debe ser un SHA-256 completo en hexadecimal minúscula")
        if "/" in self.archivo or "\\" in self.archivo or ".." in self.archivo:
            raise ModeloInvalido("El archivo no puede incluir rutas")

    @property
    def clave_objeto(self) -> str:
        """Dónde vive en el almacén de objetos: `modelos/placas/<version>/<archivo>` (bucket `modelos`)."""
        return f"placas/{self.version}/{self.archivo}"
