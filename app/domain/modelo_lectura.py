"""Reconocedor de texto de placas publicado para correr fuera del servicio (A4, O6).

Dominio puro: sin FastAPI, S3 ni librerías de ML. Complementa `modelos.py` (el detector de placas).
"""

import re
from dataclasses import dataclass

from app.domain.modelos import ModeloInvalido

_SHA256 = re.compile(r"[0-9a-f]{64}")
FORMATOS_LECTURA = frozenset(
    {"onnx-fp32", "onnx-int8"}
)  # lo único que declara el contrato `ModeloLecturaPlacas`


def _sin_rutas(nombre: str) -> bool:
    return bool(nombre) and "/" not in nombre and "\\" not in nombre and ".." not in nombre


@dataclass(frozen=True)
class ModeloDeLectura:
    """Reconocedor ONNX y su ficha JSON (caracteres y preproceso) para la PWA. Valida al construirse."""

    nombre: str
    version: str
    formato: str
    archivo: str
    sha256: str
    ficha: str

    def __post_init__(self) -> None:
        if not self.nombre or not self.version or not self.archivo or not self.ficha:
            raise ModeloInvalido("El reconocedor necesita nombre, versión, archivo y ficha")
        if self.formato not in FORMATOS_LECTURA:
            raise ModeloInvalido(f"Formato no soportado por el contrato: {self.formato}")
        if not _SHA256.fullmatch(self.sha256):
            raise ModeloInvalido("El hash debe ser un SHA-256 completo en hexadecimal minúscula")
        if not _sin_rutas(self.archivo) or not _sin_rutas(self.ficha):
            raise ModeloInvalido("El archivo y la ficha no pueden incluir rutas")

    @property
    def clave_objeto(self) -> str:
        """Dónde vive el ONNX en el almacén (bucket `modelos`): `lectura-placas/<version>/<archivo>`."""
        return f"lectura-placas/{self.version}/{self.archivo}"

    @property
    def clave_ficha(self) -> str:
        """Dónde vive la ficha, junto al ONNX."""
        return f"lectura-placas/{self.version}/{self.ficha}"
