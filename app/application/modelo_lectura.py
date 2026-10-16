"""Entrega del reconocedor de texto de placas a la PWA del guardia (A4, O6)."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from app.application.modelos import FirmadorUrls, ModeloNoDisponible, Reloj
from app.domain.modelo_lectura import ModeloDeLectura


class CatalogoLectura(Protocol):
    def lectura_activo(self) -> ModeloDeLectura | None:
        """El reconocedor vigente, o `None` si no hay ninguno publicado."""
        ...


@dataclass(frozen=True)
class ModeloLecturaDescargable:
    modelo: str
    version: str
    formato: str
    url: str
    sha256: str
    ficha_url: str
    expira_en: datetime


class ObtenerModeloLectura:
    """Caso de uso de `GET /modelos/lectura-placas/actual`."""

    def __init__(
        self, catalogo: CatalogoLectura, firmador: FirmadorUrls, reloj: Reloj, vigencia: timedelta
    ) -> None:
        if vigencia <= timedelta(0):
            raise ValueError("La vigencia de la URL debe ser positiva")
        self._catalogo, self._firmador, self._reloj, self._vigencia = catalogo, firmador, reloj, vigencia

    def ejecutar(self) -> ModeloLecturaDescargable:
        modelo = self._catalogo.lectura_activo()
        if modelo is None:
            raise ModeloNoDisponible("No hay un reconocedor de placas publicado")
        ahora = self._reloj.ahora()
        return ModeloLecturaDescargable(
            modelo=modelo.nombre,
            version=modelo.version,
            formato=modelo.formato,
            url=self._firmador.firmar_descarga(modelo.clave_objeto, self._vigencia, ahora),
            sha256=modelo.sha256,
            ficha_url=self._firmador.firmar_descarga(modelo.clave_ficha, self._vigencia, ahora),
            expira_en=ahora + self._vigencia,
        )
