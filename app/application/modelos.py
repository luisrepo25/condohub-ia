"""Entrega del modelo de placas a la PWA del guardia (A4, O6)."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from app.domain.modelos import ModeloDePlacas


class ModeloNoDisponible(Exception):
    """No hay un modelo de placas publicado."""


class AlmacenNoDisponible(Exception):
    """El almacén de objetos no está configurado o no se puede firmar contra él."""


class Reloj(Protocol):
    def ahora(self) -> datetime:
        """Ahora, con zona horaria UTC."""
        ...


class CatalogoModelos(Protocol):
    def placas_activo(self) -> ModeloDePlacas | None:
        """El modelo de placas vigente, o `None` si no hay ninguno publicado."""
        ...


class FirmadorUrls(Protocol):
    def firmar_descarga(self, clave: str, vigencia: timedelta, ahora: datetime) -> str:
        """URL temporal de lectura del objeto `clave`. Lanza `AlmacenNoDisponible` si no se puede firmar."""
        ...


@dataclass(frozen=True)
class ModeloDescargable:
    modelo: str
    version: str
    formato: str
    url: str
    sha256: str
    expira_en: datetime


class ObtenerModeloPlacas:
    """Caso de uso de `GET /modelos/placas/actual`."""

    def __init__(
        self, catalogo: CatalogoModelos, firmador: FirmadorUrls, reloj: Reloj, vigencia: timedelta
    ) -> None:
        if vigencia <= timedelta(0):
            raise ValueError("La vigencia de la URL debe ser positiva")
        self._catalogo, self._firmador, self._reloj, self._vigencia = catalogo, firmador, reloj, vigencia

    def ejecutar(self) -> ModeloDescargable:
        modelo = self._catalogo.placas_activo()
        if modelo is None:
            raise ModeloNoDisponible("No hay un modelo de placas publicado")
        ahora = self._reloj.ahora()
        url = self._firmador.firmar_descarga(modelo.clave_objeto, self._vigencia, ahora)
        return ModeloDescargable(
            modelo.nombre, modelo.version, modelo.formato, url, modelo.sha256, ahora + self._vigencia
        )
