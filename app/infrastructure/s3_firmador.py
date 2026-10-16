from datetime import datetime, timedelta
from urllib.parse import urlsplit

from app.application.modelos import AlmacenNoDisponible
from app.infrastructure.sigv4 import url_firmada_get

_MAXIMO_S3_SEGUNDOS = 7 * 24 * 3600   # S3 no acepta URLs prefirmadas de más de 7 días


class FirmadorS3:
    """Firma URLs de descarga (estilo `endpoint/bucket/clave`) para el almacén de objetos del proyecto."""

    def __init__(self, endpoint: str, bucket: str, acceso: str, secreto: str, region: str) -> None:
        partes = urlsplit(endpoint)
        self._esquema, self._host = partes.scheme or "http", partes.netloc
        self._bucket, self._acceso, self._secreto, self._region = bucket, acceso, secreto, region

    def firmar_descarga(self, clave: str, vigencia: timedelta, ahora: datetime) -> str:
        if not self._acceso or not self._secreto:
            raise AlmacenNoDisponible("El almacén de modelos no tiene credenciales configuradas")
        segundos = int(vigencia.total_seconds())
        if not 0 < segundos <= _MAXIMO_S3_SEGUNDOS:
            raise ValueError("La vigencia debe estar entre 1 segundo y 7 días")
        return url_firmada_get(
            esquema=self._esquema,
            host=self._host,
            ruta=f"/{self._bucket}/{clave}",
            acceso=self._acceso,
            secreto=self._secreto,
            region=self._region,
            ahora=ahora,
            vigencia_s=segundos,
        )
