"""Archivos de los reportes en un bucket privado; se entregan con una URL firmada y temporal."""

from datetime import timedelta

from app.application.modelos import Reloj
from app.infrastructure.s3_cliente import ClienteS3
from app.infrastructure.s3_firmador import FirmadorS3


class AlmacenReportesS3:
    def __init__(self, cliente: ClienteS3, firmador: FirmadorS3, bucket: str, reloj: Reloj) -> None:
        self._cliente, self._firmador, self._bucket, self._reloj = cliente, firmador, bucket, reloj
        self._bucket_listo = False

    def guardar(self, clave: str, datos: bytes, tipo: str) -> None:
        if not self._bucket_listo:
            self._cliente.crear_bucket(self._bucket)
            self._bucket_listo = True
        self._cliente.subir(self._bucket, clave, datos, tipo)

    def firmar_descarga(self, clave: str, vigencia: timedelta) -> str:
        return self._firmador.firmar_descarga(clave, vigencia, self._reloj.ahora())