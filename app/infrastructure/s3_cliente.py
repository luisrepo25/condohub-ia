"""Cliente mínimo del almacén de objetos S3 (SeaweedFS en desarrollo) con URLs prefirmadas.

Solo usa la biblioteca estándar.
Las fotos de los rostros y los pesos de los modelos viven aquí y nunca en git ni en las bases (I6).
"""

import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from app.application.modelos import AlmacenNoDisponible
from app.infrastructure.sigv4 import url_firmada

_SEGUNDOS_DE_FIRMA = 300


class ClienteS3:
    """Crea buckets y sube, descarga y borra objetos. `abrir` y `ahora` se inyectan para probarlo sin red."""

    def __init__(
        self,
        endpoint: str,
        acceso: str,
        secreto: str,
        region: str = "us-east-1",
        abrir: Callable[..., Any] | None = None,
        ahora: Callable[[], datetime] | None = None,
    ) -> None:
        partes = urlsplit(endpoint)
        self._esquema, self._host = partes.scheme or "http", partes.netloc
        self._acceso, self._secreto, self._region = acceso, secreto, region
        self._abrir = abrir or urllib.request.urlopen
        self._ahora = ahora or (lambda: datetime.now(UTC))

    def _llamar(
        self,
        metodo: str,
        ruta: str,
        cuerpo: bytes | None = None,
        tipo: str | None = None,
        aceptar: tuple[int, ...] = (200, 204),
    ) -> tuple[int, bytes]:
        if not self._acceso or not self._secreto:
            raise AlmacenNoDisponible("El almacén de objetos no tiene credenciales configuradas")
        url = url_firmada(
            metodo=metodo,
            esquema=self._esquema,
            host=self._host,
            ruta=ruta,
            acceso=self._acceso,
            secreto=self._secreto,
            region=self._region,
            ahora=self._ahora(),
            vigencia_s=_SEGUNDOS_DE_FIRMA,
        )
        peticion = urllib.request.Request(url, data=cuerpo, method=metodo)
        if tipo:
            peticion.add_header("Content-Type", tipo)
        try:
            with self._abrir(peticion, timeout=15) as respuesta:
                return int(respuesta.status), bytes(respuesta.read())
        except urllib.error.HTTPError as e:
            if e.code in aceptar:
                return e.code, b""
            raise AlmacenNoDisponible(f"El almacén de objetos respondió HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise AlmacenNoDisponible("No se pudo conectar al almacén de objetos") from e

    def crear_bucket(self, bucket: str) -> None:
        """Idempotente: si ya existe (409) no es un error."""
        self._llamar("PUT", f"/{bucket}", aceptar=(200, 204, 409))

    def subir(self, bucket: str, clave: str, datos: bytes, tipo: str = "application/octet-stream") -> None:
        self._llamar("PUT", f"/{bucket}/{clave}", datos, tipo, aceptar=(200, 201, 204))

    def descargar(self, bucket: str, clave: str) -> bytes | None:
        """Los bytes del objeto, o `None` si no existe."""
        estado, cuerpo = self._llamar("GET", f"/{bucket}/{clave}", aceptar=(200, 404))
        return cuerpo if estado == 200 else None

    def borrar(self, bucket: str, clave: str) -> None:
        """Idempotente: borrar algo que no existe no es un error."""
        self._llamar("DELETE", f"/{bucket}/{clave}", aceptar=(200, 204, 404))


class AlmacenFotosS3:
    """Puerto `AlmacenFotos`: fotos de rostros en un bucket privado, bajo la carpeta de cada residente."""

    def __init__(self, cliente: ClienteS3, bucket: str) -> None:
        self._cliente, self._bucket = cliente, bucket
        self._bucket_listo = False

    def guardar(self, residente_id: UUID, fotos: list[bytes]) -> list[str]:
        if not self._bucket_listo:
            self._cliente.crear_bucket(self._bucket)
            self._bucket_listo = True
        subidas: list[str] = []
        try:
            for foto in fotos:
                clave = f"{residente_id}/{uuid4()}.jpg"
                self._cliente.subir(self._bucket, clave, foto, "image/jpeg")
                subidas.append(clave)
        except Exception:
            self.borrar(subidas, ignorar_errores=True)   # I1: ninguna foto huérfana si algo falla a medias
            raise
        return subidas

    def borrar(self, claves: list[str], ignorar_errores: bool = False) -> None:
        """Intenta borrar todas; si alguna falla, avisa al final (a menos que se pida ignorarlo)."""
        fallo: AlmacenNoDisponible | None = None
        for clave in claves:
            try:
                self._cliente.borrar(self._bucket, clave)
            except AlmacenNoDisponible as e:
                fallo = e
        if fallo is not None and not ignorar_errores:
            raise fallo
