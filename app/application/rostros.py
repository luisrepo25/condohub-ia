"""Casos de uso de rostros: registrar con consentimiento (I1), identificar (I2, I3, I5) y retirarlo (I1).

La aplicación orquesta el dominio y los puertos; no conoce FastAPI, MongoDB, S3 ni PyTorch.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID

from app.application.errores import SinPermiso
from app.application.modelos import Reloj
from app.domain.errores import ReglaViolada
from app.domain.roles import Rol, Solicitante
from app.domain.rostros import (
    Consentimiento,
    Desconocido,
    Embedding,
    EmbeddingRegistrado,
    PerfilFacial,
    Prediccion,
    ReconocedorFacial,
    Zona,
)

SIN_ROSTRO = "SIN_ROSTRO"


class ImagenInvalida(Exception):
    """La imagen no se puede leer."""


class SinRostrosUtilizables(Exception):
    """Ninguna de las fotos tiene un rostro detectable."""


class PerfilNoEncontrado(Exception):
    """No existe un perfil facial para ese residente."""


class ModeloFacialNoDisponible(Exception):
    """El modelo facial no está cargado o configurado en este servicio."""


# ------------------------------------------------------------------ puertos

class ModeloFacial(Protocol):
    nombre: str
    version: str

    def embeber(self, imagen: bytes) -> Embedding | None:
        """Embedding normalizado del rostro principal, o `None` si no hay rostro. Lanza `ImagenInvalida`."""
        ...


class RepositorioPerfiles(Protocol):
    def guardar(self, perfil: PerfilFacial) -> None: ...
    def por_residente(self, residente_id: UUID) -> PerfilFacial | None: ...
    def con_consentimiento_vigente(self) -> list[PerfilFacial]: ...


class RepositorioDesconocidos(Protocol):
    def guardar(self, desconocido: Desconocido) -> None: ...


class RepositorioPredicciones(Protocol):
    def guardar(self, prediccion: Prediccion) -> str:
        """Guarda la predicción (I5) y devuelve su id."""
        ...


class AlmacenFotos(Protocol):
    def guardar(self, residente_id: UUID, fotos: list[bytes]) -> list[str]:
        """Guarda las fotos y devuelve una clave de objeto por foto, en el mismo orden."""
        ...

    def borrar(self, claves: list[str]) -> None: ...


# ------------------------------------------------------------------ resultados

@dataclass(frozen=True)
class PerfilRegistrado:
    residente_id: UUID
    embeddings: int
    modelo: str
    version: str


@dataclass(frozen=True)
class ResultadoIdentificacion:
    resultado: str                 # RECONOCIDO, DUDOSO, DESCONOCIDO o SIN_ROSTRO
    residente_id: UUID | None
    confianza: float               # 0..1 (I5)
    modelo: str
    version: str
    prediccion_id: str


@dataclass(frozen=True)
class RetiroAceptado:
    residente_id: UUID
    borrado_antes_de: datetime


# ------------------------------------------------------------------ casos de uso

class RegistrarRostro:
    """Registra (o amplía) el perfil de un residente. Solo con consentimiento explícito (I1)."""

    def __init__(
        self, modelo: ModeloFacial, perfiles: RepositorioPerfiles, fotos: AlmacenFotos, reloj: Reloj
    ) -> None:
        self._modelo, self._perfiles, self._fotos, self._reloj = modelo, perfiles, fotos, reloj

    def ejecutar(
        self,
        residente_id: UUID,
        residente_sub: str,
        unidad_id: UUID,
        consentimiento_fecha: datetime | None,
        consentimiento_version: str | None,
        fotos: list[bytes],
    ) -> PerfilRegistrado:
        ahora = self._reloj.ahora()
        if consentimiento_fecha is None or consentimiento_version is None:
            raise ReglaViolada("I1", "Falta el consentimiento: se necesitan la fecha y la versión del texto")
        # I1: se valida antes de tocar nada
        consentimiento = Consentimiento(consentimiento_fecha, consentimiento_version)
        usables: list[tuple[bytes, Embedding]] = []
        for foto in fotos:
            vector = self._modelo.embeber(foto)
            if vector is not None:
                usables.append((foto, vector))
        if not usables:
            raise SinRostrosUtilizables("Ninguna foto tiene un rostro detectable")

        claves = self._fotos.guardar(residente_id, [f for f, _ in usables])
        nuevos = [
            EmbeddingRegistrado(v, self._modelo.nombre, self._modelo.version, clave, ahora)
            for (_, v), clave in zip(usables, claves, strict=True)
        ]
        perfil = self._perfiles.por_residente(residente_id)
        if perfil is not None and perfil.vigente:
            # TODO(regla): ¿registrar de nuevo a un residente vigente amplía su perfil (hoy) o lo reemplaza?
            perfil.agregar(nuevos)
        else:
            perfil = PerfilFacial.registrar(
                residente_id, residente_sub, unidad_id, consentimiento, nuevos, ahora
            )
        self._perfiles.guardar(perfil)
        return PerfilRegistrado(
            residente_id, len(perfil.embeddings), self._modelo.nombre, self._modelo.version
        )


class IdentificarRostro:
    """Identifica el rostro de una imagen: asiste al guardia, nunca abre por sí solo (A5)."""

    def __init__(
        self,
        modelo: ModeloFacial,
        reconocedor: ReconocedorFacial,
        perfiles: RepositorioPerfiles,
        desconocidos: RepositorioDesconocidos,
        predicciones: RepositorioPredicciones,
        reloj: Reloj,
        retencion_desconocidos: timedelta,
    ) -> None:
        self._modelo, self._reconocedor, self._perfiles = modelo, reconocedor, perfiles
        self._desconocidos, self._predicciones, self._reloj = desconocidos, predicciones, reloj
        self._retencion = retencion_desconocidos

    def ejecutar(self, imagen: bytes, camara: str | None = None) -> ResultadoIdentificacion:
        inicio = time.perf_counter()
        vector = self._modelo.embeber(imagen)
        ahora = self._reloj.ahora()
        umbrales = self._reconocedor.umbrales

        if vector is None:
            resultado, residente_id, confianza = SIN_ROSTRO, None, 0.0
        else:
            r = self._reconocedor.identificar(vector, self._perfiles.con_consentimiento_vigente())
            resultado, residente_id = r.zona.value, r.residente_id
            confianza = min(1.0, max(0.0, r.similitud))
            if r.zona is Zona.DESCONOCIDO:   # I3: solo los desconocidos; un DUDOSO puede ser un residente
                self._desconocidos.guardar(
                    Desconocido.capturar(
                        vector, self._modelo.nombre, self._modelo.version, camara, ahora, self._retencion
                    )
                )
        prediccion_id = self._predicciones.guardar(
            Prediccion(
                tipo="ROSTRO",
                resultado=resultado,
                residente_id=residente_id,
                confianza=confianza,
                umbral_alto=umbrales.alto,
                umbral_bajo=umbrales.bajo,
                modelo=self._modelo.nombre,
                version_modelo=self._modelo.version,
                camara=camara,
                latencia_ms=(time.perf_counter() - inicio) * 1000,
                fecha=ahora,
            )
        )
        return ResultadoIdentificacion(
            resultado, residente_id, confianza, self._modelo.nombre, self._modelo.version, prediccion_id
        )


class RetirarConsentimiento:
    """Retira el consentimiento y borra fotos y embeddings (I1: en menos de 24 h; aquí, de inmediato)."""

    def __init__(
        self, perfiles: RepositorioPerfiles, fotos: AlmacenFotos, reloj: Reloj, plazo: timedelta
    ) -> None:
        self._perfiles, self._fotos, self._reloj, self._plazo = perfiles, fotos, reloj, plazo

    def ejecutar(self, solicitante: Solicitante, residente_id: UUID) -> RetiroAceptado:
        perfil = self._perfiles.por_residente(residente_id)
        if perfil is None:
            raise PerfilNoEncontrado("No hay un perfil facial para ese residente")
        # Un RESIDENTE solo retira SU consentimiento: el `sub` del token debe ser el del perfil
        if not solicitante.tiene_algun_rol(Rol.ADMIN) and perfil.residente_sub != solicitante.sub:
            raise SinPermiso("Solo el propio residente o un ADMIN puede retirar este consentimiento")
        if perfil.vigente:
            claves = perfil.claves_fotos()
            perfil.retirar_consentimiento(self._reloj.ahora())
            self._fotos.borrar(claves)
            self._perfiles.guardar(perfil)
        retirado_en = perfil.retirado_en
        assert retirado_en is not None          # tras retirar, siempre hay fecha
        return RetiroAceptado(residente_id, retirado_en + self._plazo)


@dataclass(frozen=True)
class CasosRostros:
    """Los tres casos de uso de rostros, listos para los routers."""

    registrar: RegistrarRostro
    identificar: IdentificarRostro
    retirar: RetirarConsentimiento
