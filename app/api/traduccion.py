"""Traduce los errores de dominio y de aplicación a `application/problem+json` (RFC 7807)."""

from collections.abc import Iterator
from contextlib import contextmanager

from app.api.problemas import ProblemaError
from app.application.leer_placa import LectorPlacasNoDisponible
from app.application.modelos import AlmacenNoDisponible
from app.application.rostros import (
    ImagenInvalida,
    ModeloFacialNoDisponible,
    PerfilNoEncontrado,
    SinRostrosUtilizables,
)
from app.domain.errores import ReglaViolada


@contextmanager
def traduciendo_errores() -> Iterator[None]:
    """`SinPermiso` no se traduce aquí: ya lo atiende el manejador global (403)."""
    try:
        yield
    except ReglaViolada as e:
        raise ProblemaError(422, "Regla de negocio violada", e.mensaje, code=e.codigo) from e
    except SinRostrosUtilizables as e:
        raise ProblemaError(422, "Sin rostros utilizables", str(e), code="SIN_ROSTRO") from e
    except ImagenInvalida as e:
        raise ProblemaError(400, "Solicitud inválida", str(e)) from e
    except PerfilNoEncontrado as e:
        raise ProblemaError(404, "No encontrado", str(e)) from e
    except (ModeloFacialNoDisponible, AlmacenNoDisponible, LectorPlacasNoDisponible) as e:
        raise ProblemaError(503, "Servicio no disponible", str(e)) from e
    except Exception as e:
        # Un fallo de MongoDB (servidor caído, tiempo agotado) es «no disponible», no un error del servicio
        if type(e).__module__.startswith("pymongo"):
            raise ProblemaError(503, "Base de datos no disponible", "No se pudo acceder a MongoDB") from e
        raise
