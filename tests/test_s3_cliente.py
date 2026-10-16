"""Cliente del almacén S3 (SeaweedFS en desarrollo): URLs prefirmadas, errores y fotos de rostros (I1, I6)."""

import io
import urllib.error
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest

from app.application.modelos import AlmacenNoDisponible
from app.infrastructure.s3_cliente import AlmacenFotosS3, ClienteS3

AHORA = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
RESIDENTE = UUID(int=7)


class RespuestaFalsa:
    def __init__(self, status: int, cuerpo: bytes = b"") -> None:
        self.status, self._cuerpo = status, cuerpo

    def read(self) -> bytes:
        return self._cuerpo

    def __enter__(self) -> "RespuestaFalsa":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class AlmacenHttpFalso:
    """Imita el servidor: guarda objetos por ruta y registra las peticiones; `fallos` fuerza un error HTTP."""

    def __init__(self) -> None:
        self.objetos: dict[str, bytes] = {}
        self.peticiones: list[Any] = []
        self.fallos: dict[str, int] = {}  # "METODO ruta" -> código HTTP
        self.cae_la_red = False

    def __call__(self, peticion: Any, timeout: float = 0) -> RespuestaFalsa:
        if self.cae_la_red:
            raise urllib.error.URLError("sin red")
        self.peticiones.append(peticion)
        ruta = urlsplit(peticion.full_url).path
        clave = f"{peticion.get_method()} {ruta}"
        if clave in self.fallos:
            raise urllib.error.HTTPError(peticion.full_url, self.fallos[clave], "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]
        if peticion.get_method() == "PUT":
            if ruta.count("/") >= 2:
                self.objetos[ruta] = peticion.data
            return RespuestaFalsa(200)
        if peticion.get_method() == "GET":
            if ruta not in self.objetos:
                raise urllib.error.HTTPError(peticion.full_url, 404, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]
            return RespuestaFalsa(200, self.objetos[ruta])
        if peticion.get_method() == "DELETE":
            if ruta not in self.objetos:
                raise urllib.error.HTTPError(peticion.full_url, 404, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]
            del self.objetos[ruta]
            return RespuestaFalsa(204)
        raise AssertionError(peticion.get_method())


def _cliente(servidor: AlmacenHttpFalso, acceso: str = "clave", secreto: str = "secreto") -> ClienteS3:
    return ClienteS3(
        "http://almacen.test:9000", acceso, secreto, "us-east-1", abrir=servidor, ahora=lambda: AHORA
    )


# ------------------------------------------------------------------ cliente


def test_I6_la_peticion_va_prefirmada_sin_el_secreto_y_con_el_metodo_correcto() -> None:
    servidor = AlmacenHttpFalso()
    _cliente(servidor).subir("rostros", "a/b.jpg", b"datos", "image/jpeg")
    [p] = servidor.peticiones
    consulta = parse_qs(urlsplit(p.full_url).query)
    assert p.get_method() == "PUT" and urlsplit(p.full_url).path == "/rostros/a/b.jpg" and p.data == b"datos"
    assert consulta["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"] and consulta["X-Amz-Date"] == [
        "20261020T120000Z"
    ]
    assert "X-Amz-Signature" in consulta and "secreto" not in p.full_url
    assert p.get_header("Content-type") == "image/jpeg"


def test_subir_descargar_y_borrar_un_objeto() -> None:
    servidor = AlmacenHttpFalso()
    c = _cliente(servidor)
    c.subir("rostros", "x.jpg", b"foto")
    assert c.descargar("rostros", "x.jpg") == b"foto"
    c.borrar("rostros", "x.jpg")
    assert c.descargar("rostros", "x.jpg") is None


def test_borrar_algo_que_no_existe_no_es_un_error() -> None:
    _cliente(AlmacenHttpFalso()).borrar("rostros", "no-existe.jpg")


def test_crear_un_bucket_que_ya_existe_no_es_un_error() -> None:
    servidor = AlmacenHttpFalso()
    servidor.fallos["PUT /rostros"] = 409
    _cliente(servidor).crear_bucket("rostros")


@pytest.mark.parametrize("codigo", [403, 500, 503])
def test_un_error_http_del_almacen_se_convierte_en_almacen_no_disponible(codigo: int) -> None:
    servidor = AlmacenHttpFalso()
    servidor.fallos["PUT /rostros/x.jpg"] = codigo
    with pytest.raises(AlmacenNoDisponible, match=str(codigo)):
        _cliente(servidor).subir("rostros", "x.jpg", b"foto")


def test_si_no_hay_red_se_avisa_que_el_almacen_no_esta_disponible() -> None:
    servidor = AlmacenHttpFalso()
    servidor.cae_la_red = True
    with pytest.raises(AlmacenNoDisponible):
        _cliente(servidor).subir("rostros", "x.jpg", b"foto")


def test_sin_credenciales_no_se_intenta_ninguna_peticion() -> None:
    servidor = AlmacenHttpFalso()
    with pytest.raises(AlmacenNoDisponible, match="credenciales"):
        _cliente(servidor, acceso="", secreto="").subir("rostros", "x.jpg", b"foto")
    assert servidor.peticiones == []


# ------------------------------------------------------------------ fotos de rostros


def test_I1_las_fotos_se_guardan_en_el_bucket_de_rostros_bajo_la_carpeta_del_residente() -> None:
    servidor = AlmacenHttpFalso()
    almacen = AlmacenFotosS3(_cliente(servidor), "rostros")
    claves = almacen.guardar(RESIDENTE, [b"uno", b"dos"])
    assert len(claves) == 2 and len(set(claves)) == 2
    assert all(c.startswith(f"{RESIDENTE}/") and c.endswith(".jpg") for c in claves)
    assert {servidor.objetos[f"/rostros/{c}"] for c in claves} == {b"uno", b"dos"}


def test_el_bucket_se_crea_una_sola_vez_y_antes_de_la_primera_foto() -> None:
    servidor = AlmacenHttpFalso()
    almacen = AlmacenFotosS3(_cliente(servidor), "rostros")
    almacen.guardar(RESIDENTE, [b"uno"])
    almacen.guardar(RESIDENTE, [b"dos"])
    creaciones = [p for p in servidor.peticiones if urlsplit(p.full_url).path == "/rostros"]
    assert len(creaciones) == 1 and servidor.peticiones[0] is creaciones[0]


def test_I1_si_falla_una_subida_no_quedan_fotos_huerfanas() -> None:
    servidor = AlmacenHttpFalso()
    almacen = AlmacenFotosS3(_cliente(servidor), "rostros")
    original = servidor.__call__
    subidas = {"n": 0}

    def con_fallo(peticion: Any, timeout: float = 0) -> RespuestaFalsa:
        if peticion.get_method() == "PUT" and urlsplit(peticion.full_url).path.count("/") >= 2:
            subidas["n"] += 1
            if subidas["n"] == 3:
                raise urllib.error.HTTPError(peticion.full_url, 500, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]
        return original(peticion, timeout)

    almacen = AlmacenFotosS3(
        ClienteS3("http://almacen.test:9000", "c", "s", abrir=con_fallo, ahora=lambda: AHORA), "rostros"
    )
    with pytest.raises(AlmacenNoDisponible):
        almacen.guardar(RESIDENTE, [b"uno", b"dos", b"tres"])
    assert servidor.objetos == {}  # las dos primeras se borraron: no queda ninguna foto


def test_I1_borrar_quita_todas_las_fotos_y_repetirlo_no_falla() -> None:
    servidor = AlmacenHttpFalso()
    almacen = AlmacenFotosS3(_cliente(servidor), "rostros")
    claves = almacen.guardar(RESIDENTE, [b"uno", b"dos"])
    almacen.borrar(claves)
    assert servidor.objetos == {}
    almacen.borrar(claves)


def test_I1_si_el_almacen_falla_al_borrar_se_avisa_y_se_intentan_todas() -> None:
    servidor = AlmacenHttpFalso()
    almacen = AlmacenFotosS3(_cliente(servidor), "rostros")
    claves = almacen.guardar(RESIDENTE, [b"uno", b"dos"])
    servidor.fallos[f"DELETE /rostros/{claves[0]}"] = 500
    with pytest.raises(AlmacenNoDisponible):
        almacen.borrar(claves)
    assert f"/rostros/{claves[1]}" not in servidor.objetos  # la segunda sí se borró aunque la primera falló
