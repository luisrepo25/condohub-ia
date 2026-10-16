"""Entrega del modelo de placas a la PWA del guardia (A4, O6): dominio, caso de uso, firma S3 y endpoint."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.application.modelos import AlmacenNoDisponible, ModeloNoDisponible, ObtenerModeloPlacas
from app.domain.modelos import ModeloDePlacas, ModeloInvalido
from app.infrastructure.catalogo_ficha import CatalogoFichaJson
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.s3_firmador import FirmadorS3
from app.infrastructure.sigv4 import url_firmada_get
from app.main import create_app
from tests.conftest import ISSUER, CrearToken

SHA = "081f6e646ac660d4f4fe4107ea5631bdbc92d0d56f83d4de9709321fe6af692d"
AHORA = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
FICHA_REAL = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "experiments"
    / "2026-10-15_placas_onnx"
    / "modelo_placas.json"
)


def _modelo(**cambios: str) -> ModeloDePlacas:
    datos = {
        "nombre": "yolov8n-placas",
        "version": "1.0.0",
        "formato": "onnx-int8",
        "archivo": "yolov8n_placas_int8.onnx",
        "sha256": SHA,
        **cambios,
    }
    return ModeloDePlacas(**datos)


# ---------------------------------------------------------------- dominio


def test_O6_el_modelo_se_guarda_bajo_modelos_placas_version_archivo() -> None:
    assert _modelo().clave_objeto == "placas/1.0.0/yolov8n_placas_int8.onnx"


@pytest.mark.parametrize(
    "cambio",
    [
        {"formato": "onnx-fp32"},  # el contrato solo declara onnx-int8
        {"sha256": "abc"},  # el hash de verificación debe ser SHA-256 completo
        {"sha256": SHA.upper()},
        {"archivo": "../secreto.onnx"},  # la clave del objeto no puede salirse de su carpeta
        {"archivo": "a/b.onnx"},
        {"version": ""},
        {"nombre": ""},
    ],
)
def test_O6_un_modelo_mal_formado_se_rechaza(cambio: dict[str, str]) -> None:
    with pytest.raises(ModeloInvalido):
        _modelo(**cambio)


# ---------------------------------------------------------------- caso de uso


class _CatalogoFalso:
    def __init__(self, modelo: ModeloDePlacas | None) -> None:
        self._modelo = modelo

    def placas_activo(self) -> ModeloDePlacas | None:
        return self._modelo


class _FirmadorFalso:
    def __init__(self) -> None:
        self.llamadas: list[tuple[str, timedelta, datetime]] = []

    def firmar_descarga(self, clave: str, vigencia: timedelta, ahora: datetime) -> str:
        self.llamadas.append((clave, vigencia, ahora))
        return f"https://almacen.test/{clave}?firma=x"


class _RelojFijo:
    def ahora(self) -> datetime:
        return AHORA


def test_O6_se_entrega_url_firmada_hash_y_vencimiento() -> None:
    firmador = _FirmadorFalso()
    caso = ObtenerModeloPlacas(
        _CatalogoFalso(_modelo()), firmador, _RelojFijo(), vigencia=timedelta(minutes=15)
    )
    r = caso.ejecutar()
    assert r.url == "https://almacen.test/placas/1.0.0/yolov8n_placas_int8.onnx?firma=x"
    assert (r.modelo, r.version, r.formato, r.sha256) == ("yolov8n-placas", "1.0.0", "onnx-int8", SHA)
    assert r.expira_en == AHORA + timedelta(minutes=15)
    assert firmador.llamadas == [("placas/1.0.0/yolov8n_placas_int8.onnx", timedelta(minutes=15), AHORA)]


def test_O6_sin_modelo_publicado_no_hay_nada_que_descargar() -> None:
    caso = ObtenerModeloPlacas(
        _CatalogoFalso(None), _FirmadorFalso(), _RelojFijo(), vigencia=timedelta(minutes=15)
    )
    with pytest.raises(ModeloNoDisponible):
        caso.ejecutar()


def test_la_vigencia_de_la_url_es_un_parametro_positivo() -> None:
    with pytest.raises(ValueError):
        ObtenerModeloPlacas(_CatalogoFalso(None), _FirmadorFalso(), _RelojFijo(), vigencia=timedelta(0))


# ---------------------------------------------------------------- adaptadores


def test_la_firma_v4_coincide_con_el_ejemplo_oficial_de_aws() -> None:
    """Ejemplo de la documentación de AWS: autenticar con parámetros de consulta (Signature Version 4)."""
    url = url_firmada_get(
        esquema="https",
        host="examplebucket.s3.amazonaws.com",
        ruta="/test.txt",
        acceso="AKIAIOSFODNN7EXAMPLE",
        secreto="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        ahora=datetime(2013, 5, 24, 0, 0, 0, tzinfo=UTC),
        vigencia_s=86400,
    )
    assert url.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404")
    assert "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request" in url


def test_el_firmador_s3_arma_la_url_con_ruta_bucket_clave() -> None:
    f = FirmadorS3("http://localhost:9000", "modelos", "clave", "secreto", "us-east-1")
    url = f.firmar_descarga("placas/1.0.0/yolov8n_placas_int8.onnx", timedelta(minutes=15), AHORA)
    assert url.startswith("http://localhost:9000/modelos/placas/1.0.0/yolov8n_placas_int8.onnx?")
    assert "X-Amz-Expires=900" in url and "X-Amz-Date=20261020T120000Z" in url
    assert "secreto" not in url  # el secreto nunca viaja en la URL


def test_el_firmador_s3_no_permite_vigencias_fuera_de_lo_que_acepta_s3() -> None:
    f = FirmadorS3("http://localhost:9000", "modelos", "clave", "secreto", "us-east-1")
    for segundos in (0, 7 * 24 * 3600 + 1):
        with pytest.raises(ValueError):
            f.firmar_descarga("x", timedelta(seconds=segundos), AHORA)


def test_el_firmador_sin_credenciales_avisa_que_el_almacen_no_esta_configurado() -> None:
    f = FirmadorS3("http://localhost:9000", "modelos", "", "", "us-east-1")
    with pytest.raises(AlmacenNoDisponible):
        f.firmar_descarga("x", timedelta(minutes=1), AHORA)


def test_el_catalogo_lee_la_ficha_json(tmp_path: Path) -> None:
    ficha = tmp_path / "modelo_placas.json"
    ficha.write_text(
        json.dumps(
            {
                "modelo": "yolov8n-placas",
                "version": "2.0.0",
                "formato": "onnx-int8",
                "archivo": "m.onnx",
                "sha256": SHA,
                "extra": "se ignora",
            }
        ),
        encoding="utf-8",
    )
    m = CatalogoFichaJson(str(ficha)).placas_activo()
    assert m is not None and m.version == "2.0.0" and m.archivo == "m.onnx"


def test_el_catalogo_sin_ficha_devuelve_nada(tmp_path: Path) -> None:
    assert CatalogoFichaJson("").placas_activo() is None
    assert CatalogoFichaJson(str(tmp_path / "no_existe.json")).placas_activo() is None


@pytest.mark.skipif(not FICHA_REAL.exists(), reason="repo abierto sin los resultados de ml/")
def test_el_catalogo_acepta_la_ficha_real_que_genera_el_notebook_07() -> None:
    m = CatalogoFichaJson(str(FICHA_REAL)).placas_activo()
    assert m is not None and m.formato == "onnx-int8" and m.sha256 == SHA


# ------------------------------------------------ endpoint (la autorización por rol está en test_roles)


def _cliente(caso: ObtenerModeloPlacas | None, clave: object) -> TestClient:
    return TestClient(create_app(JwtVerificador(ISSUER, lambda _t: clave), obtener_modelo_placas=caso))  # type: ignore[arg-type]


def test_O6_el_guardia_recibe_el_modelo_con_los_campos_del_contrato(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    caso = ObtenerModeloPlacas(
        _CatalogoFalso(_modelo()), _FirmadorFalso(), _RelojFijo(), vigencia=timedelta(minutes=15)
    )
    r = _cliente(caso, _clave.public_key()).get(
        "/modelos/placas/actual", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    )
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"modelo", "version", "formato", "url", "sha256", "expiraEn"}
    assert cuerpo["sha256"] == SHA and cuerpo["formato"] == "onnx-int8"
    assert cuerpo["expiraEn"].startswith("2026-10-20T12:15:00")


def test_O6_sin_modelo_publicado_el_endpoint_responde_404_problem_json(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    caso = ObtenerModeloPlacas(
        _CatalogoFalso(None), _FirmadorFalso(), _RelojFijo(), vigencia=timedelta(minutes=15)
    )
    r = _cliente(caso, _clave.public_key()).get(
        "/modelos/placas/actual", headers={"Authorization": f"Bearer {crear_token('ADMIN')}"}
    )
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


def test_O6_si_el_almacen_no_esta_configurado_el_endpoint_responde_503(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    firmador = FirmadorS3("http://localhost:9000", "modelos", "", "", "us-east-1")
    caso = ObtenerModeloPlacas(
        _CatalogoFalso(_modelo()), firmador, _RelojFijo(), vigencia=timedelta(minutes=15)
    )
    r = _cliente(caso, _clave.public_key()).get(
        "/modelos/placas/actual", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    )
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/problem+json")
