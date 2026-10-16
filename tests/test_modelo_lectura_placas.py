"""Reconocedor de texto de placas para la PWA del guardia (A4, O6): dominio, caso de uso y endpoint."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.application.modelo_lectura import ObtenerModeloLectura
from app.application.modelos import ModeloNoDisponible
from app.domain.modelo_lectura import ModeloDeLectura
from app.domain.modelos import ModeloInvalido
from app.infrastructure.catalogo_ficha_lectura import CatalogoLecturaFichaJson
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.publicar_lectura import publicar
from app.infrastructure.s3_firmador import FirmadorS3
from app.main import create_app
from tests.conftest import ISSUER, CrearToken

SHA = "31cd118ec9ddce4eafbccfe93b38dc5dbc3482bc325d8a061ef2c4e9e0280760"
AHORA = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
FICHA_REAL = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "experiments"
    / "2026-10-16_reconocedor_placas_onnx"
    / "ficha_reconocedor.json"
)


def _modelo(**cambios: str) -> ModeloDeLectura:
    datos = {
        "nombre": "easyocr-reconocedor-placas",
        "version": "1.0.0",
        "formato": "onnx-fp32",
        "archivo": "reconocedor_placas.onnx",
        "sha256": SHA,
        "ficha": "ficha_reconocedor.json",
        **cambios,
    }
    return ModeloDeLectura(**datos)


# ---------------------------------------------------------------- dominio


def test_O6_el_reconocedor_y_su_ficha_se_guardan_bajo_lectura_placas_version() -> None:
    m = _modelo()
    assert m.clave_objeto == "lectura-placas/1.0.0/reconocedor_placas.onnx"
    assert m.clave_ficha == "lectura-placas/1.0.0/ficha_reconocedor.json"


def test_O6_se_aceptan_los_formatos_del_contrato() -> None:
    assert _modelo(formato="onnx-int8").formato == "onnx-int8"


@pytest.mark.parametrize(
    "cambio",
    [
        {"formato": "tflite"},  # el contrato solo declara onnx-fp32 y onnx-int8
        {"sha256": "abc"},
        {"sha256": SHA.upper()},
        {"archivo": "../secreto.onnx"},
        {"archivo": "a/b.onnx"},
        {"ficha": "../ficha.json"},
        {"ficha": ""},
        {"version": ""},
        {"nombre": ""},
    ],
)
def test_O6_un_reconocedor_mal_formado_se_rechaza(cambio: dict[str, str]) -> None:
    with pytest.raises(ModeloInvalido):
        _modelo(**cambio)


# ---------------------------------------------------------------- caso de uso


class _CatalogoFalso:
    def __init__(self, modelo: ModeloDeLectura | None) -> None:
        self._modelo = modelo

    def lectura_activo(self) -> ModeloDeLectura | None:
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


def _caso(modelo: ModeloDeLectura | None, firmador: object | None = None) -> ObtenerModeloLectura:
    return ObtenerModeloLectura(
        _CatalogoFalso(modelo),
        firmador or _FirmadorFalso(),
        _RelojFijo(),
        vigencia=timedelta(minutes=15),  # type: ignore[arg-type]
    )


def test_O6_se_entregan_las_dos_urls_firmadas_hash_y_vencimiento() -> None:
    firmador = _FirmadorFalso()
    r = _caso(_modelo(), firmador).ejecutar()
    assert r.url == "https://almacen.test/lectura-placas/1.0.0/reconocedor_placas.onnx?firma=x"
    assert r.ficha_url == "https://almacen.test/lectura-placas/1.0.0/ficha_reconocedor.json?firma=x"
    assert (r.modelo, r.version, r.formato, r.sha256) == (
        "easyocr-reconocedor-placas",
        "1.0.0",
        "onnx-fp32",
        SHA,
    )
    assert r.expira_en == AHORA + timedelta(minutes=15)
    assert [c[0] for c in firmador.llamadas] == [
        "lectura-placas/1.0.0/reconocedor_placas.onnx",
        "lectura-placas/1.0.0/ficha_reconocedor.json",
    ]


def test_O6_sin_reconocedor_publicado_no_hay_nada_que_descargar() -> None:
    with pytest.raises(ModeloNoDisponible):
        _caso(None).ejecutar()


def test_la_vigencia_de_las_urls_es_un_parametro_positivo() -> None:
    with pytest.raises(ValueError):
        ObtenerModeloLectura(_CatalogoFalso(None), _FirmadorFalso(), _RelojFijo(), vigencia=timedelta(0))  # type: ignore[arg-type]


# ---------------------------------------------------------------- catálogo


def test_el_catalogo_lee_la_ficha_json_y_toma_su_nombre_de_archivo(tmp_path: Path) -> None:
    ficha = tmp_path / "ficha_reconocedor.json"
    ficha.write_text(
        json.dumps(
            {
                "modelo": "easyocr-reconocedor-placas",
                "version": "2.0.0",
                "formato": "onnx-int8",
                "archivo": "r.onnx",
                "sha256": SHA,
                "caracteres": ["[blank]", "0"],  # el resto de la ficha lo lee la PWA, no el servicio
            }
        ),
        encoding="utf-8",
    )
    m = CatalogoLecturaFichaJson(str(ficha)).lectura_activo()
    assert m is not None and (m.version, m.archivo, m.ficha) == ("2.0.0", "r.onnx", "ficha_reconocedor.json")


def test_el_catalogo_sin_ficha_devuelve_nada(tmp_path: Path) -> None:
    assert CatalogoLecturaFichaJson("").lectura_activo() is None
    assert CatalogoLecturaFichaJson(str(tmp_path / "no_existe.json")).lectura_activo() is None


@pytest.mark.skipif(not FICHA_REAL.exists(), reason="repo abierto sin los resultados de ml/")
def test_el_catalogo_acepta_la_ficha_real_que_genera_el_notebook_11() -> None:
    m = CatalogoLecturaFichaJson(str(FICHA_REAL)).lectura_activo()
    assert (
        m is not None
        and m.formato == "onnx-fp32"
        and m.sha256 == SHA
        and m.clave_ficha.endswith("/ficha_reconocedor.json")
    )


# ---------------------------------------------------------------- publicación


class _ClienteFalso:
    def __init__(self) -> None:
        self.buckets: list[str] = []
        self.subidos: dict[tuple[str, str], bytes] = {}

    def crear_bucket(self, bucket: str) -> None:
        self.buckets.append(bucket)

    def subir(self, bucket: str, clave: str, contenido: bytes) -> None:
        self.subidos[(bucket, clave)] = contenido


def _preparar(tmp_path: Path, contenido: bytes = b"onnx-de-mentira") -> tuple[Path, Path]:
    archivo = tmp_path / "reconocedor_placas.onnx"
    archivo.write_bytes(contenido)
    ficha = tmp_path / "ficha_reconocedor.json"
    ficha.write_text(
        json.dumps(
            {
                "modelo": "easyocr-reconocedor-placas",
                "version": "1.0.0",
                "formato": "onnx-fp32",
                "archivo": archivo.name,
                "sha256": hashlib.sha256(b"onnx-de-mentira").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return ficha, archivo


def test_O6_publicar_sube_el_modelo_y_su_ficha(tmp_path: Path) -> None:
    ficha, archivo = _preparar(tmp_path)
    cliente = _ClienteFalso()
    claves = publicar(ficha, archivo, cliente, "modelos")  # type: ignore[arg-type]
    assert claves == (
        "lectura-placas/1.0.0/reconocedor_placas.onnx",
        "lectura-placas/1.0.0/ficha_reconocedor.json",
    )
    assert cliente.subidos[("modelos", claves[0])] == b"onnx-de-mentira"
    assert cliente.subidos[("modelos", claves[1])] == ficha.read_bytes()


def test_O6_publicar_se_niega_si_el_archivo_no_es_el_de_la_ficha(tmp_path: Path) -> None:
    ficha, archivo = _preparar(tmp_path, contenido=b"otro-contenido")
    cliente = _ClienteFalso()
    with pytest.raises(ValueError, match="sha256"):
        publicar(ficha, archivo, cliente, "modelos")  # type: ignore[arg-type]
    assert cliente.subidos == {}


# ------------------------------------------------ endpoint


def _cliente(caso: ObtenerModeloLectura, clave: object) -> TestClient:
    return TestClient(create_app(JwtVerificador(ISSUER, lambda _t: clave), obtener_modelo_lectura=caso))  # type: ignore[arg-type]


def test_O6_el_guardia_recibe_el_reconocedor_con_los_campos_del_contrato(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    r = _cliente(_caso(_modelo()), _clave.public_key()).get(
        "/modelos/lectura-placas/actual", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    )
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"modelo", "version", "formato", "url", "sha256", "fichaUrl", "expiraEn"}
    assert cuerpo["sha256"] == SHA and cuerpo["formato"] == "onnx-fp32"
    assert cuerpo["fichaUrl"].endswith("ficha_reconocedor.json?firma=x")
    assert cuerpo["expiraEn"].startswith("2026-10-20T12:15:00")


def test_O6_el_admin_tambien_puede_pedirlo_y_el_residente_no(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    cliente = _cliente(_caso(_modelo()), _clave.public_key())
    url = "/modelos/lectura-placas/actual"
    assert cliente.get(url, headers={"Authorization": f"Bearer {crear_token('ADMIN')}"}).status_code == 200
    assert (
        cliente.get(url, headers={"Authorization": f"Bearer {crear_token('RESIDENTE')}"}).status_code == 403
    )
    assert cliente.get(url).status_code == 401


def test_O6_sin_reconocedor_publicado_el_endpoint_responde_404_problem_json(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    r = _cliente(_caso(None), _clave.public_key()).get(
        "/modelos/lectura-placas/actual", headers={"Authorization": f"Bearer {crear_token('ADMIN')}"}
    )
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


def test_O6_si_el_almacen_no_esta_configurado_el_endpoint_responde_503(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    firmador = FirmadorS3("http://localhost:9000", "modelos", "", "", "us-east-1")
    r = _cliente(_caso(_modelo(), firmador), _clave.public_key()).get(
        "/modelos/lectura-placas/actual", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    )
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/problem+json")


def test_el_servicio_sin_ficha_configurada_responde_404_y_no_se_cae(crear_token: CrearToken) -> None:
    from tests.conftest import _clave

    caso = ObtenerModeloLectura(
        CatalogoLecturaFichaJson(""),
        FirmadorS3("http://localhost:9000", "modelos", "k", "s", "us-east-1"),
        _RelojFijo(),
        timedelta(minutes=15),
    )
    r = _cliente(caso, _clave.public_key()).get(
        "/modelos/lectura-placas/actual", headers={"Authorization": f"Bearer {crear_token('GUARDIA')}"}
    )
    assert r.status_code == 404
