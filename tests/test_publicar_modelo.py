"""Publicar el modelo de placas en el almacén: la ficha y el hash mandan (O6)."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Config
from app.infrastructure.composicion import construir_almacen_fotos
from app.infrastructure.memoria import AlmacenFotosMemoria
from app.infrastructure.publicar_modelo import publicar
from app.infrastructure.s3_cliente import AlmacenFotosS3, ClienteS3
from tests.test_s3_cliente import AlmacenHttpFalso

AHORA = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
CONTENIDO = b"modelo onnx de prueba"


def _preparar(tmp_path: Path, sha: str | None = None, nombre_archivo: str = "m.onnx") -> tuple[Path, Path]:
    archivo = tmp_path / nombre_archivo
    archivo.write_bytes(CONTENIDO)
    ficha = tmp_path / "modelo_placas.json"
    ficha.write_text(
        json.dumps(
            {
                "modelo": "yolov8n-placas",
                "version": "1.0.0",
                "formato": "onnx-int8",
                "archivo": "m.onnx",
                "sha256": sha or hashlib.sha256(CONTENIDO).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return ficha, archivo


def _cliente(servidor: AlmacenHttpFalso) -> ClienteS3:
    return ClienteS3("http://almacen.test:9000", "c", "s", abrir=servidor, ahora=lambda: AHORA)


def test_O6_el_modelo_se_sube_a_su_clave_de_objeto(tmp_path: Path) -> None:
    servidor = AlmacenHttpFalso()
    ficha, archivo = _preparar(tmp_path)
    clave = publicar(ficha, archivo, _cliente(servidor), "modelos")
    assert clave == "placas/1.0.0/m.onnx"
    assert servidor.objetos["/modelos/placas/1.0.0/m.onnx"] == CONTENIDO


def test_O6_si_el_archivo_no_coincide_con_el_hash_de_la_ficha_no_se_publica(tmp_path: Path) -> None:
    servidor = AlmacenHttpFalso()
    ficha, archivo = _preparar(tmp_path, sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        publicar(ficha, archivo, _cliente(servidor), "modelos")
    assert servidor.objetos == {} and servidor.peticiones == []


def test_O6_el_archivo_debe_llamarse_como_dice_la_ficha(tmp_path: Path) -> None:
    servidor = AlmacenHttpFalso()
    ficha, archivo = _preparar(tmp_path, nombre_archivo="otro.onnx")
    with pytest.raises(ValueError, match="espera"):
        publicar(ficha, archivo, _cliente(servidor), "modelos")
    assert servidor.peticiones == []


def test_la_configuracion_elige_donde_se_guardan_las_fotos() -> None:
    assert isinstance(construir_almacen_fotos(Config("i", "j")), AlmacenFotosMemoria)
    assert isinstance(construir_almacen_fotos(Config("i", "j", almacen_fotos="s3")), AlmacenFotosS3)
    with pytest.raises(ValueError, match="ALMACEN_FOTOS"):
        construir_almacen_fotos(Config("i", "j", almacen_fotos="disco"))
