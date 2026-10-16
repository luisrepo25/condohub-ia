"""Adaptador del modelo facial real: lo que se prueba sin el modelo (el modelo se valida en Colab)."""

import io
import sys
from importlib.util import find_spec

import pytest

from app.application.rostros import ImagenInvalida, ModeloFacialNoDisponible
from app.config import Config
from app.infrastructure.composicion import ModeloFacialNoConfigurado, construir_casos_rostros

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from app.infrastructure.facenet import ModeloFacialFacenet  # noqa: E402

FACENET_INSTALADO = find_spec("facenet_pytorch") is not None


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buffer, "PNG")
    return buffer.getvalue()


def test_el_adaptador_se_identifica_con_modelo_y_version_para_la_bitacora_de_predicciones() -> None:
    m = ModeloFacialFacenet(version="2.0.0")
    assert (m.nombre, m.version) == ("facenet-vggface2", "2.0.0")


def test_construirlo_no_carga_pytorch_para_que_el_servicio_arranque_rapido() -> None:
    ModeloFacialFacenet()
    if not FACENET_INSTALADO:
        assert "facenet_pytorch" not in sys.modules


@pytest.mark.parametrize("basura", [b"", b"esto no es una imagen", b"\x89PNG\r\n\x1a\n corrupta"])
def test_una_imagen_ilegible_se_rechaza_sin_necesidad_del_modelo(basura: bytes) -> None:
    with pytest.raises(ImagenInvalida):
        ModeloFacialFacenet().embeber(basura)


@pytest.mark.skipif(FACENET_INSTALADO, reason="solo aplica donde facenet-pytorch no está instalado")
def test_sin_facenet_el_adaptador_avisa_que_el_modelo_no_esta_disponible() -> None:
    with pytest.raises(ModeloFacialNoDisponible, match="PyTorch"):
        ModeloFacialFacenet().embeber(_png())


def test_la_configuracion_elige_el_adaptador_real_sin_cargarlo() -> None:
    casos = construir_casos_rostros(Config("i", "j", modelo_facial="facenet", modelo_facial_version="9.9"))
    assert casos.identificar._modelo.nombre == "facenet-vggface2"  # noqa: SLF001
    assert casos.identificar._modelo.version == "9.9"  # noqa: SLF001


def test_sin_configurar_el_modelo_el_marcador_responde_503_al_usarse() -> None:
    casos = construir_casos_rostros(Config("i", "j"))
    assert isinstance(casos.identificar._modelo, ModeloFacialNoConfigurado)  # noqa: SLF001


def test_un_valor_desconocido_de_modelo_facial_falla_al_arrancar_y_no_en_pleno_uso() -> None:
    with pytest.raises(ValueError, match="MODELO_FACIAL"):
        construir_casos_rostros(Config("i", "j", modelo_facial="inventado"))
