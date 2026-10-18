"""Lectura de placas en el servidor: caso de uso (A4, I5), endpoint, pre/posproceso y documento de MongoDB."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.application.leer_placa import (
    PLACA_LEIDA,
    SIN_PLACA,
    LectorPlacasNoDisponible,
    LecturaCruda,
    LeerPlaca,
)
from app.application.rostros import ImagenInvalida
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.lector_placas_onnx import (
    ALFABETO,
    LectorPlacasOnnx,
    _decodificar,
    _decodificar_texto,
    _letterbox,
)
from app.infrastructure.predicciones_placas import (
    RepositorioPrediccionesPlacasMemoria,
    prediccion_placa_a_documento,
)
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave
from tests.dobles import AHORA, RelojFijo

CAJA = (10.0, 20.0, 110.0, 60.0)


class LectorFalso:
    nombre = "lector-falso"
    version = "9.9"

    def __init__(self, lectura: LecturaCruda | None = None, error: Exception | None = None) -> None:
        self._lectura, self._error = lectura, error

    def leer(self, imagen: bytes) -> LecturaCruda | None:
        if self._error:
            raise self._error
        return self._lectura


def _caso(lector: LectorFalso) -> tuple[LeerPlaca, RepositorioPrediccionesPlacasMemoria]:
    repo = RepositorioPrediccionesPlacasMemoria()
    return LeerPlaca(lector, repo, RelojFijo()), repo


# ------------------------------------------------------------------ caso de uso


def test_A4_lectura_devuelve_texto_y_confianza_sin_decidir_el_ingreso() -> None:
    caso, _ = _caso(LectorFalso(LecturaCruda("2602SIT", 0.97, CAJA)))
    r = caso.ejecutar(b"img")
    assert (r.resultado, r.placa, r.confianza) == (PLACA_LEIDA, "2602SIT", 0.97)
    assert (r.modelo, r.version) == ("lector-falso", "9.9")


def test_I5_toda_lectura_guarda_modelo_version_confianza_y_fecha() -> None:
    caso, repo = _caso(LectorFalso(LecturaCruda("2602SIT", 0.97, CAJA)))
    r = caso.ejecutar(b"img", camara="porteria-1")
    (guardada_id, p), = repo.guardadas
    assert guardada_id == r.prediccion_id
    assert (p.modelo, p.version_modelo, p.confianza, p.fecha) == ("lector-falso", "9.9", 0.97, AHORA)
    assert (p.texto, p.caja, p.camara) == ("2602SIT", CAJA, "porteria-1")
    assert p.latencia_ms >= 0


def test_A4_sin_placa_detectada_es_SIN_PLACA_y_tambien_se_guarda() -> None:
    caso, repo = _caso(LectorFalso(None))
    r = caso.ejecutar(b"img")
    assert (r.resultado, r.placa, r.confianza) == (SIN_PLACA, None, 0.0)
    assert len(repo.guardadas) == 1


def test_A4_una_lectura_demasiado_corta_no_es_una_placa() -> None:
    caso, _ = _caso(LectorFalso(LecturaCruda("12", 0.99, CAJA)))
    assert caso.ejecutar(b"img").resultado == SIN_PLACA


def test_I5_la_confianza_queda_entre_0_y_1() -> None:
    caso, _ = _caso(LectorFalso(LecturaCruda("2602SIT", 1.0000001, CAJA)))
    assert caso.ejecutar(b"img").confianza == 1.0


def test_un_lector_no_disponible_no_guarda_nada() -> None:
    caso, repo = _caso(LectorFalso(error=LectorPlacasNoDisponible("sin modelos")))
    with pytest.raises(LectorPlacasNoDisponible):
        caso.ejecutar(b"img")
    assert repo.guardadas == []


# ------------------------------------------------------------------ endpoint


def _api(lector: LectorFalso) -> tuple[TestClient, RepositorioPrediccionesPlacasMemoria]:
    caso, repo = _caso(lector)
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    return TestClient(create_app(verificador, leer_placa=caso)), repo


def _auth(crear_token: CrearToken, rol: str = "GUARDIA") -> dict[str, str]:
    return {"Authorization": f"Bearer {crear_token(rol)}"}


IMAGEN = {"imagen": ("a.jpg", b"x", "image/jpeg")}


def test_A4_el_guardia_lee_una_placa_con_el_formato_del_contrato(crear_token: CrearToken) -> None:
    api, repo = _api(LectorFalso(LecturaCruda("2602SIT", 0.97, CAJA)))
    r = api.post("/placas/leer", headers=_auth(crear_token), files=IMAGEN, data={"camara": "porteria-1"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"resultado", "placa", "confianza", "modelo", "version", "prediccionId"}
    assert (cuerpo["resultado"], cuerpo["placa"], cuerpo["confianza"]) == ("PLACA_LEIDA", "2602SIT", 0.97)
    assert repo.guardadas[0][1].camara == "porteria-1"


def test_A4_sin_placa_responde_200_con_SIN_PLACA(crear_token: CrearToken) -> None:
    api, _ = _api(LectorFalso(None))
    cuerpo = api.post("/placas/leer", headers=_auth(crear_token, "ADMIN"), files=IMAGEN).json()
    assert (cuerpo["resultado"], cuerpo["placa"]) == ("SIN_PLACA", None)


def test_imagen_ilegible_es_400_problem_json(crear_token: CrearToken) -> None:
    api, _ = _api(LectorFalso(error=ImagenInvalida("no es una imagen")))
    r = api.post("/placas/leer", headers=_auth(crear_token), files=IMAGEN)
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/problem+json")


def test_modelos_ausentes_son_503_problem_json(crear_token: CrearToken) -> None:
    api, _ = _api(LectorFalso(error=LectorPlacasNoDisponible("sin modelos")))
    r = api.post("/placas/leer", headers=_auth(crear_token), files=IMAGEN)
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/problem+json")


@pytest.mark.parametrize("rol", ["RESIDENTE"])
def test_un_residente_no_lee_placas(crear_token: CrearToken, rol: str) -> None:
    api, repo = _api(LectorFalso(LecturaCruda("2602SIT", 0.97, CAJA)))
    assert api.post("/placas/leer", headers=_auth(crear_token, rol), files=IMAGEN).status_code == 403
    assert repo.guardadas == []


# ------------------------------------------------------------------ pre y posproceso


def _probabilidades(texto: str, maximo: float = 0.9) -> np.ndarray:
    """[10, 37] con `maximo` en la letra de cada posición y relleno `_` en el resto."""
    p = np.full((10, len(ALFABETO)), (1 - maximo) / (len(ALFABETO) - 1))
    for i in range(10):
        p[i, ALFABETO.index(texto[i] if i < len(texto) else "_")] = maximo
    return p


def test_el_lector_decodifica_el_texto_y_promedia_la_confianza_sin_contar_el_relleno() -> None:
    texto, confianza = _decodificar_texto(_probabilidades("2602SIT", 0.9))
    assert texto == "2602SIT"
    assert confianza == pytest.approx(0.9)


def test_un_lector_que_solo_ve_relleno_devuelve_texto_vacio_y_confianza_0() -> None:
    assert _decodificar_texto(_probabilidades("")) == ("", 0.0)


def test_letterbox_deja_la_imagen_en_640_sin_deformarla() -> None:
    tensor, escala, px, py = _letterbox(np.zeros((320, 640, 3), dtype=np.uint8))
    assert tensor.shape == (1, 3, 640, 640)
    assert (escala, px, py) == (1.0, 0, 160)
    assert tensor.dtype == np.float32 and 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_el_detector_devuelve_la_caja_en_coordenadas_de_la_imagen_original_y_aplica_nms() -> None:
    # dos candidatas casi iguales y una de confianza baja; salida cruda [1, 5, 3] (cx, cy, w, h, conf)
    salida = np.array([[[320.0, 322.0, 100.0], [320.0, 320.0, 100.0], [100.0, 100.0, 50.0],
                        [40.0, 40.0, 50.0], [0.9, 0.8, 0.1]]])
    cajas = _decodificar(salida, escala=1.0, px=0, py=0, forma=(640, 640))
    assert len(cajas) == 1                                  # NMS descarta la repetida y el filtro, la débil
    assert tuple(np.round(cajas[0][:4])) == (270.0, 300.0, 370.0, 340.0)
    assert cajas[0][4] == pytest.approx(0.9)


# ------------------------------------------------------------------ adaptador


def test_una_imagen_que_no_lo_es_se_rechaza_antes_de_cargar_los_modelos() -> None:
    lector = LectorPlacasOnnx("no-existe.onnx", "tampoco.onnx")
    with pytest.raises(ImagenInvalida):
        lector.leer(b"esto no es una imagen")


def test_sin_los_archivos_de_modelo_el_servicio_dice_503_no_revienta() -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, "PNG")
    lector = LectorPlacasOnnx("no-existe.onnx", "tampoco.onnx")
    with pytest.raises(LectorPlacasNoDisponible):
        lector.leer(buffer.getvalue())


MODELOS = Path(__file__).resolve().parents[1] / "modelos"
HAY_MODELOS = (MODELOS / "yolov8n_placas_int8.onnx").exists() and (MODELOS / "cct_xs_v2_global.onnx").exists()


@pytest.mark.skipif(not HAY_MODELOS, reason="faltan los .onnx en modelos/ (no van a git, I6)")
def test_con_los_modelos_reales_una_imagen_sin_placa_da_None() -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (120, 120, 120)).save(buffer, "JPEG")
    detector, lector_texto = MODELOS / "yolov8n_placas_int8.onnx", MODELOS / "cct_xs_v2_global.onnx"
    assert LectorPlacasOnnx(str(detector), str(lector_texto)).leer(buffer.getvalue()) is None


# ------------------------------------------------------------------ MongoDB


def test_I5_el_documento_de_la_prediccion_usa_los_nombres_del_validador() -> None:
    caso, repo = _caso(LectorFalso(LecturaCruda("2602SIT", 0.97, CAJA)))
    caso.ejecutar(b"img", camara="porteria-1")
    doc = prediccion_placa_a_documento(repo.guardadas[0][1])
    assert doc["tipo"] == "PLACA"
    assert doc["resultado"]["texto"] == "2602SIT" and doc["resultado"]["bbox"] == list(CAJA)
    assert doc["modelo"] == {"nombre": "lector-falso", "version": "9.9"}
    assert doc["confianza"] == 0.97 and doc["fecha"] == AHORA and doc["camara"] == "porteria-1"
