import json

import numpy as np
import pytest

from ml.src import propios


def _foto(carpeta, nombre="a.jpg"):
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / nombre).write_bytes(b"x")


def _consentir(carpeta_persona, fecha="2026-10-14", version="borrador-1"):
    carpeta_persona.mkdir(parents=True, exist_ok=True)
    (carpeta_persona / "consentimiento.json").write_text(json.dumps({"fecha": fecha, "version": version}))


def test_I1_sin_consentimiento_no_se_lee_ninguna_foto(tmp_path) -> None:
    _foto(tmp_path / "luis" / "sesion_a")
    with pytest.raises(propios.SinConsentimiento):
        propios.listar_fotos(tmp_path)


def test_I1_el_consentimiento_exige_fecha_y_version(tmp_path) -> None:
    _consentir(tmp_path / "luis", version="")
    _foto(tmp_path / "luis" / "sesion_a")
    with pytest.raises(propios.SinConsentimiento):
        propios.listar_fotos(tmp_path)


def test_se_listan_fotos_por_persona_y_sesion(tmp_path) -> None:
    for persona in ("luis", "aldana"):
        _consentir(tmp_path / persona)
    _foto(tmp_path / "luis" / "2026-10-14_luz-baja", "1.jpg")
    _foto(tmp_path / "luis" / "2026-10-14_luz-alta", "2.PNG")
    _foto(tmp_path / "aldana" / "sesion_x", "3.jpeg")
    (tmp_path / "luis" / "2026-10-14_luz-baja" / "notas.txt").write_text("no es foto")
    fotos = propios.listar_fotos(tmp_path)
    assert [(f.persona, f.sesion) for f in fotos] == [
        ("aldana", "sesion_x"),
        ("luis", "2026-10-14_luz-alta"),
        ("luis", "2026-10-14_luz-baja"),
    ]


def test_fotos_sueltas_sin_subcarpeta_van_a_una_sesion_generica(tmp_path) -> None:
    _consentir(tmp_path / "luis")
    _foto(tmp_path / "luis")
    assert propios.listar_fotos(tmp_path)[0].sesion == propios.SIN_SESION


def test_cargar_imagen_devuelve_rgb_y_reduce_las_grandes(tmp_path) -> None:
    from PIL import Image

    ruta = tmp_path / "grande.jpg"
    Image.new("RGB", (3000, 2000), (200, 100, 50)).save(ruta)
    imagen = propios.cargar_imagen(ruta, lado_maximo=1000)
    assert imagen.dtype == np.uint8 and imagen.shape[2] == 3 and max(imagen.shape[:2]) == 1000
