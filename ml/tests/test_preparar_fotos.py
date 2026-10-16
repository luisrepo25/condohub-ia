from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ml.src import preparar_fotos


def _fechas(n: int) -> list[datetime]:
    return [datetime(2022, 1, 1) + timedelta(days=7 * i) for i in range(n)]


def test_las_sesiones_son_bloques_consecutivos_por_fecha() -> None:
    sesiones = preparar_fotos.agrupar_en_sesiones(_fechas(40), min_por_sesion=8, max_sesiones=4)
    assert sorted(set(sesiones)) == [0, 1, 2, 3]
    assert sesiones == sorted(sesiones)                       # ya vienen ordenadas por fecha
    assert all(sesiones.count(s) == 10 for s in range(4))


def test_el_orden_de_entrada_no_importa() -> None:
    fechas = _fechas(24)
    barajadas = fechas[::-1]
    a = preparar_fotos.agrupar_en_sesiones(fechas)
    b = preparar_fotos.agrupar_en_sesiones(barajadas)
    assert b == a[::-1]


def test_pocas_fotos_dejan_una_sola_sesion_y_sin_fotos_no_hay_sesiones() -> None:
    assert set(preparar_fotos.agrupar_en_sesiones(_fechas(5))) == {0}
    assert preparar_fotos.agrupar_en_sesiones([]) == []


def test_I1_sin_consentimiento_no_se_prepara_a_nadie(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    origen = tmp_path / "origen"
    origen.mkdir()
    Image.new("RGB", (100, 100)).save(origen / "a.jpg")
    with pytest.raises(ValueError, match="consentimiento"):
        preparar_fotos.preparar(origen, tmp_path / "salida", "ana")


def test_si_la_persona_trae_sesiones_en_subcarpetas_se_respetan(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    origen = tmp_path / "origen"
    for nombre, n in (("Luz alta", 5), ("Luz baja", 3)):
        (origen / nombre).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (1600, 1200)).save(origen / nombre / f"f{i}.jpg")
    (origen / "consentimiento.json").write_text('{"fecha": "2026-10-15", "version": "v1"}', encoding="utf-8")
    por_sesion = preparar_fotos.preparar(origen, tmp_path / "salida", "luis")
    assert por_sesion == {"Luz alta": 5, "Luz baja": 3}
    assert len(list((tmp_path / "salida" / "luis" / "Luz baja").glob("*.jpg"))) == 3


def test_prepara_reduce_y_deja_el_consentimiento(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    origen = tmp_path / "origen"
    origen.mkdir()
    for i in range(10):
        Image.new("RGB", (2000, 3000), (i * 20, 0, 0)).save(origen / f"f{i}.jpg")
    (origen / "consentimiento.json").write_text('{"fecha": "2026-10-15", "version": "v1"}', encoding="utf-8")
    por_sesion = preparar_fotos.preparar(origen, tmp_path / "salida", "ana")
    assert sum(por_sesion.values()) == 10
    fotos = list((tmp_path / "salida" / "ana").rglob("*.jpg"))
    assert len(fotos) == 10
    with Image.open(fotos[0]) as img:
        assert max(img.size) == 1280
        assert not img.getexif()                               # sin metadatos
    assert (tmp_path / "salida" / "ana" / "consentimiento.json").exists()
    assert (tmp_path / "salida" / "ana.zip").exists()
    assert len(list(origen.glob("*.jpg"))) == 10               # los originales no se tocan
