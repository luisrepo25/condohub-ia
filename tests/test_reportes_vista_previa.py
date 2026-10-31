# ruff: noqa: F811, F401  (se reutilizan los fixtures `api` y `mundo` de tests/test_reportes.py)
"""Vista previa del reporte (G4): el resultado se visualiza en pantalla además de exportarse a Excel o PDF."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.domain.reportes import vista_previa_desde
from app.infrastructure.mongo.reportes import documento_a_reporte, reporte_a_documento
from tests.conftest import CrearToken
from tests.test_reportes import ADMIN, CONSULTA, DATOS, GestionFalsa, Mundo, api, mundo


def test_G4_la_vista_previa_alinea_las_filas_con_las_columnas_y_pasa_los_valores_a_texto() -> None:
    filas = [
        {"codigo": "B-101", "deuda": "540.00", "n": 2},
        {"codigo": "B-102", "deuda": None, "extra": True},
    ]

    v = vista_previa_desde(filas)

    assert v.columnas == ["codigo", "deuda", "n", "extra"]
    assert v.filas == [["B-101", "540.00", "2", None], ["B-102", None, None, "True"]]
    assert v.total == 2


def test_G4_la_vista_previa_se_limita_pero_el_total_cuenta_todas_las_filas() -> None:
    filas = [{"i": i} for i in range(120)]

    v = vista_previa_desde(filas, limite=50)

    assert len(v.filas) == 50
    assert v.total == 120


def test_G4_sin_resultados_la_vista_previa_esta_vacia() -> None:
    v = vista_previa_desde([])

    assert (v.columnas, v.filas, v.total) == ([], [], 0)


def test_G4_al_generar_el_reporte_guarda_las_primeras_filas_para_verlas_en_pantalla() -> None:
    m = Mundo()
    reporte = m.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    m.casos.generar.generar(reporte.id, "tok")

    guardado = m.repositorio.por_id(reporte.id)
    assert guardado is not None and guardado.vista_previa is not None
    assert guardado.vista_previa.columnas == ["codigo", "deudaVencida", "propietarioNombre"]
    assert guardado.vista_previa.filas[0] == ["B-101", "540.00", "Ana"]
    assert guardado.vista_previa.total == len(DATOS["morosos"])


def test_G4_un_reporte_fallido_no_trae_vista_previa() -> None:
    m = Mundo(gestion=GestionFalsa(falla=True))
    reporte = m.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    m.casos.generar.generar(reporte.id, "tok")

    guardado = m.repositorio.por_id(reporte.id)
    assert guardado is not None and guardado.vista_previa is None


def test_G4_la_vista_previa_sobrevive_a_ir_y_volver_de_mongo() -> None:
    m = Mundo()
    reporte = m.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)
    m.casos.generar.generar(reporte.id, "tok")
    original = m.repositorio.por_id(reporte.id)
    assert original is not None

    doc = reporte_a_documento(original)
    restaurado = documento_a_reporte(doc)

    assert doc["vistaPrevia"]["columnas"] == ["codigo", "deudaVencida", "propietarioNombre"]
    assert restaurado.vista_previa == original.vista_previa


@pytest.mark.usefixtures("mundo")
def test_G4_api_el_reporte_listo_devuelve_la_vista_previa(api: TestClient, crear_token: CrearToken) -> None:  # noqa: F811
    cabeceras = {"Authorization": f"Bearer {crear_token('ADMIN')}"}
    creado = api.post("/reportes/voz", headers=cabeceras, files={"audio": ("a.wav", b"audio", "audio/wav")})

    cuerpo: dict[str, Any] = api.get(f"/reportes/{creado.json()['id']}", headers=cabeceras).json()

    assert cuerpo["estado"] == "LISTO"
    assert cuerpo["vistaPrevia"]["columnas"] == ["codigo", "deudaVencida", "propietarioNombre"]
    assert cuerpo["vistaPrevia"]["filas"][1] == ["B-102", "270.00", "Beto"]
    assert cuerpo["vistaPrevia"]["total"] == 2
    assert CONSULTA == cuerpo["consultaGraphql"]


def test_G4_api_mientras_procesa_todavia_no_hay_vista_previa(
    api: TestClient, crear_token: CrearToken
) -> None:
    cabeceras = {"Authorization": f"Bearer {crear_token('ADMIN')}"}

    creado = api.post(
        "/reportes/voz", headers=cabeceras, files={"audio": ("a.wav", b"audio", "audio/wav")}
    ).json()

    assert creado.get("vistaPrevia") is None
