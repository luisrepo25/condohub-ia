"""Inyección en los archivos de los reportes por voz (G4): los datos son texto, no fórmulas ni marcado.

Un nombre guardado como `=HYPERLINK(...)` se ejecutaría al abrir el Excel; un `<img src=...>` en el PDF
haría que reportlab lea archivos del servidor; y un marcado mal cerrado hacía fallar el reporte entero.
"""

import base64
import io
import re
import zipfile
from pathlib import Path

from openpyxl import load_workbook

from app.infrastructure.reportes_archivos import GeneradorExcel, GeneradorPdf

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAfbLI3wAAAABJRU5ErkJggg=="
)


def _hoja(contenido: bytes):
    return load_workbook(io.BytesIO(contenido)).active


def test_G4_excel_una_formula_en_un_dato_se_guarda_como_texto() -> None:
    peligrosa = '=HYPERLINK("http://malo.example/x","Haz clic")'
    contenido = GeneradorExcel().generar("t", [{"codigo": "B-101", "propietarioNombre": peligrosa}])

    celda = _hoja(contenido).cell(row=2, column=2)

    assert celda.value == peligrosa  # el texto llega intacto, sin apóstrofos añadidos
    assert celda.data_type == "s"  # y es texto, no una fórmula
    xml = zipfile.ZipFile(io.BytesIO(contenido)).read("xl/worksheets/sheet1.xml").decode()
    assert not re.search(r"<f[ >]", xml)  # el XML no contiene ninguna fórmula


def test_G4_excel_los_otros_valores_no_cambian() -> None:
    filas = [{"codigo": "B-101", "deuda": "540.00", "nota": "texto normal", "vacio": None, "n": 7}]

    hoja = _hoja(GeneradorExcel().generar("t", filas))

    assert [c.value for c in hoja[2]] == ["B-101", "540.00", "texto normal", None, 7]


def test_G4_pdf_el_marcado_dentro_de_una_celda_no_se_interpreta(tmp_path: Path) -> None:
    imagen = tmp_path / "secreto.png"
    imagen.write_bytes(PNG_1X1)
    celda = f"<img src='{imagen.as_posix()}' width='20' height='20'/>"

    pdf = GeneradorPdf().generar("t", [{"nombre": celda}])

    assert b"/Subtype /Image" not in pdf  # no se leyó ningún archivo del servidor


def test_G4_pdf_un_marcado_roto_no_hace_fallar_el_reporte() -> None:
    pdf = GeneradorPdf().generar("t", [{"nombre": "texto <b>sin cerrar", "otro": "a & b < c"}])

    assert pdf.startswith(b"%PDF")


def test_G4_pdf_el_titulo_tampoco_se_interpreta_como_marcado() -> None:
    pdf = GeneradorPdf().generar("Morosos <b>del mes", [{"codigo": "A-1"}])

    assert pdf.startswith(b"%PDF")
