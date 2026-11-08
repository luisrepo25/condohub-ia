"""Excel y PDF con una tabla por lista de la consulta (G4). Extienden a los de una tabla sin modificarlos."""

import io
import re

from app.domain.tablas import Tabla
from app.infrastructure.reportes_archivos import GeneradorExcel, GeneradorPdf
from app.infrastructure.saneado_celdas import fijar_como_texto, texto_pdf

_NO_HOJA = re.compile(r"[\[\]:*?/\\]")


def _nombre_hoja(nombre: str, usados: set[str]) -> str:
    """Excel limita el nombre a 31 caracteres, sin `[]:*?/\\`, y no admite dos hojas con el mismo nombre."""
    base = (_NO_HOJA.sub(" ", nombre).strip() or "Hoja")[:31]
    candidato, n = base, 2
    while candidato.casefold() in usados:
        sufijo = f" {n}"
        candidato, n = base[: 31 - len(sufijo)] + sufijo, n + 1
    usados.add(candidato.casefold())
    return candidato


class ExcelPorTablas(GeneradorExcel):
    def generar_tablas(self, titulo: str, tablas: list[Tabla]) -> bytes:
        from openpyxl import Workbook
        from openpyxl.styles import Font

        libro = Workbook()
        libro.properties.title = titulo[:250]
        libro.remove(libro.active)
        usados: set[str] = set()
        for tabla in tablas:
            hoja = libro.create_sheet(_nombre_hoja(tabla.nombre, usados))
            columnas = tabla.columnas
            hoja.append(columnas or ["Sin resultados"])
            for celda in hoja[1]:
                celda.font = Font(bold=True)
            for fila in tabla.filas:
                hoja.append([fila.get(c) for c in columnas])
                for celda in hoja[hoja.max_row]:
                    fijar_como_texto(celda)
            for i, columna in enumerate(columnas, start=1):
                ancho = max([len(columna)] + [len(str(f.get(columna, ""))) for f in tabla.filas]) + 2
                hoja.column_dimensions[hoja.cell(row=1, column=i).column_letter].width = min(ancho, 60)
        salida = io.BytesIO()
        libro.save(salida)
        return salida.getvalue()


class PdfPorTablas(GeneradorPdf):
    def generar_tablas(self, titulo: str, tablas: list[Tabla]) -> bytes:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        estilos = getSampleStyleSheet()
        celda = estilos["BodyText"].clone("celda", fontSize=7, leading=9)
        salida = io.BytesIO()
        documento = SimpleDocTemplate(salida, pagesize=landscape(A4), title=titulo[:250], author="CondoHub")
        contenido: list[object] = [Paragraph(texto_pdf(titulo), estilos["Title"]), Spacer(1, 12)]
        for tabla in tablas:
            contenido.append(Paragraph(texto_pdf(tabla.nombre), estilos["Heading2"]))
            columnas = tabla.columnas
            if not tabla.filas:
                contenido.append(Paragraph("Sin resultados.", estilos["Normal"]))
                continue
            datos = [[Paragraph(texto_pdf(c), celda) for c in columnas]]
            datos += [[Paragraph(texto_pdf(f.get(c)), celda) for c in columnas] for f in tabla.filas]
            cuadro = Table(datos, repeatRows=1)
            cuadro.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            contenido += [cuadro, Spacer(1, 14)]
        documento.build(contenido)
        return salida.getvalue()
