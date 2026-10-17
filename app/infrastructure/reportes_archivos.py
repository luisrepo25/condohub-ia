"""Genera el Excel (openpyxl) o el PDF (reportlab) de un reporte a partir de filas planas."""

import io

from app.domain.reportes import Fila


def _columnas(filas: list[Fila]) -> list[str]:
    columnas: list[str] = []
    for fila in filas:
        for clave in fila:
            if clave not in columnas:
                columnas.append(clave)
    return columnas


class GeneradorExcel:
    extension = "xlsx"
    tipo = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def generar(self, titulo: str, filas: list[Fila]) -> bytes:
        from openpyxl import Workbook
        from openpyxl.styles import Font

        libro = Workbook()
        libro.properties.title = titulo[:250]
        hoja = libro.active
        hoja.title = "Reporte"
        columnas = _columnas(filas)
        hoja.append(columnas or ["Sin resultados"])
        for celda in hoja[1]:
            celda.font = Font(bold=True)
        for fila in filas:
            hoja.append([fila.get(c) for c in columnas])
        for i, columna in enumerate(columnas, start=1):
            ancho = max([len(columna)] + [len(str(f.get(columna, ""))) for f in filas]) + 2
            hoja.column_dimensions[hoja.cell(row=1, column=i).column_letter].width = min(ancho, 60)
        salida = io.BytesIO()
        libro.save(salida)
        return salida.getvalue()


class GeneradorPdf:
    extension = "pdf"
    tipo = "application/pdf"

    def generar(self, titulo: str, filas: list[Fila]) -> bytes:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        estilos = getSampleStyleSheet()
        salida = io.BytesIO()
        documento = SimpleDocTemplate(salida, pagesize=landscape(A4), title=titulo[:250], author="CondoHub")
        contenido: list[object] = [Paragraph(titulo, estilos["Title"]), Spacer(1, 12)]
        columnas = _columnas(filas)
        if not filas:
            contenido.append(Paragraph("Sin resultados.", estilos["Normal"]))
        else:
            celda = estilos["BodyText"].clone("celda", fontSize=7, leading=9)
            datos = [[Paragraph(str(c), celda) for c in columnas]]
            for f in filas:
                datos.append([Paragraph("" if f.get(c) is None else str(f.get(c)), celda) for c in columnas])
            tabla = Table(datos, repeatRows=1)
            tabla.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            contenido.append(tabla)
        documento.build(contenido)
        return salida.getvalue()