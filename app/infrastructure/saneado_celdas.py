"""Los datos de Gestión que van a un reporte son texto: nunca una fórmula de Excel ni marcado de PDF (G4)."""

from typing import Any
from xml.sax.saxutils import escape


def texto_pdf(valor: object) -> str:
    """Escapa `&`, `<` y `>` para que reportlab muestre el texto tal cual y no lo interprete como marcado.

    Sin esto, un `<img src=...>` lee archivos del servidor y un `<b>` sin cerrar tira el reporte."""
    return escape("" if valor is None else str(valor))


def fijar_como_texto(celda: Any) -> None:
    """openpyxl guarda como fórmula el texto que empieza con «=»; se fuerza a cadena (queda intacto)."""
    if isinstance(celda.value, str) and celda.value.startswith("="):
        celda.data_type = "s"
