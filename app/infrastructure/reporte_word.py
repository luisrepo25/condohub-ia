"""Genera el Word (.docx) de un reporte sin librerías externas: un .docx es un ZIP con XML (Office Open XML).

Mismo criterio de seguridad que Excel y PDF (G4): los datos son texto; se escapa el marcado y se quitan
los caracteres que XML no admite.
"""

import io
import re
import zipfile
from xml.sax.saxutils import escape

from app.domain.reportes import Fila
from app.domain.tablas import Tabla

_NO_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
_ESPACIOS = re.compile(r"[\t\r\n]+")
_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_ANCHO_UTIL = 15398  # A4 apaisado (16838) menos márgenes de 720 twips, en twips

_TIPOS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/docProps/core.xml" '
    'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    "</Types>"
)
_RELACIONES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    '<Relationship Id="rId2" '
    'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
    'Target="docProps/core.xml"/>'
    "</Relationships>"
)
_BORDES = "".join(
    f'<w:{lado} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV")
)


def _texto(valor: object) -> str:
    """Texto seguro para XML: sin caracteres de control, en una línea y con `&`, `<` y `>` escapados."""
    bruto = "" if valor is None else str(valor)
    return escape(_ESPACIOS.sub(" ", _NO_XML.sub("", bruto)))


def _parrafo(texto: object, negrita: bool = False, tam: int = 18) -> str:
    formato = f'<w:rPr>{"<w:b/>" if negrita else ""}<w:sz w:val="{tam}"/></w:rPr>'
    return f'<w:p><w:r>{formato}<w:t xml:space="preserve">{_texto(texto)}</w:t></w:r></w:p>'


def _celda(texto: object, ancho: int, encabezado: bool) -> str:
    fondo = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if encabezado else ""
    propiedades = f'<w:tcPr><w:tcW w:w="{ancho}" w:type="dxa"/>{fondo}</w:tcPr>'
    return f"<w:tc>{propiedades}{_parrafo(texto, negrita=encabezado, tam=16)}</w:tc>"


def _tabla(tabla: Tabla) -> str:
    columnas = tabla.columnas
    if not columnas:
        return _parrafo("Sin resultados.")
    ancho = max(_ANCHO_UTIL // len(columnas), 600)
    cuadricula = "".join(f'<w:gridCol w:w="{ancho}"/>' for _ in columnas)
    encabezado = (
        "<w:tr><w:trPr><w:tblHeader/></w:trPr>"
        + "".join(_celda(c, ancho, True) for c in columnas)
        + "</w:tr>"
    )
    filas = "".join(
        "<w:tr>" + "".join(_celda(f.get(c), ancho, False) for c in columnas) + "</w:tr>" for f in tabla.filas
    )
    propiedades = f'<w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>{_BORDES}</w:tblBorders></w:tblPr>'
    return f"<w:tbl>{propiedades}<w:tblGrid>{cuadricula}</w:tblGrid>{encabezado}{filas}</w:tbl>"


def _documento(titulo: str, tablas: list[Tabla]) -> str:
    cuerpo = [_parrafo(titulo, negrita=True, tam=36)]
    for tabla in tablas:
        if len(tablas) > 1:
            cuerpo.append(_parrafo(tabla.nombre, negrita=True, tam=26))
        cuerpo.append(_tabla(tabla))
        cuerpo.append(_parrafo(""))  # Word exige un párrafo entre dos tablas y al final del cuerpo
    pagina = (
        '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
        '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    contenido = "".join(cuerpo)
    cabecera = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    return f"{cabecera}<w:document {_NS}><w:body>{contenido}{pagina}</w:body></w:document>"


def _propiedades(titulo: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{_texto(titulo[:250])}</dc:title><dc:creator>CondoHub</dc:creator></cp:coreProperties>"
    )


class GeneradorWord:
    extension = "docx"
    tipo = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def generar(self, titulo: str, filas: list[Fila]) -> bytes:
        return self.generar_tablas(titulo, [Tabla("Reporte", filas)])

    def generar_tablas(self, titulo: str, tablas: list[Tabla]) -> bytes:
        salida = io.BytesIO()
        with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as docx:
            docx.writestr("[Content_Types].xml", _TIPOS)
            docx.writestr("_rels/.rels", _RELACIONES)
            docx.writestr("word/document.xml", _documento(titulo, tablas))
            docx.writestr("docProps/core.xml", _propiedades(titulo))
        return salida.getvalue()
