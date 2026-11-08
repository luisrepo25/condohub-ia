"""G4 · formato Word, varias tablas por archivo y criterios de punta a punta (casos de uso y API)."""

import io
import re
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.application.reportes import (
    CasosReportes,
    GenerarArchivoReporte,
    GestionNoDisponible,
    IniciarReporte,
    ObtenerReporte,
)
from app.domain.errores import ReglaViolada
from app.domain.reportes import EstadoReporte, FormatoReporte, formato_desde_instruccion
from app.domain.roles import Rol, Solicitante
from app.domain.tablas import Tabla
from app.infrastructure.graphql_analizador import AnalizadorGraphqlCore
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.memoria_reportes import AlmacenReportesMemoria, RepositorioReportesMemoria
from app.infrastructure.reporte_word import GeneradorWord
from app.infrastructure.reportes_por_tablas import ExcelPorTablas, PdfPorTablas
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave

AHORA = datetime(2026, 11, 8, 15, 0, tzinfo=UTC)
ADMIN = Solicitante("sub-admin", frozenset({Rol.ADMIN}))
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
DATOS = {
    "morosos": [
        {"codigo": "A-101", "bloque": "A", "deudaVencida": "135.00"},
        {"codigo": "B-101", "bloque": "B", "deudaVencida": "540.00"},
        {"codigo": "B-102", "bloque": "B", "deudaVencida": "1000.00"},
    ],
    "pagosPendientes": [
        {"monto": "300.00", "estado": "PENDIENTE"},
        {"monto": "50.00", "estado": "PENDIENTE"},
    ],
}


class _Reloj:
    def ahora(self) -> datetime:
        return AHORA


class _Transcriptor:
    def transcribir(self, audio: bytes) -> str:
        return "morosos del bloque B en Word"


class _Constructor:
    def __init__(self, consulta: str) -> None:
        self.consulta = consulta

    def construir(self, instruccion: str, esquema: str) -> str:
        return self.consulta


class _Gestion:
    def __init__(self, datos: dict[str, Any]) -> None:
        self.datos = datos
        self.ejecutadas: list[str] = []

    def esquema(self, token: str) -> str:
        return "type Query { morosos: [Moroso!]! }"

    def ejecutar(self, consulta: str, token: str) -> dict[str, Any]:
        self.ejecutadas.append(consulta)
        if "falla" in consulta:
            raise GestionNoDisponible("Gestión respondió HTTP 502")
        return self.datos


class Mundo:
    def __init__(self, consulta: str, datos: dict[str, Any] | None = None) -> None:
        self.repositorio = RepositorioReportesMemoria()
        self.almacen = AlmacenReportesMemoria()
        self.gestion = _Gestion(datos or DATOS)
        reloj = _Reloj()
        iniciar = IniciarReporte(
            self.repositorio,
            _Transcriptor(),
            _Constructor(consulta),
            AnalizadorGraphqlCore(),
            self.gestion,
            reloj,
        )
        self.casos = CasosReportes(
            iniciar=iniciar,
            iniciar_texto=iniciar,
            generar=GenerarArchivoReporte(
                self.repositorio,
                self.gestion,
                {
                    FormatoReporte.EXCEL: ExcelPorTablas(),
                    FormatoReporte.PDF: PdfPorTablas(),
                    FormatoReporte.WORD: GeneradorWord(),
                },
                self.almacen,
                reloj,
            ),
            obtener=ObtenerReporte(self.repositorio, self.almacen, vigencia=timedelta(minutes=15)),
        )

    def generar(self, formato: FormatoReporte | None = None) -> tuple[Any, bytes]:
        reporte = self.casos.iniciar.iniciar(ADMIN, b"audio", "tok", formato)
        self.casos.generar.generar(reporte.id, "tok")
        listo = self.casos.obtener.obtener(reporte.id).reporte
        clave = listo.archivo_key
        return listo, self.almacen.archivos[clave] if clave else b""


def _docx(contenido: bytes) -> ElementTree.Element:
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        assert z.testzip() is None
        assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= set(z.namelist())
        for nombre in z.namelist():
            ElementTree.fromstring(z.read(nombre))  # todo el paquete es XML bien formado
        return ElementTree.fromstring(z.read("word/document.xml"))


def _filas_word(documento: ElementTree.Element) -> list[list[str]]:
    return [
        ["".join(t.text or "" for t in celda.iter(f"{{{NS['w']}}}t")) for celda in fila.findall("w:tc", NS)]
        for fila in documento.iter(f"{{{NS['w']}}}tr")
    ]


# ------------------------------------------------------------------ formato


def test_G4_word_se_toma_de_la_instruccion_pero_el_formato_elegido_y_pdf_mandan() -> None:
    assert formato_desde_instruccion("morosos en Word", None) is FormatoReporte.WORD
    assert formato_desde_instruccion("exporta a docx", None) is FormatoReporte.WORD
    assert formato_desde_instruccion("morosos en PDF", None) is FormatoReporte.PDF
    assert formato_desde_instruccion("morosos en Word", FormatoReporte.EXCEL) is FormatoReporte.EXCEL
    assert formato_desde_instruccion("morosos por bloque", None) is FormatoReporte.EXCEL


def test_G4_el_reporte_en_word_es_un_docx_valido_con_encabezado_y_filas() -> None:
    mundo = Mundo("{ morosos { codigo deudaVencida } }", {"morosos": DATOS["morosos"]})

    reporte, archivo = mundo.generar(FormatoReporte.WORD)

    assert reporte.estado is EstadoReporte.LISTO and reporte.archivo_key.endswith(".docx")
    filas = _filas_word(_docx(archivo))
    assert filas[0] == ["codigo", "bloque", "deudaVencida"]
    assert filas[1] == ["A-101", "A", "135.00"]
    assert len(filas) == 4  # encabezado + 3 morosos


def test_G4_word_trata_los_datos_como_texto_sin_romper_el_xml() -> None:
    datos = [
        {
            "codigo": "<w:p>&inyectado</w:p>",
            "nota": '=HYPERLINK("x")\x00\x07 con\nsalto',
            "unicode": "Ñandú – año",
        }
    ]

    archivo = GeneradorWord().generar("Título <b>&</b>", datos)

    documento = _docx(archivo)
    celdas = _filas_word(documento)[1]
    assert celdas[0] == "<w:p>&inyectado</w:p>"  # llega como texto, no como marcado
    assert celdas[1] == '=HYPERLINK("x") con salto'  # sin caracteres de control ni saltos
    assert celdas[2] == "Ñandú – año"
    assert len(list(documento.iter(f"{{{NS['w']}}}tbl"))) == 1


def test_G4_word_sin_resultados_y_con_varias_tablas() -> None:
    vacio = _docx(GeneradorWord().generar("Reporte", []))
    assert "Sin resultados." in "".join(t.text or "" for t in vacio.iter(f"{{{NS['w']}}}t"))

    doble = _docx(
        GeneradorWord().generar_tablas(
            "Reporte", [Tabla("morosos", [{"a": "1"}]), Tabla("pagosPendientes", [{"b": "2"}])]
        )
    )
    textos = [t.text for t in doble.iter(f"{{{NS['w']}}}t")]
    assert "morosos" in textos and "pagosPendientes" in textos  # un título por tabla
    assert len(list(doble.iter(f"{{{NS['w']}}}tbl"))) == 2


# ------------------------------------------------------------------ criterios y varias tablas


def test_G4_los_criterios_filtran_el_archivo_y_las_columnas_auxiliares_no_aparecen() -> None:
    mundo = Mundo(
        "{ morosos { codigo bloque deudaVencida } }\n"
        '# ESPECIFICACION: {"filtros": [{"campo": "bloque", "op": "=", "valor": "B"}], '
        '"orden": [{"campo": "deudaVencida", "sentido": "desc"}], "ocultar": ["bloque"]}'
    )

    reporte, archivo = mundo.generar(FormatoReporte.EXCEL)

    hoja = load_workbook(io.BytesIO(archivo)).active
    assert [[c.value for c in fila] for fila in hoja.iter_rows()] == [
        ["codigo", "deudaVencida"],
        ["B-102", "1000.00"],
        ["B-101", "540.00"],
    ]
    assert reporte.vista_previa and reporte.vista_previa.total == 2
    assert (
        "ESPECIFICACION" in reporte.consulta_graphql
    )  # la consulta guardada incluye los criterios (auditoría, G4)


def test_G4_agrupar_por_bloque_produce_una_fila_por_bloque_con_su_total() -> None:
    mundo = Mundo(
        "{ morosos { bloque deudaVencida } }\n"
        '# ESPECIFICACION: {"agrupar": {"por": ["bloque"], '
        '"calcular": [{"fn": "suma", "campo": "deudaVencida", '
        '"como": "Deuda total"}, {"fn": "contar", "como": "Unidades"}]}}',
        {"morosos": DATOS["morosos"]},
    )

    _, archivo = mundo.generar(FormatoReporte.WORD)

    assert _filas_word(_docx(archivo)) == [
        ["bloque", "Deuda total", "Unidades"],
        ["A", "135.00", "1"],
        ["B", "1540.00", "2"],
    ]


def test_G4_varias_listas_son_varias_hojas_en_excel() -> None:
    mundo = Mundo("{ morosos { codigo deudaVencida } pagosPendientes { monto estado } }")

    _, archivo = mundo.generar(FormatoReporte.EXCEL)

    libro = load_workbook(io.BytesIO(archivo))
    assert libro.sheetnames == ["morosos", "pagosPendientes"]
    assert [c.value for c in libro["pagosPendientes"][1]] == ["monto", "estado"]
    assert libro["morosos"].max_row == 4


def test_G4_varias_listas_son_varias_secciones_en_word_y_en_pdf() -> None:
    mundo = Mundo("{ morosos { codigo } pagosPendientes { monto } }")

    _, word = mundo.generar(FormatoReporte.WORD)
    _, pdf = mundo.generar(FormatoReporte.PDF)

    assert len(list(_docx(word).iter(f"{{{NS['w']}}}tbl"))) == 2
    assert pdf.startswith(b"%PDF")


def test_G4_dos_hojas_con_el_mismo_nombre_o_nombres_invalidos_no_rompen_el_excel() -> None:
    archivo = ExcelPorTablas().generar_tablas(
        "R", [Tabla("a/b:c", [{"x": "1"}]), Tabla("a/b:c", [{"x": "2"}]), Tabla("", [{"x": "3"}])]
    )

    nombres = load_workbook(io.BytesIO(archivo)).sheetnames

    assert len(set(n.casefold() for n in nombres)) == 3 and all(len(n) <= 31 for n in nombres)


def test_G4_el_criterio_con_un_campo_inexistente_deja_el_reporte_fallido_con_su_motivo() -> None:
    mundo = Mundo(
        '{ morosos { codigo } }\n# ESPECIFICACION: {"filtros": [{"campo": "piso", "op": "=", "valor": "3"}]}'
    )

    reporte, _ = mundo.generar()

    assert reporte.estado is EstadoReporte.FALLIDO
    assert "piso" in reporte.motivo_fallo and "G4" not in "".join(mundo.gestion.ejecutadas)


def test_G4_una_especificacion_mal_formada_se_rechaza_al_iniciar_y_queda_el_rastro_sin_ejecutarse() -> None:
    mundo = Mundo("{ morosos { codigo } }\n# ESPECIFICACION: esto no es json")

    with pytest.raises(ReglaViolada) as e:
        mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    assert e.value.codigo == "G4"
    assert mundo.gestion.ejecutadas == []  # nunca llegó a Gestión
    (rastro,) = mundo.repositorio._reportes.values()  # type: ignore[attr-defined]
    assert rastro.estado is EstadoReporte.FALLIDO


def test_G4_un_comentario_no_puede_colar_una_mutation_ni_cambiar_que_solo_se_lee() -> None:
    mundo = Mundo(
        'mutation { enlazarCuenta(personaId: "1", keycloakSub: "x") { nombre } }\n# ESPECIFICACION: {}'
    )

    with pytest.raises(ReglaViolada, match="mutation"):
        mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    assert mundo.gestion.ejecutadas == []


def test_G4_la_consulta_con_la_especificacion_primero_se_guarda_empezando_por_query_o_llave() -> None:
    mundo = Mundo('# ESPECIFICACION: {"limite": 1}\n{ morosos { codigo } }')

    reporte = mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    assert re.match(
        r"^\s*(query\b|\{)", reporte.consulta_graphql
    )  # lo exige el validador de la colección `reportes`


# ------------------------------------------------------------------ API


@pytest.fixture
def mundo() -> Mundo:
    return Mundo("{ morosos { codigo deudaVencida } }")


@pytest.fixture
def api(mundo: Mundo) -> TestClient:
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    return TestClient(create_app(verificador, casos_reportes=mundo.casos))


def test_G4_api_acepta_formato_word_y_entrega_un_docx(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    cabecera = {"Authorization": f"Bearer {crear_token('ADMIN')}"}

    r = api.post("/reportes/voz", headers=cabecera, data={"instruccion": "morosos", "formato": "WORD"})

    assert r.status_code == 202 and r.json()["formato"] == "WORD"
    listo = api.get(f"/reportes/{r.json()['id']}", headers=cabecera).json()
    assert listo["estado"] == "LISTO" and listo["archivoUrl"].endswith(".docx")


def test_G4_api_sigue_rechazando_formatos_que_no_existen(api: TestClient, crear_token: CrearToken) -> None:
    cabecera = {"Authorization": f"Bearer {crear_token('ADMIN')}"}

    r = api.post("/reportes/voz", headers=cabecera, data={"instruccion": "morosos", "formato": "ODT"})

    assert r.status_code == 422 or r.status_code == 400


def test_G4_api_responde_422_con_code_G4_si_los_criterios_estan_mal_formados(
    crear_token: CrearToken,
) -> None:
    mundo = Mundo('{ morosos { codigo } }\n# ESPECIFICACION: {"limite": 0}')
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    api = TestClient(create_app(verificador, casos_reportes=mundo.casos))

    r = api.post(
        "/reportes/voz",
        headers={"Authorization": f"Bearer {crear_token('ADMIN')}"},
        data={"instruccion": "morosos limite cero"},
    )

    assert r.status_code == 422 and r.json()["code"] == "G4"
