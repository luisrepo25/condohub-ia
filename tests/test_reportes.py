"""Reportes por voz (G4): dominio, casos de uso con dobles, adaptadores reales sin red y la API por HTTP."""

import io
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.application.reportes import (
    AudioInvalido,
    CasosReportes,
    ConstructorNoDisponible,
    ConsultaInvalida,
    GenerarArchivoReporte,
    GestionNoDisponible,
    IniciarReporte,
    ObtenerReporte,
    ReporteNoEncontrado,
)
from app.domain.errores import ReglaViolada
from app.domain.reportes import (
    EstadoReporte,
    FormatoReporte,
    exigir_solo_lectura,
    filas_desde_respuesta,
    formato_desde_instruccion,
)
from app.domain.roles import Rol, Solicitante
from app.infrastructure.gemini import ConstructorConsultaGemini, limpiar_respuesta
from app.infrastructure.gestion_http import GestionHttp
from app.infrastructure.graphql_analizador import AnalizadorGraphqlCore
from app.infrastructure.groq import ConstructorConsultaGroq
from app.infrastructure.keycloak import JwtVerificador
from app.infrastructure.memoria_reportes import AlmacenReportesMemoria, RepositorioReportesMemoria
from app.infrastructure.mongo.reportes import documento_a_reporte, reporte_a_documento
from app.infrastructure.reportes_archivos import GeneradorExcel, GeneradorPdf
from app.main import create_app
from tests.conftest import ISSUER, CrearToken, _clave

AHORA = datetime(2026, 10, 16, 15, 0, tzinfo=UTC)
ADMIN = Solicitante("sub-admin", frozenset({Rol.ADMIN}))
CONSULTA = "{ morosos { codigo deudaVencida propietarioNombre } }"
MUTATION = 'mutation { enlazarCuenta(personaId: "1", keycloakSub: "x") { nombre } }'
DATOS = {
    "morosos": [
        {"codigo": "B-101", "deudaVencida": "540.00", "propietarioNombre": "Ana"},
        {"codigo": "B-102", "deudaVencida": "270.00", "propietarioNombre": "Beto"},
    ]
}


# ------------------------------------------------------------------ dobles


class RelojFijo:
    def ahora(self) -> datetime:
        return AHORA


class TranscriptorFalso:
    def __init__(self, texto: str = "Dame los morosos del mes") -> None:
        self.texto = texto

    def transcribir(self, audio: bytes) -> str:
        return self.texto


class ConstructorFalso:
    def __init__(self, consulta: str = CONSULTA) -> None:
        self.consulta = consulta
        self.esquema_recibido = ""

    def construir(self, instruccion: str, esquema: str) -> str:
        self.esquema_recibido = esquema
        return self.consulta


class GestionFalsa:
    def __init__(self, datos: dict[str, Any] | None = None, falla: bool = False) -> None:
        self.datos = datos if datos is not None else DATOS
        self.falla = falla
        self.ejecutadas: list[tuple[str, str]] = []

    def esquema(self, token: str) -> str:
        return "type Query { morosos: [Moroso!]! }"

    def ejecutar(self, consulta: str, token: str) -> dict[str, Any]:
        self.ejecutadas.append((consulta, token))
        if self.falla:
            raise GestionNoDisponible("Gestión respondió HTTP 502")
        return self.datos


class Mundo:
    def __init__(
        self,
        constructor: ConstructorFalso | None = None,
        gestion: GestionFalsa | None = None,
        transcriptor: TranscriptorFalso | None = None,
    ) -> None:
        self.repositorio = RepositorioReportesMemoria()
        self.almacen = AlmacenReportesMemoria()
        self.gestion = gestion or GestionFalsa()
        self.constructor = constructor or ConstructorFalso()
        reloj = RelojFijo()
        self.casos = CasosReportes(
            iniciar=IniciarReporte(
                self.repositorio,
                transcriptor or TranscriptorFalso(),
                self.constructor,
                AnalizadorGraphqlCore(),
                self.gestion,
                reloj,
            ),
            generar=GenerarArchivoReporte(
                self.repositorio,
                self.gestion,
                {FormatoReporte.EXCEL: GeneradorExcel(), FormatoReporte.PDF: GeneradorPdf()},
                self.almacen,
                reloj,
            ),
            obtener=ObtenerReporte(self.repositorio, self.almacen, vigencia=timedelta(minutes=15)),
        )


# ------------------------------------------------------------------ dominio


def test_G4_una_query_es_de_solo_lectura() -> None:
    exigir_solo_lectura(["query"])


@pytest.mark.parametrize("operaciones", [["mutation"], ["query", "mutation"], ["subscription"], []])
def test_G4_cualquier_operacion_que_no_sea_query_se_rechaza(operaciones: list[str]) -> None:
    with pytest.raises(ReglaViolada) as e:
        exigir_solo_lectura(operaciones)
    assert e.value.codigo == "G4"


def test_el_formato_pedido_manda_luego_la_instruccion_y_por_defecto_excel() -> None:
    assert formato_desde_instruccion("morosos en pdf", None) is FormatoReporte.PDF
    assert formato_desde_instruccion("morosos en PDF", FormatoReporte.EXCEL) is FormatoReporte.EXCEL
    assert formato_desde_instruccion("morosos del mes", None) is FormatoReporte.EXCEL


def test_las_filas_salen_de_la_primera_lista_y_se_aplanan() -> None:
    datos = {"morosos": [{"codigo": "A", "titular": {"nombre": "Ana"}, "etiquetas": ["x", "y"]}]}
    assert filas_desde_respuesta(datos) == [{"codigo": "A", "titular.nombre": "Ana", "etiquetas": "x, y"}]
    assert filas_desde_respuesta({"salud": {"estado": "OK"}}) == [{"salud.estado": "OK"}]


# ------------------------------------------------------------------ analizador y Gemini


def test_el_analizador_distingue_query_de_mutation() -> None:
    a = AnalizadorGraphqlCore()
    assert a.operaciones("{ morosos { codigo } }") == ["query"]
    assert a.operaciones(MUTATION) == ["mutation"]
    assert a.operaciones("query A { a } mutation B { b }") == ["query", "mutation"]
    with pytest.raises(ConsultaInvalida):
        a.operaciones("{ morosos {")


def test_se_quita_el_cercado_de_codigo_de_la_respuesta_del_llm() -> None:
    assert limpiar_respuesta("```graphql\n{ a }\n```") == "{ a }"
    assert limpiar_respuesta("  { a }  ") == "{ a }"


def _gemini(
    respuesta: httpx.Response | Exception, clave: str = "k"
) -> tuple[ConstructorConsultaGemini, list[httpx.Request]]:
    vistas: list[httpx.Request] = []

    def atender(peticion: httpx.Request) -> httpx.Response:
        vistas.append(peticion)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    cliente = httpx.Client(transport=httpx.MockTransport(atender))
    return ConstructorConsultaGemini(clave, "modelo-x", "https://llm.test", cliente), vistas


def test_gemini_envia_el_esquema_y_la_clave_por_header_y_devuelve_la_consulta() -> None:
    cuerpo = {"candidates": [{"content": {"parts": [{"text": "```graphql\n{ morosos { codigo } }\n```"}]}}]}
    gemini, vistas = _gemini(httpx.Response(200, json=cuerpo))

    assert gemini.construir("los morosos", "type Query { morosos: [M!]! }") == "{ morosos { codigo } }"

    peticion = vistas[0]
    assert str(peticion.url) == "https://llm.test/v1beta/models/modelo-x:generateContent"
    assert peticion.headers["x-goog-api-key"] == "k"
    assert "k" not in str(peticion.url)
    assert b"type Query { morosos" in peticion.content


def test_gemini_sin_clave_o_con_error_responde_no_disponible_sin_filtrar_la_clave() -> None:
    sin_clave, _ = _gemini(httpx.Response(200, json={}), clave="")
    with pytest.raises(ConstructorNoDisponible):
        sin_clave.construir("x", "y")
    denegado, _ = _gemini(httpx.Response(403, json={"error": {"message": "denied"}}), clave="secreta")
    with pytest.raises(ConstructorNoDisponible) as e:
        denegado.construir("x", "y")
    assert "secreta" not in str(e.value)
    caido, _ = _gemini(httpx.ConnectError("sin red"))
    with pytest.raises(ConstructorNoDisponible):
        caido.construir("x", "y")


def _groq(respuesta: httpx.Response, clave: str = "k") -> tuple[ConstructorConsultaGroq, list[httpx.Request]]:
    vistas: list[httpx.Request] = []

    def atender(peticion: httpx.Request) -> httpx.Response:
        vistas.append(peticion)
        return respuesta

    cliente = httpx.Client(transport=httpx.MockTransport(atender))
    return ConstructorConsultaGroq(clave, "modelo-g", "https://groq.test", cliente), vistas


def test_groq_envia_el_esquema_con_bearer_y_devuelve_la_consulta() -> None:
    contenido = "```graphql\n{ morosos { codigo } }\n```"
    groq, vistas = _groq(httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]}))

    assert groq.construir("los morosos", "type Query { morosos: [M!]! }") == "{ morosos { codigo } }"

    assert str(vistas[0].url) == "https://groq.test/v1/chat/completions"
    assert vistas[0].headers["authorization"] == "Bearer k"
    assert b"type Query { morosos" in vistas[0].content


def test_groq_sin_clave_o_con_error_responde_no_disponible_sin_filtrar_la_clave() -> None:
    sin_clave, _ = _groq(httpx.Response(200, json={}), clave="")
    with pytest.raises(ConstructorNoDisponible, match="GROQ_API_KEY"):
        sin_clave.construir("x", "y")
    caido, _ = _groq(httpx.Response(500), clave="secreta")
    with pytest.raises(ConstructorNoDisponible) as e:
        caido.construir("x", "y")
    assert "secreta" not in str(e.value)


# ------------------------------------------------------------------ cliente de Gestión


def test_el_cliente_de_gestion_reenvia_el_token_y_traduce_los_errores() -> None:
    vistas: list[httpx.Request] = []

    def atender(peticion: httpx.Request) -> httpx.Response:
        vistas.append(peticion)
        if b"falla" in peticion.content:
            return httpx.Response(200, json={"errors": [{"message": "campo inexistente"}]})
        return httpx.Response(200, json={"data": {"salud": {"estado": "OK"}}})

    gestion = GestionHttp("http://gestion.test/graphql", httpx.Client(transport=httpx.MockTransport(atender)))

    assert gestion.ejecutar("{ salud { estado } }", "tok") == {"salud": {"estado": "OK"}}
    assert vistas[0].headers["authorization"] == "Bearer tok"
    with pytest.raises(GestionNoDisponible, match="campo inexistente"):
        gestion.ejecutar("{ falla }", "tok")


# ------------------------------------------------------------------ casos de uso


def test_G4_se_guardan_el_texto_la_consulta_y_el_archivo() -> None:
    mundo = Mundo()
    reporte = mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok-admin", None)

    assert reporte.estado is EstadoReporte.PROCESANDO
    assert reporte.transcripcion == "Dame los morosos del mes"
    assert reporte.consulta_graphql == CONSULTA
    assert "type Query" in mundo.constructor.esquema_recibido   # el LLM recibe el esquema real de Gestión

    mundo.casos.generar.generar(reporte.id, "tok-admin")

    listo = mundo.casos.obtener.obtener(reporte.id)
    assert listo.reporte.estado is EstadoReporte.LISTO
    assert listo.archivo_url == f"memoria://reportes/{reporte.id}.xlsx"
    assert mundo.gestion.ejecutadas == [(CONSULTA, "tok-admin")]   # se ejecuta con el token del ADMIN
    hoja = load_workbook(io.BytesIO(mundo.almacen.archivos[f"{reporte.id}.xlsx"])).active
    assert [c.value for c in hoja[1]] == ["codigo", "deudaVencida", "propietarioNombre"]
    assert [c.value for c in hoja[2]] == ["B-101", "540.00", "Ana"]


def test_G4_una_mutation_se_rechaza_y_queda_el_rastro_sin_ejecutarse() -> None:
    mundo = Mundo(constructor=ConstructorFalso(MUTATION))

    with pytest.raises(ReglaViolada) as e:
        mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    assert e.value.codigo == "G4"
    assert mundo.gestion.ejecutadas == []
    (rastro,) = mundo.repositorio._reportes.values()
    assert rastro.estado is EstadoReporte.FALLIDO
    assert "mutation" in (rastro.motivo_fallo or "")


def test_si_gestion_falla_el_reporte_queda_fallido_con_su_motivo() -> None:
    mundo = Mundo(gestion=GestionFalsa(falla=True))
    reporte = mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)

    mundo.casos.generar.generar(reporte.id, "tok")   # no lanza: el fallo queda en el reporte

    fallido = mundo.casos.obtener.obtener(reporte.id)
    assert fallido.reporte.estado is EstadoReporte.FALLIDO
    assert "502" in (fallido.reporte.motivo_fallo or "")
    assert fallido.archivo_url is None


def test_el_formato_pdf_se_toma_de_la_instruccion() -> None:
    mundo = Mundo(transcriptor=TranscriptorFalso("morosos en PDF"))
    reporte = mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)
    mundo.casos.generar.generar(reporte.id, "tok")

    assert mundo.almacen.archivos[f"{reporte.id}.pdf"].startswith(b"%PDF")


def test_audio_vacio_y_reporte_inexistente() -> None:
    mundo = Mundo()
    with pytest.raises(AudioInvalido):
        mundo.casos.iniciar.iniciar(ADMIN, b"", "tok", None)
    with pytest.raises(ReporteNoEncontrado):
        mundo.casos.obtener.obtener(uuid4())


# ------------------------------------------------------------------ archivos y MongoDB


def test_los_archivos_tienen_el_formato_pedido_aun_sin_resultados() -> None:
    assert GeneradorExcel().generar("t", [])[:2] == b"PK"
    assert GeneradorPdf().generar("t", [])[:4] == b"%PDF"
    assert GeneradorPdf().generar("t", [{"a": 1, "b": None}])[:4] == b"%PDF"


def test_el_documento_de_mongo_ida_y_vuelta_y_no_guarda_consultas_que_el_validador_rechaza() -> None:
    mundo = Mundo()
    reporte = mundo.casos.iniciar.iniciar(ADMIN, b"audio", "tok", None)
    assert documento_a_reporte(reporte_a_documento(reporte)) == reporte

    reporte.registrar_consulta("mutation { x }")
    assert "consultaGraphql" not in reporte_a_documento(reporte)   # mongo-init.js solo admite query o {


# ------------------------------------------------------------------ API


@pytest.fixture
def mundo() -> Mundo:
    return Mundo()


@pytest.fixture
def api(mundo: Mundo) -> TestClient:
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    return TestClient(create_app(verificador, casos_reportes=mundo.casos))


def _auth(crear_token: CrearToken, rol: str = "ADMIN") -> dict[str, str]:
    return {"Authorization": f"Bearer {crear_token(rol)}"}


def _voz(api: TestClient, crear_token: CrearToken, **form: str) -> Any:
    return api.post(
        "/reportes/voz",
        headers=_auth(crear_token),
        files={"audio": ("a.wav", b"audio", "audio/wav")},
        data=form,
    )


def test_G4_api_crea_el_reporte_y_el_archivo_queda_listo_con_url(
    api: TestClient, crear_token: CrearToken, mundo: Mundo
) -> None:
    r = _voz(api, crear_token)

    assert r.status_code == 202
    cuerpo = r.json()
    assert cuerpo["estado"] == "PROCESANDO"
    assert cuerpo["transcripcion"] == "Dame los morosos del mes"
    assert cuerpo["consultaGraphql"] == CONSULTA
    # TestClient ejecuta la tarea en segundo plano antes de devolver el control
    consulta = api.get(f"/reportes/{cuerpo['id']}", headers=_auth(crear_token))
    assert consulta.status_code == 200
    assert consulta.json()["estado"] == "LISTO"
    assert consulta.json()["archivoUrl"].endswith(".xlsx")
    assert mundo.gestion.ejecutadas[0][1].count(".") == 2   # Gestión recibe el JWT del ADMIN que lo pidió


def test_G4_api_una_mutation_da_422_con_code_G4(api: TestClient, crear_token: CrearToken) -> None:
    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    casos = Mundo(constructor=ConstructorFalso(MUTATION)).casos
    cliente = TestClient(create_app(verificador, casos_reportes=casos))

    r = _voz(cliente, crear_token)

    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "G4"


def test_api_los_errores_de_infraestructura_son_503_y_el_audio_vacio_400(
    api: TestClient, crear_token: CrearToken
) -> None:
    vacio = api.post(
        "/reportes/voz", headers=_auth(crear_token), files={"audio": ("a.wav", b"", "audio/wav")}
    )
    assert vacio.status_code == 400

    class Caido(ConstructorFalso):
        def construir(self, instruccion: str, esquema: str) -> str:
            raise ConstructorNoDisponible("El LLM no está configurado (falta GEMINI_API_KEY)")

    verificador = JwtVerificador(ISSUER, lambda _t: _clave.public_key())
    sin_llm = TestClient(create_app(verificador, casos_reportes=Mundo(constructor=Caido()).casos))
    r = _voz(sin_llm, crear_token)
    assert r.status_code == 503
    assert "GEMINI_API_KEY" in r.json()["detail"]


def test_api_reporte_inexistente_da_404_y_solo_admin_entra(api: TestClient, crear_token: CrearToken) -> None:
    assert api.get(f"/reportes/{UUID(int=7)}", headers=_auth(crear_token)).status_code == 404
    assert api.get(f"/reportes/{UUID(int=7)}", headers=_auth(crear_token, "RESIDENTE")).status_code == 403
    assert _voz(api, crear_token, formato="PDF").json()["formato"] == "PDF"
