"""Arma los casos de uso de los reportes por voz con sus adaptadores reales a partir de la configuración."""

import importlib
from datetime import timedelta

from app.application.reportes import (
    AlmacenReportes,
    CasosReportes,
    ConstructorConsulta,
    GenerarArchivoReporte,
    IniciarReporte,
    ObtenerReporte,
    RepositorioReportes,
)
from app.config import Config
from app.domain.reportes import FormatoReporte
from app.infrastructure.composicion import RelojSistema
from app.infrastructure.gemini import ConstructorConsultaGemini
from app.infrastructure.gestion_http import GestionHttp
from app.infrastructure.graphql_analizador import AnalizadorGraphqlCore
from app.infrastructure.groq import ConstructorConsultaGroq
from app.infrastructure.memoria_reportes import AlmacenReportesMemoria, RepositorioReportesMemoria
from app.infrastructure.mongo.reportes import RepositorioReportesMongo
from app.infrastructure.reportes_archivos import GeneradorExcel, GeneradorPdf
from app.infrastructure.s3_cliente import ClienteS3
from app.infrastructure.s3_firmador import FirmadorS3
from app.infrastructure.s3_reportes import AlmacenReportesS3
from app.infrastructure.whisper import TranscriptorWhisper


def _repositorio(config: Config) -> RepositorioReportes:
    if config.persistencia == "memoria":
        return RepositorioReportesMemoria()
    if config.persistencia == "mongo":
        cliente = importlib.import_module("pymongo").MongoClient(config.mongo_uri, tz_aware=True)
        return RepositorioReportesMongo(cliente[config.mongo_db]["reportes"])
    raise ValueError(f"PERSISTENCIA desconocida: {config.persistencia!r} (use «memoria» o «mongo»)")


def _almacen(config: Config, reloj: RelojSistema) -> AlmacenReportes:
    if config.almacen_reportes == "memoria":
        return AlmacenReportesMemoria()
    if config.almacen_reportes == "s3":
        cliente = ClienteS3(config.s3_endpoint, config.s3_acceso, config.s3_secreto, config.s3_region)
        firmador = FirmadorS3(
            config.s3_endpoint,
            config.s3_bucket_reportes,
            config.s3_acceso,
            config.s3_secreto,
            config.s3_region,
        )
        return AlmacenReportesS3(cliente, firmador, config.s3_bucket_reportes, reloj)
    raise ValueError(f"ALMACEN_REPORTES desconocido: {config.almacen_reportes!r} (use «memoria» o «s3»)")


def _constructor(config: Config) -> ConstructorConsulta:
    """`LLM_PROVEEDOR=gemini` (por defecto) o `groq`; la clave de cada uno viene solo del entorno."""
    if config.llm_proveedor == "gemini":
        return ConstructorConsultaGemini(config.gemini_api_key, config.llm_modelo, config.llm_url)
    if config.llm_proveedor == "groq":
        return ConstructorConsultaGroq(config.groq_api_key, config.groq_modelo, config.groq_url)
    raise ValueError(f"LLM_PROVEEDOR desconocido: {config.llm_proveedor!r} (use «gemini» o «groq»)")


def construir_casos_reportes(config: Config) -> CasosReportes:
    reloj = RelojSistema()
    repositorio = _repositorio(config)
    gestion = GestionHttp(config.gestion_graphql_url)
    almacen = _almacen(config, reloj)
    return CasosReportes(
        iniciar=IniciarReporte(
            repositorio,
            TranscriptorWhisper(
                config.whisper_modelo, "es", config.whisper_dispositivo, config.whisper_calculo
            ),
            _constructor(config),
            AnalizadorGraphqlCore(),
            gestion,
            reloj,
        ),
        generar=GenerarArchivoReporte(
            repositorio,
            gestion,
            {FormatoReporte.EXCEL: GeneradorExcel(), FormatoReporte.PDF: GeneradorPdf()},
            almacen,
            reloj,
        ),
        obtener=ObtenerReporte(repositorio, almacen, timedelta(seconds=config.url_firmada_segundos)),
    )