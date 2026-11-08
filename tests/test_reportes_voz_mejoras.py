"""Reportes por voz (G4): se entiende mejor lo dicho y se pide solo lo que se pidió.

El 08/11/2026 «genera una lista de unidades morosas con su deuda en Excel» salió con el correo y el teléfono
de cada propietario y el UUID de la unidad (nadie lo pidió), y Whisper transcribió «excelente» por «Excel»
y se comió «genera».
"""

import json
from typing import Any

import httpx

from app.infrastructure.gemini import INSTRUCCION_DEL_SISTEMA
from app.infrastructure.groq import ConstructorConsultaGroq
from app.infrastructure.whisper import TranscriptorWhisper


class _Segmento:
    text = " lista de morosos "


class _ModeloFalso:
    def __init__(self) -> None:
        self.argumentos: dict[str, Any] = {}

    def transcribe(self, _audio: Any, **kwargs: Any) -> tuple[list[_Segmento], None]:
        self.argumentos = kwargs
        return [_Segmento()], None


def test_G4_el_prompt_pide_solo_los_campos_mencionados_y_no_datos_de_contacto_ni_ids() -> None:
    texto = " ".join(INSTRUCCION_DEL_SISTEMA.lower().split())

    assert "solo los campos que la instrucción" in texto
    assert "unidadid" in texto and "correo" in texto and "teléfono" in texto
    assert "salvo que la instrucción los pida" in texto


def test_G4_el_prompt_avisa_que_la_instruccion_viene_de_una_transcripcion_con_errores() -> None:
    assert "excelente" in INSTRUCCION_DEL_SISTEMA.lower()


def test_G4_el_prompt_nuevo_llega_al_llm_de_groq_junto_con_el_esquema() -> None:
    cuerpos: list[dict[str, Any]] = []

    def atender(p: httpx.Request) -> httpx.Response:
        cuerpos.append(json.loads(p.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{ morosos { codigo } }"}}]})

    cliente = httpx.Client(transport=httpx.MockTransport(atender))
    ConstructorConsultaGroq("k", "m", cliente=cliente).construir(
        "morosos con su deuda", "type Query { x: Int }"
    )

    sistema = cuerpos[0]["messages"][0]["content"]
    assert "solo los campos que la instrucción" in " ".join(sistema.lower().split())
    assert "type Query { x: Int }" in sistema


def test_G4_whisper_recibe_vocabulario_del_dominio_para_no_oir_excelente_por_excel() -> None:
    transcriptor = TranscriptorWhisper()
    modelo = _ModeloFalso()
    transcriptor._cargado = modelo

    transcriptor.transcribir(b"audio")

    ayuda = modelo.argumentos["initial_prompt"]
    assert "Excel" in ayuda and "PDF" in ayuda and "unidades morosas" in ayuda
    assert modelo.argumentos["language"] == "es"
    assert modelo.argumentos["beam_size"] == 5
