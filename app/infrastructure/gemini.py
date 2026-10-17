"""Arma la consulta GraphQL con Gemini (API REST de Google AI Studio, sin SDK).

La clave viene de GEMINI_API_KEY."""

import re

import httpx

from app.application.reportes import ConstructorNoDisponible

INSTRUCCION_DEL_SISTEMA = """Traduces instrucciones en español a consultas GraphQL de un condominio.
Reglas estrictas:
- Responde SOLO con el texto de una consulta GraphQL, sin explicaciones ni bloques de código markdown.
- Usa únicamente operaciones `query`. Nunca `mutation` ni `subscription`.
- Usa solo tipos, campos y argumentos que existan en el esquema dado; no inventes ninguno.
- Pide los campos necesarios para responder la instrucción.

Esquema GraphQL (SDL):
"""

_CERCADO = re.compile(r"^```(?:graphql)?\s*|\s*```$", re.IGNORECASE)


def limpiar_respuesta(texto: str) -> str:
    """Quita el cercado de código que algunos modelos agregan aunque se les pida lo contrario."""
    return _CERCADO.sub("", texto.strip()).strip()


class ConstructorConsultaGemini:
    def __init__(
        self,
        api_key: str,
        modelo: str,
        url_base: str = "https://generativelanguage.googleapis.com",
        cliente: httpx.Client | None = None,
        tiempo_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._modelo = modelo
        self._url_base = url_base.rstrip("/")
        self._cliente = cliente or httpx.Client(timeout=tiempo_s)

    def construir(self, instruccion: str, esquema: str) -> str:
        if not self._api_key:
            raise ConstructorNoDisponible("El LLM no está configurado (falta GEMINI_API_KEY)")
        cuerpo = {
            "systemInstruction": {"parts": [{"text": INSTRUCCION_DEL_SISTEMA + esquema}]},
            "contents": [{"role": "user", "parts": [{"text": instruccion}]}],
            "generationConfig": {"temperature": 0},
        }
        try:
            r = self._cliente.post(
                f"{self._url_base}/v1beta/models/{self._modelo}:generateContent",
                json=cuerpo,
                headers={"x-goog-api-key": self._api_key},
            )
        except httpx.HTTPError as e:
            raise ConstructorNoDisponible("No se pudo conectar con el LLM") from e
        if r.status_code != 200:
            raise ConstructorNoDisponible(f"El LLM respondió HTTP {r.status_code}")
        try:
            texto = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ConstructorNoDisponible("El LLM no devolvió una respuesta utilizable") from e
        return limpiar_respuesta(str(texto))