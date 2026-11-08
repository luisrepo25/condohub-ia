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
- Pide solo los campos que la instrucción menciona. Por ejemplo, «unidades morosas con su deuda» pide
  codigo, bloque, deudaVencida y expensasVencidas. No pidas identificadores internos (unidadId), correos ni
  teléfonos, salvo que la instrucción los pida expresamente.
- La instrucción viene de una transcripción de voz y puede traer errores («excelente» por «Excel»):
  interpreta la intención.

Criterios (solo si la instrucción pide filtrar, agrupar, ordenar o limitar):
- El esquema no tiene argumentos de filtro. Pide los campos en la consulta y declara los criterios en UNA línea de comentario
  al FINAL, con JSON: # ESPECIFICACION: {...}  La consulta SIEMPRE empieza por `query` o `{`, nunca por el comentario.
- Claves opcionales: "tabla" (nombre del campo raíz al que se aplica; si hay varias listas y no la pones, vale para todas las
  que tengan los campos), "filtros", "agrupar", "orden", "limite" y "ocultar".
- "filtros": [{"campo": "bloque", "op": "=", "valor": "B"}]. Operadores: =, !=, >, >=, <, <=, contiene. Los números van como
  número (3), no como texto. Los campos son los de la respuesta, con punto si están anidados (titular.nombre).
- "agrupar": {"por": ["bloque"], "calcular": [{"fn": "suma", "campo": "deudaVencida", "como": "Deuda total"},
  {"fn": "contar", "como": "Unidades"}]}. Funciones: contar, suma, promedio, minimo, maximo. Úsalo para «por bloque», «total por…».
- "orden": [{"campo": "deudaVencida", "sentido": "desc"}] (sentido asc o desc; tras agrupar se puede ordenar por el nombre de "como").
- "limite": 10 para «los 10 primeros», «el mayor», etc.
- Si filtras u ordenas por un campo que la instrucción no pide como columna, inclúyelo en la consulta y agrégalo a
  "ocultar": ["bloque"] para que no aparezca. No inventes campos que no estén en el esquema.
- Para varias tablas (por ejemplo morosos y pagos pendientes) pide varios campos raíz en la MISMA consulta.
- Ejemplo: «morosos del bloque B con más de 2 expensas vencidas, solo código y deuda»:
  { morosos { codigo bloque deudaVencida expensasVencidas } }
  # ESPECIFICACION: {"filtros": [{"campo": "bloque", "op": "=", "valor": "B"}, {"campo": "expensasVencidas", "op": ">", "valor": 2}], "ocultar": ["bloque", "expensasVencidas"]}

Esquema GraphQL (SDL):
"""  # noqa: E501 - es el texto del prompt: se mantiene en líneas completas

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