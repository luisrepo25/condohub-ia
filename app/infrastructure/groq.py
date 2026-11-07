"""Arma la consulta GraphQL con un modelo alojado en Groq (API compatible con OpenAI, sin SDK).

La clave viene de GROQ_API_KEY.
"""

import httpx

from app.application.reportes import ConstructorNoDisponible
from app.infrastructure.claves_groq import CODIGOS_DE_CLAVE_AGOTADA, ClavesGroq
from app.infrastructure.gemini import INSTRUCCION_DEL_SISTEMA, limpiar_respuesta


class ConstructorConsultaGroq:
    def __init__(
        self,
        api_key: str,
        modelo: str,
        url_base: str = "https://api.groq.com/openai",
        cliente: httpx.Client | None = None,
        tiempo_s: float = 60.0,
    ) -> None:
        self._claves = ClavesGroq(api_key)  # una clave, o varias separadas por comas
        self._modelo = modelo
        self._url_base = url_base.rstrip("/")
        self._cliente = cliente or httpx.Client(timeout=tiempo_s)

    def construir(self, instruccion: str, esquema: str) -> str:
        if not self._claves:
            raise ConstructorNoDisponible("El LLM no está configurado (falta GROQ_API_KEY)")
        cuerpo = {
            "model": self._modelo,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": INSTRUCCION_DEL_SISTEMA + esquema},
                {"role": "user", "content": instruccion},
            ],
        }
        for rotadas in range(len(self._claves)):
            try:
                r = self._cliente.post(
                    f"{self._url_base}/v1/chat/completions",
                    json=cuerpo,
                    headers={"Authorization": f"Bearer {self._claves.actual()}"},
                )
            except httpx.HTTPError as e:
                raise ConstructorNoDisponible("No se pudo conectar con el LLM") from e
            if r.status_code not in CODIGOS_DE_CLAVE_AGOTADA or rotadas == len(self._claves) - 1:
                break
            self._claves.rotar()  # límite agotado o clave rechazada: se prueba la siguiente
        if r.status_code != 200:
            raise ConstructorNoDisponible(f"El LLM respondió HTTP {r.status_code}")
        try:
            texto = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ConstructorNoDisponible("El LLM no devolvió una respuesta utilizable") from e
        return limpiar_respuesta(str(texto))