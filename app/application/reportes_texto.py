"""Reportes por texto (G4): la orden escrita entra al mismo flujo que la hablada, sin pasar por Whisper."""

from app.application.reportes import AudioInvalido

MAXIMO_CARACTERES = 1000


class TranscriptorDeTexto:
    """Pasa el texto por el puerto `Transcriptor`: el flujo, las validaciones y el rastro son los mismos."""

    def transcribir(self, audio: bytes) -> str:
        texto = audio.decode("utf-8", errors="replace").strip()
        if not texto:
            raise AudioInvalido("La instrucción está vacía")
        if len(texto) > MAXIMO_CARACTERES:
            raise AudioInvalido(f"La instrucción no puede pasar de {MAXIMO_CARACTERES} caracteres")
        return texto
