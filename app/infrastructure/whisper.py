"""Transcripción local con faster-whisper (Whisper). El modelo se descarga y carga al primer uso."""

import io
from typing import Any

from app.application.reportes import AudioInvalido, TranscripcionNoDisponible


class TranscriptorWhisper:
    def __init__(
        self, modelo: str = "base", idioma: str = "es", dispositivo: str = "cpu", tipo_calculo: str = "int8"
    ) -> None:
        self._modelo, self._idioma = modelo, idioma
        self._dispositivo, self._tipo_calculo = dispositivo, tipo_calculo
        self._cargado: Any = None

    def _modelo_cargado(self) -> Any:
        if self._cargado is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise TranscripcionNoDisponible(
                    "faster-whisper no está instalado: pip install -e \".[voz]\""
                ) from e
            self._cargado = WhisperModel(
                self._modelo, device=self._dispositivo, compute_type=self._tipo_calculo
            )
        return self._cargado

    def transcribir(self, audio: bytes) -> str:
        modelo = self._modelo_cargado()
        try:
            segmentos, _ = modelo.transcribe(io.BytesIO(audio), language=self._idioma, vad_filter=True)
            texto = " ".join(s.text.strip() for s in segmentos).strip()
        except Exception as e:  # noqa: BLE001 - PyAV lanza varios tipos según el formato dañado
            raise AudioInvalido("No se pudo leer el audio (use WAV, MP3 o WebM)") from e
        if not texto:
            raise AudioInvalido("No se entendió ninguna instrucción en el audio")
        return texto