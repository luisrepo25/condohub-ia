"""Modelo facial real: MTCNN (detección y alineación) + FaceNet InceptionResnetV1 preentrenado en VGGFace2.

Es el mismo pipeline que `ml/src/embeddings.py`, de modo que lo medido en los notebooks vale para el servicio.
PyTorch y facenet-pytorch se importan al primer uso (no al arrancar), y si faltan, el servicio responde 503.
"""

import importlib
import io
from threading import Lock
from typing import Any

from app.application.rostros import ImagenInvalida, ModeloFacialNoDisponible
from app.domain.rostros import Embedding, normalizar

_LADO_MAXIMO = 1600   # una foto de 12 Mpx no mejora el rostro y sí retrasa a MTCNN


def _decodificar(imagen: bytes) -> Any:
    """Bytes → imagen PIL RGB, con la orientación EXIF corregida y reducida si es muy grande."""
    try:
        pil = importlib.import_module("PIL.Image")
        pil_ops = importlib.import_module("PIL.ImageOps")
    except ImportError as e:
        raise ModeloFacialNoDisponible("Falta Pillow en este servicio") from e
    try:
        with pil.open(io.BytesIO(imagen)) as original:
            rgb = pil_ops.exif_transpose(original).convert("RGB")
    except (pil.UnidentifiedImageError, pil.DecompressionBombError, OSError, ValueError) as e:
        raise ImagenInvalida("No se pudo leer la imagen") from e
    rgb.thumbnail((_LADO_MAXIMO, _LADO_MAXIMO))
    return rgb


class ModeloFacialFacenet:
    """Puerto `ModeloFacial`: `embeber(imagen)` → embedding normalizado de 512 dimensiones o `None`."""

    def __init__(
        self, dispositivo: str | None = None, nombre: str = "facenet-vggface2", version: str = "1.0.0"
    ) -> None:
        self.nombre = nombre
        self.version = version
        self._dispositivo = dispositivo
        self._motor: tuple[Any, Any, Any, str] | None = None
        self._cerrojo = Lock()

    def embeber(self, imagen: bytes) -> Embedding | None:
        rgb = _decodificar(imagen)                 # primero lo barato: una imagen ilegible no usa el modelo
        with self._cerrojo:
            mtcnn, facenet, torch, dispositivo = self._cargar()
            cara = mtcnn(rgb)                      # tensor [3, 160, 160] del rostro más probable, o None
            if cara is None:
                return None
            with torch.no_grad():
                vector = facenet(cara.unsqueeze(0).to(dispositivo)).squeeze(0).cpu().tolist()
        return normalizar(vector)

    def _cargar(self) -> tuple[Any, Any, Any, str]:
        if self._motor is None:
            try:
                torch = importlib.import_module("torch")
                facenet_pytorch = importlib.import_module("facenet_pytorch")
            except ImportError as e:
                raise ModeloFacialNoDisponible(
                    "Faltan PyTorch y facenet-pytorch (pip install torch torchvision; "
                    "pip install facenet-pytorch --no-deps)"
                ) from e
            dispositivo = self._dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
            mtcnn = facenet_pytorch.MTCNN(image_size=160, margin=0, post_process=True, device=dispositivo)
            red = facenet_pytorch.InceptionResnetV1(pretrained="vggface2").eval().to(dispositivo)
            self._motor = (mtcnn, red, torch, dispositivo)
        return self._motor
