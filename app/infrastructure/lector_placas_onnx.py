"""Lector de placas en el servidor: detector YOLOv8n (ONNX int8) + lector global preentrenado (ONNX).

Hace lo mismo que la demo del navegador (`ml/web/demo_webcam/`): detecta la placa, la recorta con un 15 %
de margen, la deforma a 128 × 64 RGB y lee hasta 10 caracteres de cualquier formato. Los modelos se cargan
al primer uso; si faltan los archivos o `onnxruntime`, lanza `LectorPlacasNoDisponible` (503).
Los `.onnx` no van a git (I6).
"""

import io
import threading
from typing import Any

from app.application.leer_placa import LectorPlacasNoDisponible, LecturaCruda
from app.application.rostros import ImagenInvalida

TAMANO = 640            # el detector recibe 640 × 640
RELLENO = 114           # gris del letterbox
IOU_NMS = 0.45
CONFIANZA_DETECTOR = 0.25
MARGEN = 0.15           # margen alrededor de la caja del detector
ANCHO_OCR, ALTO_OCR = 128, 64
ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
RELLENO_OCR = "_"
MAX_LADO = 4096         # una foto más grande no se procesa entera: se reduce antes de detectar


class LectorPlacasOnnx:
    nombre = "yolov8n-placas+fast-plate-ocr-global"

    def __init__(self, ruta_detector: str, ruta_lector: str, version: str = "1.0.0") -> None:
        self.version = version
        self._rutas = (ruta_detector, ruta_lector)
        self._sesiones: tuple[Any, Any] | None = None
        self._cerrojo = threading.Lock()

    # ------------------------------------------------------------------ carga

    def _cargar(self) -> tuple[Any, Any]:
        with self._cerrojo:
            if self._sesiones is None:
                try:
                    import onnxruntime as ort  # type: ignore[import-untyped]
                except ImportError as e:   # pragma: no cover - depende del entorno
                    raise LectorPlacasNoDisponible(
                        'Falta onnxruntime: pip install -e ".[placas]"'
                    ) from e
                try:
                    detector, lector = (
                        ort.InferenceSession(r, providers=["CPUExecutionProvider"]) for r in self._rutas
                    )
                    self._sesiones = (detector, lector)
                except Exception as e:   # archivo ausente o corrupto
                    raise LectorPlacasNoDisponible(
                        "No se pudieron cargar los modelos de lectura de placas"
                    ) from e
            assert self._sesiones is not None
            return self._sesiones

    # ------------------------------------------------------------------ lectura

    def leer(self, imagen: bytes) -> LecturaCruda | None:
        import numpy as np
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(imagen)) as abierta:
                abierta.load()
                pil = abierta.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ImagenInvalida("La imagen no se puede leer (se espera JPEG o PNG)") from e
        sesion_detector, sesion_lector = self._cargar()

        if max(pil.size) > MAX_LADO:
            pil.thumbnail((MAX_LADO, MAX_LADO))
        rgb = np.asarray(pil)
        alto, ancho = rgb.shape[:2]

        tensor, escala, px, py = _letterbox(rgb)
        salida = sesion_detector.run(None, {sesion_detector.get_inputs()[0].name: tensor})[0]
        cajas = _decodificar(salida, escala, px, py, (alto, ancho))
        if len(cajas) == 0:
            return None
        x1, y1, x2, y2, _ = cajas[int(np.argmax(cajas[:, 4]))]

        w, h = x2 - x1, y2 - y1
        sx, sy = max(0.0, x1 - MARGEN * w), max(0.0, y1 - MARGEN * h)
        ex, ey = min(float(ancho), x2 + MARGEN * w), min(float(alto), y2 + MARGEN * h)
        if ex - sx < 2 or ey - sy < 2:
            return None
        recorte = pil.crop((round(sx), round(sy), round(ex), round(ey))).resize(
            (ANCHO_OCR, ALTO_OCR), Image.Resampling.BILINEAR
        )
        x = np.asarray(recorte, dtype=np.uint8)[None]   # [1, 64, 128, 3] RGB
        nombres = [s.name for s in sesion_lector.get_outputs()]
        probabilidades = sesion_lector.run(
            [nombres[0] if "plate" not in nombres else "plate"],
            {sesion_lector.get_inputs()[0].name: x},
        )[0]
        texto, confianza = _decodificar_texto(probabilidades[0])
        return LecturaCruda(texto, confianza, (float(x1), float(y1), float(x2), float(y2)))


# ---------------------------------------------------------------------- detector (numpy)

def _letterbox(rgb: Any) -> tuple[Any, float, int, int]:
    """RGB uint8 [h, w, 3] → (tensor float32 [1, 3, 640, 640] en 0..1, escala, relleno_x, relleno_y)."""
    import numpy as np
    from PIL import Image

    h, w = rgb.shape[:2]
    escala = min(TAMANO / h, TAMANO / w)
    nh, nw = max(1, round(h * escala)), max(1, round(w * escala))
    reducida = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
    lienzo = np.full((TAMANO, TAMANO, 3), RELLENO, dtype=np.uint8)
    px, py = (TAMANO - nw) // 2, (TAMANO - nh) // 2
    lienzo[py : py + nh, px : px + nw] = reducida
    return lienzo.astype(np.float32).transpose(2, 0, 1)[None] / 255.0, float(escala), px, py


def _iou(caja: Any, cajas: Any) -> Any:
    import numpy as np

    x1, y1 = np.maximum(caja[0], cajas[:, 0]), np.maximum(caja[1], cajas[:, 1])
    x2, y2 = np.minimum(caja[2], cajas[:, 2]), np.minimum(caja[3], cajas[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (caja[2] - caja[0]) * (caja[3] - caja[1])
    areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _decodificar(salida: Any, escala: float, px: int, py: int, forma: tuple[int, int]) -> Any:
    """Salida cruda [1, 5, N] → detecciones [n, 5] (x1, y1, x2, y2, confianza), ya en la imagen original."""
    import numpy as np

    p = np.squeeze(salida, axis=0) if salida.ndim == 3 else salida
    if p.ndim != 2 or p.shape[0] != 5:
        raise LectorPlacasNoDisponible(f"El detector devolvió una salida inesperada: {salida.shape}")
    p = p.T
    p = p[p[:, 4] >= CONFIANZA_DETECTOR]
    if len(p) == 0:
        return np.zeros((0, 5))
    alto, ancho = forma
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    cajas = np.stack(
        [
            (cx - w / 2 - px) / escala,
            (cy - h / 2 - py) / escala,
            (cx + w / 2 - px) / escala,
            (cy + h / 2 - py) / escala,
        ],
        axis=1,
    )
    cajas[:, [0, 2]] = np.clip(cajas[:, [0, 2]], 0, ancho)
    cajas[:, [1, 3]] = np.clip(cajas[:, [1, 3]], 0, alto)
    orden = list(np.argsort(-p[:, 4]))
    quedan: list[int] = []
    while orden:
        mejor = orden.pop(0)
        quedan.append(int(mejor))
        if orden:
            solape = _iou(cajas[mejor], cajas[orden])
            orden = [i for i, s in zip(orden, solape, strict=True) if s <= IOU_NMS]
    return np.column_stack([cajas[quedan], p[quedan, 4]])


# ---------------------------------------------------------------------- lector de texto

def _decodificar_texto(probabilidades: Any) -> tuple[str, float]:
    """[10, 37] → (texto, confianza): media de la probabilidad máxima de las posiciones que no son relleno."""
    import numpy as np

    mejores = np.argmax(probabilidades, axis=1)
    maximos = np.max(probabilidades, axis=1)
    texto, suma, n = "", 0.0, 0
    for indice, maximo in zip(mejores, maximos, strict=True):
        letra = ALFABETO[int(indice)]
        if letra != RELLENO_OCR:
            texto += letra
            suma += float(maximo)
            n += 1
    return texto, (suma / n if n else 0.0)
