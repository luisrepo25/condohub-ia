"""Modelos faciales preentrenados: MTCNN (detección y alineación a 160×160) + InceptionResnetV1 (VGGFace2).

Necesita PyTorch y facenet-pytorch (se instalan en Colab; no están en el servicio). Los embeddings salen de 512
dimensiones normalizados L2, de modo que el producto punto es la similitud coseno.
"""

from collections.abc import Sequence

import numpy as np


def _dispositivo(dispositivo: str | None) -> str:
    import torch

    return dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")


def crear_embebedor(dispositivo: str | None = None, lote: int = 64):  # noqa: ANN201 - depende de torch (import diferido)
    """Devuelve `embeber(imagenes) -> (embeddings, detectado)`; `detectado[i]` es False sin rostro."""
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image

    dev = _dispositivo(dispositivo)
    mtcnn = MTCNN(image_size=160, margin=0, post_process=True, device=dev)
    modelo = InceptionResnetV1(pretrained="vggface2").eval().to(dev)

    def embeber(imagenes: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        salida = np.zeros((len(imagenes), 512), dtype=np.float32)
        detectado = np.zeros(len(imagenes), dtype=bool)
        for inicio in range(0, len(imagenes), lote):
            caras, indices = [], []
            for k in range(inicio, min(inicio + lote, len(imagenes))):
                cara = mtcnn(Image.fromarray(imagenes[k]))
                if cara is not None:
                    caras.append(cara)
                    indices.append(k)
            if caras:
                with torch.no_grad():
                    vectores = modelo(torch.stack(caras).to(dev)).cpu().numpy()
                vectores /= np.linalg.norm(vectores, axis=1, keepdims=True)
                salida[indices] = vectores
                detectado[indices] = True
        return salida, detectado

    return embeber


def crear_recortador(dispositivo: str | None = None):  # noqa: ANN201
    """Devuelve `recortar(imagenes) -> (caras uint8 [n,160,160,3], detectado)`: solo MTCNN, sin FaceNet.

    Las caras se recortan UNA vez y se reutilizan en el entrenamiento y en la evaluación, para no repetir la detección.
    """
    from facenet_pytorch import MTCNN
    from PIL import Image

    mtcnn = MTCNN(image_size=160, margin=0, post_process=False, device=_dispositivo(dispositivo))

    def recortar(imagenes: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        caras = np.zeros((len(imagenes), 160, 160, 3), dtype=np.uint8)
        detectado = np.zeros(len(imagenes), dtype=bool)
        for k, imagen in enumerate(imagenes):
            cara = mtcnn(Image.fromarray(imagen))
            if cara is not None:
                caras[k] = cara.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
                detectado[k] = True
        return caras, detectado

    return recortar


def crear_modelo_base(dispositivo: str | None = None):  # noqa: ANN201
    """FaceNet preentrenado en VGGFace2 (el mismo punto de partida de la línea base)."""
    from facenet_pytorch import InceptionResnetV1

    return InceptionResnetV1(pretrained="vggface2").eval().to(_dispositivo(dispositivo))
