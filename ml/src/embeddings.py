"""Embeddings faciales: MTCNN (detección y alineación a 160×160) + InceptionResnetV1 preentrenado en VGGFace2.

Necesita PyTorch y facenet-pytorch (se instalan en Colab; no están en el servicio). Devuelve vectores de 512
dimensiones normalizados L2, de modo que el producto punto es la similitud coseno.
"""

from collections.abc import Sequence

import numpy as np


def crear_embebedor(dispositivo: str | None = None, lote: int = 64):  # noqa: ANN201 - depende de torch (import diferido)
    """Devuelve `embeber(imagenes) -> (embeddings, detectado)`; `detectado[i]` es False sin rostro."""
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image

    dev = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
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
