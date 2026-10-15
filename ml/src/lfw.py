"""LFW (Labeled Faces in the Wild): dataset público para evaluar. Se descarga en Colab, no va al repo."""

import numpy as np

from ml.src.paths import dir_datos


def a_uint8(imagenes: np.ndarray, bloque: int = 500) -> np.ndarray:
    """Convierte a uint8 por bloques: así no se duplica varias veces en RAM un dataset de varios GB."""
    escala = (
        255.0 if imagenes.max() <= 1.0 else 1.0
    )  # según la versión de scikit-learn vienen en 0-1 o en 0-255
    salida = np.empty(imagenes.shape, dtype=np.uint8)
    for inicio in range(0, len(imagenes), bloque):
        salida[inicio : inicio + bloque] = np.clip(
            np.rint(imagenes[inicio : inicio + bloque] * escala), 0, 255
        )
    return salida


def cargar_lfw(min_fotos_por_persona: int = 20) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Devuelve (imagenes uint8 [n,250,250,3], etiquetas [n], nombres de personas)."""
    from sklearn.datasets import fetch_lfw_people

    datos = fetch_lfw_people(
        min_faces_per_person=min_fotos_por_persona,
        color=True,
        resize=1.0,
        slice_=None,
        data_home=str(dir_datos() / "sklearn"),
    )
    imagenes = a_uint8(np.asarray(datos.images))
    etiquetas, nombres = np.asarray(datos.target), list(datos.target_names)
    del datos  # libera los float32 originales (varios GB)
    return imagenes, etiquetas, nombres
