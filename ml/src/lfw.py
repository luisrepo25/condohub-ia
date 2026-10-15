"""LFW (Labeled Faces in the Wild): dataset público para evaluar. Se descarga en Colab, no va al repo."""

import numpy as np

from ml.src.paths import dir_datos


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
    imagenes = np.asarray(datos.images)
    if imagenes.max() <= 1.0:  # según la versión de scikit-learn vienen en 0-1 o en 0-255
        imagenes = imagenes * 255
    return imagenes.clip(0, 255).astype(np.uint8), np.asarray(datos.target), list(datos.target_names)
