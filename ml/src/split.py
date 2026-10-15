"""Separación por PERSONA: nunca la misma persona en entrenamiento, validación y prueba (verificación)."""

import numpy as np


def dividir_por_persona(
    etiquetas: np.ndarray, proporciones: tuple[float, float, float] = (0.6, 0.2, 0.2), semilla: int = 42
) -> dict[str, np.ndarray]:
    """Devuelve los índices de cada partición (`train`, `val`, `test`); las personas no se comparten."""
    if abs(sum(proporciones) - 1.0) > 1e-9:
        raise ValueError("Las proporciones deben sumar 1")
    personas = np.unique(etiquetas)
    if len(personas) < 3:
        raise ValueError("Se necesitan al menos 3 personas para separar train/val/test")
    rng = np.random.default_rng(semilla)
    rng.shuffle(personas)
    n_train = max(1, round(len(personas) * proporciones[0]))
    n_val = max(1, round(len(personas) * proporciones[1]))
    grupos = {
        "train": personas[:n_train],
        "val": personas[n_train : n_train + n_val],
        "test": personas[n_train + n_val :],
    }
    if len(grupos["test"]) == 0:
        raise ValueError("Muy pocas personas para las proporciones pedidas")
    return {nombre: np.flatnonzero(np.isin(etiquetas, ids)) for nombre, ids in grupos.items()}
