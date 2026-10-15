"""Métricas de verificación facial: FAR, FRR, EER, ROC y calibración del umbral (I2).

El umbral se calibra en VALIDACIÓN y recién se reporta en PRUEBA; nunca al revés.
"""

from dataclasses import dataclass

import numpy as np


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similitud coseno fila a fila entre dos matrices de embeddings."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def generar_pares(
    etiquetas: np.ndarray, n_pares: int, semilla: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pares (i, j, misma_persona) a medias positivos y negativos."""
    rng = np.random.default_rng(semilla)
    por_persona = {p: np.flatnonzero(etiquetas == p) for p in np.unique(etiquetas)}
    con_pares = [p for p, idx in por_persona.items() if len(idx) >= 2]
    if not con_pares or len(por_persona) < 2:
        raise ValueError("Se necesitan al menos 2 personas y alguna con 2 o más fotos")
    mitad = n_pares // 2
    i, j, y = [], [], []
    for _ in range(mitad):  # positivos: dos fotos distintas de la misma persona
        p = con_pares[rng.integers(len(con_pares))]
        a, b = rng.choice(por_persona[p], size=2, replace=False)
        i.append(a), j.append(b), y.append(1)
    personas = list(por_persona)
    for _ in range(n_pares - mitad):  # negativos: fotos de personas distintas
        p, q = rng.choice(len(personas), size=2, replace=False)
        (
            i.append(rng.choice(por_persona[personas[p]])),
            j.append(rng.choice(por_persona[personas[q]])),
            y.append(0),
        )
    return np.array(i), np.array(j), np.array(y)


@dataclass(frozen=True)
class Curva:
    umbrales: np.ndarray
    far: np.ndarray  # falsos aceptados: desconocido tomado por residente
    frr: np.ndarray  # falsos rechazados: residente tomado por desconocido


def curva_far_frr(scores: np.ndarray, misma_persona: np.ndarray) -> Curva:
    """FAR y FRR para cada umbral posible (aceptar si similitud >= umbral)."""
    pos, neg = scores[misma_persona == 1], scores[misma_persona == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Hacen falta pares positivos y negativos")
    umbrales = np.unique(np.concatenate([scores, [scores.min() - 1e-6, scores.max() + 1e-6]]))
    far = np.array([(neg >= u).mean() for u in umbrales])
    frr = np.array([(pos < u).mean() for u in umbrales])
    return Curva(umbrales, far, frr)


def eer(curva: Curva) -> tuple[float, float]:
    """Punto donde FAR = FRR: devuelve (eer, umbral)."""
    k = int(np.argmin(np.abs(curva.far - curva.frr)))
    return float((curva.far[k] + curva.frr[k]) / 2), float(curva.umbrales[k])


def umbral_para_far(curva: Curva, far_maximo: float) -> float:
    """Menor umbral cuyo FAR no supera `far_maximo` (en un acceso, un falso aceptado cuesta más)."""
    validos = np.flatnonzero(curva.far <= far_maximo)
    if len(validos) == 0:
        return float(curva.umbrales[-1])
    return float(curva.umbrales[validos[0]])


def evaluar_en_umbral(scores: np.ndarray, misma_persona: np.ndarray, umbral: float) -> dict[str, float]:
    """FAR, FRR y exactitud de los pares con un umbral ya fijado (calibrado en validación)."""
    aceptado = scores >= umbral
    pos, neg = misma_persona == 1, misma_persona == 0
    return {
        "umbral": float(umbral),
        "far": float(aceptado[neg].mean()),
        "frr": float((~aceptado[pos]).mean()),
        "exactitud": float((aceptado == pos).mean()),
    }
