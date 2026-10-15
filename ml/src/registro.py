"""Registro de personas por embeddings (sin reentrenar la red) y su evaluación honesta.

Reconocer = la mayor similitud coseno contra las fotos REGISTRADAS de cada persona >= umbral (I2).
Sumar una persona nueva es añadir sus embeddings: no hay nada que volver a entrenar.
"""

from typing import Any

import numpy as np


def _normalizar(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def puntuar(emb: np.ndarray, registro: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Para cada embedding: la persona registrada más parecida y su similitud (máx. sobre sus fotos registradas)."""
    nombres = list(registro)
    sim = np.stack([(_normalizar(emb) @ _normalizar(registro[n]).T).max(axis=1) for n in nombres], axis=1)
    mejor = sim.argmax(axis=1)
    return np.array(nombres, dtype=object)[mejor], sim[np.arange(len(emb)), mejor]


def umbral_para_far(scores_impostores: np.ndarray, far_maximo: float) -> float:
    """Menor umbral con el que, a lo sumo, `far_maximo` de los impostores serían aceptados."""
    s = np.sort(scores_impostores)
    permitidos = int(np.floor(far_maximo * len(s)))
    return float(np.nextafter(s[len(s) - permitidos - 1], np.inf))


def evaluar(
    emb: np.ndarray,
    personas: np.ndarray,
    sesiones: np.ndarray,
    emb_impostores: np.ndarray,
    etiquetas_impostores: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Deja una sesión fuera a la vez (registro con las demás) y mide con ella; impostores aparte.

    - El umbral se calibra SOLO con impostores de validación (no con las fotos propias de prueba).
    - El FAR se reporta con impostores de prueba: otras personas que nunca se calibraron.
    - FRR y confusiones salen de las sesiones dejadas fuera.
    """
    rng = np.random.default_rng(cfg["semilla"])
    ids = np.unique(etiquetas_impostores)
    rng.shuffle(ids)
    if len(ids) < 4:
        raise ValueError("Se necesitan al menos 4 personas impostoras (2 para calibrar y 2 para probar)")
    mitad = len(ids) // 2
    val = np.flatnonzero(np.isin(etiquetas_impostores, ids[:mitad]))
    prueba = np.flatnonzero(np.isin(etiquetas_impostores, ids[mitad:]))

    registro_completo = {p: emb[personas == p] for p in np.unique(personas)}
    umbral = umbral_para_far(puntuar(emb_impostores[val], registro_completo)[1], cfg["far_maximo"])
    far_prueba = float((puntuar(emb_impostores[prueba], registro_completo)[1] >= umbral).mean())

    por_sesion, omitidas = [], []
    for persona in np.unique(personas):
        for sesion in np.unique(sesiones[personas == persona]):
            fuera = (personas == persona) & (sesiones == sesion)
            resto = (personas == persona) & (sesiones != sesion)
            if not resto.any():  # única sesión de la persona: no hay con qué registrar
                omitidas.append(f"{persona}/{sesion}")
                continue
            registro = {q: emb[personas == q] for q in np.unique(personas) if q != persona}
            registro[persona] = emb[resto]
            mejor, score = puntuar(emb[fuera], registro)
            aceptado = score >= umbral
            correcto, confundido = aceptado & (mejor == persona), aceptado & (mejor != persona)
            por_sesion.append(
                {
                    "persona": str(persona),
                    "sesion": str(sesion),
                    "fotos": int(fuera.sum()),
                    "reconocidas": int(correcto.sum()),
                    "rechazadas": int((~aceptado).sum()),
                    "confundidas_con_otra_persona": int(confundido.sum()),
                }
            )
    total = sum(f["fotos"] for f in por_sesion)
    return {
        "umbral_calibrado_con_impostores_de_validacion": umbral,
        "far_maximo_pedido": cfg["far_maximo"],
        "far_prueba": far_prueba,
        "impostores": {"validacion": int(len(val)), "prueba": int(len(prueba))},
        "frr": 1 - sum(f["reconocidas"] for f in por_sesion) / total if total else None,
        "confusiones": sum(f["confundidas_con_otra_persona"] for f in por_sesion),
        "por_sesion": por_sesion,
        "sesiones_omitidas_por_ser_la_unica": omitidas,
    }
