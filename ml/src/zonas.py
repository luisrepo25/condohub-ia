"""Las tres zonas de la regla I2: RECONOCIDO / DUDOSO / DESCONOCIDO según la similitud coseno con el perfil más parecido.

- similitud ≥ alto        → RECONOCIDO (se muestra quién es)
- bajo ≤ similitud < alto → DUDOSO (se muestra el candidato y decide el guardia; no se guarda como desconocido)
- similitud < bajo        → DESCONOCIDO (entra al agrupamiento nocturno, I3/I4)
El reconocimiento asiste al guardia y nunca abre por sí solo (A5).

Complementa `registro.py` (que calibra un solo umbral): aquí se calibran dos y se mide a dónde cae cada tipo de rostro.
"""

from typing import Any

import numpy as np

from ml.src import registro

RECONOCIDO, DUDOSO, DESCONOCIDO = "RECONOCIDO", "DUDOSO", "DESCONOCIDO"


def clasificar(similitudes: np.ndarray, alto: float, bajo: float) -> np.ndarray:
    """Zona de cada similitud. Los umbrales son parámetros (I2), no constantes."""
    if bajo > alto:
        raise ValueError("El umbral bajo no puede ser mayor que el alto")
    return np.where(similitudes >= alto, RECONOCIDO, np.where(similitudes >= bajo, DUDOSO, DESCONOCIDO))


def reparto(similitudes: np.ndarray, alto: float, bajo: float) -> dict[str, float]:
    """Fracción de rostros que cae en cada zona."""
    if len(similitudes) == 0:
        raise ValueError("No hay similitudes que repartir")
    zonas = clasificar(similitudes, alto, bajo)
    return {z: float((zonas == z).mean()) for z in (RECONOCIDO, DUDOSO, DESCONOCIDO)}


def calibrar(similitudes_impostores: np.ndarray, far_alto: float = 0.01, far_bajo: float = 0.10) -> tuple[float, float]:
    """`alto`: a lo sumo `far_alto` de los extraños llegaría a RECONOCIDO. `bajo`: a lo sumo `far_bajo` llegaría a DUDOSO o más.

    Se calibra solo con extraños (no con las fotos de los residentes), como en `registro.evaluar`.
    """
    if not 0 < far_alto < far_bajo < 1:
        raise ValueError("Se requiere 0 < far_alto < far_bajo < 1")
    return (
        registro.umbral_para_far(similitudes_impostores, far_alto),
        registro.umbral_para_far(similitudes_impostores, far_bajo),
    )


def puntajes_equipo(
    emb: np.ndarray,
    personas: np.ndarray,
    sesiones: np.ndarray,
    emb_impostores: np.ndarray,
    etiquetas_impostores: np.ndarray,
    semilla: int = 42,
) -> dict[str, np.ndarray]:
    """Similitudes de cada tipo de rostro, listas para calibrar y medir.

    - `genuinos` / `genuinos_correcto`: fotos de una sesión dejada fuera contra el registro hecho con las demás sesiones
      (y el resto del equipo registrado); `correcto` = el perfil más parecido es el de la propia persona.
    - `companeros`: fotos de una persona que NO está registrada contra el resto del equipo (el «amigo parecido»).
    - `impostores_validacion` / `impostores_prueba`: extraños (LFW), mitad para calibrar y mitad para medir, sin personas comunes.
    """
    rng = np.random.default_rng(semilla)
    ids = np.unique(etiquetas_impostores)
    rng.shuffle(ids)
    if len(ids) < 4:
        raise ValueError("Se necesitan al menos 4 personas impostoras (2 para calibrar y 2 para probar)")
    mitad = len(ids) // 2
    validacion = np.flatnonzero(np.isin(etiquetas_impostores, ids[:mitad]))
    prueba = np.flatnonzero(np.isin(etiquetas_impostores, ids[mitad:]))

    nombres = np.unique(personas)
    completo = {p: emb[personas == p] for p in nombres}
    genuinos, correcto, omitidas = [], [], []
    for persona in nombres:
        for sesion in np.unique(sesiones[personas == persona]):
            fuera = (personas == persona) & (sesiones == sesion)
            resto = (personas == persona) & (sesiones != sesion)
            if not resto.any():
                omitidas.append(f"{persona}/{sesion}")
                continue
            reg = {q: emb[personas == q] for q in nombres if q != persona}
            reg[persona] = emb[resto]
            mejor, score = registro.puntuar(emb[fuera], reg)
            genuinos.append(score)
            correcto.append(mejor == persona)
    companeros = [
        registro.puntuar(emb[personas == p], {q: emb[personas == q] for q in nombres if q != p})[1]
        for p in nombres
        if len(nombres) >= 2
    ]
    return {
        "genuinos": np.concatenate(genuinos) if genuinos else np.array([]),
        "genuinos_correcto": np.concatenate(correcto) if correcto else np.array([], dtype=bool),
        "companeros": np.concatenate(companeros) if companeros else np.array([]),
        "impostores_validacion": registro.puntuar(emb_impostores[validacion], completo)[1],
        "impostores_prueba": registro.puntuar(emb_impostores[prueba], completo)[1],
        "sesiones_omitidas": np.array(omitidas, dtype=object),
    }


def evaluar_zonas(
    p: dict[str, np.ndarray],
    umbrales_fijos: dict[str, tuple[float, float]],
    far_alto: float = 0.01,
    far_bajo: float = 0.10,
) -> dict[str, Any]:
    """Compara los umbrales de la regla (p. ej. 0.70 y 0.50) con los calibrados con extraños, zona por zona."""
    alto_cal, bajo_cal = calibrar(p["impostores_validacion"], far_alto, far_bajo)
    configuraciones = {**umbrales_fijos, "calibrado_con_extranos": (alto_cal, bajo_cal)}
    salida: dict[str, Any] = {}
    for nombre, (alto, bajo) in configuraciones.items():
        g, ok = p["genuinos"], p["genuinos_correcto"]
        zonas = clasificar(g, alto, bajo) if len(g) else np.array([])
        salida[nombre] = {
            "alto": float(alto),
            "bajo": float(bajo),
            "genuinos": None
            if not len(g)
            else {
                "reconocido_correcto": float(((zonas == RECONOCIDO) & ok).mean()),
                "reconocido_otra_persona": float(((zonas == RECONOCIDO) & ~ok).mean()),
                "dudoso_candidato_correcto": float(((zonas == DUDOSO) & ok).mean()),
                "dudoso_candidato_otra_persona": float(((zonas == DUDOSO) & ~ok).mean()),
                "desconocido": float((zonas == DESCONOCIDO).mean()),
                "fotos": int(len(g)),
            },
            "companeros_no_registrados": reparto(p["companeros"], alto, bajo) if len(p["companeros"]) else None,
            "extranos_prueba": reparto(p["impostores_prueba"], alto, bajo),
        }
    return salida
