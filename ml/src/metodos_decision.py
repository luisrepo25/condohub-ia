"""Otras formas de decidir «quién es» a partir de los embeddings, y calidad de las fotos de registro.

Hoy el servicio puntúa con la similitud MÁXIMA contra las fotos registradas de cada persona (`registro.puntuar`). Aquí se comparan
alternativas con el mismo protocolo de evaluación de `zonas.puntajes_equipo` (una sesión fuera; compañeros no registrados; extraños):

- `maximo`: el actual.
- `top_k_promedio`: promedio de las k similitudes más altas de cada persona (menos sensible a una sola foto parecida por azar).
- `centroide`: similitud con el embedding promedio de la persona.
- `svm` / `knn`: clasificadores entrenados con los embeddings registrados; el puntaje es la probabilidad de la clase elegida.

Como las escalas de los puntajes no son comparables entre métodos, se miden cosas que no dependen de la escala: AUC entre propios y
compañeros / extraños, y la tasa de reconocimiento de propios cuando se deja pasar a lo sumo un 5 % de los compañeros parecidos.
No cambia `registro.py` ni `zonas.py` (código en uso): los extiende.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from ml.src import registro

Registro = dict[str, np.ndarray]
Puntuador = Callable[[np.ndarray, Registro], tuple[np.ndarray, np.ndarray]]


def _sim(emb: np.ndarray, reg: np.ndarray) -> np.ndarray:
    return registro._normalizar(emb) @ registro._normalizar(reg).T


def _elegir(sim_por_persona: np.ndarray, nombres: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mejor = sim_por_persona.argmax(axis=1)
    return np.array(nombres, dtype=object)[mejor], sim_por_persona[np.arange(len(sim_por_persona)), mejor]


def maximo(emb: np.ndarray, reg: Registro) -> tuple[np.ndarray, np.ndarray]:
    return registro.puntuar(emb, reg)


def top_k_promedio(k: int = 3) -> Puntuador:
    def puntuar(emb: np.ndarray, reg: Registro) -> tuple[np.ndarray, np.ndarray]:
        nombres = list(reg)
        columnas = []
        for n in nombres:
            s = np.sort(_sim(emb, reg[n]), axis=1)[:, ::-1]
            columnas.append(s[:, : min(k, s.shape[1])].mean(axis=1))
        return _elegir(np.stack(columnas, axis=1), nombres)

    return puntuar


def centroide(emb: np.ndarray, reg: Registro) -> tuple[np.ndarray, np.ndarray]:
    nombres = list(reg)
    centros = np.stack([registro._normalizar(reg[n]).mean(axis=0, keepdims=True)[0] for n in nombres])
    return _elegir(_sim(emb, centros), nombres)


def _clasificador(crear: Callable[[], Any]) -> Puntuador:
    def puntuar(emb: np.ndarray, reg: Registro) -> tuple[np.ndarray, np.ndarray]:
        nombres = list(reg)
        x = np.concatenate([registro._normalizar(reg[n]) for n in nombres])
        y = np.concatenate([[i] * len(reg[n]) for i, n in enumerate(nombres)])
        modelo = crear().fit(x, y)
        proba = modelo.predict_proba(registro._normalizar(emb))
        columnas = np.zeros((len(emb), len(nombres)))
        columnas[:, modelo.classes_.astype(int)] = proba
        return _elegir(columnas, nombres)

    return puntuar


def svm() -> Puntuador:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    return _clasificador(lambda: CalibratedClassifierCV(SVC(kernel="linear"), cv=3, ensemble=False))


def knn(k: int = 5) -> Puntuador:
    from sklearn.neighbors import KNeighborsClassifier

    return _clasificador(lambda: KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance"))


def puntajes_equipo_con(
    puntuar: Puntuador,
    emb: np.ndarray,
    personas: np.ndarray,
    sesiones: np.ndarray,
    emb_impostores: np.ndarray,
    etiquetas_impostores: np.ndarray,
    semilla: int = 42,
    usable_en_registro: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Mismo protocolo que `zonas.puntajes_equipo`, con el método de puntuación a elección.

    `usable_en_registro` (bool por foto): las fotos en False no se usan para REGISTRAR (sí se usan para probar). Sirve para medir
    si descartar fotos de mala calidad en el registro mejora el reconocimiento.
    """
    usable = np.ones(len(emb), dtype=bool) if usable_en_registro is None else usable_en_registro
    rng = np.random.default_rng(semilla)
    ids = np.unique(etiquetas_impostores)
    rng.shuffle(ids)
    if len(ids) < 4:
        raise ValueError("Se necesitan al menos 4 personas impostoras (2 para calibrar y 2 para probar)")
    prueba = np.flatnonzero(np.isin(etiquetas_impostores, ids[len(ids) // 2 :]))
    nombres = np.unique(personas)
    completo = {p: emb[(personas == p) & usable] for p in nombres}
    genuinos, correcto, indices = [], [], []
    for persona in nombres:
        for sesion in np.unique(sesiones[personas == persona]):
            fuera = (personas == persona) & (sesiones == sesion)
            resto = (personas == persona) & (sesiones != sesion) & usable
            if not resto.any():
                continue
            reg = {q: emb[(personas == q) & usable] for q in nombres if q != persona}
            reg[persona] = emb[resto]
            mejor, score = puntuar(emb[fuera], reg)
            genuinos.append(score)
            correcto.append(mejor == persona)
            indices.append(np.flatnonzero(fuera))
    companeros = [
        puntuar(emb[personas == p], {q: emb[(personas == q) & usable] for q in nombres if q != p})[1]
        for p in nombres
    ]
    return {
        "genuinos": np.concatenate(genuinos),
        "genuinos_correcto": np.concatenate(correcto),
        "genuinos_indice": np.concatenate(indices),
        "companeros": np.concatenate(companeros),
        "impostores_prueba": puntuar(emb_impostores[prueba], completo)[1],
    }


def auc(positivos: np.ndarray, negativos: np.ndarray) -> float:
    """Probabilidad de que un positivo puntúe más que un negativo (Mann-Whitney; los empates cuentan la mitad)."""
    if len(positivos) == 0 or len(negativos) == 0:
        raise ValueError("Se necesitan positivos y negativos")
    todos = np.concatenate([positivos, negativos])
    _, inverso, cuentas = np.unique(todos, return_inverse=True, return_counts=True)
    rangos = (np.cumsum(cuentas) - (cuentas - 1) / 2)[inverso]  # rango medio: los empates comparten posición
    n_pos, n_neg = len(positivos), len(negativos)
    return float((rangos[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def resumen(p: dict[str, np.ndarray], far_companeros: float = 0.05) -> dict[str, float]:
    """Métricas que no dependen de la escala del puntaje, para comparar métodos."""
    umbral = registro.umbral_para_far(p["companeros"], far_companeros)
    acierta = (p["genuinos"] >= umbral) & p["genuinos_correcto"]
    return {
        "identidad_correcta": float(p["genuinos_correcto"].mean()),
        "auc_propios_vs_companeros": auc(p["genuinos"], p["companeros"]),
        "auc_propios_vs_extranos": auc(p["genuinos"], p["impostores_prueba"]),
        f"reconocidos_con_{int(far_companeros * 100)}pct_companeros": float(acierta.mean()),
        "umbral_en_ese_punto": float(umbral),
    }


def nitidez(rgb: np.ndarray, lado: int = 256) -> float:
    """Varianza del laplaciano de la imagen en grises reducida a `lado` px de ancho: más alta = más nítida."""
    from PIL import Image

    gris = np.asarray(
        Image.fromarray(rgb).convert("L").resize((lado, max(1, round(rgb.shape[0] * lado / rgb.shape[1])))),
        dtype=float,
    )
    lap = gris[:-2, 1:-1] + gris[2:, 1:-1] + gris[1:-1, :-2] + gris[1:-1, 2:] - 4 * gris[1:-1, 1:-1]
    return float(lap.var())


def correlacion_rangos(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de Spearman (sin scipy)."""
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
