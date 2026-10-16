"""Escenario simulado de desconocidos en portería, con la respuesta conocida, para validar DBSCAN (I3, I4).

Cada «aparición» es una foto distinta de una persona real de LFW en un día de la ventana de 30 días. Como se sabe quién es
quién, se puede medir si las alertas aciertan. Es una simulación: declararlo así en el informe.

Tipos de persona (lo que pone a prueba la regla I4: ≥ 3 apariciones en ≥ 3 días distintos):
- recurrente: 3 a 6 días distintos, 1 o 2 fotos por día  → DEBE alertar.
- mismo_dia: 4 apariciones el mismo día                  → NO debe alertar (un solo día).
- dos_dias: 2 días con 2 fotos cada uno                  → NO debe alertar (solo 2 días).
- suelto: una sola aparición                             → ruido, NO debe alertar.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass(frozen=True)
class Escenario:
    embeddings: np.ndarray   # [n, 512], normalizados L2
    personas: np.ndarray     # id de la persona real: SOLO para evaluar, el algoritmo no lo ve
    dias: list[str]          # yyyy-mm-dd de cada aparición


def construir(
    embeddings: np.ndarray,
    etiquetas: np.ndarray,
    personas_elegibles: np.ndarray,
    recurrentes: int,
    mismo_dia: int,
    dos_dias: int,
    sueltos: int,
    ventana_dias: int = 30,
    semilla: int = 42,
    inicio: date = date(2026, 10, 1),
) -> Escenario:
    """Arma el escenario eligiendo personas distintas (sin repetir) entre las elegibles."""
    total = recurrentes + mismo_dia + dos_dias + sueltos
    if total > len(personas_elegibles):
        raise ValueError(f"Se piden {total} personas y hay {len(personas_elegibles)} elegibles")
    if ventana_dias < 6:
        raise ValueError("La ventana debe tener al menos 6 días")
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(personas_elegibles)[:total]
    tipos = ["recurrente"] * recurrentes + ["mismo_dia"] * mismo_dia + ["dos_dias"] * dos_dias + ["suelto"] * sueltos

    vectores: list[np.ndarray] = []
    ids: list[int] = []
    dias: list[str] = []

    def agregar(persona: int, fotos: np.ndarray, dia: int) -> None:
        for f in fotos:
            vectores.append(embeddings[f])
            ids.append(persona)
            dias.append((inicio + timedelta(days=int(dia))).isoformat())

    for persona, tipo in zip(orden, tipos, strict=True):
        fotos = rng.permutation(np.flatnonzero(etiquetas == persona))
        if tipo == "recurrente":
            dias_elegidos = rng.choice(ventana_dias, size=int(rng.integers(3, 7)), replace=False)
            usadas = 0
            for d in dias_elegidos:
                n = int(rng.integers(1, 3))
                agregar(int(persona), fotos[usadas : usadas + n], int(d))
                usadas += n
        elif tipo == "mismo_dia":
            agregar(int(persona), fotos[:4], int(rng.integers(ventana_dias)))
        elif tipo == "dos_dias":
            d1, d2 = rng.choice(ventana_dias, size=2, replace=False)
            agregar(int(persona), fotos[:2], int(d1))
            agregar(int(persona), fotos[2:4], int(d2))
        else:
            agregar(int(persona), fotos[:1], int(rng.integers(ventana_dias)))

    return Escenario(np.vstack(vectores), np.asarray(ids), dias)
