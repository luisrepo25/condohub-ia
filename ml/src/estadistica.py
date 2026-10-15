"""Estadística mínima para decir si una mejora es real o es ruido. Solo numpy."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# Valor crítico t de Student (95 %, dos colas) por grados de libertad. Para más de 10 se usa 2.0 (conservador).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def resumen_pareado(error_base: Sequence[float], error_ajustado: Sequence[float]) -> dict[str, Any]:
    """Compara el error de la base y del ajustado EN LAS MISMAS repeticiones (una por semilla).

    `mejora` = base - ajustado (positiva = el ajustado es mejor). Con el intervalo de confianza del 95 % de la mejora
    media se decide: si el intervalo queda por encima de 0, la mejora es concluyente; si cruza el 0, no se puede
    distinguir de la suerte.
    """
    base, ajustado = np.asarray(error_base, dtype=float), np.asarray(error_ajustado, dtype=float)
    if base.shape != ajustado.shape or base.size == 0:
        raise ValueError("Hacen falta los mismos resultados (>= 1) para la base y el ajustado")
    mejora = base - ajustado
    n = int(mejora.size)
    resumen: dict[str, Any] = {
        "repeticiones": n,
        "eer_base_media": float(base.mean()),
        "eer_ajustado_media": float(ajustado.mean()),
        "mejora_media": float(mejora.mean()),
        "repeticiones_donde_mejora": int((mejora > 0).sum()),
    }
    if n < 2:
        return {**resumen, "desviacion": None, "ic95": None, "concluyente": False}
    desviacion = float(mejora.std(ddof=1))
    margen = _T95.get(n - 1, 2.0) * desviacion / math.sqrt(n)
    ic = (float(mejora.mean() - margen), float(mejora.mean() + margen))
    return {**resumen, "desviacion": desviacion, "ic95": ic, "concluyente": bool(ic[0] > 0)}
