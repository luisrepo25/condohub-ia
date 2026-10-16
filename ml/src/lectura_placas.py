"""Lectura de la placa a partir de varias franjas del recorte, por votación (A4).

El reconocedor ONNX lee una línea de texto. El recorte de la placa trae el rótulo «BOLIVIA» arriba y los caracteres abajo, así que se
lee varias veces desde distintas alturas (`FRANJAS`) y una placa solo se acepta si aparece en al menos `VOTOS_MINIMOS` franjas.
Probar todas las ventanas de 7 caracteres con corrección letra/dígito sin votar acepta «placas» desfasadas un carácter que cumplen el
formato pero son falsas (medido: 2 de 10 aceptadas equivocadas); con 3 votos ninguna.

Es la referencia en Python de la lógica que la PWA del guardia repite en JavaScript (`ml/web/lectura_placas.mjs`). Extiende `placas.py` sin modificarlo.
"""

from collections import defaultdict
from collections.abc import Sequence

from ml.src import placas

FRANJAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # desde qué fracción de la altura del recorte se lee hasta abajo
VOTOS_MINIMOS = 3
MAX_CORRECCIONES = 2
PENALIZACION_POR_CORRECCION = 0.9


def candidatas(texto: str, confianza: float, max_correcciones: int = MAX_CORRECCIONES) -> dict[str, float]:
    """Placas válidas (A4) en cualquier ventana de 7 caracteres de la lectura, con la corrección letra/dígito por posición.

    Devuelve `{placa: confianza}`; la confianza baja un 10 % por cada carácter corregido.
    """
    t = placas.normalizar(texto)
    resultado: dict[str, float] = {}
    for i in range(max(0, len(t) - 6)):
        ventana = t[i : i + 7]
        corregida = placas.corregir_por_posicion(ventana)
        cambios = sum(a != b for a, b in zip(ventana, corregida, strict=True))
        if cambios <= max_correcciones and placas.es_placa_valida(corregida):
            resultado[corregida] = max(
                resultado.get(corregida, 0.0), confianza * PENALIZACION_POR_CORRECCION**cambios
            )
    return resultado


def elegir_por_votos(
    lecturas: Sequence[tuple[str, float]], votos_minimos: int = VOTOS_MINIMOS
) -> tuple[str | None, float]:
    """De las lecturas `(texto, confianza)` de cada franja, la placa con más votos si llega al mínimo; si no, `(None, 0.0)`.

    Un voto = una franja en la que la placa aparece como candidata. Empata por la mayor confianza.
    """
    if votos_minimos < 1:
        raise ValueError("Se necesita al menos un voto")
    votos: dict[str, int] = defaultdict(int)
    confianza: dict[str, float] = defaultdict(float)
    for texto, conf in lecturas:
        for placa, c in candidatas(texto, conf).items():
            votos[placa] += 1
            confianza[placa] = max(confianza[placa], c)
    validas = [p for p in votos if votos[p] >= votos_minimos]
    if not validas:
        return None, 0.0
    mejor = max(validas, key=lambda p: (votos[p], confianza[p]))
    return mejor, confianza[mejor]
