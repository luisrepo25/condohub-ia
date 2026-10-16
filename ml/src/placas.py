"""Placas bolivianas (desde 1999): 4 dígitos + 3 letras (A4). Utilidades puras para entrenar y medir la lectura de placas.

No importan torch ni ultralytics: se prueban en local. La detección (YOLOv8n) y el OCR (EasyOCR) se ejecutan en Colab.
"""

import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence

FORMATO = re.compile(r"[0-9]{4}[A-Z]{3}")

# Confusiones típicas del OCR según lo que la posición exige (dígito o letra)
_A_DIGITO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
_A_LETRA = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def normalizar(texto: str) -> str:
    """Mayúsculas y sin separadores (espacios, guiones, puntos)."""
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def es_placa_valida(texto: str) -> bool:
    return FORMATO.fullmatch(normalizar(texto)) is not None


def extraer_placa(texto: str) -> str:
    """Quita caracteres sobrantes (marco, tornillos, ruido) buscando una ventana con el formato de A4.

    Si la lectura ya cumple el formato, o no contiene ninguna ventana válida, se devuelve normalizada sin cambios.
    """
    t = normalizar(texto)
    if FORMATO.fullmatch(t):
        return t
    ventana = FORMATO.search(t)
    return ventana.group(0) if ventana else t


def corregir_por_posicion(texto: str) -> str:
    """Corrige confusiones letra/dígito según el formato (posiciones 0-3 dígitos, 4-6 letras).

    Si la lectura no tiene 7 caracteres no se inventa nada: se devuelve normalizada.
    """
    t = normalizar(texto)
    if len(t) != 7:
        return t
    digitos = "".join(_A_DIGITO.get(c, c) for c in t[:4])
    letras = "".join(_A_LETRA.get(c, c) for c in t[4:])
    return digitos + letras


_MAX_CORRECCIONES = 2


def _es_rotulo_bolivia(texto: str) -> bool:
    t = normalizar(texto)
    return "BOLIV" in t or distancia_edicion(t, "BOLIVIA") <= 2


def elegir_placa(bloques: Sequence[tuple[str, float]]) -> tuple[str | None, float]:
    """De los bloques de texto del OCR (de izquierda a derecha) devuelve `(placa, confianza)` o `(None, 0.0)`.

    Primero cada bloque por separado (el válido de mayor confianza; la confianza es la de ESE bloque, sin el ruido de los
    demás). Solo si ninguno vale, se unen los bloques que no son el rótulo «BOLIVIA» (el OCR a veces parte la placa).
    Una lectura que no cumple el formato de A4 no se acepta.
    """
    def pulir(t: str) -> str:
        extraida = extraer_placa(t)
        corregida = corregir_por_posicion(extraida)
        # Una corrección que cambia más de 2 caracteres convierte cualquier palabra en «placa»: no se acepta
        return corregida if sum(a != b for a, b in zip(extraida, corregida, strict=True)) <= _MAX_CORRECCIONES else extraida

    utiles = [(t, c) for t, c in bloques if not _es_rotulo_bolivia(t)]
    validos = [(pulir(t), c) for t, c in utiles if es_placa_valida(pulir(t))]
    if validos:
        return max(validos, key=lambda v: v[1])
    if utiles:
        unido = pulir("".join(t for t, _ in utiles))
        if es_placa_valida(unido):
            return unido, min(c for _, c in utiles)
    return None, 0.0


def voc_a_yolo(xml_texto: str, clase: int = 0) -> list[str]:
    """Anotación Pascal VOC → líneas YOLO `clase cx cy w h` normalizadas (0..1)."""
    raiz = ET.fromstring(xml_texto)
    ancho = float(raiz.findtext("size/width", "0"))
    alto = float(raiz.findtext("size/height", "0"))
    if ancho <= 0 or alto <= 0:
        raise ValueError("La anotación no trae el tamaño de la imagen")
    lineas = []
    for obj in raiz.findall("object"):
        caja = obj.find("bndbox")
        if caja is None:
            continue
        x1, y1, x2, y2 = (float(caja.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, ancho)), max(0.0, min(x2, ancho))
        y1, y2 = max(0.0, min(y1, alto)), max(0.0, min(y2, alto))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy, w, h = (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto, (x2 - x1) / ancho, (y2 - y1) / alto
        lineas.append(f"{clase} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lineas


def dividir_imagenes(
    nombres: Sequence[str], proporciones: tuple[float, float, float] = (0.7, 0.15, 0.15), semilla: int = 42
) -> dict[str, list[str]]:
    """Reparte imágenes en train/val/test de forma reproducible; ninguna imagen está en dos partes."""
    if abs(sum(proporciones) - 1.0) > 1e-9:
        raise ValueError("Las proporciones deben sumar 1")
    if len(set(nombres)) != len(nombres):
        raise ValueError("Hay nombres de imagen repetidos")
    if len(nombres) < 10:
        raise ValueError("Muy pocas imágenes para separar train/val/test")
    orden = sorted(nombres)
    random.Random(semilla).shuffle(orden)
    n_train = round(len(orden) * proporciones[0])
    n_val = round(len(orden) * proporciones[1])
    return {"train": orden[:n_train], "val": orden[n_train : n_train + n_val], "test": orden[n_train + n_val :]}


def distancia_edicion(a: str, b: str) -> int:
    """Distancia de Levenshtein (inserciones, borrados y sustituciones)."""
    previa = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        actual = [i]
        for j, cb in enumerate(b, 1):
            actual.append(min(previa[j] + 1, actual[j - 1] + 1, previa[j - 1] + (ca != cb)))
        previa = actual
    return previa[-1]


def metricas_lectura(reales: Sequence[str], leidas: Sequence[str]) -> dict[str, float]:
    """Precisión por carácter (1 − errores de edición / caracteres reales) y exactitud de placa completa."""
    if len(reales) != len(leidas) or not reales:
        raise ValueError("Se necesitan listas no vacías del mismo tamaño")
    caracteres = sum(len(r) for r in reales)
    errores = sum(distancia_edicion(r, normalizar(p)) for r, p in zip(reales, leidas, strict=True))
    completas = sum(r == normalizar(p) for r, p in zip(reales, leidas, strict=True))
    return {
        "precision_por_caracter": max(0.0, 1 - errores / caracteres),
        "placa_completa": completas / len(reales),
        "placas": float(len(reales)),
    }
