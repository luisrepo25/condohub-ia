"""Placas bolivianas sintéticas (texto conocido) para medir el OCR sin etiquetar a mano. Requiere Pillow.

Son un sustituto razonable en un trabajo académico, pero NO igualan la tipografía ni las condiciones de una placa real:
las métricas obtenidas con ellas son optimistas y así deben declararse en el informe.
"""

import random
import string

from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FUENTES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Colab / Linux
    "C:/Windows/Fonts/arialbd.ttf",                           # Windows
    "DejaVuSans-Bold.ttf",
)
AZUL = (20, 50, 140)


def texto_aleatorio(rng: random.Random) -> str:
    """Una placa con el formato de A4: 4 dígitos y 3 letras."""
    return "".join(rng.choices(string.digits, k=4)) + "".join(rng.choices(string.ascii_uppercase, k=3))


def _fuente(tamano: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for ruta in _FUENTES:
        try:
            return ImageFont.truetype(ruta, tamano)
        except OSError:
            continue
    return ImageFont.load_default(tamano)


def dibujar(texto: str, rng: random.Random, degradar: bool = True) -> Image.Image:
    """Placa tamaño americano (2:1), caracteres y reborde azul sobre blanco, con «BOLIVIA» arriba."""
    img = Image.new("RGB", (360, 180), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((6, 6, 353, 173), outline=AZUL, width=6)
    d.text((180, 34), "BOLIVIA", fill=AZUL, font=_fuente(26), anchor="mm")
    d.text((180, 108), f"{texto[:4]} {texto[4:]}", fill=AZUL, font=_fuente(66), anchor="mm")
    if degradar:
        img = img.rotate(rng.uniform(-5, 5), expand=False, fillcolor="white", resample=Image.BICUBIC)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.6)))
        factor = rng.uniform(0.55, 1.15)   # luz baja ... luz alta
        img = img.point(lambda v: min(255, int(v * factor)))
    return img
