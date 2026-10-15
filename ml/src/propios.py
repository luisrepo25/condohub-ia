"""Fotos propias del equipo: `propios/<persona>/<sesion>/foto.jpg`, siempre con consentimiento (I1).

Las fotos viven en Drive (privado) y nunca en git (I6). Cada persona tiene un `consentimiento.json` con
`fecha` y `version` del texto aceptado; sin él, el cargador se niega a leerla.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXTENSIONES = {".jpg", ".jpeg", ".png"}
SIN_SESION = "sin_sesion"


class SinConsentimiento(Exception):
    """La persona no tiene un `consentimiento.json` válido: no se pueden usar sus fotos (I1)."""


@dataclass(frozen=True)
class Foto:
    persona: str
    sesion: str
    ruta: Path


def leer_consentimiento(carpeta_persona: Path) -> dict[str, str]:
    archivo = carpeta_persona / "consentimiento.json"
    if not archivo.is_file():
        raise SinConsentimiento(f"{carpeta_persona.name}: falta consentimiento.json")
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    if not datos.get("fecha") or not datos.get("version"):
        raise SinConsentimiento(f"{carpeta_persona.name}: el consentimiento necesita `fecha` y `version`")
    return {"fecha": str(datos["fecha"]), "version": str(datos["version"])}


def listar_fotos(raiz: Path) -> list[Foto]:
    """Todas las fotos de `raiz`, ordenadas. Una sesión es una subcarpeta (ej. `2026-10-14_luz-baja`)."""
    fotos: list[Foto] = []
    for persona in sorted(p for p in raiz.iterdir() if p.is_dir()):
        leer_consentimiento(persona)  # I1: sin consentimiento no se lee ni una foto
        for ruta in sorted(persona.rglob("*")):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES:
                relativa = ruta.relative_to(persona)
                sesion = relativa.parts[0] if len(relativa.parts) > 1 else SIN_SESION
                fotos.append(Foto(persona.name, sesion, ruta))
    return fotos


def cargar_imagen(ruta: Path, lado_maximo: int = 1600) -> np.ndarray:
    """RGB uint8, con la orientación EXIF corregida y reducida si es muy grande (acelera MTCNN)."""
    from PIL import Image, ImageOps

    with Image.open(ruta) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        imagen.thumbnail((lado_maximo, lado_maximo))
        return np.asarray(imagen)
