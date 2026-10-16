"""Prepara las fotos de una persona antes de subirlas a Colab: sesiones por fecha, 1280 px y sin metadatos (I1, I6).

Uso (en local, es ligero; los originales NO se tocan):
    python -m ml.src.preparar_fotos <carpeta_origen> <carpeta_salida> <persona>
Deja `<salida>/<persona>/<sesion>/fotoN.jpg` + `consentimiento.json` y `<salida>/<persona>.zip` (rutas con «/», listo para Colab).
Sesión = bloque de fotos consecutivas por fecha de captura (EXIF, o fecha del archivo si no tiene): varias fotos de un mismo
día se parecen mucho entre sí, así que la evaluación deja una sesión entera fuera.
"""

import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}
LADO_MAXIMO = 1280


def agrupar_en_sesiones(fechas: list[datetime], min_por_sesion: int = 8, max_sesiones: int = 4) -> list[int]:
    """Número de sesión (0, 1, ...) de cada foto: bloques consecutivos por fecha, de tamaño parecido."""
    if not fechas:
        return []
    n_sesiones = max(1, min(max_sesiones, len(fechas) // min_por_sesion))
    orden = np.argsort([f.timestamp() for f in fechas], kind="stable")
    sesion = [0] * len(fechas)
    for numero, bloque in enumerate(np.array_split(orden, n_sesiones)):
        for indice in bloque:
            sesion[int(indice)] = numero
    return sesion


def fecha_captura(ruta: Path) -> datetime:
    """Fecha EXIF de la foto; si no tiene, la fecha de modificación del archivo."""
    from PIL import Image

    try:
        with Image.open(ruta) as imagen:
            exif = imagen.getexif()
            texto = exif.get(306) or exif.get_ifd(0x8769).get(36867)
        if texto:
            return datetime.strptime(str(texto), "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError):
        pass
    return datetime.fromtimestamp(ruta.stat().st_mtime)


def preparar(origen: Path, salida: Path, persona: str) -> dict[str, int]:
    from PIL import Image, ImageOps

    consentimiento = origen / "consentimiento.json"
    if not consentimiento.exists():
        raise ValueError("Falta consentimiento.json (I1): sin él no se prepara a nadie")
    destino = salida / persona
    nombres: dict[int, str] = {}

    carpetas = sorted(
        d for d in origen.iterdir() if d.is_dir() and any(f.suffix.lower() in EXTENSIONES for f in d.iterdir())
    )
    if carpetas:   # la persona ya trajo sus sesiones en subcarpetas (ej. «Luz alta», «Luz baja»): se respetan
        pares = [(p, numero) for numero, d in enumerate(carpetas) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTENSIONES]
        fotos, sesiones = [p for p, _ in pares], [n for _, n in pares]
        shutil.rmtree(destino, ignore_errors=True)
        for numero, d in enumerate(carpetas):
            nombres[numero] = d.name
            (destino / d.name).mkdir(parents=True)
    else:          # fotos sueltas: sesiones por fecha de captura
        fotos = sorted(p for p in origen.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)
        if not fotos:
            raise ValueError(f"No hay fotos en {origen}")
        fechas = [fecha_captura(p) for p in fotos]
        sesiones = agrupar_en_sesiones(fechas)
        shutil.rmtree(destino, ignore_errors=True)
        for numero in sorted(set(sesiones)):
            dias = [f for f, s in zip(fechas, sesiones, strict=True) if s == numero]
            nombres[numero] = f"sesion{numero + 1}_{min(dias):%Y-%m-%d}_a_{max(dias):%Y-%m-%d}"
            (destino / nombres[numero]).mkdir(parents=True)

    por_sesion: dict[str, int] = {}
    for contador, (ruta, sesion) in enumerate(zip(fotos, sesiones, strict=True), 1):
        with Image.open(ruta) as imagen:
            imagen = ImageOps.exif_transpose(imagen).convert("RGB")   # endereza y descarta el modo original
            imagen.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
            carpeta = destino / nombres[sesion]
            imagen.save(carpeta / f"foto{contador:03d}.jpg", "JPEG", quality=90)   # sin exif: no se pasa
        por_sesion[nombres[sesion]] = por_sesion.get(nombres[sesion], 0) + 1
    shutil.copy(consentimiento, destino / "consentimiento.json")

    archivo_zip = salida / f"{persona}.zip"
    with zipfile.ZipFile(archivo_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destino.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(salida).as_posix())
    return por_sesion


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    resultado = preparar(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    for nombre, cantidad in resultado.items():
        print(f"{nombre}: {cantidad} fotos")
    print("zip:", Path(sys.argv[2]) / f"{sys.argv[3]}.zip")
