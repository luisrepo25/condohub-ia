"""Publica el reconocedor de texto de placas y su ficha en el almacén de objetos.

Valida la ficha y el hash y sube los dos archivos a `modelos/lectura-placas/<version>/`.

Uso (con las variables S3_ENDPOINT, S3_ACCESS_KEY y S3_SECRET_KEY del entorno):
    python -m app.infrastructure.publicar_lectura <ficha_reconocedor.json> <reconocedor_placas.onnx>
"""

import argparse
import hashlib
import json
from pathlib import Path

from app.config import Config
from app.domain.modelo_lectura import ModeloDeLectura
from app.infrastructure.s3_cliente import ClienteS3


def publicar(ficha: Path, archivo: Path, cliente: ClienteS3, bucket: str) -> tuple[str, str]:
    """Sube el ONNX y la ficha y devuelve sus claves. Se niega si el archivo no es el que dice la ficha."""
    datos = json.loads(ficha.read_text(encoding="utf-8"))
    modelo = ModeloDeLectura(
        nombre=datos["modelo"],
        version=datos["version"],
        formato=datos["formato"],
        archivo=datos["archivo"],
        sha256=datos["sha256"],
        ficha=ficha.name,
    )
    if archivo.name != modelo.archivo:
        raise ValueError(f"El archivo se llama {archivo.name} y la ficha espera {modelo.archivo}")
    contenido = archivo.read_bytes()
    if hashlib.sha256(contenido).hexdigest() != modelo.sha256:
        raise ValueError("El archivo no coincide con el sha256 de la ficha: no se publica")
    cliente.crear_bucket(bucket)
    cliente.subir(bucket, modelo.clave_objeto, contenido)
    cliente.subir(bucket, modelo.clave_ficha, ficha.read_bytes())
    return modelo.clave_objeto, modelo.clave_ficha


def main() -> None:
    parser = argparse.ArgumentParser(description="Publica el reconocedor de placas en el almacén de objetos")
    parser.add_argument("ficha", type=Path)
    parser.add_argument("archivo", type=Path)
    args = parser.parse_args()
    config = Config.desde_entorno()
    cliente = ClienteS3(config.s3_endpoint, config.s3_acceso, config.s3_secreto, config.s3_region)
    claves = publicar(args.ficha, args.archivo, cliente, config.s3_bucket_modelos)
    for clave in claves:
        print(f"Publicado: {config.s3_bucket_modelos}/{clave}")
    print(f"Para usarlo: MODELO_LECTURA_FICHA={args.ficha}")


if __name__ == "__main__":
    main()
