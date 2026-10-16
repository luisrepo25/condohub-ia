"""Publica un modelo de placas en el almacén de objetos.

Valida su ficha y su hash y lo sube a `modelos/placas/<version>/`.

Uso (con las variables S3_ENDPOINT, S3_ACCESS_KEY y S3_SECRET_KEY del entorno):
    python -m app.infrastructure.publicar_modelo <modelo_placas.json> <archivo.onnx>
"""

import argparse
import hashlib
import json
from pathlib import Path

from app.config import Config
from app.domain.modelos import ModeloDePlacas
from app.infrastructure.s3_cliente import ClienteS3


def publicar(ficha: Path, archivo: Path, cliente: ClienteS3, bucket: str) -> str:
    """Sube el archivo a su clave y la devuelve. Se niega si el archivo no es el que dice la ficha."""
    datos = json.loads(ficha.read_text(encoding="utf-8"))
    modelo = ModeloDePlacas(
        nombre=datos["modelo"],
        version=datos["version"],
        formato=datos["formato"],
        archivo=datos["archivo"],
        sha256=datos["sha256"],
    )
    if archivo.name != modelo.archivo:
        raise ValueError(f"El archivo se llama {archivo.name} y la ficha espera {modelo.archivo}")
    contenido = archivo.read_bytes()
    if hashlib.sha256(contenido).hexdigest() != modelo.sha256:
        raise ValueError("El archivo no coincide con el sha256 de la ficha: no se publica")
    cliente.crear_bucket(bucket)
    cliente.subir(bucket, modelo.clave_objeto, contenido)
    return modelo.clave_objeto


def main() -> None:
    parser = argparse.ArgumentParser(description="Publica un modelo de placas en el almacén de objetos")
    parser.add_argument("ficha", type=Path)
    parser.add_argument("archivo", type=Path)
    args = parser.parse_args()
    config = Config.desde_entorno()
    cliente = ClienteS3(config.s3_endpoint, config.s3_acceso, config.s3_secreto, config.s3_region)
    clave = publicar(args.ficha, args.archivo, cliente, config.s3_bucket_modelos)
    print(f"Publicado: {config.s3_bucket_modelos}/{clave}")
    print(f"Para usarlo: MODELO_PLACAS_FICHA={args.ficha}")


if __name__ == "__main__":
    main()
