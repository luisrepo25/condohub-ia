import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuración por variables de entorno (nunca valores de negocio fijos en el código)."""

    keycloak_issuer: str
    keycloak_jwks: str
    # Modelo de placas para la PWA del guardia (O6) y almacén de objetos (SeaweedFS con API S3; ver ADR 0002)
    modelo_placas_ficha: str = ""
    s3_endpoint: str = "http://localhost:9000"
    s3_acceso: str = ""
    s3_secreto: str = ""
    s3_bucket_modelos: str = "modelos"
    s3_region: str = "us-east-1"
    url_firmada_segundos: int = 900

    @classmethod
    def desde_entorno(cls) -> "Config":
        issuer = os.environ.get("KEYCLOAK_ISSUER", "http://localhost:8180/realms/condohub")
        return cls(
            keycloak_issuer=issuer,
            keycloak_jwks=os.environ.get("KEYCLOAK_JWKS", f"{issuer}/protocol/openid-connect/certs"),
            modelo_placas_ficha=os.environ.get("MODELO_PLACAS_FICHA", ""),
            s3_endpoint=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
            s3_acceso=os.environ.get("S3_ACCESS_KEY", os.environ.get("MINIO_ROOT_USER", "")),
            s3_secreto=os.environ.get("S3_SECRET_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "")),
            s3_bucket_modelos=os.environ.get("S3_BUCKET_MODELOS", "modelos"),
            s3_region=os.environ.get("S3_REGION", "us-east-1"),
            url_firmada_segundos=int(os.environ.get("URL_FIRMADA_SEGUNDOS", "900")),
        )
