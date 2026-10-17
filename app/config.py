import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Configuración por variables de entorno (nunca valores de negocio fijos en el código)."""

    keycloak_issuer: str
    keycloak_jwks: str
    # Modelo de placas para la PWA del guardia (O6) y almacén de objetos (SeaweedFS con API S3; ver ADR 0002)
    modelo_placas_ficha: str = ""
    modelo_lectura_ficha: str = ""  # ficha del reconocedor de texto de placas (notebook 11)
    s3_endpoint: str = "http://localhost:9000"
    s3_endpoint_publico: str = ""   # el que ve el cliente en las URLs firmadas; vacío = el mismo s3_endpoint
    s3_acceso: str = ""
    s3_secreto: str = ""
    s3_bucket_modelos: str = "modelos"
    s3_region: str = "us-east-1"
    url_firmada_segundos: int = 900
    # Rostros: parámetros de negocio de I1, I2 e I3 (valores iniciales de reglas-de-negocio.md)
    umbral_reconocido: float = 0.70
    umbral_dudoso: float = 0.50
    desconocidos_retencion_dias: int = 30
    retiro_plazo_horas: int = 24
    # Modelo facial: "" = sin configurar (503) o "facenet" (MTCNN + FaceNet preentrenado; necesita PyTorch)
    modelo_facial: str = ""
    modelo_facial_version: str = "1.0.0"
    modelo_facial_dispositivo: str = ""
    # Persistencia: "memoria" (se pierde al reiniciar; desarrollo y pruebas) o "mongo"
    persistencia: str = "memoria"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "ia"
    # Fotos de los rostros: "memoria" (se pierden al reiniciar) o "s3" (el almacén de objetos del proyecto)
    almacen_fotos: str = "memoria"
    s3_bucket_rostros: str = "rostros"
    # I4: visitantes recurrentes (parámetros del job nocturno; valores de ml/README.md, notebook 05)
    dbscan_eps: float = 0.30
    dbscan_min_samples: int = 3
    i4_min_apariciones: int = 3
    i4_min_dias: int = 3
    # Eventos de dominio (S4): exchange `condohub.events`; el usuario es el de desarrollo de infra/CLAUDE.md
    rabbitmq_url: str = "amqp://condohub:condohub@localhost:5672/%2F"
    rabbitmq_exchange: str = "condohub.events"
    # G4: reportes por voz. Whisper local, LLM de Gemini (clave solo por entorno) y Gestión como fuente
    gestion_graphql_url: str = "http://localhost:8081/graphql"
    gemini_api_key: str = field(default="", repr=False)
    llm_proveedor: str = "gemini"   # "gemini" o "groq"
    llm_modelo: str = "gemini-flash-latest"
    llm_url: str = "https://generativelanguage.googleapis.com"
    groq_api_key: str = field(default="", repr=False)
    groq_modelo: str = "openai/gpt-oss-20b"
    groq_url: str = "https://api.groq.com/openai"
    whisper_modelo: str = "base"
    whisper_dispositivo: str = "cpu"
    whisper_calculo: str = "int8"
    almacen_reportes: str = "memoria"   # "memoria" o "s3"
    s3_bucket_reportes: str = "reportes"

    @property
    def s3_endpoint_firma(self) -> str:
        """Host con el que se firman las URLs del cliente (dentro de Docker, `minio:9000` no se alcanza)."""
        return self.s3_endpoint_publico or self.s3_endpoint

    @classmethod
    def desde_entorno(cls) -> "Config":
        issuer = os.environ.get("KEYCLOAK_ISSUER", "http://localhost:8180/realms/condohub")
        return cls(
            keycloak_issuer=issuer,
            keycloak_jwks=os.environ.get("KEYCLOAK_JWKS", f"{issuer}/protocol/openid-connect/certs"),
            modelo_placas_ficha=os.environ.get("MODELO_PLACAS_FICHA", ""),
            modelo_lectura_ficha=os.environ.get("MODELO_LECTURA_FICHA", ""),
            s3_endpoint=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
            s3_endpoint_publico=os.environ.get("S3_ENDPOINT_PUBLICO", ""),
            s3_acceso=os.environ.get("S3_ACCESS_KEY", os.environ.get("MINIO_ROOT_USER", "")),
            s3_secreto=os.environ.get("S3_SECRET_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "")),
            s3_bucket_modelos=os.environ.get("S3_BUCKET_MODELOS", "modelos"),
            s3_region=os.environ.get("S3_REGION", "us-east-1"),
            url_firmada_segundos=int(os.environ.get("URL_FIRMADA_SEGUNDOS", "900")),
            umbral_reconocido=float(os.environ.get("UMBRAL_RECONOCIDO", "0.70")),
            umbral_dudoso=float(os.environ.get("UMBRAL_DUDOSO", "0.50")),
            desconocidos_retencion_dias=int(os.environ.get("DESCONOCIDOS_RETENCION_DIAS", "30")),
            retiro_plazo_horas=int(os.environ.get("RETIRO_PLAZO_HORAS", "24")),
            modelo_facial=os.environ.get("MODELO_FACIAL", ""),
            modelo_facial_version=os.environ.get("MODELO_FACIAL_VERSION", "1.0.0"),
            modelo_facial_dispositivo=os.environ.get("MODELO_FACIAL_DISPOSITIVO", ""),
            persistencia=os.environ.get("PERSISTENCIA", "memoria"),
            mongo_uri=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
            mongo_db=os.environ.get("MONGO_DB", "ia"),
            almacen_fotos=os.environ.get("ALMACEN_FOTOS", "memoria"),
            s3_bucket_rostros=os.environ.get("S3_BUCKET_ROSTROS", "rostros"),
            dbscan_eps=float(os.environ.get("DBSCAN_EPS", "0.30")),
            dbscan_min_samples=int(os.environ.get("DBSCAN_MIN_SAMPLES", "3")),
            i4_min_apariciones=int(os.environ.get("I4_MIN_APARICIONES", "3")),
            i4_min_dias=int(os.environ.get("I4_MIN_DIAS", "3")),
            rabbitmq_url=os.environ.get("RABBITMQ_URL", "amqp://condohub:condohub@localhost:5672/%2F"),
            rabbitmq_exchange=os.environ.get("RABBITMQ_EXCHANGE", "condohub.events"),
            gestion_graphql_url=os.environ.get("GESTION_GRAPHQL_URL", "http://localhost:8081/graphql"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            llm_proveedor=os.environ.get("LLM_PROVEEDOR", "gemini"),
            llm_modelo=os.environ.get("LLM_MODELO", "gemini-flash-latest"),
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            groq_modelo=os.environ.get("GROQ_MODELO", "openai/gpt-oss-20b"),
            groq_url=os.environ.get("GROQ_URL", "https://api.groq.com/openai"),
            llm_url=os.environ.get("LLM_URL", "https://generativelanguage.googleapis.com"),
            whisper_modelo=os.environ.get("WHISPER_MODELO", "base"),
            whisper_dispositivo=os.environ.get("WHISPER_DISPOSITIVO", "cpu"),
            whisper_calculo=os.environ.get("WHISPER_CALCULO", "int8"),
            almacen_reportes=os.environ.get("ALMACEN_REPORTES", "memoria"),
            s3_bucket_reportes=os.environ.get("S3_BUCKET_REPORTES", "reportes"),
        )
