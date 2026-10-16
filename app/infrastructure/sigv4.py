"""URL prefirmada de lectura con AWS Signature Version 4 (S3, MinIO y SeaweedFS). Solo biblioteca estándar."""

import hashlib
import hmac
from datetime import UTC, datetime
from urllib.parse import quote


def _hmac(clave: bytes, mensaje: str) -> bytes:
    return hmac.new(clave, mensaje.encode("utf-8"), hashlib.sha256).digest()


def url_firmada_get(
    *,
    esquema: str,
    host: str,
    ruta: str,
    acceso: str,
    secreto: str,
    region: str,
    ahora: datetime,
    vigencia_s: int,
) -> str:
    """`GET` prefirmado de `ruta` (con barra inicial) en `host` (con puerto si no es el estándar).

    El secreto no viaja en la URL.
    """
    ahora = ahora.astimezone(UTC)
    fecha_hora, fecha = ahora.strftime("%Y%m%dT%H%M%SZ"), ahora.strftime("%Y%m%d")
    alcance = f"{fecha}/{region}/s3/aws4_request"
    parametros = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{acceso}/{alcance}",
        "X-Amz-Date": fecha_hora,
        "X-Amz-Expires": str(vigencia_s),
        "X-Amz-SignedHeaders": "host",
    }
    consulta = "&".join(f"{quote(k, safe='~')}={quote(v, safe='~')}" for k, v in sorted(parametros.items()))
    ruta_codificada = quote(ruta, safe="/~")
    canonica = "\n".join(["GET", ruta_codificada, consulta, f"host:{host}", "", "host", "UNSIGNED-PAYLOAD"])
    a_firmar = "\n".join(
        ["AWS4-HMAC-SHA256", fecha_hora, alcance, hashlib.sha256(canonica.encode("utf-8")).hexdigest()]
    )
    llave = _hmac(_hmac(_hmac(_hmac(f"AWS4{secreto}".encode(), fecha), region), "s3"), "aws4_request")
    firma = hmac.new(llave, a_firmar.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{esquema}://{host}{ruta_codificada}?{consulta}&X-Amz-Signature={firma}"
