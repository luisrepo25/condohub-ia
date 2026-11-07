"""I1: las fotos de rostros pueden vivir en un almacén distinto al de los reportes y los modelos.

Los reportes y modelos usan S3_ENDPOINT (p. ej. AWS); las fotos conservan su almacén de siempre (el de la VM)
con S3_ROSTROS_*. Sin esas variables todo se comporta como antes: las fotos usan el mismo almacén general.
"""

from app.config import Config
from app.infrastructure.composicion import construir_almacen_fotos
from app.infrastructure.s3_cliente import AlmacenFotosS3

AWS = {
    "S3_ENDPOINT": "https://s3.us-east-1.amazonaws.com",
    "S3_ACCESS_KEY": "aws-acceso",
    "S3_SECRET_KEY": "aws-secreto",
}
VM = {
    "S3_ROSTROS_ENDPOINT": "http://minio:9000",
    "S3_ROSTROS_ACCESS_KEY": "vm-acceso",
    "S3_ROSTROS_SECRET_KEY": "vm-secreto",
}


def _config(monkeypatch, *grupos: dict[str, str]) -> Config:
    for clave in (*AWS, *VM, "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"):
        monkeypatch.delenv(clave, raising=False)
    for grupo in grupos:
        for clave, valor in grupo.items():
            monkeypatch.setenv(clave, valor)
    return Config.desde_entorno()


def test_I1_sin_variables_propias_las_fotos_usan_el_almacen_general(monkeypatch):
    config = _config(monkeypatch, AWS)

    assert config.almacen_rostros == ("https://s3.us-east-1.amazonaws.com", "aws-acceso", "aws-secreto")


def test_I1_con_variables_propias_las_fotos_van_a_su_almacen_y_el_resto_al_general(monkeypatch):
    config = _config(monkeypatch, AWS, VM)

    assert config.almacen_rostros == ("http://minio:9000", "vm-acceso", "vm-secreto")
    assert config.s3_endpoint == "https://s3.us-east-1.amazonaws.com"
    assert (config.s3_acceso, config.s3_secreto) == ("aws-acceso", "aws-secreto")


def test_I1_cada_dato_propio_que_falte_cae_al_general(monkeypatch):
    solo_endpoint = {"S3_ROSTROS_ENDPOINT": "http://minio:9000"}

    config = _config(monkeypatch, AWS, solo_endpoint)

    assert config.almacen_rostros == ("http://minio:9000", "aws-acceso", "aws-secreto")


def test_I1_el_almacen_de_fotos_s3_se_construye_con_el_endpoint_y_las_claves_propias(monkeypatch):
    config = _config(monkeypatch, AWS, VM)
    config = Config(**{**config.__dict__, "almacen_fotos": "s3"})

    fotos = construir_almacen_fotos(config)

    assert isinstance(fotos, AlmacenFotosS3)
    assert fotos._cliente._host == "minio:9000"
    assert fotos._cliente._acceso == "vm-acceso"
    assert fotos._cliente._secreto == "vm-secreto"


def test_I1_sin_almacen_propio_las_fotos_s3_usan_el_cliente_general_como_antes(monkeypatch):
    config = _config(monkeypatch, AWS)
    config = Config(**{**config.__dict__, "almacen_fotos": "s3"})

    fotos = construir_almacen_fotos(config)

    assert fotos._cliente._host == "s3.us-east-1.amazonaws.com"
    assert fotos._cliente._acceso == "aws-acceso"


def test_I1_el_bucket_de_rostros_sigue_siendo_el_configurado(monkeypatch):
    config = _config(monkeypatch, AWS, VM, {"S3_BUCKET_ROSTROS": "rostros"})
    config = Config(**{**config.__dict__, "almacen_fotos": "s3"})

    assert construir_almacen_fotos(config)._bucket == "rostros"
