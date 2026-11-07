"""G4: los reportes pueden guardarse bajo una carpeta (prefijo) del bucket, p. ej. «reportes/».

Sin prefijo, todo queda igual que antes.
"""

from datetime import UTC, datetime, timedelta

from app.config import Config
from app.infrastructure.s3_reportes import AlmacenReportesS3


class _Cliente:
    def __init__(self) -> None:
        self.subidos: list[tuple[str, str]] = []

    def crear_bucket(self, bucket: str) -> None:
        pass

    def subir(self, bucket: str, clave: str, datos: bytes, tipo: str) -> None:
        self.subidos.append((bucket, clave))


class _Firmador:
    def __init__(self) -> None:
        self.firmadas: list[str] = []

    def firmar_descarga(self, clave: str, vigencia: timedelta, ahora: datetime) -> str:
        self.firmadas.append(clave)
        return f"https://almacen/{clave}"


class _Reloj:
    def ahora(self) -> datetime:
        return datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


def _almacen(prefijo: str | None = None):
    cliente, firmador = _Cliente(), _Firmador()
    args = (cliente, firmador, "condohub-documentos", _Reloj())
    almacen = AlmacenReportesS3(*args) if prefijo is None else AlmacenReportesS3(*args, prefijo=prefijo)
    return almacen, cliente, firmador


def test_G4_con_prefijo_el_reporte_se_sube_dentro_de_la_carpeta():
    almacen, cliente, _ = _almacen("reportes/")

    almacen.guardar("abc.xlsx", b"x", "application/vnd.ms-excel")

    assert cliente.subidos == [("condohub-documentos", "reportes/abc.xlsx")]


def test_G4_con_prefijo_la_url_firmada_apunta_a_la_misma_clave_de_la_carpeta():
    almacen, _, firmador = _almacen("reportes/")

    almacen.firmar_descarga("abc.xlsx", timedelta(minutes=15))

    assert firmador.firmadas == ["reportes/abc.xlsx"]


def test_G4_sin_prefijo_se_comporta_como_siempre():
    almacen, cliente, firmador = _almacen()

    almacen.guardar("abc.xlsx", b"x", "application/vnd.ms-excel")
    almacen.firmar_descarga("abc.xlsx", timedelta(minutes=15))

    assert cliente.subidos == [("condohub-documentos", "abc.xlsx")]
    assert firmador.firmadas == ["abc.xlsx"]


def test_G4_el_prefijo_se_lee_del_entorno_y_por_omision_esta_vacio(monkeypatch):
    monkeypatch.delenv("S3_PREFIJO_REPORTES", raising=False)
    assert Config.desde_entorno().s3_prefijo_reportes == ""

    monkeypatch.setenv("S3_PREFIJO_REPORTES", "reportes/")
    assert Config.desde_entorno().s3_prefijo_reportes == "reportes/"
