"""Reglas de rostros en el dominio, sin infraestructura: I1 (consentimiento), I2 (zonas) e I3."""

from datetime import datetime, timedelta
from uuid import UUID

import pytest

from app.domain.errores import ReglaViolada
from app.domain.rostros import (
    DIMENSION,
    Consentimiento,
    Desconocido,
    EmbeddingRegistrado,
    PerfilFacial,
    ReconocedorFacial,
    Umbrales,
    Zona,
    normalizar,
)
from tests.dobles import AHORA, con_similitud, eje

ANA, BEA = UUID(int=1), UUID(int=2)
UMBRALES = Umbrales(alto=0.70, bajo=0.50)


def _consentimiento(version: str = "v1", fecha: datetime = AHORA - timedelta(days=1)) -> Consentimiento:
    return Consentimiento(fecha, version)


def _registro(vector: tuple[float, ...], foto: str = "f1") -> EmbeddingRegistrado:
    return EmbeddingRegistrado(vector, "facenet", "1.0", foto, AHORA)


def _perfil(
    residente: UUID = ANA, vector: tuple[float, ...] | None = None, sub: str = "sub-ana"
) -> PerfilFacial:
    return PerfilFacial.registrar(
        residente, sub, UUID(int=99), _consentimiento(), [_registro(vector or eje(0))], AHORA
    )


# ------------------------------------------------------------------ vectores


def test_los_vectores_se_normalizan_y_deben_tener_512_dimensiones() -> None:
    v = normalizar([3.0] + [0.0] * (DIMENSION - 1))
    assert v[0] == pytest.approx(1.0) and len(v) == DIMENSION
    with pytest.raises(ValueError):
        normalizar([1.0, 2.0])
    with pytest.raises(ValueError):
        normalizar([0.0] * DIMENSION)


# ------------------------------------------------------------------ I2: tres zonas


@pytest.mark.parametrize(
    ("similitud", "zona"),
    [
        (0.95, Zona.RECONOCIDO),
        (0.70, Zona.RECONOCIDO),
        (0.699, Zona.DUDOSO),
        (0.50, Zona.DUDOSO),
        (0.499, Zona.DESCONOCIDO),
        (0.0, Zona.DESCONOCIDO),
        (-0.3, Zona.DESCONOCIDO),
    ],
)
def test_I2_la_zona_depende_de_los_umbrales_y_los_bordes_son_inclusivos(similitud: float, zona: Zona) -> None:
    assert UMBRALES.clasificar(similitud) is zona


@pytest.mark.parametrize(("alto", "bajo"), [(0.5, 0.7), (1.2, 0.5), (0.7, 0.0), (0.7, -0.1)])
def test_I2_umbrales_incoherentes_se_rechazan(alto: float, bajo: float) -> None:
    with pytest.raises(ValueError):
        Umbrales(alto=alto, bajo=bajo)


def test_I2_se_reconoce_al_residente_mas_parecido() -> None:
    perfiles = [_perfil(ANA, eje(0)), _perfil(BEA, eje(5), "sub-bea")]
    r = ReconocedorFacial(UMBRALES).identificar(con_similitud(0.9, base=5, otro=6), perfiles)
    assert r.zona is Zona.RECONOCIDO and r.residente_id == BEA
    assert r.similitud == pytest.approx(0.9)


def test_I2_en_la_zona_dudosa_se_devuelve_el_candidato_y_decide_el_guardia() -> None:
    r = ReconocedorFacial(UMBRALES).identificar(con_similitud(0.6), [_perfil(ANA, eje(0))])
    assert r.zona is Zona.DUDOSO and r.residente_id == ANA


def test_I2_por_debajo_del_umbral_bajo_es_desconocido_sin_candidato() -> None:
    r = ReconocedorFacial(UMBRALES).identificar(con_similitud(0.3), [_perfil(ANA, eje(0))])
    assert r.zona is Zona.DESCONOCIDO and r.residente_id is None
    assert r.similitud == pytest.approx(0.3)


def test_I2_sin_perfiles_todo_rostro_es_desconocido() -> None:
    r = ReconocedorFacial(UMBRALES).identificar(eje(0), [])
    assert r.zona is Zona.DESCONOCIDO and r.residente_id is None and r.similitud == 0.0


def test_I2_se_toma_la_foto_registrada_mas_parecida_de_cada_perfil() -> None:
    perfil = PerfilFacial.registrar(
        ANA,
        "sub-ana",
        UUID(int=99),
        _consentimiento(),
        [_registro(eje(0), "a"), _registro(eje(7), "b")],
        AHORA,
    )
    r = ReconocedorFacial(UMBRALES).identificar(con_similitud(0.95, base=7, otro=8), [perfil])
    assert r.zona is Zona.RECONOCIDO


def test_I1_un_perfil_con_el_consentimiento_retirado_nunca_se_reconoce() -> None:
    perfil = _perfil(ANA, eje(0))
    perfil.retirar_consentimiento(AHORA)
    r = ReconocedorFacial(UMBRALES).identificar(eje(0), [perfil])
    assert r.zona is Zona.DESCONOCIDO and r.residente_id is None


# ------------------------------------------------------------------ I1: consentimiento


def test_I1_el_consentimiento_necesita_la_version_del_texto_aceptado() -> None:
    for version in ("", "   "):
        with pytest.raises(ReglaViolada) as e:
            Consentimiento(AHORA, version)
        assert e.value.codigo == "I1"


def test_I1_el_consentimiento_lleva_zona_horaria() -> None:
    with pytest.raises(ReglaViolada) as e:
        Consentimiento(datetime(2026, 10, 19, 12, 0), "v1")  # sin tzinfo
    assert e.value.codigo == "I1"


def test_I1_no_se_registra_con_un_consentimiento_fechado_en_el_futuro() -> None:
    futuro = Consentimiento(AHORA + timedelta(days=1), "v1")
    with pytest.raises(ReglaViolada) as e:
        PerfilFacial.registrar(ANA, "sub-ana", UUID(int=99), futuro, [_registro(eje(0))], AHORA)
    assert e.value.codigo == "I1"


def test_I1_el_perfil_se_registra_con_consentimiento_y_embeddings() -> None:
    p = _perfil()
    assert p.vigente and len(p.embeddings) == 1 and p.consentimiento.version == "v1"
    assert p.claves_fotos() == ["f1"]
    with pytest.raises(ValueError):
        PerfilFacial.registrar(ANA, "sub-ana", UUID(int=99), _consentimiento(), [], AHORA)


def test_I1_retirar_el_consentimiento_borra_los_embeddings_y_es_idempotente() -> None:
    p = _perfil()
    p.retirar_consentimiento(AHORA)
    assert not p.vigente and p.embeddings == [] and p.claves_fotos() == [] and p.retirado_en == AHORA
    p.retirar_consentimiento(AHORA + timedelta(hours=5))  # repetirlo no cambia nada
    assert p.retirado_en == AHORA


def test_I1_a_un_perfil_retirado_no_se_le_agregan_embeddings() -> None:
    p = _perfil()
    p.retirar_consentimiento(AHORA)
    with pytest.raises(ReglaViolada) as e:
        p.agregar([_registro(eje(3), "otra")])
    assert e.value.codigo == "I1"


def test_I1_agregar_fotos_a_un_perfil_vigente_suma_embeddings() -> None:
    p = _perfil()
    p.agregar([_registro(eje(3), "otra")])
    assert len(p.embeddings) == 2 and p.claves_fotos() == ["f1", "otra"]


# ------------------------------------------------------------------ I3: desconocidos


def test_I3_el_desconocido_vence_a_los_30_dias_de_ser_capturado() -> None:
    d = Desconocido.capturar(eje(0), "facenet", "1.0", "portada", AHORA, timedelta(days=30))
    assert d.expira_en == AHORA + timedelta(days=30) and d.dia == "2026-10-20" and d.camara == "portada"
    with pytest.raises(ValueError):
        Desconocido.capturar(eje(0), "facenet", "1.0", None, AHORA, timedelta(0))
