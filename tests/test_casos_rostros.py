"""Casos de uso de rostros con dobles: I1 (consentimiento y retiro), I2 (zonas), I3 (desconocidos), I5."""

from datetime import datetime, timedelta

import pytest

from app.application.errores import SinPermiso
from app.application.rostros import ImagenInvalida, PerfilNoEncontrado, SinRostrosUtilizables
from app.domain.errores import ReglaViolada
from app.domain.roles import Rol, Solicitante
from tests.dobles import AHORA, ANA, BEA, FECHA, UNIDAD, Mundo


@pytest.fixture
def mundo() -> Mundo:
    return Mundo()


def _solicitante(sub: str, *roles: Rol) -> Solicitante:
    return Solicitante(sub, frozenset(roles))


# ------------------------------------------------------------------ registrar (I1)


def test_I1_se_registra_un_perfil_con_consentimiento_y_las_fotos_van_al_almacen(mundo: Mundo) -> None:
    r = mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"ana-1", b"ana-2"])
    assert (r.residente_id, r.embeddings, r.modelo, r.version) == (ANA, 2, "facenet-falso", "0.0.1")
    perfil = mundo.perfiles.por_residente(ANA)
    assert perfil is not None and perfil.consentimiento.version == "v1" and perfil.residente_sub == "sub-ana"
    assert len(mundo.fotos.objetos) == 2 and set(perfil.claves_fotos()) == set(mundo.fotos.objetos)


def test_I1_sin_version_del_consentimiento_no_se_registra_y_no_se_toca_nada(mundo: Mundo) -> None:
    with pytest.raises(ReglaViolada) as e:
        mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "", [b"ana-1"])
    assert e.value.codigo == "I1"
    assert (
        mundo.modelo.llamadas == 0 and mundo.fotos.objetos == {} and mundo.perfiles.por_residente(ANA) is None
    )


@pytest.mark.parametrize(("fecha", "version"), [(None, "v1"), (FECHA, None), (None, None)])
def test_I1_si_falta_la_fecha_o_la_version_no_hay_consentimiento(
    mundo: Mundo, fecha: datetime | None, version: str | None
) -> None:
    with pytest.raises(ReglaViolada) as e:
        mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, fecha, version, [b"ana-1"])
    assert e.value.codigo == "I1" and mundo.modelo.llamadas == 0


def test_las_fotos_sin_rostro_se_omiten(mundo: Mundo) -> None:
    r = mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"ana-1", b"sin-rostro"])
    assert r.embeddings == 1 and len(mundo.fotos.objetos) == 1


def test_si_ninguna_foto_tiene_rostro_no_se_guarda_nada(mundo: Mundo) -> None:
    with pytest.raises(SinRostrosUtilizables):
        mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"sin-rostro"])
    assert mundo.fotos.objetos == {} and mundo.perfiles.por_residente(ANA) is None


def test_una_imagen_ilegible_se_rechaza(mundo: Mundo) -> None:
    with pytest.raises(ImagenInvalida):
        mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"basura"])


def test_registrar_de_nuevo_a_un_residente_vigente_amplia_su_perfil(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v1", [b"ana-otra"])
    assert r.embeddings == 3 and len(mundo.fotos.objetos) == 3


def test_I1_tras_retirar_el_consentimiento_se_puede_registrar_con_uno_nuevo(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    mundo.retirar.ejecutar(_solicitante("sub-ana", Rol.RESIDENTE), ANA)
    r = mundo.registrar.ejecutar(ANA, "sub-ana", UNIDAD, FECHA, "v2", [b"ana-1"])
    perfil = mundo.perfiles.por_residente(ANA)
    assert (
        r.embeddings == 1 and perfil is not None and perfil.vigente and perfil.consentimiento.version == "v2"
    )


# ------------------------------------------------------------------ identificar (I2, I3, I5)


def test_I2_un_rostro_muy_parecido_se_reconoce_con_su_confianza(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.identificar.ejecutar(b"ana-otra", camara="porteria")
    assert r.resultado == "RECONOCIDO" and r.residente_id == ANA
    assert r.confianza == pytest.approx(0.95) and (r.modelo, r.version) == ("facenet-falso", "0.0.1")
    assert mundo.desconocidos.guardados == []


def test_I2_en_la_zona_dudosa_se_devuelve_el_candidato_y_no_se_guarda_como_desconocido(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.identificar.ejecutar(b"parecido")
    assert r.resultado == "DUDOSO" and r.residente_id == ANA and r.confianza == pytest.approx(0.60)
    assert mundo.desconocidos.guardados == []  # I3: un DUDOSO puede ser un residente


def test_I3_un_desconocido_se_guarda_30_dias_para_el_agrupamiento_nocturno(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.identificar.ejecutar(b"extrano", camara="porteria")
    assert r.resultado == "DESCONOCIDO" and r.residente_id is None
    [d] = mundo.desconocidos.guardados
    assert d.expira_en == AHORA + timedelta(days=30) and d.camara == "porteria"


def test_I2_si_no_hay_nadie_registrado_todo_rostro_es_desconocido(mundo: Mundo) -> None:
    r = mundo.identificar.ejecutar(b"ana-1")
    assert r.resultado == "DESCONOCIDO" and r.confianza == 0.0 and len(mundo.desconocidos.guardados) == 1


def test_sin_rostro_en_la_imagen_se_responde_sin_rostro_y_no_se_guarda_desconocido(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.identificar.ejecutar(b"sin-rostro")
    assert r.resultado == "SIN_ROSTRO" and r.residente_id is None and r.confianza == 0.0
    assert mundo.desconocidos.guardados == [] and len(mundo.predicciones.guardadas) == 1


def test_I5_toda_prediccion_guarda_modelo_version_confianza_y_fecha(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.identificar.ejecutar(b"parecido", camara="porteria")
    [(prediccion_id, p)] = mundo.predicciones.guardadas
    assert prediccion_id == r.prediccion_id
    assert (p.tipo, p.resultado, p.residente_id) == ("ROSTRO", "DUDOSO", ANA)
    assert (p.modelo, p.version_modelo, p.fecha, p.camara) == ("facenet-falso", "0.0.1", AHORA, "porteria")
    assert p.confianza == pytest.approx(0.60) and (p.umbral_alto, p.umbral_bajo) == (0.70, 0.50)
    assert p.latencia_ms >= 0


def test_la_confianza_nunca_es_negativa(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    assert mundo.identificar.ejecutar(b"opuesto").confianza == 0.0


def test_una_imagen_ilegible_no_deja_rastro(mundo: Mundo) -> None:
    with pytest.raises(ImagenInvalida):
        mundo.identificar.ejecutar(b"basura")
    assert mundo.predicciones.guardadas == [] and mundo.desconocidos.guardados == []


# ------------------------------------------------------------------ retirar (I1)


def test_I1_el_residente_retira_su_consentimiento_y_se_borran_fotos_y_embeddings(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    r = mundo.retirar.ejecutar(_solicitante("sub-ana", Rol.RESIDENTE), ANA)
    assert r.residente_id == ANA and r.borrado_antes_de == AHORA + timedelta(hours=24)
    assert mundo.fotos.objetos == {}
    perfil = mundo.perfiles.por_residente(ANA)
    assert perfil is not None and not perfil.vigente and perfil.embeddings == []


def test_I1_tras_el_retiro_el_residente_ya_no_se_reconoce(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    mundo.retirar.ejecutar(_solicitante("sub-ana", Rol.RESIDENTE), ANA)
    assert mundo.identificar.ejecutar(b"ana-1").resultado == "DESCONOCIDO"


def test_I1_un_admin_puede_retirar_el_consentimiento_de_cualquiera(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    mundo.retirar.ejecutar(_solicitante("sub-admin", Rol.ADMIN), ANA)
    perfil = mundo.perfiles.por_residente(ANA)
    assert perfil is not None and not perfil.vigente


def test_I1_un_residente_no_puede_retirar_el_consentimiento_de_otro(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    with pytest.raises(SinPermiso):
        mundo.retirar.ejecutar(_solicitante("sub-bea", Rol.RESIDENTE), ANA)
    assert len(mundo.fotos.objetos) == 2  # nada se borró
    perfil = mundo.perfiles.por_residente(ANA)
    assert perfil is not None and perfil.vigente


def test_retirar_el_consentimiento_de_quien_no_tiene_perfil_es_no_encontrado(mundo: Mundo) -> None:
    with pytest.raises(PerfilNoEncontrado):
        mundo.retirar.ejecutar(_solicitante("sub-admin", Rol.ADMIN), BEA)


def test_I1_retirar_dos_veces_devuelve_lo_mismo_y_borra_una_sola_vez(mundo: Mundo) -> None:
    mundo.registrar_a_ana()
    solicitante = _solicitante("sub-ana", Rol.RESIDENTE)
    primera = mundo.retirar.ejecutar(solicitante, ANA)
    mundo.reloj.avanzar(timedelta(hours=5))
    segunda = mundo.retirar.ejecutar(solicitante, ANA)
    assert segunda == primera
    assert mundo.fotos.borrados == 1
