import numpy as np
import pytest

from ml.src import zonas


def _mundo(personas: int = 3, sesiones: int = 2, fotos: int = 15):
    """Personas con sesiones que cambian un poco; y 60 extraños distintos de todas ellas."""
    rng = np.random.default_rng(7)
    centros = rng.normal(size=(personas, 512))
    emb, p, s = [], [], []
    for i in range(personas):
        for j in range(sesiones):
            desplazamiento = rng.normal(scale=0.2, size=512)
            emb.append(centros[i] + desplazamiento + rng.normal(scale=0.3, size=(fotos, 512)))
            p += [f"p{i}"] * fotos
            s += [f"s{j}"] * fotos
    et = np.repeat(np.arange(60), 5)
    imp = rng.normal(size=(60, 512))[et] + rng.normal(scale=0.3, size=(300, 512))
    return np.vstack(emb), np.array(p), np.array(s), imp, et


def test_I2_las_zonas_respetan_los_umbrales_incluido_el_borde() -> None:
    sim = np.array([0.95, 0.70, 0.699, 0.50, 0.499, 0.0])
    assert zonas.clasificar(sim, 0.70, 0.50).tolist() == [
        "RECONOCIDO", "RECONOCIDO", "DUDOSO", "DUDOSO", "DESCONOCIDO", "DESCONOCIDO",
    ]


def test_I2_el_umbral_bajo_no_puede_superar_al_alto() -> None:
    with pytest.raises(ValueError):
        zonas.clasificar(np.array([0.5]), alto=0.4, bajo=0.6)


def test_I2_el_reparto_suma_uno() -> None:
    r = zonas.reparto(np.random.default_rng(0).uniform(size=200), 0.7, 0.5)
    assert sum(r.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        zonas.reparto(np.array([]), 0.7, 0.5)


def test_I2_calibrar_deja_pasar_a_lo_sumo_el_far_pedido_y_el_bajo_no_supera_al_alto() -> None:
    extranos = np.random.default_rng(1).normal(0.3, 0.1, 2000)
    alto, bajo = zonas.calibrar(extranos, far_alto=0.01, far_bajo=0.10)
    assert (extranos >= alto).mean() <= 0.01
    assert (extranos >= bajo).mean() <= 0.10
    assert bajo <= alto


def test_I2_calibrar_rechaza_fars_incoherentes() -> None:
    with pytest.raises(ValueError):
        zonas.calibrar(np.arange(100) / 100, far_alto=0.2, far_bajo=0.1)


def test_las_personas_propias_se_reconocen_y_los_extranos_quedan_como_desconocidos() -> None:
    emb, p, s, imp, et = _mundo()
    puntos = zonas.puntajes_equipo(emb, p, s, imp, et)
    r = zonas.evaluar_zonas(puntos, {"regla": (0.70, 0.50)})["regla"]
    assert r["genuinos"]["reconocido_correcto"] > 0.9
    assert r["genuinos"]["reconocido_otra_persona"] == 0.0
    assert r["extranos_prueba"]["DESCONOCIDO"] > 0.9
    assert sum(v for k, v in r["genuinos"].items() if k != "fotos") == pytest.approx(1.0)


def test_los_companeros_no_registrados_no_deben_ser_reconocidos_como_otro() -> None:
    emb, p, s, imp, et = _mundo()
    puntos = zonas.puntajes_equipo(emb, p, s, imp, et)
    r = zonas.evaluar_zonas(puntos, {"regla": (0.70, 0.50)})["regla"]
    assert r["companeros_no_registrados"]["RECONOCIDO"] == 0.0


def test_con_una_sola_persona_no_hay_companeros() -> None:
    emb, p, s, imp, et = _mundo(personas=1)
    puntos = zonas.puntajes_equipo(emb, p, s, imp, et)
    assert len(puntos["companeros"]) == 0
    assert zonas.evaluar_zonas(puntos, {"regla": (0.7, 0.5)})["regla"]["companeros_no_registrados"] is None


def test_una_persona_con_una_sola_sesion_se_omite_y_se_avisa() -> None:
    emb, p, s, imp, et = _mundo(personas=1, sesiones=1)
    puntos = zonas.puntajes_equipo(emb, p, s, imp, et)
    assert len(puntos["genuinos"]) == 0 and puntos["sesiones_omitidas"].tolist() == ["p0/s0"]
    assert zonas.evaluar_zonas(puntos, {"regla": (0.7, 0.5)})["regla"]["genuinos"] is None


def test_el_calibrado_con_extranos_se_agrega_a_los_umbrales_fijos() -> None:
    emb, p, s, imp, et = _mundo()
    r = zonas.evaluar_zonas(zonas.puntajes_equipo(emb, p, s, imp, et), {"regla": (0.70, 0.50)})
    assert set(r) == {"regla", "calibrado_con_extranos"}
    assert r["calibrado_con_extranos"]["bajo"] <= r["calibrado_con_extranos"]["alto"]


def test_pocos_impostores_se_rechazan() -> None:
    emb, p, s, _, _ = _mundo()
    with pytest.raises(ValueError):
        zonas.puntajes_equipo(emb, p, s, emb[:6], np.array([0, 0, 1, 1, 2, 2]))
