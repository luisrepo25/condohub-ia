import numpy as np
import pytest

from ml.src import registro

CFG = {"semilla": 42, "far_maximo": 0.02}


def _mundo(n_personas: int, con_sesion_extra: bool = True):
    """Personas con 2 sesiones: la 2ª cambia un poco de condiciones (desplazamiento común), como luz baja vs alta."""
    rng = np.random.default_rng(5)
    centros = rng.normal(size=(n_personas, 512))
    emb, personas, sesiones = [], [], []
    for p in range(n_personas):
        for s, nombre in enumerate(("luz-baja", "luz-alta")):
            if s == 1 and not con_sesion_extra:
                continue
            desplazamiento = rng.normal(scale=0.2, size=512) * s
            emb.append(centros[p] + desplazamiento + rng.normal(scale=0.3, size=(25, 512)))
            personas += [f"p{p}"] * 25
            sesiones += [nombre] * 25
    imp_etiquetas = np.repeat(np.arange(60), 5)
    imp = rng.normal(size=(60, 512))[imp_etiquetas] + rng.normal(scale=0.3, size=(300, 512))
    return np.vstack(emb), np.array(personas), np.array(sesiones), imp, imp_etiquetas


def test_I2_el_umbral_respeta_el_far_maximo_en_los_impostores_de_validacion() -> None:
    s = np.random.default_rng(1).normal(0.3, 0.1, 1000)
    umbral = registro.umbral_para_far(s, 0.02)
    assert (s >= umbral).mean() <= 0.02


def test_una_sola_persona_se_reconoce_entre_sesiones_y_los_impostores_no_pasan() -> None:
    emb, personas, sesiones, imp, imp_et = _mundo(1)
    r = registro.evaluar(emb, personas, sesiones, imp, imp_et, CFG)
    assert r["frr"] < 0.1
    assert r["far_prueba"] <= 0.05
    assert r["confusiones"] == 0  # con una persona no hay a quién confundir
    assert {f["sesion"] for f in r["por_sesion"]} == {"luz-baja", "luz-alta"}


def test_I1_sumar_una_segunda_persona_no_exige_reentrenar_y_se_miden_las_confusiones() -> None:
    emb, personas, sesiones, imp, imp_et = _mundo(2)
    r = registro.evaluar(emb, personas, sesiones, imp, imp_et, CFG)
    assert {f["persona"] for f in r["por_sesion"]} == {"p0", "p1"}
    assert r["confusiones"] == 0 and r["frr"] < 0.1


def test_una_persona_con_una_sola_sesion_se_omite_y_se_avisa() -> None:
    emb, personas, sesiones, imp, imp_et = _mundo(1, con_sesion_extra=False)
    r = registro.evaluar(emb, personas, sesiones, imp, imp_et, CFG)
    assert r["por_sesion"] == [] and r["frr"] is None
    assert r["sesiones_omitidas_por_ser_la_unica"] == ["p0/luz-baja"]


def test_A5_un_desconocido_sin_parecido_queda_por_debajo_del_umbral() -> None:
    emb, personas, sesiones, imp, imp_et = _mundo(1)
    r = registro.evaluar(emb, personas, sesiones, imp, imp_et, CFG)
    nombres, scores = registro.puntuar(imp[:20], {"p0": emb[personas == "p0"]})
    assert (scores < r["umbral_calibrado_con_impostores_de_validacion"]).mean() > 0.9


def test_la_evaluacion_es_reproducible_con_la_misma_semilla() -> None:
    args = _mundo(2)
    assert registro.evaluar(*args, CFG) == registro.evaluar(*args, CFG)


def test_pocos_impostores_se_rechazan() -> None:
    emb, personas, sesiones, _, _ = _mundo(1)
    with pytest.raises(ValueError):
        registro.evaluar(emb, personas, sesiones, emb[:6], np.array([0, 0, 1, 1, 2, 2]), CFG)
