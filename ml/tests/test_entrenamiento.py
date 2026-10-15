import numpy as np
import pytest

from ml.src import entrenamiento

torch = pytest.importorskip("torch")  # estas pruebas necesitan PyTorch; en Colab ya está instalado


def test_el_lote_tiene_p_personas_con_k_fotos_cada_una() -> None:
    etiquetas = np.repeat(np.arange(10), 6)
    idx = entrenamiento.muestrear_lote(etiquetas, 4, 3, np.random.default_rng(0))
    assert len(idx) == 12 and len(set(etiquetas[idx])) == 4
    assert all((etiquetas[idx] == p).sum() == 3 for p in set(etiquetas[idx]))


def test_una_persona_con_pocas_fotos_se_repite_en_vez_de_fallar() -> None:
    etiquetas = np.array([0, 0, 1, 1, 2, 2])
    assert len(entrenamiento.muestrear_lote(etiquetas, 3, 4, np.random.default_rng(0))) == 12


def test_la_perdida_es_casi_cero_con_clases_bien_separadas_y_alta_si_estan_mezcladas() -> None:
    etiquetas = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    centros = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
    separadas = torch.nn.functional.normalize(centros[etiquetas] + 0.01 * torch.randn(9, 16), dim=1)
    mezcladas = torch.nn.functional.normalize(torch.randn(9, 16), dim=1)
    perdida_ok, _ = entrenamiento.perdida_triplet_semi_hard(separadas, etiquetas, 0.2)
    perdida_mal, activas = entrenamiento.perdida_triplet_semi_hard(mezcladas, etiquetas, 0.2)
    assert perdida_ok < 0.05 < perdida_mal and activas > 0.3


def test_la_perdida_se_puede_derivar() -> None:
    emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=1).requires_grad_()
    perdida, _ = entrenamiento.perdida_triplet_semi_hard(emb, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), 0.2)
    perdida.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


def test_los_aumentos_mantienen_forma_y_rango_y_son_reproducibles() -> None:
    x = torch.rand(6, 3, 32, 32) * 255
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a, b = entrenamiento.aumentar(x, g1), entrenamiento.aumentar(x, g2)
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 255
    assert torch.equal(a, b) and not torch.equal(a, x)


def _modelo_de_juguete() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
    )


def _caras_de_juguete(personas: int = 6, por_persona: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    patrones = rng.integers(0, 255, size=(personas, 16, 16, 3))
    etiquetas = np.repeat(np.arange(personas), por_persona)
    ruido = rng.integers(-20, 20, size=(len(etiquetas), 16, 16, 3))
    return np.clip(patrones[etiquetas] + ruido, 0, 255).astype(np.uint8), etiquetas


CFG = {
    "semilla": 42,
    "entrenables": ["3"],
    "personas_por_lote": 6,
    "fotos_por_persona": 4,
    "epocas": 6,
    "pasos_por_epoca": 15,
    "tasa_aprendizaje": 0.01,
    "margen": 0.2,
}


def test_congelar_deja_entrenable_solo_lo_pedido() -> None:
    modelo = _modelo_de_juguete()
    assert entrenamiento.congelar_salvo(modelo, ["3"]) == ["3.weight", "3.bias"]
    assert not any(p.requires_grad for n, p in modelo.named_parameters() if not n.startswith("3."))


def test_el_ajuste_baja_la_perdida_y_solo_cambia_las_capas_entrenables() -> None:
    modelo = _modelo_de_juguete()
    antes = {n: p.detach().clone() for n, p in modelo.named_parameters()}
    caras, etiquetas = _caras_de_juguete()
    historial = entrenamiento.ajustar(modelo, caras, etiquetas, CFG, "cpu")
    assert historial[-1]["perdida"] < historial[0]["perdida"]
    for nombre, parametro in modelo.named_parameters():
        cambio = not torch.equal(parametro, antes[nombre])
        assert cambio == nombre.startswith("3."), f"{nombre}: cambió={cambio}"


def test_el_ajuste_es_reproducible_con_la_misma_semilla() -> None:
    caras, etiquetas = _caras_de_juguete()
    a = entrenamiento.ajustar(_modelo_de_juguete(), caras, etiquetas, CFG, "cpu")
    b = entrenamiento.ajustar(_modelo_de_juguete(), caras, etiquetas, CFG, "cpu")
    assert a == b


def test_sin_capas_entrenables_se_rechaza() -> None:
    caras, etiquetas = _caras_de_juguete()
    with pytest.raises(ValueError):
        entrenamiento.ajustar(
            _modelo_de_juguete(), caras, etiquetas, {**CFG, "entrenables": ["no_existe"]}, "cpu"
        )


def test_los_embeddings_salen_normalizados() -> None:
    caras, _ = _caras_de_juguete()
    emb = entrenamiento.embeber_caras(_modelo_de_juguete(), caras, "cpu")
    assert emb.shape == (len(caras), 8) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)


def test_la_adopcion_exige_superar_a_la_linea_base() -> None:
    mejor = entrenamiento.decidir_adopcion({"eer_prueba": 0.05}, {"eer_prueba": 0.03}, 0.0)
    peor = entrenamiento.decidir_adopcion({"eer_prueba": 0.05}, {"eer_prueba": 0.06}, 0.0)
    igual = entrenamiento.decidir_adopcion({"eer_prueba": 0.05}, {"eer_prueba": 0.05}, 0.0)
    assert mejor["adoptar"] and not peor["adoptar"] and not igual["adoptar"]


def test_la_repeticion_con_semillas_da_una_fila_por_semilla_y_es_reproducible() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    patrones = rng.integers(0, 255, size=(20, 16, 16, 3))
    etiquetas = np.repeat(np.arange(20), 12)
    caras = np.clip(
        patrones[etiquetas] + rng.integers(-120, 120, size=(len(etiquetas), 16, 16, 3)), 0, 255
    ).astype(np.uint8)
    base = _modelo_de_juguete()
    emb_base = entrenamiento.embeber_caras(base, caras, "cpu")
    cfg = {
        **CFG,
        "proporciones": [0.6, 0.2, 0.2],
        "pares_por_parte": 200,
        "far_maximo": 0.05,
        "epocas": 2,
        "pasos_por_epoca": 5,
    }
    a = entrenamiento.comparar_con_semillas(
        [1, 2, 3], caras, etiquetas, emb_base, cfg, _modelo_de_juguete, "cpu"
    )
    b = entrenamiento.comparar_con_semillas(
        [1, 2, 3], caras, etiquetas, emb_base, cfg, _modelo_de_juguete, "cpu"
    )
    assert [f["semilla"] for f in a] == [1, 2, 3] and a == b
    assert all(f["mejora"] == pytest.approx(f["eer_base"] - f["eer_ajustado"]) for f in a)
    assert len({f["eer_base"] for f in a}) > 1  # cada semilla cambia la partición de personas
