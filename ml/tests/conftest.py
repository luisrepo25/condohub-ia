import numpy as np
import pytest


@pytest.fixture
def embeddings_sinteticos() -> tuple[np.ndarray, np.ndarray]:
    """20 personas con 12 fotos cada una: una nube alrededor del centro de cada persona (512-d)."""
    rng = np.random.default_rng(0)
    centros = rng.normal(size=(20, 512))
    etiquetas = np.repeat(np.arange(20), 12)
    emb = centros[etiquetas] + rng.normal(scale=0.35, size=(len(etiquetas), 512))
    return emb, etiquetas
