"""Rutas de trabajo. Los datos NUNCA van al repositorio (I6): en Colab viven en Google Drive."""

import os
from datetime import date
from pathlib import Path

RAIZ_ML = Path(__file__).resolve().parents[1]


def dir_datos() -> Path:
    """Carpeta de datasets y pesos. `ML_DATA_DIR` la fija; en Colab apunta a una carpeta de Drive."""
    return Path(os.environ.get("ML_DATA_DIR", RAIZ_ML / "data"))


def dir_experimento(nombre: str, base: Path | None = None) -> Path:
    """`ml/experiments/<fecha>_<nombre>/`: métricas y gráficos (sí se versionan; los pesos no)."""
    carpeta = (base or RAIZ_ML / "experiments") / f"{date.today().isoformat()}_{nombre}"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta
