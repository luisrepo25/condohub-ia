"""Ajuste fino (fine-tuning) de FaceNet con triplet loss y minado semi-hard. Se ejecuta en Google Colab (GPU).

Parte de InceptionResnetV1 preentrenado en VGGFace2 y reentrena SOLO las últimas capas. Se entrena con personas de
LFW; las fotos propias no entran al entrenamiento, para poder medir con ellas si el modelo generaliza.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

MEDIA, ESCALA = 127.5, 128.0  # misma estandarización que usa facenet-pytorch (fixed_image_standardization)


def muestrear_lote(
    etiquetas: np.ndarray, personas_por_lote: int, fotos_por_persona: int, rng: np.random.Generator
) -> np.ndarray:
    """Índices de un lote P×K: P personas distintas con K fotos cada una (con repetición si tiene menos de K)."""
    personas = np.unique(etiquetas)
    elegidas = rng.choice(personas, size=min(personas_por_lote, len(personas)), replace=False)
    indices: list[int] = []
    for persona in elegidas:
        propias = np.flatnonzero(etiquetas == persona)
        indices.extend(rng.choice(propias, size=fotos_por_persona, replace=len(propias) < fotos_por_persona))
    return np.array(indices)


def perdida_triplet_semi_hard(emb: Any, etiquetas: Any, margen: float) -> tuple[Any, float]:
    """Triplet loss con minado semi-hard dentro del lote.

    Para cada par (ancla, positivo) se elige el negativo MÁS CERCANO que todavía queda más lejos que el positivo
    (d_an > d_ap) y dentro del margen. Si no hay ninguno, se usa el negativo más difícil. Devuelve (pérdida,
    fracción de tripletas que aún violan el margen).
    """
    import torch

    n = len(etiquetas)
    d = torch.cdist(emb, emb)
    mismas = etiquetas[:, None] == etiquetas[None, :]
    positivos = mismas & ~torch.eye(n, dtype=torch.bool, device=emb.device)
    negativos = ~mismas
    d_ap = d.unsqueeze(2)  # (ancla, positivo, 1)
    d_an = d.unsqueeze(1).expand(-1, n, -1)  # (ancla, 1, negativo) -> (ancla, positivo, negativo)
    semi = (d_an > d_ap) & (d_an < d_ap + margen) & negativos.unsqueeze(1)
    infinito = torch.tensor(float("inf"), device=emb.device)
    d_neg_semi = torch.where(semi, d_an, infinito).min(dim=2).values  # (ancla, positivo)
    d_neg_dificil = torch.where(negativos, d, infinito).min(dim=1).values  # (ancla,)
    d_neg = torch.where(torch.isfinite(d_neg_semi), d_neg_semi, d_neg_dificil[:, None])
    perdidas = torch.relu(d - d_neg + margen)[positivos]
    if perdidas.numel() == 0:
        return emb.sum() * 0.0, 0.0
    return perdidas.mean(), float((perdidas > 0).float().mean())


def aumentar(x: Any, generador: Any) -> Any:
    """Aumentos que simulan la cámara de portería: brillo, contraste, reflejo de foco, desenfoque y espejo.

    `x`: lote (B,3,H,W) en 0-255. Devuelve el lote aumentado, recortado a 0-255.
    """
    import torch
    import torch.nn.functional as F

    b, _, alto, ancho = x.shape
    dev = x.device
    media = x.mean(dim=(1, 2, 3), keepdim=True)
    contraste = 0.6 + 0.8 * torch.rand(b, 1, 1, 1, generator=generador, device=dev)
    brillo = 0.4 + 1.2 * torch.rand(b, 1, 1, 1, generator=generador, device=dev)
    x = (x - media) * contraste + media * brillo

    # reflejo de foco (mancha brillante difusa), como en las fotos con contraluz
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, alto, device=dev), torch.linspace(0, 1, ancho, device=dev), indexing="ij"
    )
    cx = torch.rand(b, 1, 1, generator=generador, device=dev)
    cy = torch.rand(b, 1, 1, generator=generador, device=dev)
    radio = 0.1 + 0.2 * torch.rand(b, 1, 1, generator=generador, device=dev)
    intensidad = (80 + 120 * torch.rand(b, 1, 1, generator=generador, device=dev)) * (
        torch.rand(b, 1, 1, generator=generador, device=dev) < 0.3
    )
    mancha = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radio**2)))
    x = x + (intensidad * mancha).unsqueeze(1)

    # desenfoque gaussiano en una parte del lote
    sigma = float(0.5 + 1.5 * torch.rand(1, generator=generador, device=dev))
    ejes = torch.arange(-2, 3, device=dev, dtype=x.dtype)
    nucleo1d = torch.exp(-(ejes**2) / (2 * sigma**2))
    nucleo1d = nucleo1d / nucleo1d.sum()
    nucleo = (nucleo1d[:, None] * nucleo1d[None, :]).expand(3, 1, 5, 5)
    borroso = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), nucleo, groups=3)
    usar_borroso = (torch.rand(b, 1, 1, 1, generator=generador, device=dev) < 0.3).to(x.dtype)
    x = usar_borroso * borroso + (1 - usar_borroso) * x

    espejo = torch.rand(b, generator=generador, device=dev) < 0.5
    x = torch.where(espejo[:, None, None, None], x.flip(-1), x)
    return x.clamp(0, 255)


def congelar_salvo(modelo: Any, entrenables: Sequence[str]) -> list[str]:
    """Congela todo menos los módulos cuyo nombre empieza por alguno de `entrenables`. Devuelve los nombres entrenables."""
    nombres = []
    for nombre, parametro in modelo.named_parameters():
        parametro.requires_grad = any(nombre == e or nombre.startswith(e + ".") for e in entrenables)
        if parametro.requires_grad:
            nombres.append(nombre)
    return nombres


def a_tensor(caras_uint8: np.ndarray, dispositivo: str) -> Any:
    """(N,160,160,3) uint8 -> (N,3,160,160) float32 en 0-255, en el dispositivo."""
    import torch

    return torch.from_numpy(caras_uint8).to(dispositivo).permute(0, 3, 1, 2).float()


def embeber_caras(modelo: Any, caras_uint8: np.ndarray, dispositivo: str, lote: int = 128) -> np.ndarray:
    """Embeddings normalizados L2 (el producto punto es la similitud coseno)."""
    import torch

    modelo.eval()
    salida = []
    with torch.no_grad():
        for inicio in range(0, len(caras_uint8), lote):
            x = (a_tensor(caras_uint8[inicio : inicio + lote], dispositivo) - MEDIA) / ESCALA
            e = modelo(x)
            salida.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(salida)


def ajustar(
    modelo: Any, caras_uint8: np.ndarray, etiquetas: np.ndarray, cfg: dict[str, Any], dispositivo: str
) -> list[dict[str, float]]:
    """Ajuste fino. `caras_uint8`/`etiquetas` son SOLO de las personas de entrenamiento. Devuelve el historial por época."""
    import torch

    torch.manual_seed(cfg["semilla"])
    rng = np.random.default_rng(cfg["semilla"])
    generador = torch.Generator(device=dispositivo).manual_seed(cfg["semilla"])
    modelo.to(dispositivo)
    entrenables = congelar_salvo(modelo, cfg["entrenables"])
    if not entrenables:
        raise ValueError("Ningún parámetro quedó entrenable: revisa `entrenables` en la configuración")
    modelo.eval()  # las capas congeladas (y sus BatchNorm) no deben cambiar sus estadísticas
    optimizador = torch.optim.Adam(
        [p for p in modelo.parameters() if p.requires_grad], lr=cfg["tasa_aprendizaje"]
    )
    historial = []
    for epoca in range(cfg["epocas"]):
        perdidas, activas = [], []
        for _ in range(cfg["pasos_por_epoca"]):
            idx = muestrear_lote(etiquetas, cfg["personas_por_lote"], cfg["fotos_por_persona"], rng)
            x = aumentar(a_tensor(caras_uint8[idx], dispositivo), generador)
            emb = torch.nn.functional.normalize(modelo((x - MEDIA) / ESCALA), dim=1)
            y = torch.from_numpy(etiquetas[idx]).to(dispositivo)
            perdida, fraccion = perdida_triplet_semi_hard(emb, y, cfg["margen"])
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            perdidas.append(float(perdida.detach()))
            activas.append(fraccion)
        historial.append(
            {
                "epoca": epoca + 1,
                "perdida": float(np.mean(perdidas)),
                "tripletas_activas": float(np.mean(activas)),
            }
        )
    return historial


def decidir_adopcion(base: dict[str, Any], ajustado: dict[str, Any], mejora_minima: float) -> dict[str, Any]:
    """Regla del proyecto: un modelo entra al servicio solo si SUPERA a la línea base (EER en personas no vistas)."""
    eer_base, eer_ajustado = base["eer_prueba"], ajustado["eer_prueba"]
    return {
        "eer_base": eer_base,
        "eer_ajustado": eer_ajustado,
        "mejora_absoluta": eer_base - eer_ajustado,
        "adoptar": bool(eer_base - eer_ajustado > mejora_minima),
    }


def comparar_con_semillas(
    semillas: Sequence[int],
    caras_uint8: np.ndarray,
    etiquetas: np.ndarray,
    emb_base: np.ndarray,
    cfg: dict[str, Any],
    crear_modelo: Any,
    dispositivo: str,
) -> list[dict[str, float]]:
    """Repite el experimento completo con distintas semillas: cada una cambia la partición de personas, los pares de
    prueba y el entrenamiento. Devuelve el EER en personas NO vistas de la base y del ajustado, por semilla.
    """
    import torch

    from ml.src import linea_base
    from ml.src.split import dividir_por_persona

    filas = []
    for semilla in semillas:
        c = {**cfg, "semilla": semilla}
        partes = dividir_por_persona(etiquetas, tuple(c["proporciones"]), semilla)
        eer_base = linea_base.verificacion(emb_base, etiquetas, c)["eer_prueba"]
        modelo = crear_modelo()
        ajustar(modelo, caras_uint8[partes["train"]], etiquetas[partes["train"]], c, dispositivo)
        eer_ajustado = linea_base.verificacion(embeber_caras(modelo, caras_uint8, dispositivo), etiquetas, c)[
            "eer_prueba"
        ]
        filas.append(
            {
                "semilla": semilla,
                "eer_base": eer_base,
                "eer_ajustado": eer_ajustado,
                "mejora": eer_base - eer_ajustado,
            }
        )
        del modelo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return filas
