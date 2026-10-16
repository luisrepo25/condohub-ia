"""Pre y posproceso del detector de placas en ONNX (YOLOv8n) y medidas para comparar modelos.

Es la **referencia en Python** de lo que la PWA del guardia debe hacer en JavaScript (carpeta `ml/` de condohub-guardia):
1. `letterbox`: imagen RGB → tensor [1, 3, 640, 640] float32 en 0..1, redimensionada sin deformar y rellena con gris 114.
2. el modelo devuelve [1, 5, 8400]: por cada candidata (cx, cy, w, h, confianza) en píxeles del tensor de 640.
3. `decodificar`: descarta confianza baja, pasa a coordenadas de la imagen original y aplica NMS (IoU 0,45).
Solo usa numpy (y Pillow para redimensionar): se prueba en local; el modelo se ejecuta en Colab.
"""

import numpy as np

TAMANO = 640
RELLENO = 114
IOU_NMS = 0.45


def letterbox(imagen: np.ndarray, tamano: int = TAMANO) -> tuple[np.ndarray, float, int, int]:
    """RGB uint8 [h, w, 3] → (tensor float32 [1, 3, t, t] en 0..1, escala, relleno_x, relleno_y)."""
    from PIL import Image

    h, w = imagen.shape[:2]
    escala = min(tamano / h, tamano / w)
    nh, nw = max(1, round(h * escala)), max(1, round(w * escala))
    reducida = np.asarray(Image.fromarray(imagen).resize((nw, nh), Image.BILINEAR))
    lienzo = np.full((tamano, tamano, 3), RELLENO, dtype=np.uint8)
    px, py = (tamano - nw) // 2, (tamano - nh) // 2
    lienzo[py : py + nh, px : px + nw] = reducida
    tensor = lienzo.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, float(escala), px, py


def iou(caja: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    """IoU de una caja [4] (x1, y1, x2, y2) contra varias [n, 4]."""
    if len(cajas) == 0:
        return np.zeros(0)
    x1, y1 = np.maximum(caja[0], cajas[:, 0]), np.maximum(caja[1], cajas[:, 1])
    x2, y2 = np.minimum(caja[2], cajas[:, 2]), np.minimum(caja[3], cajas[:, 3])
    interseccion = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (caja[2] - caja[0]) * (caja[3] - caja[1])
    areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    union = area + areas - interseccion
    return np.where(union > 0, interseccion / np.where(union > 0, union, 1), 0.0)


def nms(cajas: np.ndarray, puntajes: np.ndarray, iou_max: float = IOU_NMS) -> list[int]:
    """Índices que se conservan: de cada grupo de cajas solapadas, la de mayor puntaje."""
    orden = list(np.argsort(-puntajes))
    conservadas: list[int] = []
    while orden:
        mejor = orden.pop(0)
        conservadas.append(int(mejor))
        if orden:
            solape = iou(cajas[mejor], cajas[orden])
            orden = [i for i, s in zip(orden, solape, strict=True) if s <= iou_max]
    return conservadas


def decodificar(
    salida: np.ndarray,
    escala: float,
    px: int,
    py: int,
    forma_original: tuple[int, int],
    conf_min: float = 0.25,
    iou_nms: float = IOU_NMS,
) -> np.ndarray:
    """Salida cruda del modelo → detecciones [n, 5] (x1, y1, x2, y2, confianza) en coordenadas de la imagen original."""
    p = np.squeeze(salida, axis=0) if salida.ndim == 3 else salida
    if p.ndim != 2 or p.shape[0] != 5:
        raise ValueError(f"Se esperaba una salida [1, 5, N] (una sola clase), llegó {salida.shape}")
    p = p.T
    p = p[p[:, 4] >= conf_min]
    if len(p) == 0:
        return np.zeros((0, 5))
    alto, ancho = forma_original
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    cajas = np.stack([(cx - w / 2 - px) / escala, (cy - h / 2 - py) / escala,
                      (cx + w / 2 - px) / escala, (cy + h / 2 - py) / escala], axis=1)
    cajas[:, [0, 2]] = np.clip(cajas[:, [0, 2]], 0, ancho)
    cajas[:, [1, 3]] = np.clip(cajas[:, [1, 3]], 0, alto)
    quedan = nms(cajas, p[:, 4], iou_nms)
    return np.column_stack([cajas[quedan], p[quedan, 4]])


def _emparejar(predichas: np.ndarray, referencia: np.ndarray, iou_min: float) -> list[int | None]:
    """Para cada predicción (en el orden dado) el índice de la referencia con la que empareja, o None. Cada referencia, una vez."""
    usadas: set[int] = set()
    resultado: list[int | None] = []
    for fila in predichas:
        asignada = None
        if len(referencia):
            solapes = iou(fila[:4], referencia[:, :4])
            j = int(np.argmax(solapes))
            if solapes[j] >= iou_min and j not in usadas:
                usadas.add(j)
                asignada = j
        resultado.append(asignada)
    return resultado


def ap50(predichas: list[np.ndarray], verdad: list[np.ndarray], iou_min: float = 0.5) -> float:
    """Precisión promedio a IoU 0,5 (interpolación de todos los puntos, una sola clase)."""
    total = sum(len(v) for v in verdad)
    if total == 0:
        raise ValueError("No hay placas de referencia")
    registros: list[tuple[float, int]] = []
    for pred, gt in zip(predichas, verdad, strict=True):
        pred = pred[np.argsort(-pred[:, 4])] if len(pred) else pred
        registros += [(float(f[4]), int(a is not None)) for f, a in zip(pred, _emparejar(pred, gt, iou_min), strict=True)]
    if not registros:
        return 0.0
    registros.sort(key=lambda r: -r[0])
    acierto = np.array([r[1] for r in registros])
    tp, fp = np.cumsum(acierto), np.cumsum(1 - acierto)
    recall, precision = tp / total, tp / (tp + fp)
    mrec, mpre = np.concatenate([[0.0], recall, [1.0]]), np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    cambios = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[cambios + 1] - mrec[cambios]) * mpre[cambios + 1]))


def precision_recall(
    predichas: list[np.ndarray], verdad: list[np.ndarray], conf_min: float, iou_min: float = 0.5
) -> dict[str, float | int | None]:
    """Precisión y recall en un punto de operación (solo detecciones con confianza ≥ `conf_min`)."""
    tp = fp = fn = 0
    for pred, gt in zip(predichas, verdad, strict=True):
        pred = pred[pred[:, 4] >= conf_min] if len(pred) else pred
        pred = pred[np.argsort(-pred[:, 4])] if len(pred) else pred
        asignadas = _emparejar(pred, gt, iou_min)
        tp += sum(a is not None for a in asignadas)
        fp += sum(a is None for a in asignadas)
        fn += len(gt) - sum(a is not None for a in asignadas)
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
    }


def coincidencia(
    a: list[np.ndarray], b: list[np.ndarray], conf_min: float = 0.5, iou_min: float = 0.5
) -> dict[str, float | int | None]:
    """Cuánto coinciden las detecciones de dos modelos (p. ej. PyTorch frente a ONNX int8) con confianza ≥ `conf_min`."""
    comunes = solo_a = solo_b = 0
    solapes: list[float] = []
    for da, db in zip(a, b, strict=True):
        da = da[da[:, 4] >= conf_min] if len(da) else da
        db = db[db[:, 4] >= conf_min] if len(db) else db
        da = da[np.argsort(-da[:, 4])] if len(da) else da
        asignadas = _emparejar(da, db, iou_min)
        comunes += sum(x is not None for x in asignadas)
        solo_a += sum(x is None for x in asignadas)
        solo_b += len(db) - sum(x is not None for x in asignadas)
        solapes += [float(iou(f[:4], db[:, :4])[j]) for f, j in zip(da, asignadas, strict=True) if j is not None]
    return {
        "coinciden": comunes, "solo_el_primero": solo_a, "solo_el_segundo": solo_b,
        "iou_medio": float(np.mean(solapes)) if solapes else None,
    }
