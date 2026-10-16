// Detector de placas (YOLOv8n ONNX int8) para la PWA del guardia. Misma lógica que ml/src/onnx_placas.py.
// Sin dependencias de navegador: recibe píxeles RGBA (como los da un canvas) y una sesión de onnxruntime-web.
export const TAMANO = 640;
const RELLENO = 114;
export const IOU_NMS = 0.45;

// RGBA [alto*ancho*4] -> { tensor Float32 [1,3,640,640] en 0..1, escala, px, py }
export function letterbox(rgba, ancho, alto, tamano = TAMANO) {
  const escala = Math.min(tamano / alto, tamano / ancho);
  const nw = Math.max(1, Math.round(ancho * escala));
  const nh = Math.max(1, Math.round(alto * escala));
  const px = Math.floor((tamano - nw) / 2);
  const py = Math.floor((tamano - nh) / 2);
  const plano = tamano * tamano;
  const tensor = new Float32Array(3 * plano).fill(RELLENO / 255);
  for (let y = 0; y < nh; y++) {
    // bilineal con centros de píxel, igual que Pillow al reducir
    const sy = Math.min(alto - 1, Math.max(0, (y + 0.5) / escala - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(alto - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(ancho - 1, Math.max(0, (x + 0.5) / escala - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(ancho - 1, x0 + 1), fx = sx - x0;
      for (let c = 0; c < 3; c++) {
        const a = rgba[(y0 * ancho + x0) * 4 + c], b = rgba[(y0 * ancho + x1) * 4 + c];
        const d = rgba[(y1 * ancho + x0) * 4 + c], e = rgba[(y1 * ancho + x1) * 4 + c];
        const v = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
        tensor[c * plano + (py + y) * tamano + (px + x)] = v / 255;
      }
    }
  }
  return { tensor, escala, px, py };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

// salida cruda [1,5,N] (aplanada, orden fila: cx[N], cy[N], w[N], h[N], conf[N]) -> [{caja:[x1,y1,x2,y2], confianza}]
export function decodificar(salida, n, escala, px, py, ancho, alto, confMin = 0.25, iouNms = IOU_NMS) {
  const candidatas = [];
  for (let i = 0; i < n; i++) {
    const conf = salida[4 * n + i];
    if (conf < confMin) continue;
    const cx = salida[i], cy = salida[n + i], w = salida[2 * n + i], h = salida[3 * n + i];
    const caja = [(cx - w / 2 - px) / escala, (cy - h / 2 - py) / escala, (cx + w / 2 - px) / escala, (cy + h / 2 - py) / escala];
    caja[0] = Math.min(ancho, Math.max(0, caja[0])); caja[2] = Math.min(ancho, Math.max(0, caja[2]));
    caja[1] = Math.min(alto, Math.max(0, caja[1])); caja[3] = Math.min(alto, Math.max(0, caja[3]));
    candidatas.push({ caja, confianza: conf });
  }
  candidatas.sort((a, b) => b.confianza - a.confianza);
  const quedan = [];
  for (const c of candidatas) if (quedan.every((q) => iou(q.caja, c.caja) <= iouNms)) quedan.push(c);
  return quedan;
}

// sesion: InferenceSession de onnxruntime-web; ort: el módulo (para crear el Tensor)
export async function detectarPlacas(ort, sesion, rgba, ancho, alto, confMin = 0.25) {
  const { tensor, escala, px, py } = letterbox(rgba, ancho, alto);
  const resultado = await sesion.run({ images: new ort.Tensor('float32', tensor, [1, 3, TAMANO, TAMANO]) });
  const salida = resultado.output0;
  return decodificar(salida.data, salida.dims[2], escala, px, py, ancho, alto, confMin);
}
