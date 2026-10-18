// Decisión de portería (A4) a partir de las lecturas recientes de los dos lectores. Lógica pura: sin DOM, se prueba en Node.
// Cada elemento de `reciente` es un cuadro: { cnn: { texto, confianza }, ocr: { placa | null, confianza } }.
//   - ocr: lector global (ocr_global.mjs). Lee placas de cualquier formato; su confianza NO distingue aciertos de errores (los errores son de
//     1 carácter y llegan con 0,98–1,00), así que la seguridad sale de comparar con la lista de placas REGISTRADAS y de exigir varios cuadros.
//   - cnn: red propia de TensorFlow, solo fiable con la placa de la demo (memoriza): cuenta únicamente si es exacta y segura.
import { distanciaEdicion } from './cnn.mjs';

export const PARAMETROS = {
  umbral: 0.85,           // A4: confianza de un cuadro para permitir
  cuadrosMinimos: 2,      // cuadros coincidentes (misma placa registrada) con confianza ≥ umbral
  ventanaConsenso: 8,     // últimos cuadros que se miran para el consenso
  cuadrosConsenso: 3,     // consenso (propuesta, 18/10/2026): la misma placa registrada en al menos 3 de los últimos 8 cuadros
};

export const fmt = (x) => x.toFixed(2).replace('.', ',');
const cuenta = (lista) => lista.reduce((m, t) => m.set(t, (m.get(t) || 0) + 1), new Map());

export function evaluar(reciente, reg, p = PARAMETROS) {
  const cnnSeguras = cuenta(reciente.filter((r) => r.cnn.confianza >= p.umbral).map((r) => r.cnn.texto));
  const ocrLeidas = cuenta(reciente.filter((r) => r.ocr.placa).map((r) => r.ocr.placa));
  const ocrSeguras = cuenta(reciente.filter((r) => r.ocr.placa && r.ocr.confianza >= p.umbral).map((r) => r.ocr.placa));
  const ventana = cuenta(reciente.slice(-p.ventanaConsenso).filter((r) => r.ocr.placa).map((r) => r.ocr.placa));
  const confOcr = (placa) => Math.max(...reciente.filter((r) => r.ocr.placa === placa).map((r) => r.ocr.confianza));
  const confCnn = (placa) => Math.max(...reciente.filter((r) => r.cnn.texto === placa).map((r) => r.cnn.confianza));
  const dos = (m, placa) => (m.get(placa) || 0) >= p.cuadrosMinimos;

  for (const placa of reg) {
    if (dos(cnnSeguras, placa)) return { nivel: 'ok', estado: 'PERMITIDO', texto: placa, conf: confCnn(placa), detalle: 'Placa registrada leída por la CNN, confianza ≥ 0,85 en 2 cuadros (A4).' };
    if (dos(ocrSeguras, placa)) return { nivel: 'ok', estado: 'PERMITIDO', texto: placa, conf: confOcr(placa), detalle: 'Placa registrada leída por el OCR, confianza ≥ 0,85 en 2 cuadros (A4).' };
  }
  for (const placa of reg) {
    const n = ventana.get(placa) || 0;
    if (n >= p.cuadrosConsenso) {
      return { nivel: 'ok', estado: 'PERMITIDO', texto: placa, conf: confOcr(placa), detalle: `Consenso del OCR: la misma placa registrada en ${n} de los últimos ${p.ventanaConsenso} cuadros (propuesta de A4, pendiente de confirmar con el equipo).` };
    }
  }
  for (const placa of reg) {
    if ((ocrLeidas.get(placa) || 0) >= 1) return { nivel: 'aviso', estado: 'SUGERIDA: DECIDE EL GUARDIA', texto: placa, conf: confOcr(placa), detalle: `El OCR leyó exactamente una placa registrada, pero con poca evidencia (menos de ${p.cuadrosMinimos} cuadros coincidentes): decide el guardia.` };
  }
  let cerca = null;
  for (const r of reciente) {
    if (!r.ocr.placa) continue;
    for (const placa of reg) {
      if (distanciaEdicion(r.ocr.placa, placa) === 1 && (!cerca || r.ocr.confianza > cerca.conf)) cerca = { texto: placa, leida: r.ocr.placa, conf: r.ocr.confianza };
    }
  }
  if (cerca) return { nivel: 'aviso', estado: 'SUGERIDA: DECIDE EL GUARDIA', texto: cerca.texto, conf: cerca.conf, detalle: `El OCR leyó ${cerca.leida}, a 1 carácter de una placa registrada: decide el guardia.` };
  for (const [placa] of ocrLeidas) {
    if (!reg.includes(placa) && dos(ocrLeidas, placa)) return { nivel: 'aviso', estado: 'NO REGISTRADA', texto: placa, conf: confOcr(placa), detalle: 'Placa leída por el OCR que no está en la lista de registradas: decide el guardia.' };
  }
  const ult = reciente[reciente.length - 1];
  const conLectura = reciente.slice(-p.ventanaConsenso).filter((r) => r.ocr.placa).length;
  const progreso = `Cuadros con lectura del OCR: ${conLectura} de los últimos ${Math.min(reciente.length, p.ventanaConsenso)}. Mantén la placa quieta y de frente unos segundos.`;
  return ult ? { nivel: 'mal', estado: 'SIN COINCIDENCIA', texto: ult.ocr.placa || '—', conf: ult.ocr.confianza || 0, detalle: `Ningún lector consigue una lectura segura de una placa registrada: decide el guardia. ${progreso}` } : null;
}
