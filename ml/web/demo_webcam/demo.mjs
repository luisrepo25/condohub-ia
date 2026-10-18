// Demo de portería con la webcam: detector de placas (YOLOv8n ONNX) + DOS lectores: el lector global preentrenado (cualquier formato de
// placa; es el principal) y la CNN de TensorFlow (JavaScript puro, solo fiable con la placa de la demo) + regla A4.
// Solo es una prueba local del modelo: no llama a ningún servicio ni registra accesos reales.
import * as ort from '/ml/web/node_modules/onnxruntime-web/dist/ort.min.mjs';
import { detectarPlacas } from '/ml/web/placas.mjs';
import { cargarModelo, leerPlaca, ALTO, ANCHO } from './cnn.mjs';
import { cargarOcrGlobal, leerOcrGlobal, ANCHO_OCR, ALTO_OCR } from './ocr_global.mjs';
import { evaluar as decidir, fmt } from './decision.mjs';

const MARGEN = 0.15;            // margen alrededor de la caja del detector
const DESDE = [0.35, 0.4, 0.45]; // franjas inferiores del recorte que lee la CNN (se queda la de mayor confianza)
const VENTANA = 10;             // lecturas recientes que se consideran

const $ = (id) => document.getElementById(id);
const video = $('video'), cajaCanvas = $('caja'), ctxCaja = cajaCanvas.getContext('2d');
const lienzo = document.createElement('canvas'), ctx = lienzo.getContext('2d', { willReadFrequently: true });
const franja = document.createElement('canvas'); franja.width = ANCHO; franja.height = ALTO;
const ctxFranja = franja.getContext('2d', { willReadFrequently: true });
ctxFranja.imageSmoothingEnabled = true; ctxFranja.imageSmoothingQuality = 'high';
const recorteOcr = document.createElement('canvas'); recorteOcr.width = ANCHO_OCR; recorteOcr.height = ALTO_OCR;
const ctxOcr = recorteOcr.getContext('2d', { willReadFrequently: true });
ctxOcr.imageSmoothingEnabled = true; ctxOcr.imageSmoothingQuality = 'high';

let detector, cnn, ocrGlobal, reciente = [], decisionTomada = null, cuadrosNegros = 0, abrirCamara = null;

// La lista se separa por coma, punto y coma o salto de línea; cada placa se normaliza (mayúsculas, sin espacios ni guiones) y debe tener
// de 4 a 10 letras o dígitos (el lector global lee placas de cualquier país). Lo que no cumple se avisa en pantalla, no se ignora en silencio.
const CLAVE_LISTA = 'placasRegistradas';
function analizarLista() {
  const validas = [], invalidas = [];
  for (const crudo of $('registradas').value.split(/[,;\n]+/)) {
    const p = crudo.toUpperCase().replace(/[^A-Z0-9]/g, '');
    if (!p) continue;
    if (/^[A-Z0-9]{4,10}$/.test(p)) { if (!validas.includes(p)) validas.push(p); } else invalidas.push(crudo.trim());
  }
  return { validas, invalidas };
}
function placasRegistradas() {
  return analizarLista().validas;
}
function actualizarLista() {
  const { validas, invalidas } = analizarLista();
  $('estadoRegistradas').textContent = `${validas.length} placa(s) válida(s): ${validas.join(', ') || 'ninguna'}` +
    (invalidas.length ? ` · IGNORADAS por no tener entre 4 y 10 letras o dígitos: ${invalidas.join(', ')}` : '');
  try { localStorage.setItem(CLAVE_LISTA, $('registradas').value); } catch { /* sin almacenamiento: la lista no se recuerda */ }
}
try { const guardada = localStorage.getItem(CLAVE_LISTA); if (guardada) $('registradas').value = guardada; } catch { /* idem */ }
$('registradas').addEventListener('input', actualizarLista);
actualizarLista();

function regionPlaca(caja) {
  const [x1, y1, x2, y2] = caja, w = x2 - x1, h = y2 - y1;
  const sx = Math.max(0, x1 - MARGEN * w), sy = Math.max(0, y1 - MARGEN * h);
  return { sx, sy, cw: Math.min(lienzo.width, x2 + MARGEN * w) - sx, ch: Math.min(lienzo.height, y2 + MARGEN * h) - sy };
}

function leerConCnn(caja) {
  const { sx, sy, cw, ch } = regionPlaca(caja);
  let mejor = null;
  for (const d of DESDE) {
    ctxFranja.drawImage(lienzo, sx, sy + d * ch, cw, ch * (1 - d), 0, 0, ANCHO, ALTO);
    const px = ctxFranja.getImageData(0, 0, ANCHO, ALTO).data;
    const gris = new Float32Array(ANCHO * ALTO);
    for (let i = 0; i < gris.length; i++) gris[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
    const r = leerPlaca(cnn, gris);
    if (!mejor || r.confianza > mejor.confianza) mejor = r;
  }
  return { texto: mejor.texto, confianza: mejor.confianza };
}

async function leerConOcr(caja) {
  const { sx, sy, cw, ch } = regionPlaca(caja);
  ctxOcr.drawImage(lienzo, sx, sy, cw, ch, 0, 0, ANCHO_OCR, ALTO_OCR);          // el recorte completo, deformado a 128 × 64 como pide el modelo
  const r = await leerOcrGlobal(ort, ocrGlobal, ctxOcr.getImageData(0, 0, ANCHO_OCR, ALTO_OCR).data);
  return { placa: r.texto.length >= 4 ? r.texto : null, confianza: r.confianza, bruto: r.texto };
}

const evaluar = () => decidir(reciente, placasRegistradas());

function pintar(det, ev) {
  cajaCanvas.width = video.videoWidth; cajaCanvas.height = video.videoHeight;
  ctxCaja.clearRect(0, 0, cajaCanvas.width, cajaCanvas.height);
  if (det) {
    ctxCaja.strokeStyle = '#4c9ffe'; ctxCaja.lineWidth = Math.max(3, video.videoWidth / 300);
    ctxCaja.strokeRect(det.caja[0], det.caja[1], det.caja[2] - det.caja[0], det.caja[3] - det.caja[1]);
  }
  if (!ev) return;
  const e = $('estado'); e.textContent = ev.estado; e.className = `estado ${ev.nivel}`;
  $('placa').textContent = ev.texto; $('detalle').textContent = `${ev.detalle} (confianza ${ev.conf.toFixed(2)})`;
  const hay = ev.nivel !== 'ok';
  $('confirmar').disabled = !hay || decisionTomada; $('rechazar').disabled = decisionTomada !== null;
}

function historial(r, detConf) {
  const fila = document.createElement('tr');
  fila.innerHTML = `<td>${r.cnn.texto} (${fmt(r.cnn.confianza)})</td><td>${r.ocr.bruto || '—'} (${fmt(r.ocr.confianza)})</td><td>${detConf.toFixed(2)}</td>`;
  const t = $('historial'); t.prepend(fila); while (t.children.length > 8) t.lastChild.remove();
}

function registrar(accion) {
  const ev = evaluar(); if (!ev) return;
  decisionTomada = accion;
  $('registro').textContent = `Guardia: ${accion} ${ev.texto} a las ${new Date().toLocaleTimeString()} (solo demostración, sin registro real).`;
  $('confirmar').disabled = true; $('rechazar').disabled = true;
  setTimeout(() => { decisionTomada = null; }, 4000);
}
$('guardar').onclick = () => lienzo.toBlob((b) => {            // el cuadro completo de la cámara, tal como lo ve la página
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = `cuadro_${new Date().toISOString().replace(/[:.]/g, '-')}.png`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}, 'image/png');
$('confirmar').onclick = () => registrar('CONFIRMÓ');
$('rechazar').onclick = () => registrar('RECHAZÓ');

async function ciclo() {
  try {
    if (video.videoWidth) {
      lienzo.width = video.videoWidth; lienzo.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const rgba = ctx.getImageData(0, 0, lienzo.width, lienzo.height).data;
      // Imagen negra (p. ej. la cámara infrarroja de Windows Hello): tras ~20 cuadros se prueba la siguiente cámara.
      let suma = 0; const paso = Math.max(1, Math.floor(rgba.length / 4 / 2000));
      for (let i = 0; i < rgba.length / 4; i += paso) suma += rgba[4 * i] + rgba[4 * i + 1] + rgba[4 * i + 2];
      if (suma / (rgba.length / 4 / paso) / 3 < 4) {
        if (++cuadrosNegros >= 20 && abrirCamara) { cuadrosNegros = 0; await abrirCamara(true); }
        $('estado').textContent = 'Imagen negra: probando otra cámara…'; $('estado').className = 'estado aviso';
        setTimeout(ciclo, 100); return;
      }
      cuadrosNegros = 0;
      const t0 = performance.now();
      const dets = await detectarPlacas(ort, detector, rgba, lienzo.width, lienzo.height, 0.4);
      const det = dets[0];
      if (det) {
        const r = { cnn: leerConCnn(det.caja), ocr: await leerConOcr(det.caja) };
        reciente.push(r); if (reciente.length > VENTANA) reciente.shift();
        historial(r, det.confianza);
      } else if (reciente.length) {
        reciente.shift();    // sin placa a la vista, las lecturas viejas van saliendo
      }
      const ev = evaluar();
      pintar(det, ev);
      if (!ev) { $('estado').textContent = 'Buscando placa…'; $('estado').className = 'estado'; $('placa').textContent = ' '; $('detalle').textContent = ''; }
      $('carga').textContent = `${Math.round(performance.now() - t0)} ms por cuadro`;
    }
  } catch (e) {                         // un error en un cuadro no debe congelar la página: se muestra y se sigue
    $('carga').textContent = 'Error en el cuadro: ' + e.message; console.error(e);
  }
  setTimeout(ciclo, 50);
}

async function iniciar() {
  ort.env.wasm.wasmPaths = '/ml/web/node_modules/onnxruntime-web/dist/';
  // Varios hilos de WebAssembly solo si la página está aislada (servir.py manda COOP/COEP); si no, un hilo.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  $('registro').textContent = `Hilos de cálculo: ${ort.env.wasm.numThreads}`;
  detector = await ort.InferenceSession.create('/modelos/yolov8n_placas_int8.onnx', { executionProviders: ['wasm'] });
  cnn = await cargarModelo('/modelos/placas_cnn_pesos.json', '/modelos/placas_cnn_pesos.bin');
  ocrGlobal = await cargarOcrGlobal(ort, '/modelos/cct_xs_v2_global.onnx');
  // Primer permiso (sin elegir cámara) para que el navegador revele los nombres de los dispositivos.
  (await navigator.mediaDevices.getUserMedia({ video: true, audio: false })).getTracks().forEach((t) => t.stop());
  const camaras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const sel = $('camara');
  camaras.forEach((c, i) => sel.append(new Option(c.label || `Cámara ${i + 1}`, c.deviceId)));
  const virtual = /(ir|infrared|infrarroj|nvidia|broadcast|virtual|obs|droidcam|snap)/i;   // IR de Windows Hello y cámaras virtuales dan negro
  const preferida = camaras.find((c) => !virtual.test(c.label)) || camaras[0];
  if (preferida) sel.value = preferida.deviceId;
  async function abrir(siguiente = false) {
    if (siguiente) sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
    if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
    reciente = [];
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: sel.value }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    await video.play();
    $('registro').textContent = 'Cámara en uso: ' + sel.options[sel.selectedIndex].text;
  }
  abrirCamara = abrir;
  sel.onchange = () => abrir().catch((e) => { $('carga').textContent = 'Error de cámara: ' + e.message; });
  await abrir();
  ciclo();
}
iniciar().catch((e) => { $('carga').textContent = 'Error: ' + e.message; console.error(e); });
