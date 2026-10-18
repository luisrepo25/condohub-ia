// Demo de portería con la webcam: detector de placas (YOLOv8n ONNX) + lectura con la CNN (JavaScript puro) + regla A4.
// Solo es una prueba local del modelo: no llama a ningún servicio ni registra accesos reales.
import * as ort from '/ml/web/node_modules/onnxruntime-web/dist/ort.min.mjs';
import { detectarPlacas } from '/ml/web/placas.mjs';
import { cargarModelo, leerPlaca, distanciaEdicion, ALTO, ANCHO } from './cnn.mjs';

const UMBRAL = 0.85;            // A4
const MARGEN = 0.15;            // el mismo margen con el que se recortaron las placas de entrenamiento
const DESDE = [0.35, 0.4, 0.45]; // franjas inferiores del recorte que se leen (se queda la de mayor confianza)
const VENTANA = 10;             // lecturas recientes que se consideran
const CUADROS_MINIMOS = 2;      // cuadros con confianza ≥ UMBRAL y el mismo texto para aceptar una placa (un cuadro suelto no basta)

const $ = (id) => document.getElementById(id);
const video = $('video'), cajaCanvas = $('caja'), ctxCaja = cajaCanvas.getContext('2d');
const lienzo = document.createElement('canvas'), ctx = lienzo.getContext('2d', { willReadFrequently: true });
const franja = document.createElement('canvas'); franja.width = ANCHO; franja.height = ALTO;
const ctxFranja = franja.getContext('2d', { willReadFrequently: true });
ctxFranja.imageSmoothingEnabled = true; ctxFranja.imageSmoothingQuality = 'high';

let detector, cnn, reciente = [], decisionTomada = null, cuadrosNegros = 0, abrirCamara = null;

function placasRegistradas() {
  return $('registradas').value.toUpperCase().split(/[\s,;]+/).filter((p) => /^\d{4}[A-Z]{3}$/.test(p));
}

function leerFranja(caja) {
  const [x1, y1, x2, y2] = caja, w = x2 - x1, h = y2 - y1;
  const sx = Math.max(0, x1 - MARGEN * w), sy = Math.max(0, y1 - MARGEN * h);
  const cw = Math.min(lienzo.width, x2 + MARGEN * w) - sx, ch = Math.min(lienzo.height, y2 + MARGEN * h) - sy;
  let mejor = null;
  for (const d of DESDE) {
    ctxFranja.drawImage(lienzo, sx, sy + d * ch, cw, ch * (1 - d), 0, 0, ANCHO, ALTO);
    const px = ctxFranja.getImageData(0, 0, ANCHO, ALTO).data;
    const gris = new Float32Array(ANCHO * ALTO);
    for (let i = 0; i < gris.length; i++) gris[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
    const r = leerPlaca(cnn, gris);
    if (!mejor || r.confianza > mejor.confianza) mejor = r;
  }
  return mejor;
}

function evaluar() {
  const reg = placasRegistradas();
  const seguras = reciente.filter((r) => r.confianza >= UMBRAL);
  const cuenta = {};
  for (const r of seguras) cuenta[r.texto] = (cuenta[r.texto] || 0) + 1;
  const mejores = seguras.filter((r) => cuenta[r.texto] >= CUADROS_MINIMOS);
  const exacta = mejores.find((r) => reg.includes(r.texto));
  if (exacta) return { nivel: 'ok', estado: 'PERMITIDO', texto: exacta.texto, conf: exacta.confianza, detalle: 'Placa registrada, confianza ≥ 0,85 (A4).' };
  const alta = mejores[0];
  if (alta) return { nivel: 'aviso', estado: 'NO REGISTRADA', texto: alta.texto, conf: alta.confianza, detalle: 'Lectura segura, pero la placa no está registrada: decide el guardia.' };
  let cerca = null;
  for (const r of reciente) for (const p of reg) {
    const d = distanciaEdicion(r.texto, p);
    if (d <= 1 && (!cerca || r.confianza > cerca.conf)) cerca = { texto: p, leida: r.texto, conf: r.confianza };
  }
  if (cerca) return { nivel: 'aviso', estado: 'SUGERIDA: DECIDE EL GUARDIA', texto: cerca.texto, conf: cerca.conf, detalle: `Se leyó ${cerca.leida} (confianza baja); está a 1 carácter de una placa registrada.` };
  const ult = reciente[reciente.length - 1];
  return ult ? { nivel: 'mal', estado: 'SIN COINCIDENCIA', texto: ult.texto, conf: ult.confianza, detalle: 'Confianza baja y sin placa registrada parecida: decide el guardia.' } : null;
}

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

function historial(texto, conf, detConf) {
  const fila = document.createElement('tr');
  fila.innerHTML = `<td>${texto}</td><td>${conf.toFixed(2)}</td><td>${detConf.toFixed(2)}</td>`;
  const t = $('historial'); t.prepend(fila); while (t.children.length > 8) t.lastChild.remove();
}

function registrar(accion) {
  const ev = evaluar(); if (!ev) return;
  decisionTomada = accion;
  $('registro').textContent = `Guardia: ${accion} ${ev.texto} a las ${new Date().toLocaleTimeString()} (solo demostración, sin registro real).`;
  $('confirmar').disabled = true; $('rechazar').disabled = true;
  setTimeout(() => { decisionTomada = null; }, 4000);
}
$('confirmar').onclick = () => registrar('CONFIRMÓ');
$('rechazar').onclick = () => registrar('RECHAZÓ');

async function ciclo() {
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
      const r = leerFranja(det.caja);
      reciente.push(r); if (reciente.length > VENTANA) reciente.shift();
      historial(r.texto, r.confianza, det.confianza);
    } else if (reciente.length) {
      reciente.shift();    // sin placa a la vista, las lecturas viejas van saliendo
    }
    const ev = evaluar();
    pintar(det, ev);
    if (!ev) { $('estado').textContent = 'Buscando placa…'; $('estado').className = 'estado'; $('placa').textContent = ' '; $('detalle').textContent = ''; }
    $('carga').textContent = `${Math.round(performance.now() - t0)} ms por cuadro`;
  }
  setTimeout(ciclo, 50);
}

async function iniciar() {
  ort.env.wasm.wasmPaths = '/ml/web/node_modules/onnxruntime-web/dist/';
  ort.env.wasm.numThreads = 1;   // sin SharedArrayBuffer: no hace falta COOP/COEP
  detector = await ort.InferenceSession.create('/modelos/yolov8n_placas_int8.onnx', { executionProviders: ['wasm'] });
  cnn = await cargarModelo('/modelos/placas_cnn_pesos.json', '/modelos/placas_cnn_pesos.bin');
  // Primer permiso (sin elegir cámara) para que el navegador revele los nombres de los dispositivos.
  (await navigator.mediaDevices.getUserMedia({ video: true, audio: false })).getTracks().forEach((t) => t.stop());
  const camaras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const sel = $('camara');
  camaras.forEach((c, i) => sel.append(new Option(c.label || `Cámara ${i + 1}`, c.deviceId)));
  const virtual = /(ir|infrared|infrarroj|nvidia|broadcast|virtual|obs|droidcam|snap)/i;   // IR de Windows Hello y cámaras virtuales dan negro
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
