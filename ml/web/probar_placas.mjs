// Prueba en Node (WASM, el mismo motor que el navegador) del detector de placas ONNX.
// Uso: node probar_placas.mjs <modelo.onnx> <foto.jpg> [repeticiones]
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import jpeg from 'jpeg-js';
import { detectarPlacas } from './placas.mjs';

const [modelo, foto, reps = '5'] = process.argv.slice(2);
if (!modelo || !foto) {
  console.error('Uso: node probar_placas.mjs <modelo.onnx> <foto.jpg> [repeticiones]');
  process.exit(2);
}
ort.env.wasm.numThreads = 1;
const imagen = jpeg.decode(readFileSync(foto), { useTArray: true });
const t0 = performance.now();
const sesion = await ort.InferenceSession.create(readFileSync(modelo), { executionProviders: ['wasm'] });
console.log(`modelo cargado en ${(performance.now() - t0).toFixed(0)} ms | imagen ${imagen.width}x${imagen.height}`);

let detecciones = [];
const tiempos = [];
for (let i = 0; i < Number(reps); i++) {
  const t = performance.now();
  detecciones = await detectarPlacas(ort, sesion, imagen.data, imagen.width, imagen.height, 0.25);
  tiempos.push(performance.now() - t);
}
tiempos.sort((a, b) => a - b);
console.log(`inferencia (con pre y posproceso): mediana ${tiempos[Math.floor(tiempos.length / 2)].toFixed(0)} ms, mejor ${tiempos[0].toFixed(0)} ms`);
console.log(JSON.stringify(detecciones.map((d) => ({ caja: d.caja.map((v) => Math.round(v)), confianza: Number(d.confianza.toFixed(3)) }))));
