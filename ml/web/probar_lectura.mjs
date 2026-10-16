// Prueba en Node (WASM, el mismo motor que el navegador) de la lectura de placas: reconocedor ONNX + votación entre franjas.
// Compara con la referencia de Python (ml/src/lectura_placas.py) lectura por lectura.
// Uso: node probar_lectura.mjs <reconocedor.onnx> <ficha_reconocedor.json> <referencia.json> <carpeta_de_recortes>
//   referencia.json: lista de { archivo, real, lecturas: [[texto, confianza], ...], placa, confianza } generada con el código de Python.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { PNG } from 'pngjs';
import { leerPlaca } from './lectura_placas.mjs';

const [modelo, ficha, referencia, carpeta] = process.argv.slice(2);
if (!modelo || !ficha || !referencia || !carpeta) {
  console.error('Uso: node probar_lectura.mjs <reconocedor.onnx> <ficha_reconocedor.json> <referencia.json> <carpeta_de_recortes>');
  process.exit(2);
}
ort.env.wasm.numThreads = 1;
const { caracteres } = JSON.parse(readFileSync(ficha, 'utf8'));
const esperado = JSON.parse(readFileSync(referencia, 'utf8'));
const sesion = await ort.InferenceSession.create(readFileSync(modelo), { executionProviders: ['wasm'] });

let mismoTexto = 0;
let lecturasTotales = 0;
let mismaPlaca = 0;
let exactas = 0;
let equivocadas = 0;
let mayorDifConfianza = 0;
const tiempos = [];
for (const e of esperado) {
  const img = PNG.sync.read(readFileSync(join(carpeta, e.archivo)));
  const t0 = performance.now();
  const r = await leerPlaca(ort, sesion, img.data, img.width, img.height, caracteres);
  tiempos.push(performance.now() - t0);
  r.lecturas.forEach(([texto, conf], i) => {
    lecturasTotales++;
    if (texto === e.lecturas[i][0]) mismoTexto++;
    mayorDifConfianza = Math.max(mayorDifConfianza, Math.abs(conf - e.lecturas[i][1]));
  });
  if (r.placa === e.placa) mismaPlaca++;
  if (r.placa === e.real) exactas++;
  else if (r.placa) equivocadas++;
}
tiempos.sort((a, b) => a - b);
console.log(`mismo texto que Python en ${mismoTexto} de ${lecturasTotales} lecturas (mayor diferencia de confianza ${mayorDifConfianza.toFixed(4)})`);
console.log(`misma placa elegida que Python: ${mismaPlaca} de ${esperado.length}`);
console.log(`JS: exactas ${exactas}/${esperado.length} | equivocadas aceptadas ${equivocadas} | rechazadas ${esperado.length - exactas - equivocadas}`);
console.log(`tiempo por placa (6 franjas, con pre y posproceso): mediana ${tiempos[Math.floor(tiempos.length / 2)].toFixed(0)} ms`);
process.exit(equivocadas === 0 && mismaPlaca === esperado.length ? 0 : 1);
