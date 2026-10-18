// Compara la CNN en JavaScript con las salidas de referencia de Keras (paridad.json, lo exporta el notebook 13).
// Uso: node probar_cnn.mjs <pesos.json> <pesos.bin> <paridad.json>
import { readFileSync } from 'node:fs';
import { probabilidades } from './cnn.mjs';

const [, , rutaJson, rutaBin, rutaParidad] = process.argv;
const manifiesto = JSON.parse(readFileSync(rutaJson, 'utf8'));
const buf = readFileSync(rutaBin);
const bin = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const p = {};
for (const { nombre, forma, offset } of manifiesto) p[nombre] = { forma, datos: new Float32Array(bin, offset, forma.reduce((a, b) => a * b, 1)) };

const { entradas, salidas } = JSON.parse(readFileSync(rutaParidad, 'utf8'));
let peor = 0;
entradas.forEach((e, i) => {
  const out = probabilidades(p, Float32Array.from(e));
  let d = 0;
  for (let k = 0; k < out.length; k++) d = Math.max(d, Math.abs(out[k] - salidas[i][k]));
  peor = Math.max(peor, d);
});
console.log(`${entradas.length} entradas | diferencia máxima de probabilidad: ${peor.toExponential(2)}`);
process.exit(peor < 1e-4 ? 0 : 1);
