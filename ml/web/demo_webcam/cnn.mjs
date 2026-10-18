// CNN de placas (Keras, notebook 12/13) ejecutada en JavaScript puro, sin TensorFlow.js:
// 3 × [Conv3x3 (BatchNorm plegada) + ReLU + MaxPool2] → Dense 256 + ReLU → Dense 252 → 7 × 36 con softmax por posición.
// Entrada: franja inferior de la placa en gris, 32 × 128, valores 0..1 (fila por fila). Pesos: `placas_cnn_pesos.bin/.json` (los exporta el notebook 13).
export const ALTO = 32;
export const ANCHO = 128;
export const ALFA = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

export async function cargarModelo(urlJson, urlBin) {
  const [manifiesto, bin] = await Promise.all([
    fetch(urlJson).then((r) => r.json()),
    fetch(urlBin).then((r) => r.arrayBuffer()),
  ]);
  const p = {};
  for (const { nombre, forma, offset } of manifiesto) {
    const n = forma.reduce((a, b) => a * b, 1);
    p[nombre] = { forma, datos: new Float32Array(bin, offset, n) };
  }
  return p;
}

// x: [h, w, cin] (HWC); W: [3, 3, cin, cout]; devuelve [h, w, cout] con ReLU. Relleno "same".
function conv3x3Relu(x, h, w, cin, W, b, cout) {
  const y = new Float32Array(h * w * cout);
  for (let i = 0; i < h * w; i++) y.set(b, i * cout);
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      const o = (yy * w + xx) * cout;
      for (let ky = 0; ky < 3; ky++) {
        const sy = yy + ky - 1;
        if (sy < 0 || sy >= h) continue;
        for (let kx = 0; kx < 3; kx++) {
          const sx = xx + kx - 1;
          if (sx < 0 || sx >= w) continue;
          const s = (sy * w + sx) * cin;
          const base = (ky * 3 + kx) * cin * cout;
          for (let ci = 0; ci < cin; ci++) {
            const v = x[s + ci];
            if (v === 0) continue;
            const wb = base + ci * cout;
            for (let co = 0; co < cout; co++) y[o + co] += v * W[wb + co];
          }
        }
      }
    }
  }
  for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
  return y;
}

function maxPool2(x, h, w, c) {
  const h2 = h >> 1, w2 = w >> 1;
  const y = new Float32Array(h2 * w2 * c);
  for (let i = 0; i < h2; i++) {
    for (let j = 0; j < w2; j++) {
      for (let k = 0; k < c; k++) {
        const a = x[((2 * i) * w + 2 * j) * c + k], b = x[((2 * i) * w + 2 * j + 1) * c + k];
        const d = x[((2 * i + 1) * w + 2 * j) * c + k], e = x[((2 * i + 1) * w + 2 * j + 1) * c + k];
        y[(i * w2 + j) * c + k] = Math.max(a, b, d, e);
      }
    }
  }
  return y;
}

function densa(x, W, b, entradas, salidas, relu) {
  const y = Float32Array.from(b);
  for (let i = 0; i < entradas; i++) {
    const v = x[i];
    if (v === 0) continue;
    const wb = i * salidas;
    for (let j = 0; j < salidas; j++) y[j] += v * W[wb + j];
  }
  if (relu) for (let j = 0; j < salidas; j++) if (y[j] < 0) y[j] = 0;
  return y;
}

// gris: Float32Array[32*128] en 0..1 → probabilidades Float32Array[7*36]
export function probabilidades(p, gris) {
  let x = gris, h = ALTO, w = ANCHO, c = 1;
  for (let i = 0; i < 3; i++) {
    const W = p[`conv${i}_w`], b = p[`conv${i}_b`];
    const cout = W.forma[3];
    x = conv3x3Relu(x, h, w, c, W.datos, b.datos, cout);
    x = maxPool2(x, h, w, cout);
    h >>= 1; w >>= 1; c = cout;
  }
  const d0 = p.dense0_w, d1 = p.dense1_w;
  x = densa(x, d0.datos, p.dense0_b.datos, d0.forma[0], d0.forma[1], true);
  x = densa(x, d1.datos, p.dense1_b.datos, d1.forma[0], d1.forma[1], false);
  const out = new Float32Array(7 * 36);
  for (let pos = 0; pos < 7; pos++) {
    let mx = -Infinity;
    for (let k = 0; k < 36; k++) mx = Math.max(mx, x[pos * 36 + k]);
    let suma = 0;
    for (let k = 0; k < 36; k++) { const e = Math.exp(x[pos * 36 + k] - mx); out[pos * 36 + k] = e; suma += e; }
    for (let k = 0; k < 36; k++) out[pos * 36 + k] /= suma;
  }
  return out;
}

// Formato A4: dígitos en las posiciones 0-3 y letras en 4-6. Confianza = producto de los máximos por posición.
export function decodificar(prob) {
  let texto = '', conf = 1;
  for (let pos = 0; pos < 7; pos++) {
    const desde = pos < 4 ? 0 : 10, hasta = pos < 4 ? 10 : 36;
    let suma = 0;
    for (let k = desde; k < hasta; k++) suma += prob[pos * 36 + k];
    let mejor = desde, mx = -1;
    for (let k = desde; k < hasta; k++) if (prob[pos * 36 + k] > mx) { mx = prob[pos * 36 + k]; mejor = k; }
    texto += ALFA[mejor];
    conf *= mx / suma;
  }
  return { texto, confianza: conf };
}

export function leerPlaca(p, gris) {
  return decodificar(probabilidades(p, gris));
}

export function distanciaEdicion(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}
