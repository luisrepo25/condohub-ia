// Lectura de la placa en el dispositivo: reconocedor de EasyOCR en ONNX + votación entre franjas (A4).
// Misma lógica que ml/src/lectura_placas.py y ml/src/placas.py. Sin dependencias de navegador: recibe píxeles RGBA (como los da un canvas),
// una sesión de onnxruntime-web y la ficha del modelo (ficha_reconocedor.json: caracteres y alto de entrada).
export const FRANJAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5]; // desde qué fracción de la altura del recorte se lee hasta abajo
export const VOTOS_MINIMOS = 3;
const MAX_CORRECCIONES = 2;
const PENALIZACION = 0.9;
const ALTO = 64;
const PERMITIDOS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const A_DIGITO = { O: '0', Q: '0', D: '0', I: '1', L: '1', Z: '2', S: '5', G: '6', B: '8' };
const A_LETRA = { 0: 'O', 1: 'I', 2: 'Z', 5: 'S', 6: 'G', 8: 'B' };

export const normalizar = (texto) => texto.toUpperCase().replace(/[^A-Z0-9]/g, '');
export const esPlacaValida = (texto) => /^[0-9]{4}[A-Z]{3}$/.test(normalizar(texto));

export function corregirPorPosicion(texto) {
  const t = normalizar(texto);
  if (t.length !== 7) return t;
  const digitos = [...t.slice(0, 4)].map((c) => A_DIGITO[c] ?? c).join('');
  const letras = [...t.slice(4)].map((c) => A_LETRA[c] ?? c).join('');
  return digitos + letras;
}

// Placas válidas en cualquier ventana de 7 caracteres, con la corrección letra/dígito por posición. -> Map(placa -> confianza)
export function candidatas(texto, confianza) {
  const t = normalizar(texto);
  const resultado = new Map();
  for (let i = 0; i + 7 <= t.length; i++) {
    const ventana = t.slice(i, i + 7);
    const corregida = corregirPorPosicion(ventana);
    const cambios = [...ventana].filter((c, k) => c !== corregida[k]).length;
    if (cambios <= MAX_CORRECCIONES && esPlacaValida(corregida)) {
      resultado.set(corregida, Math.max(resultado.get(corregida) ?? 0, confianza * PENALIZACION ** cambios));
    }
  }
  return resultado;
}

// lecturas: [[texto, confianza], ...] una por franja. Una placa se acepta solo si aparece en al menos `votosMinimos` franjas.
export function elegirPorVotos(lecturas, votosMinimos = VOTOS_MINIMOS) {
  if (votosMinimos < 1) throw new Error('Se necesita al menos un voto');
  const votos = new Map();
  const confianza = new Map();
  for (const [texto, conf] of lecturas) {
    for (const [placa, c] of candidatas(texto, conf)) {
      votos.set(placa, (votos.get(placa) ?? 0) + 1);
      confianza.set(placa, Math.max(confianza.get(placa) ?? 0, c));
    }
  }
  const validas = [...votos.keys()].filter((p) => votos.get(p) >= votosMinimos);
  if (!validas.length) return { placa: null, confianza: 0 };
  validas.sort((a, b) => votos.get(b) - votos.get(a) || confianza.get(b) - confianza.get(a));
  return { placa: validas[0], confianza: confianza.get(validas[0]) };
}

// ---- Preproceso (equivalente a PIL: gris, recorte, redimensión bicúbica con antialias) ----
const bicubico = (x) => {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
};
const redondear8 = (v) => Math.max(0, Math.min(255, Math.round(v)));

function coeficientes(entrada, salida) {
  const escala = entrada / salida;
  const escalaFiltro = Math.max(escala, 1);
  const soporte = 2 * escalaFiltro;
  const filas = [];
  for (let xx = 0; xx < salida; xx++) {
    const centro = (xx + 0.5) * escala;
    const xmin = Math.max(0, Math.trunc(centro - soporte + 0.5));
    const xmax = Math.min(entrada, Math.trunc(centro + soporte + 0.5));
    const pesos = [];
    let suma = 0;
    for (let x = xmin; x < xmax; x++) {
      const w = bicubico((x - centro + 0.5) / escalaFiltro);
      pesos.push(w);
      suma += w;
    }
    filas.push({ xmin, pesos: pesos.map((w) => w / suma) });
  }
  return filas;
}

// gris uint8 [alto*ancho] -> redimensionado a (anchoSalida x altoSalida), igual que Image.resize(..., BICUBIC)
function redimensionar(gris, ancho, alto, anchoSalida, altoSalida) {
  const horizontal = coeficientes(ancho, anchoSalida);
  const medio = new Float64Array(anchoSalida * alto);
  for (let y = 0; y < alto; y++) {
    for (let x = 0; x < anchoSalida; x++) {
      const { xmin, pesos } = horizontal[x];
      let s = 0;
      for (let k = 0; k < pesos.length; k++) s += gris[y * ancho + xmin + k] * pesos[k];
      medio[y * anchoSalida + x] = redondear8(s);
    }
  }
  const vertical = coeficientes(alto, altoSalida);
  const salida = new Float32Array(anchoSalida * altoSalida);
  for (let y = 0; y < altoSalida; y++) {
    const { xmin, pesos } = vertical[y];
    for (let x = 0; x < anchoSalida; x++) {
      let s = 0;
      for (let k = 0; k < pesos.length; k++) s += medio[(xmin + k) * anchoSalida + x] * pesos[k];
      salida[y * anchoSalida + x] = redondear8(s);
    }
  }
  return salida;
}

// RGBA -> tensor Float32 [1,1,64,W] en -1..1, leyendo desde la fracción `desde` de la altura hasta abajo
export function prepararFranja(rgba, ancho, alto, desde) {
  const arriba = Math.floor(desde * alto);
  const altoFranja = alto - arriba;
  const gris = new Uint8Array(ancho * altoFranja);
  for (let y = 0; y < altoFranja; y++) {
    for (let x = 0; x < ancho; x++) {
      const p = ((arriba + y) * ancho + x) * 4;
      gris[y * ancho + x] = (rgba[p] * 19595 + rgba[p + 1] * 38470 + rgba[p + 2] * 7471 + 0x8000) >> 16; // igual que PIL 'L'
    }
  }
  const anchoSalida = Math.max(8, Math.ceil((ALTO * ancho) / altoFranja));
  const px = redimensionar(gris, ancho, altoFranja, anchoSalida, ALTO);
  for (let i = 0; i < px.length; i++) px[i] = (px[i] / 255 - 0.5) / 0.5;
  return { tensor: px, ancho: anchoSalida };
}

// ---- Decodificación CTC voraz restringida a los caracteres permitidos ----
// probabilidades: Float32Array [pasos*clases]; caracteres: lista de la ficha (el 0 es el blank de CTC)
export function decodificar(probabilidades, pasos, clases, caracteres) {
  const permitido = caracteres.map((c, i) => i === 0 || PERMITIDOS.includes(c));
  let texto = '';
  let previo = 0;
  let producto = 1;
  for (let t = 0; t < pasos; t++) {
    let suma = 0;
    for (let c = 0; c < clases; c++) if (permitido[c]) suma += probabilidades[t * clases + c];
    let mejor = 0;
    let max = -1;
    for (let c = 0; c < clases; c++) {
      const p = permitido[c] ? probabilidades[t * clases + c] / suma : 0;
      if (p > max) {
        max = p;
        mejor = c;
      }
    }
    if (mejor !== 0 && mejor !== previo) texto += caracteres[mejor];
    previo = mejor;
    producto *= max;
  }
  return [texto, producto ** (2 / Math.sqrt(pasos))];
}

// Lee un recorte de placa: una lectura por franja y la placa por votación.
export async function leerPlaca(ort, sesion, rgba, ancho, alto, caracteres, franjas = FRANJAS, votosMinimos = VOTOS_MINIMOS) {
  const lecturas = [];
  for (const desde of franjas) {
    const { tensor, ancho: w } = prepararFranja(rgba, ancho, alto, desde);
    const salida = (await sesion.run({ imagen: new ort.Tensor('float32', tensor, [1, 1, ALTO, w]) })).probabilidades;
    lecturas.push(decodificar(salida.data, salida.dims[1], salida.dims[2], caracteres));
  }
  return { ...elegirPorVotos(lecturas, votosMinimos), lecturas };
}
