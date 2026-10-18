// Lector de placas «global»: modelo preentrenado `cct-xs-v2-global-model` de fast-plate-ocr (MIT), entrenado con placas de muchos países.
// Entrada: imagen RGB de 128 × 64 (uint8, NHWC). Salida `plate`: [1, 10, 37] probabilidades (10 posiciones; alfabeto 0-9, A-Z y «_» de relleno).
// Se obtiene con: pip install "fast-plate-ocr[onnx]"  y  LicensePlateRecognizer("cct-xs-v2-global-model")  (se descarga a ~/.cache/fast-plate-ocr/);
// copiar `cct_xs_v2_global.onnx` (3,2 MB) a `services/ia/modelos/` (los .onnx no van a git, I6).
export const ANCHO_OCR = 128;
export const ALTO_OCR = 64;
export const ALFABETO = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_';
const RELLENO = '_';

// probabilidades: [ranuras * clases] -> { texto, confianza }. La confianza es la media de la probabilidad máxima de las posiciones que no son relleno.
// OJO: este modelo es muy seguro de sí mismo (también cuando se equivoca por un carácter): no sirve para distinguir aciertos de errores.
export function decodificar(probabilidades, ranuras = 10, clases = 37) {
  let texto = '';
  let suma = 0;
  let n = 0;
  for (let r = 0; r < ranuras; r++) {
    let mejor = 0;
    let max = -1;
    for (let c = 0; c < clases; c++) {
      const p = probabilidades[r * clases + c];
      if (p > max) { max = p; mejor = c; }
    }
    if (ALFABETO[mejor] !== RELLENO) { texto += ALFABETO[mejor]; suma += max; n++; }
  }
  return { texto, confianza: n ? suma / n : 0 };
}

export const cargarOcrGlobal = (ort, url) => ort.InferenceSession.create(url, { executionProviders: ['wasm'] });

// rgba: Uint8ClampedArray de ANCHO_OCR × ALTO_OCR × 4 (el recorte de la placa ya redimensionado, p. ej. con un canvas)
export async function leerOcrGlobal(ort, sesion, rgba) {
  const x = new Uint8Array(ANCHO_OCR * ALTO_OCR * 3);
  for (let i = 0, j = 0; i < ANCHO_OCR * ALTO_OCR; i++) {
    x[j++] = rgba[4 * i];
    x[j++] = rgba[4 * i + 1];
    x[j++] = rgba[4 * i + 2];
  }
  const salida = (await sesion.run({ input: new ort.Tensor('uint8', x, [1, ALTO_OCR, ANCHO_OCR, 3]) })).plate;
  return decodificar(salida.data, salida.dims[1], salida.dims[2]);
}
