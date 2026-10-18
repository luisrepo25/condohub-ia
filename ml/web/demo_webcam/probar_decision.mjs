// Pruebas de la decisión de portería (A4) con los dos lectores. Uso: node probar_decision.mjs
import { evaluar } from './decision.mjs';

let fallos = 0;
function caso(nombre, reciente, registradas, esperado) {
  const r = evaluar(reciente, registradas);
  const ok = esperado === null ? r === null : r && r.estado === esperado.estado && (esperado.texto === undefined || r.texto === esperado.texto);
  console.log(`${ok ? '✓' : '✗'} ${nombre} -> ${r ? `${r.estado} ${r.texto}` : 'null'}`);
  if (!ok) fallos++;
}

const cnnMala = { texto: '6652SAT', confianza: 0.2 };                // la CNN lee basura en placas que no son la de la demo
const ocr = (placa, confianza = 0.55) => ({ cnn: cnnMala, ocr: { placa, confianza } });
const nada = { cnn: cnnMala, ocr: { placa: null, confianza: 0 } };

// Los 8 cuadros de la captura real: 4 lecturas correctas de 2534RIS (confianza 0,52–0,58) y 4 sin lectura válida
const captura = [nada, nada, ocr('2534RIS', 0.56), ocr('2534RIS', 0.58), nada, ocr('2534RIS', 0.57), ocr('2534RIS', 0.52), nada];

caso('captura real: 2534RIS leída en 4 de 8 cuadros', captura, ['2602SIT', '2534RIS'], { estado: 'PERMITIDO', texto: '2534RIS' });
caso('solo 2 de 8 cuadros: no basta, decide el guardia', [nada, nada, nada, nada, nada, nada, ocr('2534RIS'), ocr('2534RIS')], ['2534RIS'], { estado: 'SUGERIDA: DECIDE EL GUARDIA', texto: '2534RIS' });
caso('3 lecturas pero fuera de los últimos 8 cuadros', [ocr('2534RIS'), ocr('2534RIS'), ocr('2534RIS'), nada, nada, nada, nada, nada, nada, nada], ['2534RIS'], { estado: 'SUGERIDA: DECIDE EL GUARDIA' });
caso('placa NO registrada leída 4 veces: nunca permite', [ocr('3993YHA'), ocr('3993YHA'), ocr('3993YHA'), ocr('3993YHA')], ['2602SIT', '2534RIS'], { estado: 'NO REGISTRADA', texto: '3993YHA' });
caso('lecturas distintas cada vez: no se acumula consenso', [ocr('2534RIS'), ocr('2534RIT'), ocr('2634RIS'), ocr('2534RIS'), nada, nada, nada, nada], ['2534RIS'], { estado: 'SUGERIDA: DECIDE EL GUARDIA' });
caso('una sola lectura a 1 carácter de la registrada: sugerida', [ocr('2534RIT', 0.4)], ['2534RIS'], { estado: 'SUGERIDA: DECIDE EL GUARDIA', texto: '2534RIS' });
caso('la CNN lee basura cercana a la de la demo: SIN sugerencia falsa', [{ cnn: { texto: '6602SIT', confianza: 0.74 }, ocr: { placa: null, confianza: 0 } }, { cnn: { texto: '6652SIT', confianza: 0.5 }, ocr: { placa: null, confianza: 0 } }], ['2602SIT'], { estado: 'SIN COINCIDENCIA' });
caso('A4 original: OCR con confianza ≥ 0,85 en 2 cuadros', [ocr('2534RIS', 0.9), ocr('2534RIS', 0.88)], ['2534RIS'], { estado: 'PERMITIDO', texto: '2534RIS' });
caso('A4 original: CNN exacta y segura en 2 cuadros (la placa de la demo)', [{ cnn: { texto: '2602SIT', confianza: 0.99 }, ocr: { placa: null, confianza: 0 } }, { cnn: { texto: '2602SIT', confianza: 0.95 }, ocr: { placa: null, confianza: 0 } }], ['2602SIT'], { estado: 'PERMITIDO', texto: '2602SIT' });
caso('la CNN segura en UN solo cuadro no basta', [{ cnn: { texto: '2602SIT', confianza: 0.99 }, ocr: { placa: null, confianza: 0 } }], ['2602SIT'], { estado: 'SIN COINCIDENCIA' });
caso('sin cuadros todavía', [], ['2602SIT'], null);

console.log(fallos === 0 ? '\nTodas las pruebas pasan' : `\n${fallos} prueba(s) fallan`);
process.exit(fallos === 0 ? 0 : 1);
