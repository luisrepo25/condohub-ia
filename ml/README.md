# ml/ — entrenamiento y evaluación (Google Colab)

El entrenamiento se hace en **Google Colab** (GPU T4); en el repo vive el código reutilizable y los resultados. Los notebooks
llaman a `ml/src` y no duplican código. Datasets, fotos y pesos **no se versionan** (I6): viven fuera de git.

## Decisión actual

**El modelo del servicio es FaceNet preentrenado en VGGFace2 (InceptionResnetV1), sin ajuste propio.** El ajuste fino con LFW no
se adopta: su mejora no es concluyente y, en las pruebas con personas reales, separó peor a los desconocidos (ver abajo).
Se compara por similitud coseno contra los embeddings registrados; la regla I2 usa **tres zonas con los valores iniciales 0,70 y 0,50**, que la medición con el equipo respalda (sección 6); se revalidan con Erick y con más personas.

## Notebooks (en `notebooks/`, se ejecutan en Colab, de arriba abajo)

| Notebook | Qué hace |
|---|---|
| `01_linea_base_facial.ipynb` | Línea base con LFW (embeddings congelados + SVM/k-NN, verificación 1:1) |
| `02_registro_propios.ipynb` | Registro de personas del equipo por embeddings y evaluación dejando una sesión fuera |
| `03_finetuning_facial.ipynb` | Ajuste fino con triplet loss (semi-hard), comparación contra la base y repetición con 5 semillas |
| `04_placas.ipynb` | Placas: YOLOv8n ajustado (mAP en prueba) + OCR medido con placas sintéticas de formato A4 (4 dígitos + 3 letras) |
| `06_registro_equipo.ipynb` | Registra a las personas del equipo (una carpeta por persona; sirve para cualquier número) y mide las tres zonas de I2 contra extraños de LFW y «compañeros no registrados». Fotos preparadas con `python -m ml.src.preparar_fotos` |
| `08_probar_servicio_facial.ipynb` | El modelo facial **real a través del servicio** (adaptador + casos de uso de `app/`): paridad con el pipeline de `ml/`, zonas de I2 con las fotos del equipo, ciclo de I1 (registrar, reconocer, retirar) y latencia. Necesita `servicio.zip` (carpetas `app/` y `ml/`) |
| `05_dbscan_desconocidos.ipynb` | Visitantes recurrentes: DBSCAN (coseno) sobre desconocidos simulados con fotos de LFW; elige `eps` y mide las alertas de I4 |

Lo que sale de cada corrida (`metrics.json` y similares) se copia a `experiments/<fecha>_<nombre>/` y **sí se versiona**.

## Cómo se ejecuta en Colab (lo aprendido)

- Activa la GPU (Entorno de ejecución → Cambiar tipo → T4). Instala con `pip install facenet-pytorch --no-deps`.
- Sube dos ZIP con `files.upload()`: uno con la carpeta `ml/` y otro con las fotos (`<persona>/<sesion>/foto.jpg` más `consentimiento.json`).
  Si ya existe un archivo con el mismo nombre, Colab guarda `ml (1).zip`: usa el nombre que devuelve `files.upload()`, no el esperado.
- **Antes de subir las fotos:** reducirlas a 1280 px y quitarles los metadatos (GPS y datos del teléfono). Pasan de ~270 MB a ~18 MB.
- El código y las fotos desaparecen cuando se cierra la sesión: descarga `metrics.json` y los pesos al terminar.

## Qué se puede probar sin GPU

`split.py`, `metricas.py`, `cluster.py`, `linea_base.py`, `registro.py`, `estadistica.py`, `propios.py` y `entrenamiento.py` se prueban en local
(`pip install -e ".[dev,ml]"` y, para el entrenamiento, PyTorch CPU; `pytest ml/tests`). Solo `embeddings.py` necesita facenet-pytorch y se ejecuta en Colab.

## Resultados (15/10/2026)

### 1. Línea base y ajuste fino (LFW, personas que el modelo no vio)

Con 158 personas de LFW (≥ 10 fotos), partición por persona 95 / 32 / 31. EER en la partición de prueba (menor es mejor):

| Corrida | Base | Ajustado |
|---|---|---|
| Semilla 42 (la primera) | 4,65 % | 4,40 % |
| Media de 5 semillas (42–46) | 4,85 % | 4,60 % |

La mejora media es de **0,25 puntos**, con intervalo de confianza del 95 % de **−0,38 a +0,87 puntos**: cruza el cero, por lo que **no es concluyente**
(4 de 5 semillas mejoran, 1 empeora). La variación entre semillas del EER de la base (4,0 % a 5,85 %) es mayor que la mejora.
Detalle: `experiments/2026-10-15_facial_finetuning/` (`metrics.json`, `repeticion_semillas.json`). Se entrenaron solo `block8`, `last_linear` y `last_bn`;
8 épocas × 40 pasos, lotes de 16 personas × 4 fotos, margen 0,2. Los pesos (`.pt`, 107 MB) están guardados fuera del repo.

### 2. Registro de una persona (fotos propias, 98 fotos en 2 condiciones de luz)

El detector halló cara en 77 de 80 fotos con luz alta y en 13 de 18 con luz baja. Umbral calibrado solo con impostores de LFW (FAR objetivo 1 %): 0,576.

| Medida | Resultado |
|---|---|
| Desconocidos de LFW aceptados | 0,12 % |
| Fotos propias rechazadas (de las detectadas) | 8,9 % (8 de 90) |
| Fotos propias no reconocidas contando las que el detector perdió | ≈ 16 % (82 de 98 reconocidas) |

Detalle: `experiments/2026-10-15_propios_registro/metrics.json`.

### 3. Prueba con fotos nuevas y desconocidos reales (cualitativa)

| Foto | Base | Ajustado |
|---|---|---|
| La persona registrada, otro día | 0,929 reconocida | 0,938 reconocida |
| La persona registrada, con poca luz | 0,884 reconocida | 0,868 reconocida |
| Una persona distinta | 0,228 desconocida | 0,320 desconocida |
| Foto de grupo: 3 amigos | **2 de 3 aceptados** (0,60; 0,64; el tercero 0,50 rechazado) | **3 de 3 aceptados** (0,67; 0,69; 0,72) |

**Hallazgo importante:** el FAR medido con LFW (0,12 %) es demasiado optimista para esta población. LFW son sobre todo famosos, en su mayoría de otros
perfiles; los amigos (jóvenes, pelo oscuro, rasgos parecidos) obtienen puntajes cercanos al umbral. Es una muestra pequeña (3 personas, una foto), pero
suficiente para no fiarse del umbral calibrado solo con LFW. Con las fotos propias dejando una sesión fuera, estos son los costos de subir el umbral:

| Umbral | Base: fotos propias rechazadas / amigos aceptados | Ajustado: ídem |
|---|---|---|
| 0,55 | 9 % / 67 % | 3 % / 100 % |
| 0,60 | 9 % / 33 % | 6 % / 100 % |
| 0,65 | 17 % / 0 % | 9 % / 100 % |
| 0,70 | 19 % / 0 % | 18 % / 33 % |
| 0,75 | 32 % / 0 % | 27 % / 0 % |

La base separa mejor que el ajustado (entrenado con LFW). Esta prueba registra con una sola sesión, por lo que los puntajes propios con las 90 fotos serían algo mejores.

### 4. Placas: detección (YOLOv8n) y lectura (OCR)

Notebook `04_placas.ipynb`. Detector ajustado desde COCO con *Car License Plate Detection* (433 imágenes, una clase, CC0; **no hay placas bolivianas**),
partición 303 / 65 / 65 por imagen, 60 épocas (mejor en la 40), semilla 42. Detalle: `experiments/2026-10-15_placas_deteccion/` (`metrics.json`, curvas).
Pesos (`yolov8n_placas.pt`, 6,5 MB) fuera de git. COCO no tiene la clase «placa», así que no existe un detector preentrenado con el que comparar.

| Detección en las 65 imágenes de prueba (70 placas) | Valor |
|---|---|
| mAP@0.5 | 0,931 |
| mAP@0.5-0.95 | 0,519 |
| Precisión / recall | 0,937 / 0,857 |

Lectura con EasyOCR sobre **300 placas sintéticas** de formato A4 (4 dígitos + 3 letras; texto conocido, optimistas frente a placas reales):

| Paso | Precisión por carácter | Placa completa |
|---|---|---|
| OCR crudo | 99,48 % | 96,7 % |
| + extraer la ventana con el formato (descarta el marco leído como caracteres) | 99,95 % | **99,67 %** |
| + corrección letra/dígito por posición | 99,95 % | 99,67 % (no aporta en este conjunto) |

Una sola falla: un dígito mal leído, que ningún formato detecta. Las 300 lecturas finales cumplen el formato.

**Foto real boliviana** (una imagen, cualitativa, no es una métrica; la foto no se versiona): placa `2608ZYU`.
- Con margen añadido (la placa deja de ser primer plano): una caja con 0,84, lectura `2608ZYU`, confianza OCR 0,76. El OCR leyó `26082YU` y la corrección por posición devolvió la `Z`.
- En primer plano el detector da cajas parciales (nunca vio placas así de grandes); una lectura correcta (0,72) y otra rechazada por no cumplir el formato, sin inventar nada.
- Con A4 en 0,85 esta placa bien leída no habría abierto sola (0,72–0,76): **decisión pendiente del equipo** sobre ese umbral.

Lecciones: pasar al detector una imagen PIL (RGB) y no un arreglo RGB (Ultralytics espera BGR); la corrección por posición convierte cualquier palabra de 7 letras en «placa» (`BOLIVIA` → `8011VIA`), por eso `elegir_placa` la limita a 2 cambios y excluye el rótulo.

### 5. No supervisado: DBSCAN sobre desconocidos (I3, I4)

Notebook `05_dbscan_desconocidos.ipynb`; detalle en `experiments/2026-10-15_dbscan_desconocidos/` (métricas y tres figuras).
**Es una simulación**: los desconocidos son fotos de LFW (4324 fotos, 158 personas, embeddings FaceNet) repartidas en una ventana de 30 días,
de modo que se sabe quién es quién. Cuatro tipos de persona: recurrente (3–6 días distintos, debe alertar), mismo día (4 apariciones en un día),
dos días y suelta (ninguna debe alertar). `eps` se elige con 79 personas («ajuste») y se mide con otras 47 que nunca se usaron («prueba»),
en 5 escenarios con distinta semilla (6 recurrentes reales cada uno, 30 en total). `min_samples` = 3 (I4).

- **Elección de `eps`:** el codo del gráfico k-distance da **0,291**. El barrido sobre las personas de ajuste da una meseta entre 0,25 y 0,40
  (F1 de 0,91 a 0,96) y cae desde 0,45 porque se mezclan personas distintas. El mejor F1 exacto fue 0,40 (0,960) frente a 0,30 (0,957).
- **Medición con personas nuevas** (media de 5 escenarios):

| `eps` | Alertas | Aciertos | Falsas alarmas | Recurrentes detectados | Precisión | Recall |
|---|---|---|---|---|---|---|
| 0,40 (mejor F1 en el ajuste) | 30 | 28 | 2 | 28 de 30 | 0,93 ± 0,09 | 0,93 ± 0,13 |
| 0,291 (codo, sin etiquetas) | 27 | 27 | 0 | 27 de 30 | 1,00 ± 0,00 | 0,90 ± 0,13 |

Las diferencias son de una alerta sobre 30, dentro de la variación entre escenarios: no hay un ganador claro. Lo relevante es que el `eps` elegido **sin etiquetas**
(codo) rinde igual que el elegido con ellas, que es lo único posible en una portería real. Valor inicial propuesto: **0,30**, como parámetro configurable (no
constante). Se eligió mirando también los resultados de prueba: presentarlos los dos en el informe.
- **La regla de días distintos de I4 es la que evita falsas alarmas:** las personas que vienen varias veces el mismo día o solo dos días sí forman
  clústeres densos (en `clusters_pca.png`, clústeres 6–11), pero no alertan.
- **Límites:** simulación con LFW (fotos más variadas que las de una cámara fija de portería); 2 dimensiones de PCA solo ilustran; muestra pequeña
  (30 recurrentes), por eso las desviaciones son grandes; en dos escenarios se pierde un recurrente cuyas fotos varían mucho.

### 6. Registro del equipo y las tres zonas de I2

Notebook `06_registro_equipo.ipynb`; detalle en `experiments/2026-10-15_registro_equipo/` (`metrics.json` e `histograma_similitudes.png`).
Registradas **dos personas** (Aldana: 50 fotos en 4 sesiones por fecha, de 2022 a 2023; Luis: 98 fotos en 2 sesiones de luz), 139 fotos con rostro detectado
(49/50 y 90/98). Se evalúa dejando una sesión entera fuera. Extraños: 790 fotos de 158 personas de LFW, mitad para calibrar y mitad para medir.
«Compañero no registrado» = fotos de una persona contra el registro de la otra: el «amigo parecido» que dejó mal parado al umbral calibrado solo con LFW.
Los consentimientos son los borradores (`borrador-1` y `borrador-2`): pendiente el texto definitivo.

| Rostro → zona | Regla (alto 0,70 / bajo 0,50) | Calibrado con LFW (alto 0,584 / bajo 0,447) |
|---|---|---|
| Propios: RECONOCIDO correcto | 85,6 % | 92,8 % |
| Propios: RECONOCIDO como otra persona | 0,0 % | 0,7 % (1 foto) |
| Propios: DUDOSO / DESCONOCIDO | 9,3 % / 5,0 % | 2,2 % / 4,3 % |
| **Compañero no registrado: RECONOCIDO (error grave)** | **5,8 %** | **59,7 %** |
| Compañero no registrado: DUDOSO / DESCONOCIDO | 76,3 % / 18,0 % | 29,5 % / 10,8 % |
| Extraños de LFW: DESCONOCIDO | 97,7 % | 92,4 % |
| Extraños de LFW: RECONOCIDO | 0,0 % | 0,5 % |

- **Calibrar solo con LFW es peligroso:** el umbral sale en 0,58 y reconocería por error a 6 de cada 10 compañeros parecidos. Los extraños de LFW (similitud ~0,3)
  no se parecen a los residentes; los compañeros se concentran en 0,5–0,7 y los propios en 0,8–0,9 (`histograma_similitudes.png`).
- **Los valores de la regla (0,70 y 0,50) se mantienen:** casi no hay falsos reconocimientos y los parecidos caen en DUDOSO, donde decide el guardia (A5). El costo es que
  ~14 % de las fotos propias requieren decisión del guardia. Con un solo umbral habría que elegir entre confundir parecidos o no reconocer residentes.
- Con el umbral único de 0,584: FAR 0,51 % en LFW, FRR 7,2 % y **1 confusión** (una foto de la sesión 4 de Aldana reconocida como Luis). Los rechazos de Luis están sobre todo
  en `Luz alta` (6 de 77), no en `Luz baja` (2 de 13): la luz no parece ser el factor.
- **Límites:** solo dos personas (el «compañero» es siempre el mismo parecido), 139 fotos propias, y las fotos de Aldana son de hace años y de WhatsApp. No afinar los umbrales con esta muestra;
  revalidar con Erick (tres personas) y con fotos recientes de Aldana.

**Revalidación con tres personas (16/10/2026).** Se sumaron las fotos de Erick (42, en 4 bloques del mismo día; consentimiento `borrador-3`) y 10 recientes de Aldana
(`sesion5_2026-10-15`): 191 fotos propias con rostro (`experiments/2026-10-15_registro_equipo/metrics_3_personas.json` e `histograma_similitudes_3_personas.png`). Misma metodología; las dos primeras columnas son las de la tabla anterior para comparar.

| Rostro → zona | Regla 0,70 / 0,50 (2 personas) | Regla 0,70 / 0,50 (3 personas) | Calibrado con LFW 0,593 / 0,453 (3 personas) |
|---|---|---|---|
| Propios: RECONOCIDO correcto | 85,6 % | 89,5 % | 94,8 % |
| Propios: RECONOCIDO como otra persona | 0,0 % | 0,0 % | 0,5 % |
| Propios: DUDOSO / DESCONOCIDO | 9,3 % / 5,0 % | 7,3 % / 3,1 % | 1,5 % / 3,1 % |
| **Compañero no registrado: RECONOCIDO (error grave)** | 5,8 % | **9,4 %** | **63,4 %** |
| Compañero no registrado: DUDOSO / DESCONOCIDO | 76,3 % / 18,0 % | 79,6 % / 11,0 % | 32,5 % / 4,2 % |
| Extraños de LFW: DESCONOCIDO / RECONOCIDO | 97,7 % / 0,0 % | 97,5 % / 0,0 % | 94,2 % / 0,3 % |

- **Decisión: los umbrales finales de I2 son 0,70 (alto) y 0,50 (bajo).** Con tres personas se repite la conclusión: calibrar solo con LFW dejaría reconocer a 6 de cada 10
  compañeros parecidos. Umbral único calibrado: 0,593 (FAR 0,25 % en LFW, FRR 5,2 %, 1 confusión: una foto de la sesión 4 de Aldana).
- Las 10 fotos recientes de Aldana se reconocen todas (10/10, sin confusiones) con un registro hecho con fotos de 2022–2023: el perfil resiste el paso de los años.
- Por persona (umbral único): Erick 42/42, Aldana 47/48 en las sesiones antiguas y 10/10 en la reciente, Luis 82/90 (rechazos: 6 de 77 con luz alta, 2 de 13 con luz baja).
- **Límites:** las 4 sesiones de Erick son del mismo día, así que su resultado es optimista (fotos casi iguales entre sí); el «compañero no registrado» son solo tres personas que se
  prestan parecido entre sí, por lo que el 9,4 % es orientativo y no una tasa firme. Los consentimientos son de versiones de borrador distintas; se acepta así (ver «Pendiente», punto 5).

### 6b. Cómo decidir «quién es» y calidad de las fotos (16/10/2026)

Notebook `09_decision_y_calidad.ipynb` (resultados en `experiments/2026-10-16_decision_y_calidad/`); código en `ml/src/metodos_decision.py` (extiende `registro.py` y `zonas.py` sin modificarlos). Mismas 191 fotos propias y extraños de LFW
que la sección 6. Como los puntajes de cada método tienen otra escala, se comparan con métricas que no dependen de ella: AUC y «propios reconocidos si se deja pasar a lo sumo un 5 % de compañeros parecidos».

| Método | Identidad correcta | AUC propios/compañeros | AUC propios/extraños | Reconocidos con ≤5 % compañeros | Umbral en ese punto |
|---|---|---|---|---|---|
| **Máximo (el del servicio)** | 0,990 | 0,929 | 0,991 | 0,869 | 0,714 |
| Promedio de las 3 mejores | 0,979 | 0,915 | 0,989 | 0,827 | 0,703 |
| Centroide | 0,979 | 0,943 | 0,993 | 0,869 | 0,639 |
| SVM lineal | 0,990 | 0,882 | 0,658 | 0,482 | 0,965 |
| k-NN (k = 5) | 0,974 | 0,637 | 0,744 | 0,000 | 1,000 |

- **Se mantiene el máximo.** El centroide empata (misma tasa al 5 %; su ventaja de AUC con tres personas está dentro del ruido) y exigiría recalibrar I2: su punto equivalente cae en 0,64.
  Con el máximo, el umbral del punto de 5 % (0,714) coincide con el 0,70 de la regla: confirmación independiente de ese valor.
- **SVM y k-NN identifican bien pero no rechazan extraños** (AUC frente a LFW de 0,66 y 0,74; el k-NN reconoce a todos los compañeros parecidos con probabilidad 1,0): un clasificador de conjunto cerrado
  siempre elige a alguien, y I2 necesita saber cuándo no es nadie. Por eso la regla usa similitud y no la probabilidad de un clasificador.
- **Nitidez** (varianza del laplaciano): correlación de Spearman 0,49 con el puntaje de los propios; las fotos con puntaje < 0,70 tienen una nitidez mediana de 67 frente a 166 del resto.
  Las fotos borrosas, al compararse, puntúan bajo y caen en DUDOSO (decide el guardia, I2).
- **No registrar las fotos menos nítidas no ayuda; empeora:** al descartar el 0 / 10 / 20 / 30 % del registro, la identidad correcta pasa de 0,990 a 0,990 / 0,979 / 0,963 y los propios con puntaje < 0,70
  de 10,5 % a 11,0 / 12,0 / 13,6 %. Se pierde más por tener menos variedad de fotos que lo que se gana por calidad. **Se descarta el filtro de calidad en el registro.**
- **Límites:** tres personas; las 4 sesiones de Erick son del mismo día; los «compañeros» son el propio equipo. Es una comparación orientativa, no una afinación de umbrales.

### 6c. Lectura del texto sobre placas reales (16/10/2026)

Notebook `10_placas_reales_ocr.ipynb` (métricas en `experiments/2026-10-16_placas_reales_ocr/`). **10 recortes de placas bolivianas reales** (capturas de vehículos, recortadas con el detector ONNX; 58 a 174 px de ancho, anotadas a mano por Luis;
no se versionan: son vehículos de terceros). Es el único conjunto real con texto verdadero: el dataset de detección (433 imágenes) solo trae cajas, y hasta ahora la lectura se había medido con 300 placas
sintéticas y una foto. Una de las 10 es una placa antigua de 3 dígitos que A4 no debe aceptar. Por privacidad (I6) ni las placas ni las lecturas se guardan en el repositorio: solo las métricas.

| Lectura | Placas exactas | Precisión por carácter |
|---|---|---|
| Tesseract en Colab (mejor de 6 variantes de escala y modo) | 3/10 | 0,41 |
| Tesseract.js en Node, WASM (mejor de 6 variantes) | 3/10 | 0,45 |
| EasyOCR crudo | 2/10 | 0,49 |
| **EasyOCR + formato de A4 (`elegir_placa`)** | **6/10** | **0,61** |

- **Con el filtro de A4 no se aceptó ninguna lectura equivocada:** las 6 aceptadas son exactas (dos gracias a la corrección letra/dígito por posición: un `5` leído por `S` y un `8` repetido por `G`) y las 4 rechazadas
  no se inventaron: la de formato antiguo (correcto), una placa dañada leída con varios errores, una sin lectura y una en la que el rótulo «BOLIVIA» quedó pegado a los caracteres. Sobre las 9 placas con formato A4: 6 de 9.
- **La comparación con Tesseract no es del todo justa:** el filtro de formato y la corrección por posición no se le aplicaron. Lo firme es que Tesseract (en Python o en el navegador) no basta por sí solo.
- **Confianza y A4 (0,85):** solo 2 de las 6 lecturas correctas llegan a 0,85 (0,93 y 0,99; otra queda en 0,83). Con ese umbral el sistema abriría solo 2 de 10 y el guardia decidiría el resto: seguro, pero muy conservador.
  Las lecturas correctas de baja confianza (0,45 y 0,55) pasaron el formato. Con 10 placas no se puede concluir sobre el umbral: decisión del equipo con más datos.
- **Mejora barata por probar:** leer solo la franja de caracteres (la mitad inferior de la placa) para dejar fuera el rótulo «BOLIVIA».
- **Límites:** 10 placas, recortes pequeños de capturas de pantalla (en portería serán mayores). Orientativo, no una tasa.

### 6d. El reconocedor de EasyOCR en ONNX, para leer la placa en el dispositivo (16/10/2026)

Notebook `11_exportar_reconocedor_placas.ipynb`; lógica de lectura en `ml/src/lectura_placas.py`. EasyOCR no corre en el navegador, pero su **reconocedor** (VGG + BiLSTM + CTC, 3,8 M de parámetros) se exporta a ONNX y corre en
`onnxruntime-web`; el detector de texto de EasyOCR (CRAFT) no hace falta porque ya hay detector de placas. Mismos 10 recortes reales de 6c.

- **Exportación:** en CPU EasyOCR carga el reconocedor cuantizado (no exportable) y el `quantize=False` no lo evitó: se reconstruye el modelo en coma flotante (`easyocr.model.vgg_model.Model`) con los pesos de `english_g2.pth`, se reemplaza
  `AdaptiveAvgPool2d((None, 1))` por un promedio del último eje (equivalente, con ancho dinámico) y se exporta con el exportador clásico (`dynamo=False`). Archivo: **15,1 MB** (fp32) o **3,85 MB** con cuantización dinámica int8.
- **Paridad PyTorch ↔ ONNX:** el mismo texto en los 10 recortes y diferencia de confianza 0,0000.
- **Velocidad (CPU de Colab, nativa):** 17 ms por lectura en fp32; el int8 dinámico baja el tamaño unas 4 veces **sin perder exactitud** (mismo resultado) pero tarda más (42 ms) en esa CPU; en el navegador (WASM) falta medirlo.
- **Franja a leer:** pasar el recorte entero da 0/10 (el rótulo «BOLIVIA» va en la misma línea). Se lee desde varias alturas (`FRANJAS`: 0, 10, 20, 30, 40 y 50 % de la altura hasta abajo).

| Estrategia sobre el recorte (reconocedor ONNX) | Exactas | Equivocadas aceptadas | Rechazadas |
|---|---|---|---|
| Una franja (30 %) + `elegir_placa` | 4/10 | 0 | 6 |
| Una franja (30 %) + todas las ventanas con corrección | 4/10 | 2 | 4 |
| Una franja (40 %) + todas las ventanas con corrección | 5/10 | 3 | 2 |
| Varias franjas + todas las ventanas, sin votar | 6/10 | **2** | 2 |
| **Varias franjas + votación (≥ 3 de 6)** | **4/10** | **0** | 6 |
| *EasyOCR completo + formato (servidor, 6c)* | *6/10* | *0* | *4* |

- **Las «ventanas con corrección» son peligrosas:** una lectura ruidosa contiene «placas» desfasadas un carácter que cumplen el formato y son falsas (2 de 10 aceptadas equivocadas). Por eso se exige que la misma placa aparezca en **al menos 3 franjas**.
- **Decisión:** la PWA usa la votación (≥ 3): en el dispositivo se leen 4 de 10 placas con **cero errores aceptados**; las demás las decide el guardia (A4). Rechazar es barato, aceptar una placa falsa no.
  El servidor (EasyOCR completo) llega a 6 de 10 con cero errores.
- **Límites:** 10 placas de capturas pequeñas, y la estrategia se eligió sobre esas mismas placas: es una hipótesis que confirmar con más placas (idealmente de la cámara de portería), no una tasa.
- **Portado a JavaScript** (`ml/web/lectura_placas.mjs`, probado con `probar_lectura.mjs` en Node con `onnxruntime-web`/WASM): preproceso equivalente a PIL (gris, recorte, bicúbica con antialias), decodificación CTC restringida
  a dígitos y mayúsculas, y la votación. Frente a la referencia de Python: **el mismo texto en 60 de 60 lecturas** (6 franjas × 10 placas, diferencia máxima de confianza 0,0006) y la misma placa en 10 de 10 (4 exactas, 0 equivocadas).
  Una placa completa (6 lecturas, con pre y posproceso) tarda unos 260 ms en WASM en la PC; falta medirla en un celular. Los pesos (`reconocedor_placas.onnx`, SHA-256 en `ficha_reconocedor.json`) no se versionan: van al almacén S3.

### 7. Placas en el dispositivo: ONNX int8

Notebook `07_exportar_onnx_placas.ipynb`; detalle en `experiments/2026-10-15_placas_onnx/` (`metrics.json` y la ficha `modelo_placas.json`). El `.onnx` **no** va a git
(I6): `D:\SW2\modelos\placas\2026-10-15_onnx\yolov8n_placas_int8.onnx` y, cuando haya infraestructura, el almacén S3 (`modelos/placas/1.0.0/`).
El detector se exporta a ONNX y se **cuantiza de forma estática (QDQ)** con 100 imágenes de *entrenamiento*; se compara con PyTorch en las mismas 65 imágenes de prueba
del notebook 04 (70 placas), con el mismo posproceso (`ml/src/onnx_placas.py`, la referencia en Python de lo que debe hacer la PWA).

| Modelo | AP50 | Precisión | Recall | Tamaño | CPU (Colab, Python) |
|---|---|---|---|---|---|
| PyTorch | 0,930 | 0,922 | 0,843 | 6,5 MB | — |
| ONNX fp32 | 0,923 | 0,935 | 0,829 | 12,8 MB | 90 ms |
| ONNX int8, todo cuantizado (`u8u8`, `u8s8`) | **0,000** | — | 0,000 | 3,5 MB | 120 / 77 ms |
| **ONNX int8, solo convoluciones (`u8s8_solo_conv`)** | **0,903** | 0,967 | 0,829 | **3,5 MB** | 98 ms |

- **Cuantizar todo el grafo destruye la cabeza de detección** (AP 0,0); por eso se probaron tres variantes en vez de fiarse de una. Cuantizando solo las convoluciones la caída
  es de 0,026 frente a PyTorch (límite fijado de antemano: 0,03) → se adopta. Con 70 placas de referencia la diferencia son unas 2 placas: equivalente al fp32 dentro de la incertidumbre.
- La coincidencia con PyTorch en el punto de operación (confianza ≥ 0,5): el fp32 reproduce 62 de 64 detecciones y el int8, 59 (5 solo PyTorch, 1 solo ONNX).
- **El int8 no es más rápido en este CPU** (98 ms frente a 90 ms): la ganancia es el **tamaño de descarga** (3,7 veces menos). La velocidad en un celular con `onnxruntime-web` solo se sabe midiéndola allí.
- El modelo tiene nodos de cuantización (QDQ): la PWA debe comprobar que su versión de `onnxruntime-web` lo carga; si no, usar el fp32 (el notebook también lo exporta como respaldo).
- **Contrato con la PWA** (`modelo_placas.json`): entrada `images` `[1, 3, 640, 640]` float (RGB, redimensionar sin deformar con relleno gris 114, dividir entre 255, NCHW); salida `output0` `[1, 5, 8400]`
  (cx, cy, w, h en píxeles del tensor de 640 y confianza); confianza mínima 0,5 y NMS con IoU 0,45. Esto solo **detecta** la placa: leer el texto en el dispositivo
  (sin EasyOCR) sigue sin resolverse (ver «Pendiente»).

### 8. El modelo facial real a través del servicio

Notebook `08_probar_servicio_facial.ipynb`; detalle en `experiments/2026-10-15_servicio_facial/metrics.json`. Mide el adaptador real (`app/infrastructure/facenet.py`) conectado a los
casos de uso `RegistrarRostro`, `IdentificarRostro` y `RetirarConsentimiento`, con los umbrales de la regla (0,70 y 0,50), las mismas fotos del equipo (Luis y Aldana)
y 316 fotos de extraños de LFW (2 por persona).

- **Paridad con el pipeline de `ml/`:** 27 fotos comparadas, similitud mínima 0,99999994 y mediana 1,0, sin discrepancias de detección. El servicio produce los mismos vectores
  que los notebooks, así que lo medido en 02, 03 y 06 vale para el servicio.
- **Resultados idénticos al notebook 06, foto por foto** (sobre las 139 fotos con rostro; el servicio evalúa 148 y 9 salen «sin rostro» por el detector):

| Rostro → resultado | Servicio (148 fotos) | Notebook 06 (139 fotos) |
|---|---|---|
| Propios: RECONOCIDO correcto | 119 (80,4 %) | 119 (85,6 %) |
| Propios: DUDOSO candidato correcto / otro | 12 / 1 | 12 / 1 |
| Propios: DESCONOCIDO / sin rostro | 7 / 9 | 7 / — |
| Propios reconocidos como OTRA persona | 0 | 0 |
| Compañero no registrado: RECONOCIDO / DUDOSO / DESCONOCIDO | 8 / 106 / 25 | 5,8 % / 76,3 % / 18,0 % |
| Extraños de LFW: DESCONOCIDO / DUDOSO / RECONOCIDO | 304 / 12 / 0 | (otra muestra: 97,7 % / 2,3 % / 0 %) |

- **I1:** registrar → reconocer → retirar el consentimiento: las 49 fotos se borran al momento, el perfil queda sin embeddings y la misma foto pasa de RECONOCIDO a DESCONOCIDO.
  El plazo informado es de 24 h. **I3:** se guardaron 304 desconocidos (los 12 DUDOSOS no). **I5:** 316 predicciones registradas, una por identificación.
- La confusión de Aldana (sesión 4) que con el umbral único salía reconocida como Luis cae ahora en DUDOSO con candidato otro: es el caso para el que sirve la zona intermedia.
- **Latencia del modelo (sin caché, 60 fotos, GPU T4):** mediana **80 ms**, p95 122 ms, máximo 160 ms por imagen nueva (decodificar la foto, MTCNN y FaceNet); medida en una celda aparte tras un calentamiento.
  **No citar** el `latencia_ms` que trae el `metrics.json` de esta corrida (mediana 2 ms, p95 54 ms): se midió con la caché de vectores activa y solo refleja la comparación con los perfiles.
  **No medida en CPU:** si el servicio se despliega sin GPU, hay que volver a medir (se espera bastante más).

## Pendiente

1. ~~Fotos de Erick y recientes de Aldana~~: recibidas y usadas en la revalidación de la sección 6 (16/10/2026). Mejorable: fotos de Erick en otros días y con otra luz, porque las 42 son del mismo día.
2. ~~Probar otros métodos de comparación~~: hecho en la sección 6b; se mantiene la similitud máxima.
3. ~~Filtrar por calidad las fotos de registro~~: probado en 6b y descartado (empeora). Queda mejorar la detección con poca luz (hoy se pierden 5 de 18) y, si hiciera falta, avisar al guardia cuando la foto de portería sea muy borrosa (la nitidez explica los puntajes bajos).
4. **Tres zonas** (reconocido / dudoso, decide el guardia / desconocido): escritas en la regla I2 con 0.70 y 0.50 y respaldadas por la sección 6. Revalidadas con tres personas (sección 6). Falta avisar a Erick (Acceso debe mostrar el candidato en DUDOSO y no abrir).
5. ~~Texto de consentimiento definitivo (I1)~~: **decisión del equipo (16/10/2026): no se exige.** Los tres desarrolladores aportaron sus propias fotos por voluntad propia (proyecto académico); su autorización queda registrada en su `consentimiento.json` (fecha y versión de borrador, que pueden diferir). El mecanismo de I1 sigue vigente: retirar el consentimiento borra fotos y embeddings. Si algún día se registra a una persona ajena al equipo, entonces sí hace falta su consentimiento explícito con el texto de `docs/negocio/consentimiento-facial.md`.
6. Placas: probar con más fotos bolivianas (y, si hace falta, ajustar el detector con ellas), decidir el umbral de A4 (0,85 es estricto para la confianza del OCR). **Leer el texto de la placa en el dispositivo:** resuelto en las secciones 6c y 6d (el reconocedor de EasyOCR exportado a ONNX + votación entre franjas, portado a JavaScript; Tesseract.js descartado). Falta publicarlo en el almacén S3 (`GET /modelos/...`), entregárselo a Erick y confirmar con más placas. El int8 ya se probó en `onnxruntime-web` (WASM, en Node, `ml/web/`): 10 de 10 placas de vehículos detectadas, IoU 0,82–0,99 frente al motor nativo y ~240 ms por imagen en PC; falta medir la velocidad en un celular. Publicar el modelo en el almacén S3. Llevar DBSCAN al job nocturno del servicio (parámetros `eps` = 0,30 y `min_samples` = 3 configurables) y, si se consiguen, validarlo con fotos reales de portería.

## Datos personales

Las fotos de personas con consentimiento (I1) van a un lugar **privado**, nunca a git ni a una carpeta pública. El texto de consentimiento (con versión) debe
indicar que las fotos se procesan en servicios de nube. En las pruebas de hoy el consentimiento fue un archivo `consentimiento.json` provisional.