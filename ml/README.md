# ml/ — entrenamiento y evaluación (Google Colab)

El entrenamiento se hace en **Google Colab** (GPU T4); en el repo vive el código reutilizable y los resultados. Los notebooks
llaman a `ml/src` y no duplican código. Datasets, fotos y pesos **no se versionan** (I6): viven fuera de git.

## Decisión actual

**El modelo del servicio es FaceNet preentrenado en VGGFace2 (InceptionResnetV1), sin ajuste propio.** El ajuste fino con LFW no
se adopta: su mejora no es concluyente y, en las pruebas con personas reales, separó peor a los desconocidos (ver abajo).
Se compara por similitud coseno contra los embeddings registrados; el umbral final de la regla I2 **sigue sin definirse** (ver «Pendiente»).

## Notebooks (en `notebooks/`, se ejecutan en Colab, de arriba abajo)

| Notebook | Qué hace |
|---|---|
| `01_linea_base_facial.ipynb` | Línea base con LFW (embeddings congelados + SVM/k-NN, verificación 1:1) |
| `02_registro_propios.ipynb` | Registro de personas del equipo por embeddings y evaluación dejando una sesión fuera |
| `03_finetuning_facial.ipynb` | Ajuste fino con triplet loss (semi-hard), comparación contra la base y repetición con 5 semillas |

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

## Pendiente

1. **Desconocidos reales** para calibrar el umbral: 10 a 15 caras de familiares y amigos (con permiso), y las de Aldana y Erick. Subir también sus fotos para registrarlos.
2. **Probar otros métodos de comparación**: hoy se toma la foto registrada más parecida; probar el promedio de las k más parecidas o un registro con fotos de buena calidad.
3. **Filtrar por calidad** las fotos de registro (muy borrosas o con reflejo) y mejorar la detección con poca luz (hoy se pierden 5 de 18).
4. **Tres zonas en vez de un umbral** (reconocido / dudoso, decide el guardia / desconocido). Cambia la regla I2, así que primero hay que acordarlo en el equipo y escribirlo en `docs/negocio/reglas-de-negocio.md`.
5. Texto de consentimiento (I1), todavía inexistente, antes de tomar fotos de más personas.
6. Dataset de placas, DBSCAN con datos reales y exportación ONNX.

## Datos personales

Las fotos de personas con consentimiento (I1) van a un lugar **privado**, nunca a git ni a una carpeta pública. El texto de consentimiento (con versión) debe
indicar que las fotos se procesan en servicios de nube. En las pruebas de hoy el consentimiento fue un archivo `consentimiento.json` provisional.