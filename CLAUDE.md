# CLAUDE.md — condohub-ia

**Servicio IA** de CondoHub: contexto de soporte **IA**. Dos capacidades: **visión** (reconocimiento facial de residentes, lectura de placas, clustering de desconocidos) y **voz** (reportes generativos por voz a Excel/PDF). Incluye `ml/`, donde se entrenan y evalúan los modelos. Responsable: **Luis**.

Este repo es independiente. Si está dentro del espacio de trabajo `condohub`, las reglas completas están en `../../docs/negocio/reglas-de-negocio.md` y el contrato en `../../contracts/openapi/ia.yaml`.

## Stack

Python 3.11 · FastAPI · Pydantic v2 · PyTorch · facenet-pytorch (MTCNN + InceptionResnetV1) · scikit-learn (SVM/k-NN, DBSCAN, métricas) · Ultralytics YOLOv8 + EasyOCR (placas) · ONNX / onnxruntime · openai-whisper o faster-whisper · Motor (MongoDB) · MinIO (imágenes y pesos) · openpyxl + WeasyPrint/ReportLab · pytest · ruff · mypy. Dependencias con `uv`.

## Estructura

```
app/                          # el microservicio (Clean)
├── domain/                   # puro: PerfilFacial, Consentimiento, Embedding, Identificacion, ClusterDesconocidos, Confianza
│   └── ports/                # EmbeddingRepository, ModeloFacial, ModeloPlacas, Transcriptor, GeneradorConsulta, ...
├── application/              # IdentificarRostro, RegistrarRostro, RetirarConsentimiento, LeerPlaca,
│                             # AgruparDesconocidos (job nocturno), GenerarReportePorVoz
├── infrastructure/
│   ├── models/               # carga de pesos versionados desde MinIO; inferencia PyTorch/ONNX
│   ├── mongo/                # repositorios
│   ├── graphql_client/       # cliente de Gestión, SOLO queries
│   └── exporters/            # Excel y PDF
└── api/                      # routers FastAPI, validación JWT, esquemas Pydantic
ml/                           # entrenamiento (no se despliega con el servicio)
├── data/                     # IGNORADO por git: datasets crudos y procesados
├── notebooks/                # exploración; numerados 01_..., 02_...
├── src/                      # código reutilizable: dataset.py, preprocess.py, train_facial.py, train_placas.py,
│                             # evaluate.py, cluster.py, export_onnx.py
├── configs/                  # YAML de cada experimento (semilla, hiperparámetros, dataset)
├── experiments/              # métricas y gráficos por experimento (sí se versionan; los pesos no)
└── README.md                 # resultados, umbral elegido, decisiones
```

`app/domain` y `app/application` no importan `torch`, `fastapi`, `motor` ni `sklearn`.

## Reglas de negocio que implementa

| Código | Resumen |
|---|---|
| I1 | Solo rostros de residentes con consentimiento explícito (fecha + versión del texto). Retiro → borrar fotos y embeddings en < 24 h |
| I2 | Tres zonas por similitud coseno: ≥ alto (0.70) `RECONOCIDO`; entre bajo (0.50) y alto `DUDOSO` (decide el guardia, no se guarda como desconocido); < bajo `DESCONOCIDO`. Valores finales en `ml/README.md` |
| I3 | Embeddings de desconocidos se guardan 30 días (índice TTL en MongoDB) |
| I4 | DBSCAN nocturno (coseno) sobre desconocidos de 30 días; clúster con ≥ 3 apariciones en días distintos → evento `VisitanteRecurrenteDetectado` |
| I5 | Toda predicción guarda modelo, versión, confianza y fecha |
| I6 | Datasets, fotos y pesos fuera de git |
| A4/A5 (soporte) | Devuelve confianza; la decisión de abrir la toma Acceso, nunca este servicio |
| G4 | Reportes por voz solo `ADMIN`; Whisper → LLM → GraphQL **solo lectura** → Excel/PDF; guardar texto, consulta y archivo |

## Machine Learning (ml/)

**Supervisado: reconocimiento facial**
1. Pipeline: MTCNN (detección y alineación, 160×160) → InceptionResnetV1 preentrenado en VGGFace2 → embedding 512-d normalizado L2.
2. Línea base: embeddings congelados + clasificador (SVM lineal y k-NN). Es la referencia contra la que se compara todo.
3. Entrenamiento: fine-tuning de las últimas capas con **triplet loss** (minado semi-hard) o ArcFace sobre el dataset propio con consentimiento. Aumentos: brillo, desenfoque, giro leve (simulan la cámara de portería).
4. Evaluación en un split por **persona** (nunca la misma persona en train y test para verificación): accuracy, FAR, FRR, curva ROC, EER, matriz de confusión. Calibrar el umbral I2 en validación, no en test.

**Supervisado: placas**: YOLOv8n fine-tuned para detectar la placa + OCR (EasyOCR) + validación con el formato de placa. Métricas: mAP@0.5 de detección y precisión por carácter / placa completa. Export a ONNX cuantizado (int8) para la PWA del guardia; comprobar que el ONNX da el mismo resultado que PyTorch dentro de una tolerancia.

**No supervisado: DBSCAN** sobre embeddings de desconocidos, métrica coseno. Elegir `eps` con el gráfico k-distance y `min_samples` = 3; reportar silhouette y número de clústeres/ruido. Validar con un conjunto donde se sabe quién se repite.

Reglas de ML:
- Semillas fijas y config YAML por experimento; cada corrida guarda en `ml/experiments/<fecha>_<nombre>/` sus métricas (`metrics.json`) y gráficos.
- Nunca reportar métricas sobre datos usados para entrenar o calibrar.
- Un modelo pasa al servicio solo si supera a la línea base; se sube a MinIO como `modelos/<nombre>/<version>/` y el servicio lo carga por versión (variable de entorno).
- Entrenamiento pesado en Colab/GPU: los notebooks llaman a `ml/src`, no duplican código.

## Base de datos

MongoDB `ia`, colecciones con validador `$jsonSchema` e índices en `db/mongo-init.js` (idempotente): `perfiles_faciales`, `desconocidos` (TTL 30 días), `clusters_desconocidos`, `predicciones`, `modelos`, `reportes`. Imágenes, audio y pesos solo en MinIO.

## API

`POST /rostros/identificar` · `POST /rostros/registrar` · `DELETE /rostros/{residenteId}` (retiro de consentimiento) · `POST /placas/leer` · `GET /modelos/placas/actual` (URL del ONNX para la PWA) · `GET /modelos/lectura-placas/actual` (URLs del reconocedor de texto y de su ficha) · `POST /reportes/voz` (audio → archivo) · `GET /reportes/{id}`.
Respuesta de visión siempre con `{ resultado, confianza, modelo, version }`. Acceso las traduce con su capa anticorrupción: no metas conceptos de portería aquí.

## Comandos

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8083
uv run pytest
uv run ruff check . && uv run mypy app
uv run python -m ml.src.train_facial --config ml/configs/facial_triplet.yaml
uv run python -m ml.src.evaluate --config ml/configs/facial_triplet.yaml
uv run python -m ml.src.export_onnx --modelo placas
uv run python -m app.jobs.agrupar_desconocidos     # job DBSCAN (I4)
```

## Convenciones locales

- Tests: `test_I2_rostro_bajo_umbral_es_desconocido`. Los tests no descargan modelos: usan dobles del puerto `ModeloFacial`.
- Nunca loguear imágenes, embeddings ni audio; solo ids y confianza.
- El LLM de reportes nunca recibe permisos de mutation: el cliente GraphQL rechaza cualquier operación que no sea `query`.

## Estado y seguridad (esqueleto, 14/10/2026)

Los endpoints de `ia.yaml` existen y ya están protegidos por rol; su lógica de ML no: responden `501` (`NO_IMPLEMENTADO`), a quitar al implementar cada uno.

- Roles: `GET /salud` cualquiera · `POST /rostros/identificar`, `POST /placas/leer`, `GET /modelos/placas/actual`, `GET /modelos/lectura-placas/actual`: GUARDIA, ADMIN · `POST /rostros/registrar`, `POST /reportes/voz`, `GET /reportes/{id}`: ADMIN · `DELETE /rostros/{residenteId}`: ADMIN, RESIDENTE.
- Enlace token → residente (cerrado): `POST /rostros/registrar` recibe `residenteId` (uuid de Gestión) y `residenteSub`; el perfil guarda ambos y `DELETE /rostros/{residenteId}` deja pasar a un RESIDENTE solo si su `sub` es el `residenteSub` del perfil. Aún por implementar en el caso de uso.
- JWT: `app/infrastructure/keycloak.py` (JWKS, RS256, `exp`, `iss`); dependencia `requiere_rol(...)` en `app/api/seguridad.py`; errores problem+json en `app/api/problemas.py`. Un test compara las rutas con `ia.yaml`.
- Comandos reales (no hay `uv` en la máquina de Luis): `py -3.11 -m venv .venv`, `.venv\Scripts\python -m pip install -e ".[dev]"`, `... -m uvicorn app.main:app --port 8083`, `... -m pytest` (38 tests), `... -m ruff check .`, `... -m mypy`. Sin torch todavía.
- Variables: `KEYCLOAK_ISSUER`, `KEYCLOAK_JWKS`. Por el gateway: `http://localhost:8080/ia/salud` con Bearer.
- **Implementado (16/10/2026): `POST /rostros/registrar`, `POST /rostros/identificar` y `DELETE /rostros/{residenteId}`** (I1, I2, I3, I5, A5; ya no responden 501). Dominio `domain/rostros.py` (`PerfilFacial`, `Consentimiento`, `Umbrales`/`Zona`, `ReconocedorFacial`, `Desconocido`, `Prediccion`), casos de uso `application/rostros.py` (`RegistrarRostro`, `IdentificarRostro`, `RetirarConsentimiento`) con sus puertos, traducción de errores en `api/traduccion.py`. Reglas aplicadas: sin versión o fecha de consentimiento → 422 `code: I1` (aunque falten del formulario); un RESIDENTE solo retira el perfil cuyo `residenteSub` es su `sub` (si no, 403); retirar es idempotente y borra fotos y embeddings al momento; solo los DESCONOCIDOS se guardan 30 días (un DUDOSO no); toda identificación guarda su predicción. Umbrales y plazos por variable de entorno: `UMBRAL_RECONOCIDO` (0.70), `UMBRAL_DUDOSO` (0.50), `DESCONOCIDOS_RETENCION_DIAS` (30), `RETIRO_PLAZO_HORAS` (24). **Persistencia (16/10/2026):** `PERSISTENCIA=memoria` (por defecto; se pierde al reiniciar) o `PERSISTENCIA=mongo` con `MONGO_URI` (`mongodb://localhost:27017`, el compose no usa contraseña) y `MONGO_DB` (`ia`); driver **PyMongo síncrono** (decisión: los casos de uso son síncronos; se aparta de «Motor» del stack) con `pip install -e ".[persistencia]"`. Repositorios en `infrastructure/mongo/` (`documentos.py` traduce dominio ↔ documentos con los nombres del validador de `db/mongo-init.js`); un fallo de MongoDB responde 503. Pruebas: `tests/test_mongo_*.py` (sin servidor) y `tests/integracion/` contra un MongoDB real (se omiten solas si no hay): `docker compose -f infra/docker-compose.yml up -d mongo`, luego `docker compose -f infra/docker-compose.yml exec -T -e IA_DB=ia_pruebas mongo mongosh --quiet /docker-entrypoint-initdb.d/01-ia.js` y `pytest tests/integracion`. **Ojo:** el compose solo corre `mongo-init.js` al crear el volumen; si cambia el script en una base existente hay que ejecutarlo a mano (idempotente) con el comando anterior sin `IA_DB`. Las fotos siguen en memoria (falta S3).
**Reconocedor de texto de placas (16/10/2026):** `GET /modelos/lectura-placas/actual` entrega las URLs firmadas del ONNX del reconocedor (EasyOCR exportado, notebook 11 de `ml/`) y de su ficha (`ficha_reconocedor.json`: caracteres del modelo, preproceso y decodificación) con el mismo vencimiento; contrato `ModeloLecturaPlacas` en `contracts/openapi/ia.yaml`. Variable `MODELO_LECTURA_FICHA` (ruta de la ficha; sin ella responde 404). Publicarlo: `python -m app.infrastructure.publicar_lectura <ficha_reconocedor.json> <reconocedor_placas.onnx>` (valida el sha256 y sube los dos archivos a `modelos/lectura-placas/<version>/`). Código: `domain/modelo_lectura.py`, `application/modelo_lectura.py`, `infrastructure/{catalogo_ficha_lectura,publicar_lectura,composicion_lectura}.py`, `api/routers/lectura_placas.py`. La PWA lee la placa con `ml/web/lectura_placas.mjs` (votación entre franjas; A4).

**Fotos y almacén S3 (16/10/2026):** `ALMACEN_FOTOS=memoria` (por defecto) o `s3`: bucket privado `S3_BUCKET_ROSTROS` (`rostros`) del almacén del compose (**SeaweedFS local, no AWS**; `S3_ENDPOINT` `http://localhost:9000`, credenciales de desarrollo de `infra/seaweedfs/s3.json`: `S3_ACCESS_KEY`/`S3_SECRET_KEY`, o `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`). `infrastructure/s3_cliente.py` (`ClienteS3` con URLs prefirmadas SigV4 de PUT, GET y DELETE, solo biblioteca estándar; `AlmacenFotosS3`: si una subida falla a medias borra las ya subidas, y borrar es idempotente). **Verificado contra el SeaweedFS real:** las URLs prefirmadas SigV4 (PUT, GET y DELETE) que el ADR 0002 daba por no probadas funcionan, y una firma con otra clave o ya vencida da 403. Publicar el modelo de placas: `python -m app.infrastructure.publicar_modelo <modelo_placas.json> <archivo.onnx>` (valida el sha256 de la ficha); ya publicado `modelos/placas/1.0.0/yolov8n_placas_int8.onnx` y comprobado que el archivo descargado por la URL del endpoint coincide con el hash. Prueba de extremo a extremo (`tests/integracion/test_s3_real.py`): registrar guarda las fotos en el almacén y los embeddings en MongoDB; **retirar el consentimiento borra las dos cosas** (I1).
**Job nocturno de I4 (16/10/2026): `python -m app.jobs.agrupar_desconocidos`** (programarlo cada noche con cron, un CronJob de Kubernetes o el Programador de tareas de Windows; no hay planificador dentro del servicio). Lee los desconocidos que no vencen (I3), los agrupa con DBSCAN coseno (`infrastructure/dbscan.py`, scikit-learn; `eps` 0,30 del notebook 05) y, por cada visitante con ≥ 3 apariciones en ≥ 3 días distintos, deja un aviso en la colección `outbox` y lo publica en RabbitMQ (`infrastructure/rabbitmq.py`, pika) como `ia.VisitanteRecurrenteDetectado` (exchange `condohub.events`; la cola `n8n.eventos` ya lo recibe). El mismo visitante conserva su `clusterId` entre noches (los desconocidos se marcan con él) y **se avisa una sola vez**; el `eventId` sale del visitante (uuid5), así un reintento no duplica (S5). Orden a prueba de caídas: marcar miembros → outbox → registrar visitante. Si RabbitMQ está caído el aviso espera en el outbox y el comando sale con código 1; el siguiente intento lo envía. MongoDB sin réplica no da transacciones: la idempotencia viene del `eventId` único. Casos de uso `application/agrupar_desconocidos.py` y `publicar_outbox.py`; dominio `domain/agrupamiento.py`; adaptadores `infrastructure/mongo/agrupamiento.py`. Variables: `PERSISTENCIA=mongo` (obligatoria), `MONGO_URI`, `MONGO_DB`, `RABBITMQ_URL` (por defecto el usuario de desarrollo `condohub`), `RABBITMQ_EXCHANGE`, `DBSCAN_EPS`, `DBSCAN_MIN_SAMPLES`, `I4_MIN_APARICIONES`, `I4_MIN_DIAS`. Dependencias nuevas: `scikit-learn` (extra `inferencia`/`ml`) y `pika` (RabbitMQ); `jsonschema` solo para la prueba de contrato. El contrato del evento ya se valida contra `contracts/events/` en `tests/test_contrato_eventos.py`.
**Estado honesto:** con los valores por defecto (`memoria`) nada se guarda; para persistir hay que arrancar con `PERSISTENCIA=mongo ALMACEN_FOTOS=s3`. El servicio aún no tiene `/placas/leer` (501). **Reportes por voz (G4) implementados:** `POST /reportes/voz` transcribe con faster-whisper (local, `WHISPER_MODELO=base`; extra `voz`), pide la consulta al LLM por REST (`LLM_PROVEEDOR=gemini` con `GEMINI_API_KEY` y `LLM_MODELO`, o `groq` con `GROQ_API_KEY` y `GROQ_MODELO=openai/gpt-oss-20b`; sin clave → 503; Groq fue el más rápido y estable en las pruebas, Gemini dio 503 y esperas intermitentes), comprueba con `graphql-core` que solo haya `query` (si no, 422 `G4`, y queda un reporte `FALLIDO` de rastro) y ejecuta la consulta en Gestión (`GESTION_GRAPHQL_URL`) con el token del propio ADMIN; el Excel (openpyxl) o PDF (reportlab) se genera en segundo plano y `GET /reportes/{id}` entrega una URL firmada (`ALMACEN_REPORTES=memoria|s3`, bucket `reportes`). Verificado de punta a punta por el gateway con Whisper, el LLM (Groq y Gemini), Gestión, MongoDB y S3 reales: un reporte de morosos en PDF queda `LISTO` en ~4 s y se descarga con la URL firmada; una orden de escritura no llega a Gestión (422) y un RESIDENTE recibe 403. El modelo facial real es `infrastructure/facenet.py` (MTCNN + FaceNet VGGFace2, el mismo pipeline que `ml/`; carga PyTorch al primer uso); se activa con `MODELO_FACIAL=facenet` (`MODELO_FACIAL_VERSION`, `MODELO_FACIAL_DISPOSITIVO`) y requiere `pip install -e ".[inferencia]"` más `pip install facenet-pytorch --no-deps`; sin configurar, el servicio responde **503**. Se valida en Colab con el notebook `ml/notebooks/08_probar_servicio_facial.ipynb`. Falta: guardado y borrado de fotos en S3. `TODO(regla)`: registrar de nuevo a un residente vigente hoy **amplía** su perfil; confirmar si debe reemplazarlo.
- **Implementado (16/10/2026): `GET /modelos/placas/actual`** (A4, O6; ya no responde 501; los demás endpoints siguen en 501). Caso de uso `application/modelos.py` (`ObtenerModeloPlacas`), dominio `domain/modelos.py`, adaptadores `infrastructure/catalogo_ficha.py` (lee `modelo_placas.json`), `s3_firmador.py` y `sigv4.py` (URL prefirmada SigV4 con la biblioteca estándar, verificada con el ejemplo oficial de AWS). Sin modelo publicado → 404; almacén sin credenciales → 503. Variables: `MODELO_PLACAS_FICHA` (ruta de la ficha), `S3_ENDPOINT` (por defecto `http://localhost:9000`), `S3_ACCESS_KEY` y `S3_SECRET_KEY` (o `MINIO_ROOT_USER` y `MINIO_ROOT_PASSWORD`), `S3_BUCKET_MODELOS` (`modelos`), `S3_REGION`, `URL_FIRMADA_SEGUNDOS` (900). Publicar el modelo: subir el `.onnx` a `s3://modelos/placas/<version>/<archivo>` y apuntar `MODELO_PLACAS_FICHA` a su ficha. **Sin probar contra SeaweedFS real** (ADR 0002: faltan las URLs firmadas).
- ML en Google Colab (`ml/README.md`): línea base, registro con fotos propias y ajuste fino **ejecutados el 15/10/2026**. Decisión vigente: el servicio usa FaceNet preentrenado (VGGFace2) **sin ajuste propio**; el ajuste fino no mejoró de forma concluyente (IC95 de la mejora cruza el 0) y separó peor a desconocidos reales. El umbral de I2 aún no está definido: con LFW el FAR era 0,12 %, pero con amigos reales aceptó a 2 de 3 (hace falta calibrar con desconocidos reales y probablemente tres zonas, lo que cambia I2). Falta: más fotos y desconocidos, texto de consentimiento, placas, DBSCAN con datos reales, ONNX.

## Código funcional protegido (obligatorio)

Código en `main` con tests en verde **no se reescribe, refactoriza, renombra ni reformatea** si la tarea no lo pide explícitamente. Lo nuevo se agrega extendiendo (clases, métodos o archivos nuevos). Si es inevitable tocar código funcional, **detente antes de editar**, explica qué archivo/líneas y por qué, y espera confirmación; luego haz el mínimo diff. Nunca edites tests existentes para que pase código nuevo. Corre los tests antes y después: todo lo que pasaba debe seguir pasando. En el resumen, lista cada archivo existente que modificaste.
