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
| I2 | Reconocido = similitud coseno ≥ umbral calibrado (inicial 0.70; valor final documentado en `ml/README.md`) |
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

`POST /rostros/identificar` · `POST /rostros/registrar` · `DELETE /rostros/{residenteId}` (retiro de consentimiento) · `POST /placas/leer` · `GET /modelos/placas/actual` (URL del ONNX para la PWA) · `POST /reportes/voz` (audio → archivo) · `GET /reportes/{id}`.
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

- Roles: `GET /salud` cualquiera · `POST /rostros/identificar`, `POST /placas/leer`, `GET /modelos/placas/actual`: GUARDIA, ADMIN · `POST /rostros/registrar`, `POST /reportes/voz`, `GET /reportes/{id}`: ADMIN · `DELETE /rostros/{residenteId}`: ADMIN, RESIDENTE.
- `TODO(regla)`: el RESIDENTE solo debe retirar SU consentimiento (I1); falta definir cómo se enlaza el `sub` del token con `residenteId`.
- JWT: `app/infrastructure/keycloak.py` (JWKS, RS256, `exp`, `iss`); dependencia `requiere_rol(...)` en `app/api/seguridad.py`; errores problem+json en `app/api/problemas.py`. Un test compara las rutas con `ia.yaml`.
- Comandos reales (no hay `uv` en la máquina de Luis): `py -3.11 -m venv .venv`, `.venv\Scripts\python -m pip install -e ".[dev]"`, `... -m uvicorn app.main:app --port 8083`, `... -m pytest` (38 tests), `... -m ruff check .`, `... -m mypy`. Sin torch todavía.
- Variables: `KEYCLOAK_ISSUER`, `KEYCLOAK_JWKS`. Por el gateway: `http://localhost:8080/ia/salud` con Bearer.
- ML en Google Colab (`ml/README.md`): línea base, registro con fotos propias y ajuste fino **ejecutados el 15/10/2026**. Decisión vigente: el servicio usa FaceNet preentrenado (VGGFace2) **sin ajuste propio**; el ajuste fino no mejoró de forma concluyente (IC95 de la mejora cruza el 0) y separó peor a desconocidos reales. El umbral de I2 aún no está definido: con LFW el FAR era 0,12 %, pero con amigos reales aceptó a 2 de 3 (hace falta calibrar con desconocidos reales y probablemente tres zonas, lo que cambia I2). Falta: más fotos y desconocidos, texto de consentimiento, placas, DBSCAN con datos reales, ONNX.

## Código funcional protegido (obligatorio)

Código en `main` con tests en verde **no se reescribe, refactoriza, renombra ni reformatea** si la tarea no lo pide explícitamente. Lo nuevo se agrega extendiendo (clases, métodos o archivos nuevos). Si es inevitable tocar código funcional, **detente antes de editar**, explica qué archivo/líneas y por qué, y espera confirmación; luego haz el mínimo diff. Nunca edites tests existentes para que pase código nuevo. Corre los tests antes y después: todo lo que pasaba debe seguir pasando. En el resumen, lista cada archivo existente que modificaste.
