# condohub-ia

Servicio IA de CondoHub: visión (rostros, placas) y voz (reportes). FastAPI · Python 3.11 · Clean.

```bash
py -3.11 -m venv .venv && .venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn app.main:app --port 8083
.venv\Scripts\python -m pytest
```

Prueba de lectura de placas con la webcam (CNN de TensorFlow en el navegador; requiere los pesos `modelos/placas_cnn_pesos.*`, que no van a git):

```bash
.venvScriptspython mlwebdemo_webcamservir.py   # http://localhost:8090/ml/web/demo_webcam/index.html
```

Pruebas de integración (MongoDB, S3 y RabbitMQ reales): `docker compose -f ../../infra/docker-compose.yml up -d mongo minio rabbitmq` y `.venvScriptspython -m pytest tests/integracion`.

Arquitectura, reglas, estado, roles por endpoint y variables: `CLAUDE.md`.