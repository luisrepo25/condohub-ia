# Servicio IA de CondoHub (FastAPI). Imagen de producción: sin pruebas, sin ml/ y sin datos (I6).
# Construir:  docker build -t condohub-ia services/ia
# Sin el modelo facial (imagen más liviana):  docker build --build-arg INFERENCIA=false -t condohub-ia services/ia
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

ARG INFERENCIA=true
# PyTorch solo de CPU (la versión de PyPI trae CUDA y pesa varios GB); facenet-pytorch fija versiones viejas de torch,
# por eso va aparte y sin dependencias.
RUN if [ "$INFERENCIA" = "true" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY pyproject.toml ./
COPY app ./app
RUN if [ "$INFERENCIA" = "true" ]; then EXTRAS="persistencia,agrupamiento,voz,reportes,placas,inferencia"; \
    else EXTRAS="persistencia,agrupamiento,voz,reportes,placas"; fi \
 && pip install ".[${EXTRAS}]" \
 && if [ "$INFERENCIA" = "true" ]; then pip install --no-deps facenet-pytorch; fi

# El modelo de Whisper y las fichas de los modelos se montan o se descargan fuera de la imagen
ENV HF_HOME=/models/huggingface
RUN useradd --system --uid 10001 condohub && mkdir -p /models && chown -R condohub /models
USER condohub

EXPOSE 8083
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8083), 2)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8083"]