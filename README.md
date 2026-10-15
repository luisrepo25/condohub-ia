# condohub-ia

Servicio IA de CondoHub: visión (rostros, placas) y voz (reportes). FastAPI · Python 3.11 · Clean.

```bash
py -3.11 -m venv .venv && .venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn app.main:app --port 8083
.venv\Scripts\python -m pytest
```

Arquitectura, reglas, estado, roles por endpoint y variables: `CLAUDE.md`.