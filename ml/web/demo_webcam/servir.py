"""Sirve services/ia por http://localhost:8090 (la webcam exige localhost o https) con los tipos MIME que necesitan los módulos y el WASM.

Uso, desde services/ia:  .venv\\Scripts\\python ml\\web\\demo_webcam\\servir.py
Luego abrir http://localhost:8090/ml/web/demo_webcam/index.html
"""

import http.server
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]          # services/ia
PUERTO = 8090


class Manejador(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".bin": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        # Aislamiento de origen: permite SharedArrayBuffer y, con él, varios hilos de WebAssembly (onnxruntime-web ~2x más rápido).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(RAIZ)
    print(f"Sirviendo {RAIZ} en http://localhost:{PUERTO}/ml/web/demo_webcam/index.html")
    http.server.ThreadingHTTPServer(("127.0.0.1", PUERTO), Manejador).serve_forever()
