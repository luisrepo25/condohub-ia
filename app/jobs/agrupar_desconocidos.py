"""Job nocturno de I4. Se programa cada noche (cron, CronJob de Kubernetes o tareas de Windows):

    python -m app.jobs.agrupar_desconocidos

Variables: `PERSISTENCIA=mongo`, `MONGO_URI`, `MONGO_DB`, `RABBITMQ_URL`, `DBSCAN_EPS`, `DBSCAN_MIN_SAMPLES`,
`I4_MIN_APARICIONES`, `I4_MIN_DIAS`, `DESCONOCIDOS_RETENCION_DIAS`.
No imprime vectores ni fotos (I1, I6), solo conteos.
Termina con código 1 si quedaron avisos sin publicar, para que el planificador lo señale;
el siguiente intento los envía.
"""

import json
import sys
from typing import Any

from app.application.agrupar_desconocidos import AgruparDesconocidos
from app.application.publicar_outbox import PublicarOutbox
from app.config import Config
from app.infrastructure.composicion import construir_job_agrupamiento


def ejecutar(agrupar: AgruparDesconocidos, publicar: PublicarOutbox) -> dict[str, Any]:
    """Agrupa, deja los avisos nuevos en el outbox y publica todo lo pendiente (también de otras noches)."""
    agrupamiento = agrupar.ejecutar()
    publicacion = publicar.ejecutar()
    return {
        "desconocidos": agrupamiento.desconocidos,
        "visitantes_recurrentes": agrupamiento.grupos,
        "alertas_nuevas": agrupamiento.alertas,
        "avisos_publicados": publicacion.publicados,
        "avisos_pendientes": publicacion.pendientes,
    }


def main() -> int:
    agrupar, publicar = construir_job_agrupamiento(Config.desde_entorno())
    resumen = ejecutar(agrupar, publicar)
    print(json.dumps(resumen, ensure_ascii=False))
    return 1 if resumen["avisos_pendientes"] else 0


if __name__ == "__main__":
    sys.exit(main())
