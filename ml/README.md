# ml/ — entrenamiento y evaluación (Google Colab)

El entrenamiento pesado se hace en **Google Colab** (GPU); en el repo vive el código reutilizable y los resultados.
Los notebooks llaman a `ml/src` y no duplican código. Datasets, fotos y pesos **no se versionan** (I6): viven en Google Drive.

## Flujo

1. Abre `notebooks/01_linea_base_facial.ipynb` en Colab y activa la GPU (Entorno de ejecución → Cambiar tipo).
2. Monta Drive: los datos quedan en `MyDrive/condohub-ml` (variable `ML_DATA_DIR`).
3. El notebook clona el repo (o copia el código desde Drive), instala `facenet-pytorch --no-deps`, descarga LFW y corre la línea base.
4. Copia `metrics.json` y `roc.png` desde la carpeta del experimento a `ml/experiments/<fecha>_<nombre>/` (esos **sí** se versionan).
5. Anota abajo el umbral elegido y las métricas.

## Qué se puede probar sin GPU

`ml/src/split.py`, `metricas.py`, `cluster.py` y `linea_base.py` no necesitan PyTorch: se prueban en local con embeddings sintéticos
(`pip install -e ".[dev,ml]"` y `pytest ml/tests`). Solo `embeddings.py` necesita torch y se ejecuta en Colab.

## Datos personales

Las fotos de personas con consentimiento (I1) van a una carpeta **privada** del Drive del equipo, nunca a git ni a una carpeta compartida
pública. El texto de consentimiento (con su versión) debe mencionar que las fotos se procesan en servicios de nube. Todavía no existe ese texto.

## Resultados

_Pendiente: aún no se ha corrido la línea base en Colab._ Cuando se corra, registrar aquí: dataset, nº de personas y fotos,
exactitud (SVM y k-NN), EER y FAR/FRR en prueba, y el **umbral final de la regla I2** (inicial 0.70).