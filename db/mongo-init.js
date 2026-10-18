// =====================================================================
// CondoHub · Servicio IA · MongoDB (base "ia")
// Reglas: I1–I6, G4 (docs/negocio/reglas-de-negocio.md)
// Uso: mongosh "mongodb://localhost:27017" db/mongo-init.js
//      (otra base, p. ej. para pruebas: IA_DB=ia_pruebas mongosh ... db/mongo-init.js)
// Idempotente: crea colecciones e índices solo si faltan; si existen, actualiza el validador.
// Las imágenes y el audio NO se guardan aquí: van a MinIO y aquí solo su clave (I6).
// =====================================================================

const ia = db.getSiblingDB(process.env.IA_DB || 'ia');

function asegurar(nombre, schema) {
  const existe = ia.getCollectionNames().includes(nombre);
  const opciones = { validator: { $jsonSchema: schema }, validationLevel: 'strict', validationAction: 'error' };
  if (existe) ia.runCommand({ collMod: nombre, ...opciones });
  else ia.createCollection(nombre, opciones);
}

const confianza = { bsonType: 'number', minimum: 0, maximum: 1 };
const embedding512 = { bsonType: 'array', minItems: 512, maxItems: 512, items: { bsonType: 'number' } };
const modeloRef = {
  bsonType: 'object', required: ['nombre', 'version'],
  properties: { nombre: { bsonType: 'string' }, version: { bsonType: 'string' } },
};

// ---------------------------------------------------------------------
// perfiles_faciales (agregado PerfilFacial · I1, I2)
// Solo residentes con consentimiento vigente tienen embeddings.
// ---------------------------------------------------------------------
asegurar('perfiles_faciales', {
  bsonType: 'object',
  required: ['residenteId', 'residenteSub', 'unidadId', 'consentimiento', 'embeddings', 'creadoEn'],
  properties: {
    residenteId: { bsonType: 'string', description: 'id de la persona en Gestión (uuid); es el del contrato ia.yaml' },
    residenteSub: { bsonType: 'string', description: 'sub de Keycloak: un RESIDENTE solo retira su propio perfil (I1)' },
    unidadId: { bsonType: 'string', description: 'id de la unidad en Gestión (uuid)' },
    unidadCodigo: { bsonType: 'string', pattern: '^[A-Z]-[0-9]{3}$', description: 'opcional, solo informativo' },
    nombre: { bsonType: 'string' },
    consentimiento: {
      bsonType: 'object',
      required: ['otorgado', 'fecha', 'versionTexto'],
      properties: {
        otorgado: { bsonType: 'bool' },
        fecha: { bsonType: 'date' },
        versionTexto: { bsonType: 'string' },
        retiradoEn: { bsonType: ['date', 'null'] },
      },
    },
    embeddings: {
      bsonType: 'array',
      items: {
        bsonType: 'object',
        required: ['vector', 'modelo', 'fotoKey', 'creadoEn'],
        properties: {
          vector: embedding512,
          modelo: modeloRef,
          fotoKey: { bsonType: 'string', description: 'objeto en MinIO' },
          calidad: confianza,
          creadoEn: { bsonType: 'date' },
        },
      },
    },
    creadoEn: { bsonType: 'date' },
    actualizadoEn: { bsonType: 'date' },
  },
  // I1: sin consentimiento vigente no puede haber embeddings
  anyOf: [
    { properties: { consentimiento: { properties: { otorgado: { enum: [true] }, retiradoEn: { bsonType: 'null' } } } } },
    { properties: { embeddings: { maxItems: 0 } } },
  ],
});
ia.perfiles_faciales.createIndex({ residenteId: 1 }, { unique: true });
ia.perfiles_faciales.createIndex({ residenteSub: 1 }, { unique: true });
ia.perfiles_faciales.createIndex({ unidadCodigo: 1 });

// ---------------------------------------------------------------------
// desconocidos (I3): rostros sin coincidencia, se borran solos a los 30 días
// ---------------------------------------------------------------------
asegurar('desconocidos', {
  bsonType: 'object',
  required: ['vector', 'modelo', 'capturadoEn', 'expiraEn'],     // `camara` es opcional, como en ia.yaml
  properties: {
    vector: embedding512,
    modelo: modeloRef,
    mejorSimilitud: confianza,
    capturadoEn: { bsonType: 'date' },
    dia: { bsonType: 'string', pattern: '^\\d{4}-\\d{2}-\\d{2}$', description: 'para contar días distintos (I4)' },
    camara: { bsonType: 'string' },
    fotoKey: { bsonType: 'string' },
    clusterId: { bsonType: ['string', 'null'] },
    expiraEn: { bsonType: 'date' },
  },
});
ia.desconocidos.createIndex({ expiraEn: 1 }, { expireAfterSeconds: 0 });      // TTL I3
ia.desconocidos.createIndex({ capturadoEn: -1 });
ia.desconocidos.createIndex({ clusterId: 1 });

// ---------------------------------------------------------------------
// clusters_desconocidos (agregado ClusterDesconocidos · I4)
// ---------------------------------------------------------------------
asegurar('clusters_desconocidos', {
  bsonType: 'object',
  required: ['clusterId', 'ejecucionId', 'apariciones', 'diasDistintos', 'primeraVez', 'ultimaVez', 'alertado'],
  properties: {
    clusterId: { bsonType: 'string' },
    ejecucionId: { bsonType: 'string', description: 'corrida nocturna de DBSCAN' },
    parametros: {
      bsonType: 'object',
      properties: { eps: { bsonType: 'number' }, minSamples: { bsonType: ['int', 'long'] }, metrica: { enum: ['cosine'] } },
    },
    miembros: { bsonType: 'array', items: { bsonType: 'objectId' } },
    apariciones: { bsonType: ['int', 'long'], minimum: 1 },
    diasDistintos: { bsonType: ['int', 'long'], minimum: 1 },
    primeraVez: { bsonType: 'date' },
    ultimaVez: { bsonType: 'date' },
    alertado: { bsonType: 'bool', description: 'I4: true cuando diasDistintos >= 3 y se publicó la alerta' },
    alertadoEn: { bsonType: ['date', 'null'] },
  },
});
ia.clusters_desconocidos.createIndex({ ejecucionId: 1, clusterId: 1 }, { unique: true });
ia.clusters_desconocidos.createIndex({ alertado: 1, ultimaVez: -1 });

// ---------------------------------------------------------------------
// predicciones (I5): toda inferencia guarda modelo, versión y confianza
// ---------------------------------------------------------------------
asegurar('predicciones', {
  bsonType: 'object',
  required: ['tipo', 'resultado', 'confianza', 'modelo', 'fecha'],
  properties: {
    tipo: { enum: ['ROSTRO', 'PLACA'] },
    resultado: {
      bsonType: 'object',
      description: 'ROSTRO: {zona: RECONOCIDO|DUDOSO|DESCONOCIDO, residenteId|null}; PLACA: {texto, bbox}',
    },
    confianza,
    umbral: confianza,
    umbralBajo: confianza,
    modelo: modeloRef,
    origen: { enum: ['CAMARA_PORTERIA', 'APP_GUARDIA', 'PRUEBA'] },
    camara: { bsonType: 'string' },
    latenciaMs: { bsonType: 'number' },
    fecha: { bsonType: 'date' },
  },
});
ia.predicciones.createIndex({ fecha: -1 });
ia.predicciones.createIndex({ tipo: 1, 'modelo.version': 1, fecha: -1 });

// ---------------------------------------------------------------------
// modelos: registro de versiones entrenadas en ml/ (métricas para el informe)
// ---------------------------------------------------------------------
asegurar('modelos', {
  bsonType: 'object',
  required: ['nombre', 'version', 'tarea', 'artefactoKey', 'metricas', 'activo', 'creadoEn'],
  properties: {
    nombre: { bsonType: 'string' },
    version: { bsonType: 'string' },
    tarea: { enum: ['RECONOCIMIENTO_FACIAL', 'DETECCION_PLACAS', 'OCR_PLACAS', 'LECTURA_PLACAS_CNN', 'CLUSTERING_DESCONOCIDOS'] },
    tipoAprendizaje: { enum: ['SUPERVISADO', 'NO_SUPERVISADO'] },
    artefactoKey: { bsonType: 'string', description: 'modelos/<nombre>/<version>/ en MinIO' },
    formato: { enum: ['PYTORCH', 'ONNX', 'ONNX_INT8', 'SKLEARN', 'KERAS', 'TFLITE'] },
    umbral: { bsonType: ['number', 'null'] },
    metricas: { bsonType: 'object', description: 'accuracy, far, frr, eer, auc, map50, silhouette...' },
    experimento: { bsonType: 'string', description: 'carpeta ml/experiments/<fecha>_<nombre>' },
    activo: { bsonType: 'bool' },
    creadoEn: { bsonType: 'date' },
  },
});
ia.modelos.createIndex({ nombre: 1, version: 1 }, { unique: true });
// un solo modelo activo por tarea
ia.modelos.createIndex({ tarea: 1 }, { unique: true, partialFilterExpression: { activo: true } });

// ---------------------------------------------------------------------
// reportes (G4): reportes generativos por voz
// ---------------------------------------------------------------------
asegurar('reportes', {
  bsonType: 'object',
  required: ['adminSub', 'estado', 'formato', 'creadoEn'],
  properties: {
    reporteId: { bsonType: 'string', description: 'uuid que expone la API (GET /reportes/{id})' },
    adminSub: { bsonType: 'string' },
    audioKey: { bsonType: 'string' },
    transcripcion: { bsonType: 'string' },
    consultaGraphql: {
      bsonType: 'string',
      pattern: '^\\s*(query\\b|\\{)',
      description: 'G4: solo lectura; nunca mutation',
    },
    formato: { enum: ['EXCEL', 'PDF'] },
    archivoKey: { bsonType: 'string' },
    estado: { enum: ['PROCESANDO', 'LISTO', 'FALLIDO'], description: 'mismos valores que ia.yaml' },
    motivoFallo: { bsonType: 'string' },
    creadoEn: { bsonType: 'date' },
    terminadoEn: { bsonType: 'date' },
  },
});
ia.reportes.createIndex({ adminSub: 1, creadoEn: -1 });

// ---------------------------------------------------------------------
// outbox (S4): eventos de dominio pendientes de publicar en RabbitMQ (exchange condohub.events)
// El cambio y el evento no comparten transacción (MongoDB sin réplica): la idempotencia sale de `eventId` único.
// ---------------------------------------------------------------------
asegurar('outbox', {
  bsonType: 'object',
  required: ['eventId', 'type', 'occurredAt', 'version', 'data', 'publicadoEn'],
  properties: {
    eventId: { bsonType: 'string', description: 'uuid del sobre; los consumidores descartan repetidos (S5)' },
    type: { bsonType: 'string', pattern: '^[a-z]+\\.[A-Z][A-Za-z]+$', description: '<contexto>.<Evento> = routing key' },
    occurredAt: { bsonType: 'date' },
    version: { bsonType: ['int', 'long'], minimum: 1 },
    data: { bsonType: 'object' },
    publicadoEn: { bsonType: ['date', 'null'], description: 'null mientras no se haya publicado' },
  },
});
ia.outbox.createIndex({ eventId: 1 }, { unique: true });
ia.outbox.createIndex({ publicadoEn: 1, occurredAt: 1 });

print('Base "ia" lista');
