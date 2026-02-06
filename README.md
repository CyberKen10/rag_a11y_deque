# RAG de accesibilidad con Word + Mistral AI (API externa gratis con límites)

Este proyecto implementa un **RAG** para preguntar sobre accesibilidad y responder **solo con tu base documental `.docx`**.

La respuesta final se genera con **Mistral AI (API externa)** usando su plan gratuito (con límites), y además devuelve **trazabilidad** de evidencia usada.

## Plan por tareas

### Tarea 1 — Ingesta de documentos Word
- Leer automáticamente todos los `.docx` de una carpeta.
- Partir el contenido en chunks reutilizables para búsqueda.

### Tarea 2 — Indexación vectorial gratuita
- Crear embeddings TF‑IDF en Python estándar.
- Guardar chunks e índice para reutilizarlo.

### Tarea 3 — Recuperación (retrieval)
- Buscar los chunks más similares a la pregunta.
- Ordenar por score de similitud coseno.

### Tarea 4 — Generación de respuesta con Mistral AI
- Enviar solo contexto recuperado al LLM.
- Forzar prompt para responder únicamente con esa evidencia.

### Tarea 5 — Trazabilidad
- Mostrar archivo + sección + score + fragmentos usados para responder.

---

## Requisitos

- Python 3.10+
- Cuenta en Mistral AI con API key (plan gratis con límites).
- Sin dependencias Python externas obligatorias.

## Configurar API key

Opción A (recomendada): variable de entorno

```bash
export MISTRAL_API_KEY="tu_api_key"
```

Opción B: pasarla por CLI en cada ejecución

```bash
--mistral-api-key "tu_api_key"
```

## Uso

### 1) Indexar tus Word

```bash
PYTHONPATH=src python -m rag_deque.ingestion --input-dir /ruta/a/tus/word --output-dir data/index
```

### 2) Hacer una pregunta

```bash
PYTHONPATH=src python -m rag_deque.cli \
  --index-dir data/index \
  --question "¿Cuándo conviene usar aria-live?" \
  --mistral-model mistral-small-latest
```

También puedes pasar la key por parámetro:

```bash
PYTHONPATH=src python -m rag_deque.cli \
  --index-dir data/index \
  --question "¿Cuándo conviene usar aria-live?" \
  --mistral-model mistral-small-latest \
  --mistral-api-key "tu_api_key"
```

## Estructura

- `src/rag_deque/ingestion.py`: lectura de `.docx`, chunking e índice.
- `src/rag_deque/retrieval.py`: búsqueda por similitud.
- `src/rag_deque/answering.py`: llamada a Mistral AI + trazabilidad.
- `src/rag_deque/cli.py`: interfaz de línea de comandos.

## Notas

- Actualmente procesa `.docx`. Si tienes `.doc`, conviértelos a `.docx`.
- La API de Mistral en plan gratis tiene límites de uso.
