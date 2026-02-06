# RAG de accesibilidad con Word + Hugging Face (modelo remoto)

Este proyecto implementa un **RAG** para preguntar sobre accesibilidad y responder **solo con tu base documental `.docx`**.

La respuesta final se genera con **Hugging Face Inference API** usando un modelo remoto (sin instalar modelos localmente), y además devuelve **trazabilidad** de evidencia usada.

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

### Tarea 4 — Generación con Hugging Face (API externa)
- Enviar solo contexto recuperado al modelo remoto.
- Forzar prompt para responder únicamente con esa evidencia.

### Tarea 5 — Trazabilidad
- Mostrar archivo + sección + score + fragmentos usados para responder.

---

## Requisitos

- Python 3.10+
- Cuenta en Hugging Face con API token (plan gratis con límites).
- Sin dependencias Python externas obligatorias.

## Cómo conseguir tu API key/token de Hugging Face

1. Crea cuenta o inicia sesión en https://huggingface.co/
2. Entra a **Settings** → **Access Tokens**.
3. Crea un token nuevo (scope de inferencia/read).
4. Copia el token.

## Configurar token

Opción A (recomendada): variable de entorno

```bash
export HF_API_KEY="tu_token_hf"
```

También soporta:

```bash
export HUGGINGFACE_API_KEY="tu_token_hf"
```

Opción B: pasarlo por CLI

```bash
--hf-api-key "tu_token_hf"
```

## Dónde copiar tus documentos Word

Crea una carpeta local dentro del proyecto (ejemplo `docs_deque/`) y copia ahí tus `.docx`:

```bash
mkdir -p docs_deque
# Copia aquí tus archivos .docx de Deque
```

## Uso

### 1) Indexar tus Word

```bash
PYTHONPATH=src python -m rag_deque.ingestion --input-dir docs_deque --output-dir data/index
```

### 2) Hacer una pregunta en lenguaje natural

```bash
PYTHONPATH=src python -m rag_deque.cli \
  --index-dir data/index \
  --question "¿Cuándo conviene usar aria-live?" \
  --hf-model google/flan-t5-large
```

También puedes pasar el token por parámetro:

```bash
PYTHONPATH=src python -m rag_deque.cli \
  --index-dir data/index \
  --question "¿Qué recomienda la documentación sobre texto alternativo?" \
  --hf-model google/flan-t5-large \
  --hf-api-key "tu_token_hf"
```

## Estructura

- `src/rag_deque/ingestion.py`: lectura de `.docx`, chunking e índice.
- `src/rag_deque/retrieval.py`: búsqueda por similitud.
- `src/rag_deque/answering.py`: llamada a Hugging Face Inference API + trazabilidad.
- `src/rag_deque/cli.py`: interfaz de línea de comandos.

## Notas

- Actualmente procesa `.docx`. Si tienes `.doc`, conviértelos a `.docx`.
- La API de Hugging Face en plan gratis tiene límites de uso.
- Si un modelo está "cold", el código espera a que cargue (`wait_for_model=true`).
