# Documentación Técnica: Sistema de Retrieval Mejorado para Tablas Complejas

## Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura General del Sistema](#arquitectura-general-del-sistema)
3. [Flujo de Detección y Clasificación de Tablas](#flujo-de-detección-y-clasificación-de-tablas)
4. [Procesamiento de Imágenes y Document Intelligence](#procesamiento-de-imágenes-y-document-intelligence)
5. [Integración con Azure AI Search](#integración-con-azure-ai-search)
6. [Componentes y Responsabilidades](#componentes-y-responsabilidades)
7. [Implementación Técnica](#implementación-técnica)
8. [Configuración del Sistema](#configuración-del-sistema)
9. [Métricas y Monitoreo](#métricas-y-monitoreo)

---

## Resumen Ejecutivo

El Sistema de Retrieval Mejorado para Tablas Complejas es una extensión del Orchestrator IDM que permite el procesamiento y comprensión avanzada de estructuras tabulares durante el proceso de recuperación de información. Este sistema NO realiza ingesta de datos, sino que procesa y enriquece la información en tiempo real durante el retrieval.

### Capacidades Principales

- Detección automática de queries relacionadas con tablas
- Extracción de estructura de tablas desde HTML y imágenes
- Clasificación de complejidad tabular
- Procesamiento de características avanzadas (headers jerárquicos, celdas fusionadas, tablas sin bordes)
- Análisis semántico mediante IA multimodal

### Diferenciación Clave

Este sistema opera exclusivamente durante el **retrieval**, no durante la ingesta. Toda la extracción, procesamiento y enriquecimiento de tablas ocurre en tiempo real cuando se ejecuta una consulta.

---

## Arquitectura General del Sistema

### Flujo Principal de Procesamiento

```
Usuario → Query → Orchestrator IDM → Azure AI Search → Procesamiento de Tablas → Respuesta Enriquecida
```

### Componentes del Sistema

1. **Azure AI Search**: Proporciona los resultados base del retrieval, incluyendo texto y URLs de imágenes
2. **Table Processor**: Extrae y procesa estructuras de tablas desde HTML/texto
3. **Document Intelligence**: Analiza imágenes para extraer estructuras tabulares
4. **Mistral OCR**: Proporciona comprensión semántica de contenido visual
5. **Complex Table Handler**: Procesa características específicas de tablas complejas
6. **Agentic Search**: Ejecuta búsquedas paralelas para queries complejas

---

## Flujo de Detección y Clasificación de Tablas

### Nivel 1: Detección de Query de Tabla

**Archivo**: `orc/code_orchestration_enhanced.py`  
**Función**: `detect_table_query(query: str, context: Dict) -> bool`

#### Mecanismo de Detección

El sistema identifica queries relacionadas con tablas mediante análisis de palabras clave:

```python
table_indicators = [
    # Términos directos
    "table", "tabla", "quadro",
    "column", "columna", "coluna",
    "row", "fila", "linha",
    "cell", "celda", "célula",
    
    # Términos estructurales
    "header", "encabezado", "cabeçalho",
    "merged", "fusionada", "mesclada",
    "hierarchical", "nested", "jerárquico",
    
    # Términos de cálculo
    "total", "sum", "aggregate",
    
    # Términos de negocio específicos
    "authorization level", "approval level"
]
```

### Nivel 2: Extracción de Tablas del Contenido

#### Opción A: Extracción desde HTML/Texto

**Componente**: `TableStructureExtractor`  
**Archivo**: `shared/table_processor.py`

```python
class TableStructureExtractor:
    def extract_tables_from_html(self, html: str) -> List[Dict]:
        # Busca patrones de tabla HTML
        table_pattern = r'<table[^>]*>.*?</table>'
        
        # Para cada tabla encontrada
        for table_html in tables_found:
            structure = self.extract_table_structure(table_html)
            # Extrae: células, headers, merged cells, metadata
```

#### Opción B: Extracción desde Imágenes

**Componente**: `DocumentIntelligenceTableExtractor`  
**Archivo**: `shared/document_intelligence_integration.py`

```python
class DocumentIntelligenceTableExtractor:
    async def extract_tables_from_document(self, image_url: str) -> List[Dict]:
        # Azure Document Intelligence analiza la imagen
        # Detecta tablas mediante análisis visual
        # Retorna estructura completa
```

### Nivel 3: Clasificación de Complejidad

**Componente**: `ComplexTableInterpreter`  
**Archivo**: `shared/complex_table_handler.py`

#### Sistema de Puntuación de Complejidad

```python
def _assess_complexity(self, table_data: Dict) -> TableComplexity:
    score = 0
    
    # Asignación de puntos por característica
    if has_merged_cells:         score += 2
    if has_hierarchical_headers: score += 2
    if has_nested_structure:     score += 3
    if is_borderless:            score += 1
    
    # Puntos por dimensiones
    if rows > 50 or cols > 20:   score += 2
    elif rows > 20 or cols > 10: score += 1
    
    # Clasificación final
    if score >= 6: return TableComplexity.VERY_HIGH
    if score >= 4: return TableComplexity.HIGH
    if score >= 2: return TableComplexity.MEDIUM
    return TableComplexity.SIMPLE
```

### Nivel 4: Procesamiento Especializado por Característica

#### Headers Jerárquicos

**Componente**: `HierarchicalHeaderProcessor`

Detecta y procesa:
- Headers que abarcan múltiples columnas
- Estructuras anidadas multi-nivel
- Relaciones padre-hijo entre headers

#### Celdas Fusionadas

**Componente**: `MergedCellHandler`

Identifica y maneja:
- Fusiones verticales (rowspan > 1)
- Fusiones horizontales (colspan > 1)
- Propagación de valores a celdas afectadas

#### Tablas sin Bordes

**Componente**: `BorderlessTableHandler`

Infiere estructura mediante:
- Análisis de espaciado
- Detección de patrones de alineación
- Identificación de secciones lógicas

---

## Procesamiento de Imágenes y Document Intelligence

### Origen de las Imágenes

Las imágenes provienen del índice de Azure AI Search como parte del resultado del retrieval.

**Ubicación en código**: `orc/code_orchestration_enhanced.py`, líneas 600-620

```python
# Extracción de URLs de imágenes desde el resultado de AI Search
relevant_sources = await chat_completion(system_message, filter_sources_prompt, variables)

# Las imágenes vienen en formato: relatedImages: [url1, url2, url3]
image_urls = re.findall(r'relatedImages:\s*(\[[^\]]*\])', relevant_sources)

# Procesamiento de URLs
relatedImages = []
for image_url in image_urls:
    images = ast.literal_eval(image_url)
    relatedImages.extend(images)
```

### Flujos de Procesamiento de Imágenes

#### Flujo 1: Document Intelligence

**Ubicación**: `orc/code_orchestration_enhanced.py`, líneas 256-273

```python
if ENABLE_DOCUMENT_INTELLIGENCE and relatedImages:
    doc_extractor = DocumentIntelligenceTableExtractor()
    
    for image_url in relatedImages[:3]:  # Procesa máximo 3 imágenes
        extracted_tables = await doc_extractor.extract_tables_from_document(image_url)
        
        for extracted in extracted_tables:
            interpreted = complex_interpreter.interpret_table_for_retrieval(
                extracted,
                query
            )
            table_context["enhanced_tables"].append(interpreted)
```

#### Flujo 2: Mistral OCR

**Ubicación**: `orc/code_orchestration_enhanced.py`, líneas 196-229

```python
if mistral_integration and relatedImages:
    tables_with_images = []
    
    for i, table in enumerate(processed_tables):
        if i < len(relatedImages):
            table_with_image = table.copy()
            table_with_image["image_url"] = relatedImages[i]
            table_with_image["metadata"] = {
                "has_merged_cells": ...,
                "has_hierarchical_headers": ...,
                "complexity": ...
            }
            tables_with_images.append(table_with_image)
    
    mistral_results = await mistral_integration.enhance_table_retrieval_pipeline(
        query,
        tables_with_images,
        {"language": "Spanish"}
    )
```

#### Flujo 3: Conversión Base64 para LLM

**Ubicación**: `orc/code_orchestration_enhanced.py`, líneas 650-680

```python
if relatedImages:
    storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
    container_name = os.environ.get("STORAGE_CONTAINER_IMAGES")
    
    base64_images = []
    for image in relatedImages:
        blob_client = container_client.get_blob_client(blob_name)
        image_bytes = blob_client.download_blob().readall()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        base64_images.append(f"data:image/png;base64,{image_base64}")
```

---

## Integración con Azure AI Search

### Flujo de Datos

1. **Query Inicial**: El usuario realiza una consulta
2. **Búsqueda en AI Search**: Se ejecuta la búsqueda en el índice existente
3. **Resultados**: AI Search retorna:
   - `sources`: Contenido textual de los chunks relevantes
   - `relatedImages`: URLs de imágenes asociadas a los documentos

### Formato de URLs de Imágenes

```
https://[storage-account].blob.core.windows.net/[container]/[path]/[filename].png
```

### Punto Importante

El sistema **NO modifica el índice de AI Search**. Todo el procesamiento ocurre durante el retrieval, sin alterar los datos almacenados.

---

## Componentes y Responsabilidades

### Matriz de Responsabilidades

| Componente | Archivo | Responsabilidad Principal |
|------------|---------|--------------------------|
| `detect_table_query()` | `code_orchestration_enhanced.py` | Identificar queries relacionadas con tablas |
| `TableStructureExtractor` | `table_processor.py` | Extraer estructura de tablas desde HTML/texto |
| `DocumentIntelligenceTableExtractor` | `document_intelligence_integration.py` | Extraer tablas desde imágenes |
| `ComplexTableInterpreter` | `complex_table_handler.py` | Clasificar complejidad y orquestar procesamiento |
| `HierarchicalHeaderProcessor` | `complex_table_handler.py` | Procesar headers multi-nivel |
| `MergedCellHandler` | `complex_table_handler.py` | Manejar celdas fusionadas |
| `BorderlessTableHandler` | `complex_table_handler.py` | Inferir estructura de tablas sin bordes |
| `MistralOCRTableAnalyzer` | `mistral_ocr_retrieval.py` | Análisis semántico de contenido visual |
| `AgenticTableSearch` | `agentic_table_search.py` | Búsqueda paralela para queries complejas |
| `TableRetrievalMonitor` | `table_retrieval_metrics.py` | Monitoreo de performance y calidad |

---

## Implementación Técnica

### Flujo de Ejecución Completo

```python
# 1. Recepción de Query
query = "What is the Q2 revenue for Product A in the hierarchical table?"

# 2. Detección de Query de Tabla
is_table_query = await detect_table_query(query, context)  # Returns: True

# 3. Búsqueda en AI Search
search_results = await search_client.search(query)
sources = search_results.sources
relatedImages = search_results.relatedImages

# 4. Procesamiento de Tablas según Disponibilidad

# 4a. Si hay tablas en HTML
if tables_in_html:
    tables = table_extractor.extract_tables_from_html(sources)
    for table in tables:
        processed = complex_interpreter.interpret_table_for_retrieval(table, query)

# 4b. Si hay imágenes y Document Intelligence está habilitado
if ENABLE_DOCUMENT_INTELLIGENCE and relatedImages:
    for image_url in relatedImages:
        extracted = await doc_extractor.extract_tables_from_document(image_url)

# 4c. Si Mistral OCR está habilitado
if ENABLE_MISTRAL_OCR and relatedImages:
    mistral_results = await mistral_analyzer.analyze_table_for_retrieval(
        image_url, query, context
    )

# 5. Generación de Respuesta
enhanced_prompt = create_enhanced_prompt(
    original_sources,
    extracted_tables,
    mistral_insights,
    document_intelligence_data
)
response = await llm.generate(enhanced_prompt)
```

### Ejemplo de Procesamiento Real

```python
# Input
query = "Show total revenue for merged cells in Q2"
sources = "<table><td colspan='3' rowspan='2'>$1,500</td>...</table>"
relatedImages = ["https://storage.blob.core.windows.net/docs/table1.png"]

# Processing
# 1. Detección: TRUE (palabras clave: "merged cells")
# 2. Extracción: merged_cells = [{colspan: 3, rowspan: 2, value: "$1,500"}]
# 3. Clasificación: MEDIUM (score = 2)
# 4. Procesamiento: Valor $1,500 aplica a 6 celdas (3x2)

# Output
interpreted_table = {
    "complexity_assessment": "MEDIUM",
    "merged_cells_processed": [{
        "original_position": (0, 0),
        "affected_cells": [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
        "value": "$1,500",
        "interpretation": "This value applies to all Q2 cells"
    }],
    "query_answer": "$1,500 for Q2 merged region"
}
```

---

## Configuración del Sistema

### Variables de Entorno Requeridas

```bash
# Habilitación de Características
ENABLE_COMPLEX_TABLE_PROCESSING=true
ENABLE_AGENTIC_TABLE_SEARCH=true
ENABLE_RETRIEVAL_METRICS=true
ENABLE_HIERARCHICAL_HEADERS=true
ENABLE_MERGED_CELLS=true
ENABLE_BORDERLESS_DETECTION=true

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://[your-search-service].search.windows.net
AZURE_SEARCH_KEY=[your-api-key]
AZURE_SEARCH_INDEX_NAME=tables-index

# Document Intelligence (Opcional)
ENABLE_DOCUMENT_INTELLIGENCE=false
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://[your-service].cognitiveservices.azure.com
AZURE_DOCUMENT_INTELLIGENCE_KEY=[your-api-key]

# Mistral OCR via Azure AI Foundry (Opcional)
ENABLE_MISTRAL_OCR=false
AZURE_MISTRAL_OCR_ENDPOINT=https://[your-resource].openai.azure.com/v1/ocr
AZURE_MISTRAL_OCR_API_KEY=[your-api-key]
MISTRAL_OCR_MODEL=mistral-ocr-2503

# Performance
QUERY_TIMEOUT_MS=5000
COMPLEX_TABLE_TIMEOUT_MULTIPLIER=1.5
AGENTIC_MAX_PARALLEL_QUERIES=5
```

### Modos de Operación

| Modo | Configuración | Características |
|------|--------------|-----------------|
| **Standard** | `RETRIEVAL_MODE=standard` | Procesamiento básico, menor latencia |
| **Enhanced** | `RETRIEVAL_MODE=enhanced` | Procesamiento completo de estructuras complejas |
| **Agentic** | `RETRIEVAL_MODE=agentic` | Búsqueda paralela avanzada, requiere Azure AI Search |
| **Hybrid** | `RETRIEVAL_MODE=hybrid` | Combina capacidades según complejidad de query |

---

## Métricas y Monitoreo

### Métricas Clave del Sistema

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| **Tiempo de Respuesta** | < 2000ms | Tiempo total de procesamiento |
| **Precisión en Tablas Complejas** | > 90% | Exactitud en extracción de datos |
| **Detección de Estructura** | > 85% | Identificación correcta de características |
| **Calidad de Retrieval** | > 0.7 | Score de relevancia promedio |

### Implementación de Monitoreo

```python
# Tracking de métricas
if ENABLE_RETRIEVAL_METRICS:
    metrics = await retrieval_monitor.track_retrieval(
        query=query,
        results=table_context.get("enhanced_tables", []),
        execution_time=execution_time,
        query_analysis={"intents": ["table_query"], "search_method": "enhanced"},
        table_features=table_features
    )
    
    # Obtener resumen de performance
    summary = monitor.get_performance_summary()
```

### Recomendaciones de Optimización

El sistema proporciona recomendaciones automáticas basadas en métricas:

```python
recommendations = analyzer.get_optimization_recommendations()
# Ejemplo de salida:
# - "Considerar habilitar Document Intelligence para mejorar precisión"
# - "Aumentar timeout para tablas con >50 filas"
# - "Activar caché para queries frecuentes"
```

---

## Consideraciones de Performance

### Optimizaciones Implementadas

1. **Procesamiento Selectivo**: Solo se procesan tablas cuando se detecta una query relacionada
2. **Límite de Imágenes**: Máximo 3 imágenes procesadas con Document Intelligence
3. **Caché de Resultados**: Resultados frecuentes se almacenan temporalmente
4. **Procesamiento Paralelo**: Búsquedas agénticas ejecutan sub-queries en paralelo

### Escalabilidad

- El sistema puede procesar hasta 10 tablas por request
- Timeout configurable con multiplicador para tablas complejas
- Búsqueda agéntica soporta hasta 5 queries paralelas por defecto

---

## Diagrama de Flujo Completo

```
┌─────────────────┐
│  Query Usuario  │
└────────┬────────┘
         │
         v
┌────────────────────────┐
│ Detección Query Tabla  │
└────────┬───────────────┘
         │
         v
    ┌────────────┐
    │ AI Search  │
    └────┬───────┘
         │
         v
┌──────────────────────┐
│ sources + images URLs│
└──────┬───────────────┘
       │
       v
┌──────────────────────────────┐
│ Procesamiento según Tipo     │
├──────────────────────────────┤
│ • HTML → TableExtractor      │
│ • Images → Doc Intelligence  │
│ • Images → Mistral OCR       │
└──────────┬───────────────────┘
           │
           v
┌───────────────────────────┐
│ Clasificación Complejidad │
└───────────┬───────────────┘
            │
            v
┌────────────────────────────────┐
│ Aplicar Procesadores           │
├────────────────────────────────┤
│ • HierarchicalHeaderProcessor  │
│ • MergedCellHandler            │
│ • BorderlessTableHandler       │
└────────────┬───────────────────┘
             │
             v
┌──────────────────────┐
│ Contexto Enriquecido │
└──────────┬───────────┘
           │
           v
┌────────────────────┐
│ Generar Respuesta  │
└────────────────────┘
```

---

## Conclusión

El Sistema de Retrieval Mejorado para Tablas Complejas representa una evolución significativa en la capacidad del Orchestrator IDM para procesar y comprender estructuras tabulares. Al operar exclusivamente durante el retrieval, el sistema mantiene la flexibilidad de procesar diferentes tipos de tablas sin requerir re-indexación, mientras proporciona respuestas precisas y contextualizadas para queries complejas sobre datos tabulares.

La arquitectura modular permite habilitar o deshabilitar características según las necesidades específicas del deployment, asegurando un balance óptimo entre performance y capacidades de procesamiento.