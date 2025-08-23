# Plan de Mejora para Procesamiento de Tablas Complejas - Orchestrator IDM

## Resumen Ejecutivo

Este documento presenta un plan integral para mejorar el procesamiento de tablas complejas en el Orchestrator, enfocándose en la interpretación semántica avanzada sin modificar la estructura actual del sistema.

## 1. Análisis del Sistema Actual

### 1.1 Problemas Identificados

#### Limitaciones en el Procesamiento de Tablas
- **Conversión a Imágenes**: Las tablas HTML se convierten a imágenes base64, perdiendo estructura semántica
- **Pérdida de Contexto**: No se preservan las relaciones entre celdas, headers y contenido
- **Headers Jerárquicos**: Dificultad para interpretar headers multi-nivel y celdas fusionadas
- **Bordes Unificados**: Problemas con tablas sin bordes claros o con celdas combinadas

#### Flujo Actual
1. Documentos incluyen campo `relatedImages` con URLs de tablas
2. Durante retrieval se recuperan contenido textual y URLs
3. Tablas HTML se reemplazan por placeholders `[IMAGE_filename]`
4. Se usa prompt especializado `answer_tables.prompt` para interpretación

### 1.2 Fortalezas del Sistema
- Integración con Azure Blob Storage para imágenes
- Uso de modelos multimodales para procesamiento
- Sistema de prompts especializado para tablas

## 2. Tecnologías Propuestas para Mejora

### 2.1 Azure Document Intelligence Layout Model
**Capacidades Clave:**
- Extracción avanzada de estructuras tabulares
- Detección de celdas fusionadas y spans multi-columna/fila
- Identificación de headers jerárquicos
- Coordenadas precisas de bounding boxes
- Soporte para tablas con y sin bordes

**Integración Propuesta:**
- Pre-procesamiento durante indexación
- Enriquecimiento de metadata estructural
- Preservación de relaciones celda-header

### 2.2 Mistral OCR
**Capacidades Clave:**
- 96.12% precisión en procesamiento de tablas
- Comprensión semántica profunda de elementos
- Procesamiento de 2000 páginas/minuto
- Salida estructurada en JSON nativo
- Soporte multilingüe avanzado

**Integración Propuesta:**
- Extracción semántica durante indexación
- Generación de embeddings contextuales
- Procesamiento batch para actualización de índices

### 2.3 Azure Agentic Retrieval
**Capacidades Clave:**
- Descomposición de queries complejas
- Búsqueda paralela multi-query
- Re-ranking semántico de resultados
- Integración con RAG workflows

**Integración Propuesta:**
- Query decomposition para preguntas sobre tablas
- Búsqueda híbrida (estructura + contenido)
- Ranking mejorado basado en coincidencia estructural

## 3. Estrategia de Implementación

### 3.1 Fase 1: Enriquecimiento del Pipeline de Indexación

#### Componente: Table Structure Extractor
**Ubicación**: Nuevo módulo en `shared/table_processor.py`

**Funcionalidades**:
1. **Extracción Estructural Dual**
   - Aplicar Document Intelligence Layout para estructura
   - Usar Mistral OCR para comprensión semántica
   - Fusionar resultados en representación unificada

2. **Generación de Metadata Enriquecida**
   ```json
   {
     "table_id": "uuid",
     "structure": {
       "rows": 10,
       "columns": 5,
       "merged_cells": [...],
       "hierarchical_headers": {
         "level_1": [...],
         "level_2": [...] 
       }
     },
     "semantic_content": {
       "cell_relationships": [...],
       "header_mappings": {...},
       "data_types": [...]
     },
     "visual_representation": "base64_image",
     "text_representation": "markdown_table"
   }
   ```

3. **Creación de Embeddings Especializados**
   - Embeddings de estructura tabular
   - Embeddings de relaciones semánticas
   - Embeddings de contenido textual

### 3.2 Fase 2: Mejora del Proceso de Retrieval

#### Componente: Enhanced Table Retrieval
**Ubicación**: Extensión en `orc/plugins/RAG/native_function_enhanced.py`

**Funcionalidades**:
1. **Query Analysis para Tablas**
   - Detectar cuando query se refiere a datos tabulares
   - Extraer entidades y relaciones buscadas
   - Generar sub-queries estructurales

2. **Búsqueda Híbrida Mejorada**
   ```python
   # Pseudocódigo conceptual
   def enhanced_table_search(query):
       # Búsqueda por contenido textual
       text_results = search_text_index(query)
       
       # Búsqueda por estructura
       structure_results = search_structure_index(query)
       
       # Búsqueda por relaciones semánticas
       semantic_results = search_semantic_index(query)
       
       # Fusión y re-ranking
       return merge_and_rank(text_results, structure_results, semantic_results)
   ```

3. **Recuperación Contextual**
   - Recuperar tabla completa con metadata
   - Incluir contexto circundante
   - Preservar relaciones jerárquicas

### 3.3 Fase 3: Optimización del Procesamiento de Respuestas

#### Componente: Table-Aware Answer Generation
**Ubicación**: Mejoras en `orc/code_orchestration.py`

**Funcionalidades**:
1. **Procesamiento Inteligente de Tablas**
   ```python
   # Mejora propuesta para replace_tables()
   def enhanced_replace_tables(content, table_metadata):
       # Preservar estructura semántica
       structured_content = {
           "visual": generate_image_placeholder(),
           "structure": table_metadata["structure"],
           "semantic": table_metadata["semantic_content"],
           "context": extract_surrounding_context()
       }
       return structured_content
   ```

2. **Prompt Engineering Avanzado**
   - Incluir estructura JSON junto con imagen
   - Proporcionar mapeo de headers jerárquicos
   - Añadir contexto de relaciones entre celdas

3. **Validación de Respuestas**
   - Verificar coherencia con estructura tabular
   - Validar referencias a celdas específicas
   - Asegurar precisión en datos numéricos

### 3.4 Fase 4: Manejo de Casos Especiales

#### Componente: Complex Table Handler
**Ubicación**: Nuevo módulo en `shared/complex_table_handler.py`

**Funcionalidades**:
1. **Headers Jerárquicos**
   ```python
   def process_hierarchical_headers(table):
       # Identificar niveles de jerarquía
       header_levels = identify_header_levels(table)
       
       # Crear mapeo de relaciones
       header_tree = build_header_tree(header_levels)
       
       # Generar representación semántica
       return generate_semantic_representation(header_tree)
   ```

2. **Celdas Fusionadas y Bordes Unificados**
   ```python
   def handle_merged_cells(table):
       # Detectar celdas fusionadas
       merged_regions = detect_merged_regions(table)
       
       # Asignar contenido correctamente
       cell_assignments = assign_content_to_cells(merged_regions)
       
       # Preservar contexto visual
       return preserve_visual_context(cell_assignments)
   ```

3. **Tablas Sin Bordes Claros**
   - Inferir estructura mediante análisis de espaciado
   - Usar alineación de texto para detectar columnas
   - Aplicar heurísticas para identificar separadores

## 4. Configuración y Parametrización

### 4.1 Variables de Entorno Nuevas
```bash
# Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
DOCUMENT_INTELLIGENCE_MODEL=prebuilt-layout

# Mistral OCR (opcional)
MISTRAL_OCR_ENDPOINT=
MISTRAL_OCR_API_KEY=
MISTRAL_OCR_ENABLED=false

# Table Processing
TABLE_EXTRACTION_METHOD=document_intelligence  # document_intelligence | mistral | hybrid
TABLE_STRUCTURE_PRESERVATION=true
TABLE_SEMANTIC_ENRICHMENT=true
TABLE_HIERARCHICAL_HEADERS=true

# Agentic Retrieval
AGENTIC_RETRIEVAL_ENABLED=true
AGENTIC_TABLE_QUERIES=true
AGENTIC_QUERY_DECOMPOSITION=true
```

### 4.2 Configuración de Procesamiento
```json
{
  "table_processing": {
    "extraction": {
      "method": "hybrid",
      "confidence_threshold": 0.8,
      "max_table_size": 1000
    },
    "indexing": {
      "include_structure": true,
      "include_semantic": true,
      "include_visual": true
    },
    "retrieval": {
      "use_agentic": true,
      "max_subqueries": 5,
      "semantic_weight": 0.4,
      "structure_weight": 0.3,
      "content_weight": 0.3
    }
  }
}
```

## 5. Mejoras en Prompts

### 5.1 Prompt Mejorado para Tablas Complejas
**Archivo**: `orc/prompts/answer_tables_enhanced.prompt`

```text
## Enhanced Table Analysis Instructions

When analyzing tables with complex structures:

### Hierarchical Headers
- Identify multi-level header relationships
- Map data cells to ALL relevant header levels
- Preserve parent-child header relationships

### Merged Cells
- Recognize horizontally and vertically merged cells
- Understand that merged cells apply to all sub-cells
- Maintain context across merged regions

### Borderless Tables
- Use spacing and alignment to infer structure
- Identify implicit column boundaries
- Recognize section headers vs data headers

### Semantic Understanding
- Consider table context from surrounding text
- Understand domain-specific terminology
- Apply business logic when interpreting relationships

### Data Extraction
Given the table structure JSON:
{{$table_structure}}

And the visual representation:
{{$table_image}}

Extract and interpret data considering:
1. Full hierarchical context
2. Merged cell implications
3. Implicit relationships
4. Domain-specific rules
```

### 5.2 Prompt para Query Decomposition
**Archivo**: `orc/prompts/table_query_decomposition.prompt`

```text
## Table Query Analysis

Analyze the user query to identify table-related information needs:

### Query: {{$query}}

Decompose into:
1. **Structural queries**: What table structure is needed?
2. **Content queries**: What specific data is requested?
3. **Relationship queries**: What relationships between data points?
4. **Aggregation queries**: What calculations or summaries?

Output format:
{
  "is_table_query": true/false,
  "structural_requirements": [...],
  "data_requirements": [...],
  "relationships": [...],
  "aggregations": [...]
}
```

## 6. Métricas de Evaluación

### 6.1 KPIs de Mejora
1. **Precisión de Extracción**
   - Baseline actual: ~70% para tablas complejas
   - Objetivo: >90% con Document Intelligence
   - Medición: Comparación manual de muestras

2. **Tiempo de Procesamiento**
   - Baseline actual: 2-3 segundos por tabla
   - Objetivo: <1 segundo con caché optimizado
   - Medición: Logs de tiempo de respuesta

3. **Calidad de Respuestas**
   - Baseline actual: 75% satisfacción usuario
   - Objetivo: >90% para queries sobre tablas
   - Medición: Feedback de usuarios y validación manual

### 6.2 Monitoreo
```python
# Componente de monitoreo
class TableProcessingMetrics:
    def track_extraction(self, table_id, method, duration, success):
        # Log métricas de extracción
        pass
    
    def track_retrieval(self, query, results, relevance_score):
        # Log métricas de retrieval
        pass
    
    def track_answer_quality(self, question, answer, confidence):
        # Log calidad de respuestas
        pass
```

## 7. Plan de Testing

### 7.1 Test Suite para Tablas Complejas
**Ubicación**: `tests/test_complex_tables.py`

```python
class TestComplexTableProcessing:
    def test_hierarchical_headers(self):
        # Test con headers multi-nivel
        pass
    
    def test_merged_cells(self):
        # Test con celdas fusionadas
        pass
    
    def test_borderless_tables(self):
        # Test con tablas sin bordes
        pass
    
    def test_mixed_content_tables(self):
        # Test con contenido mixto
        pass
```

### 7.2 Casos de Prueba Específicos
1. **Tablas con Headers Jerárquicos**
   - 3+ niveles de headers
   - Headers que abarcan múltiples columnas
   - Headers con sub-categorías

2. **Celdas Fusionadas Complejas**
   - Fusión horizontal y vertical simultánea
   - Celdas fusionadas con contenido variable
   - Bordes parcialmente definidos

3. **Tablas Sin Estructura Clara**
   - Tablas con espaciado irregular
   - Separadores implícitos
   - Formato mixto (texto/números/fechas)

## 8. Documentación de Implementación

### 8.1 Guías de Desarrollo
1. **Integración con Document Intelligence**
   - Configuración de credenciales
   - Llamadas API optimizadas
   - Manejo de errores y reintentos

2. **Procesamiento de Metadata**
   - Formato de almacenamiento
   - Estrategias de caché
   - Actualización incremental

3. **Optimización de Queries**
   - Patrones de búsqueda eficientes
   - Índices especializados
   - Estrategias de ranking

### 8.2 Ejemplos de Código

#### Ejemplo 1: Extracción de Estructura
```python
async def extract_table_structure(document_url):
    """
    Extrae estructura completa de tabla usando Document Intelligence
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    
    client = DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        credential=DefaultAzureCredential()
    )
    
    poller = await client.begin_analyze_document(
        "prebuilt-layout",
        document_url
    )
    
    result = await poller.result()
    
    tables = []
    for table in result.tables:
        structured_table = {
            "row_count": table.row_count,
            "column_count": table.column_count,
            "cells": process_cells(table.cells),
            "merged_cells": identify_merged_cells(table.cells),
            "headers": extract_hierarchical_headers(table.cells)
        }
        tables.append(structured_table)
    
    return tables
```

#### Ejemplo 2: Query Decomposition para Tablas
```python
async def decompose_table_query(query, context):
    """
    Descompone query compleja en sub-queries para tablas
    """
    # Analizar si query se refiere a tablas
    is_table_query = await detect_table_intent(query)
    
    if not is_table_query:
        return None
    
    # Extraer componentes de la query
    components = {
        "entities": extract_entities(query),
        "relationships": extract_relationships(query),
        "filters": extract_filters(query),
        "aggregations": extract_aggregations(query)
    }
    
    # Generar sub-queries optimizadas
    subqueries = []
    
    # Query por estructura
    if components["relationships"]:
        subqueries.append(
            generate_structure_query(components["relationships"])
        )
    
    # Query por contenido
    if components["entities"]:
        subqueries.append(
            generate_content_query(components["entities"])
        )
    
    # Query por agregaciones
    if components["aggregations"]:
        subqueries.append(
            generate_aggregation_query(components["aggregations"])
        )
    
    return subqueries
```

#### Ejemplo 3: Procesamiento de Headers Jerárquicos
```python
def process_hierarchical_headers(table_cells):
    """
    Procesa headers multi-nivel en tablas complejas
    """
    header_tree = {}
    header_levels = []
    
    # Identificar filas de headers
    header_rows = identify_header_rows(table_cells)
    
    for level, row in enumerate(header_rows):
        level_headers = []
        
        for cell in row:
            header = {
                "text": cell.content,
                "span": cell.column_span,
                "children": [],
                "parent": find_parent_header(cell, header_levels)
            }
            level_headers.append(header)
        
        header_levels.append(level_headers)
    
    # Construir árbol jerárquico
    for level in header_levels:
        for header in level:
            if header["parent"]:
                header["parent"]["children"].append(header)
            else:
                header_tree[header["text"]] = header
    
    return header_tree
```

## 9. Consideraciones de Seguridad y Performance

### 9.1 Seguridad
- Sanitización de contenido extraído de tablas
- Validación de límites de tamaño de tabla
- Control de acceso a metadata estructural
- Encriptación de datos sensibles en caché

### 9.2 Performance
- Caché de estructuras de tabla procesadas
- Procesamiento asíncrono para tablas grandes
- Lazy loading de imágenes de tablas
- Índices optimizados para búsqueda estructural

### 9.3 Escalabilidad
- Procesamiento batch para múltiples tablas
- Queue management para Document Intelligence
- Rate limiting para APIs externas
- Distribución de carga para procesamiento intensivo

## 10. Cronograma de Implementación

### Fase 1: Preparación (Semana 1-2)
- Configuración de servicios Azure
- Setup de credenciales y permisos
- Creación de estructura de módulos

### Fase 2: Desarrollo Core (Semana 3-6)
- Implementación de extractores de estructura
- Integración con Document Intelligence
- Desarrollo de procesadores de metadata

### Fase 3: Integración (Semana 7-8)
- Integración con pipeline existente
- Actualización de prompts
- Ajuste de flujos de retrieval

### Fase 4: Testing y Optimización (Semana 9-10)
- Pruebas con casos complejos
- Optimización de performance
- Ajuste de parámetros

### Fase 5: Deployment (Semana 11-12)
- Deployment gradual
- Monitoreo de métricas
- Documentación final

## 11. Conclusiones y Próximos Pasos

### Beneficios Esperados
1. **Mejora en Precisión**: >90% en extracción de tablas complejas
2. **Reducción de Errores**: -60% en interpretación incorrecta
3. **Mayor Velocidad**: 3x más rápido en procesamiento
4. **Mejor UX**: Respuestas más precisas y contextuales

### Próximos Pasos Inmediatos
1. Validar plan con stakeholders
2. Obtener credenciales para servicios Azure
3. Crear ambiente de desarrollo
4. Iniciar POC con subset de documentos

### Evolución Futura
- Integración con modelos de lenguaje más avanzados
- Soporte para formatos adicionales (Excel, CSV)
- Capacidades de edición y corrección de tablas
- Analytics avanzados sobre datos tabulares

## Anexos

### A. Referencias Técnicas
- [Azure Document Intelligence Documentation](https://docs.microsoft.com/azure/ai-services/document-intelligence/)
- [Mistral OCR Technical Specs](https://mistral.ai/news/mistral-ocr)
- [Agentic Retrieval Patterns](https://learn.microsoft.com/azure/search/search-agentic-retrieval-concept)
- [Table Extraction Best Practices 2024](https://arxiv.org/html/2409.14192v1)

### B. Glosario
- **Headers Jerárquicos**: Headers que abarcan múltiples niveles de organización
- **Celdas Fusionadas**: Celdas que ocupan múltiples filas o columnas
- **Bordes Unificados**: Tablas donde los bordes no están claramente definidos
- **Interpretación Semántica**: Comprensión del significado y contexto de los datos

### C. Contactos Técnicos
- Equipo de Arquitectura: Para aprobación de cambios estructurales
- Equipo de DevOps: Para configuración de servicios Azure
- Equipo de QA: Para validación de casos de prueba