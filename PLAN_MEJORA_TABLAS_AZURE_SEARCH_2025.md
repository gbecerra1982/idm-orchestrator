# Plan de Mejora Avanzado: Integración de Azure AI Search 2025 para Tablas Complejas

## Resumen Ejecutivo Actualizado

Este documento amplía el plan de mejora integrando las capacidades más recientes de Azure AI Search (API 2025-03-01-preview) para optimizar el procesamiento y retrieval de tablas complejas en el Orchestrator IDM.

## 1. Capacidades Clave de Azure AI Search 2025

### 1.1 Tipos de Datos Complejos Nativos
Azure AI Search soporta nativamente:
- **Edm.ComplexType**: Objetos anidados simples
- **Collection(Edm.ComplexType)**: Arrays de objetos complejos
- **Facetas Jerárquicas**: Navegación multi-nivel en datos estructurados
- **Agregaciones Numéricas**: Suma y cálculos sobre campos facetables

### 1.2 Búsqueda Agéntica (Preview)
- Ejecución paralela de múltiples subconsultas
- Optimización automática para cargas RAG
- Re-ranking semántico integrado

### 1.3 Limitaciones Críticas
- **Máximo 3,000 elementos** en colecciones complejas por documento
- Las operaciones merge no fusionan elementos dentro de colecciones
- Requiere recuperación completa para actualizar colecciones

## 2. Arquitectura Propuesta con Azure AI Search

### 2.1 Modelo de Datos para Tablas Complejas

```json
{
  "id": "doc_001_table_01",
  "documentId": "doc_001",
  "tableMetadata": {
    "tableId": "table_01",
    "pageNumber": 5,
    "dimensions": {
      "rows": 15,
      "columns": 8
    }
  },
  "structure": {
    "headers": {
      "level1": ["Revenue", "Costs", "Profit"],
      "level2": ["Q1", "Q2", "Q3", "Q4"],
      "hierarchicalMap": [
        {
          "parent": "Revenue",
          "children": ["Q1", "Q2", "Q3", "Q4"],
          "span": 4
        }
      ]
    },
    "mergedCells": [
      {
        "startRow": 0,
        "startCol": 0,
        "rowSpan": 2,
        "colSpan": 1,
        "content": "Category"
      }
    ]
  },
  "cells": [
    {
      "row": 2,
      "column": 1,
      "value": "150000",
      "dataType": "currency",
      "headers": {
        "row": ["Product A"],
        "column": ["Revenue", "Q1"]
      },
      "context": "First quarter revenue for Product A"
    }
  ],
  "semanticContent": {
    "summary": "Quarterly financial performance table showing revenue, costs and profit breakdown",
    "keyInsights": ["Q2 showed highest growth", "Product A leads revenue"],
    "relationships": [
      {
        "type": "calculation",
        "formula": "Profit = Revenue - Costs",
        "cells": ["C3", "A3", "B3"]
      }
    ]
  },
  "vectorEmbeddings": {
    "structure": [0.123, 0.456, ...],
    "content": [0.789, 0.012, ...],
    "semantic": [0.345, 0.678, ...]
  }
}
```

### 2.2 Estrategia de Indexación

#### Índice Principal de Tablas
```json
{
  "name": "tables-index",
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true
    },
    {
      "name": "tableMetadata",
      "type": "Edm.ComplexType",
      "fields": [
        {"name": "tableId", "type": "Edm.String", "filterable": true},
        {"name": "pageNumber", "type": "Edm.Int32", "filterable": true},
        {"name": "dimensions", "type": "Edm.ComplexType"}
      ]
    },
    {
      "name": "structure",
      "type": "Edm.ComplexType",
      "fields": [
        {
          "name": "headers",
          "type": "Edm.ComplexType",
          "facetable": true
        },
        {
          "name": "mergedCells",
          "type": "Collection(Edm.ComplexType)"
        }
      ]
    },
    {
      "name": "cells",
      "type": "Collection(Edm.ComplexType)",
      "fields": [
        {"name": "value", "type": "Edm.String", "searchable": true},
        {"name": "dataType", "type": "Edm.String", "filterable": true},
        {"name": "headers", "type": "Edm.ComplexType"}
      ]
    },
    {
      "name": "semanticContent",
      "type": "Edm.ComplexType",
      "searchable": true
    },
    {
      "name": "vectorEmbeddings",
      "type": "Collection(Edm.Single)",
      "vectorSearchDimensions": 1536,
      "vectorSearchProfileName": "table-vector-profile"
    }
  ]
}
```

### 2.3 Implementación de Búsqueda Agéntica

```python
# shared/agentic_table_search.py
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import asyncio

class AgenticTableSearch:
    def __init__(self, search_client: SearchClient):
        self.client = search_client
        
    async def execute_agentic_search(self, query: str, context: dict):
        """
        Ejecuta búsqueda agéntica descomponiendo la query en subconsultas
        """
        # 1. Analizar query para identificar intención
        query_analysis = await self.analyze_table_query(query)
        
        # 2. Generar subconsultas paralelas
        subqueries = []
        
        if query_analysis['needs_structure_search']:
            subqueries.append(
                self.search_by_structure(query_analysis['structure_params'])
            )
        
        if query_analysis['needs_content_search']:
            subqueries.append(
                self.search_by_content(query_analysis['content_params'])
            )
        
        if query_analysis['needs_semantic_search']:
            subqueries.append(
                self.search_by_semantics(query_analysis['semantic_params'])
            )
        
        # 3. Ejecutar subconsultas en paralelo
        results = await asyncio.gather(*subqueries)
        
        # 4. Fusionar y re-rankear resultados
        merged_results = self.merge_and_rerank(results)
        
        return merged_results
    
    async def search_by_structure(self, params):
        """
        Búsqueda por estructura de tabla usando campos complejos
        """
        filter_expr = self.build_structure_filter(params)
        
        results = self.client.search(
            search_text="*",
            filter=filter_expr,
            select=["id", "structure", "tableMetadata"],
            top=10
        )
        
        return list(results)
    
    async def search_by_content(self, params):
        """
        Búsqueda por contenido de celdas usando expresiones lambda
        """
        # Usar expresiones lambda para buscar en colecciones
        filter_expr = f"cells/any(c: c/value eq '{params['value']}')"
        
        if params.get('headers'):
            filter_expr += f" and cells/any(c: c/headers/column/any(h: h eq '{params['headers']}'))"
        
        results = self.client.search(
            search_text=params.get('text', '*'),
            filter=filter_expr,
            select=["id", "cells", "semanticContent"],
            query_type="semantic",
            semantic_configuration_name="table-semantic-config",
            top=10
        )
        
        return list(results)
    
    async def search_by_semantics(self, params):
        """
        Búsqueda vectorial y semántica combinada
        """
        # Generar embedding de la query
        query_embedding = await self.generate_embedding(params['query'])
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=10,
            fields="vectorEmbeddings"
        )
        
        results = self.client.search(
            search_text=params.get('text', ''),
            vector_queries=[vector_query],
            select=["id", "semanticContent", "structure"],
            top=10
        )
        
        return list(results)
    
    def build_structure_filter(self, params):
        """
        Construye filtros OData para estructuras complejas
        """
        filters = []
        
        if params.get('min_rows'):
            filters.append(f"structure/dimensions/rows ge {params['min_rows']}")
        
        if params.get('has_merged_cells'):
            filters.append("structure/mergedCells/any()")
        
        if params.get('header_levels'):
            filters.append(f"structure/headers/level1/any(h: h eq '{params['header_levels']}')")
        
        return " and ".join(filters) if filters else None
```

### 2.4 Optimización para Límite de 3000 Elementos

```python
# shared/table_chunking_strategy.py

class TableChunkingStrategy:
    """
    Estrategia para manejar tablas que exceden el límite de 3000 elementos
    """
    MAX_CELLS_PER_DOC = 2500  # Dejar margen para metadata
    
    def chunk_large_table(self, table_data):
        """
        Divide tablas grandes en múltiples documentos relacionados
        """
        total_cells = len(table_data['cells'])
        
        if total_cells <= self.MAX_CELLS_PER_DOC:
            return [table_data]  # No necesita chunking
        
        chunks = []
        chunk_size = self.MAX_CELLS_PER_DOC
        
        for i in range(0, total_cells, chunk_size):
            chunk = {
                **table_data,
                'id': f"{table_data['id']}_chunk_{i//chunk_size}",
                'chunkInfo': {
                    'isChunked': True,
                    'chunkIndex': i // chunk_size,
                    'totalChunks': (total_cells + chunk_size - 1) // chunk_size,
                    'parentId': table_data['id']
                },
                'cells': table_data['cells'][i:i + chunk_size]
            }
            chunks.append(chunk)
        
        return chunks
    
    def use_delimiter_encoding(self, cells):
        """
        Estrategia alternativa usando delimitación de cadenas
        """
        encoded_cells = []
        
        for cell in cells:
            # Formato: |row|col|value|dataType|headers|
            encoded = f"|{cell['row']}|{cell['column']}|{cell['value']}|{cell['dataType']}|{','.join(cell['headers'])}|"
            encoded_cells.append(encoded)
        
        return encoded_cells
```

### 2.5 Integración con Document Intelligence

```python
# shared/document_intelligence_integration.py
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

class DocumentIntelligenceTableExtractor:
    def __init__(self):
        self.client = DocumentIntelligenceClient(
            endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
        )
    
    async def extract_and_index_tables(self, document_url):
        """
        Extrae tablas y las prepara para Azure AI Search
        """
        # 1. Analizar documento con Document Intelligence
        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            document_url
        )
        result = await poller.result()
        
        # 2. Procesar cada tabla
        indexed_tables = []
        for idx, table in enumerate(result.tables):
            table_data = {
                'id': f"{document_url}_table_{idx}",
                'tableMetadata': {
                    'tableId': f"table_{idx}",
                    'pageNumber': table.bounding_regions[0].page_number if table.bounding_regions else 0,
                    'dimensions': {
                        'rows': table.row_count,
                        'columns': table.column_count
                    }
                },
                'structure': self.extract_structure(table),
                'cells': self.extract_cells(table),
                'semanticContent': await self.generate_semantic_content(table)
            }
            
            # 3. Aplicar chunking si es necesario
            chunker = TableChunkingStrategy()
            chunks = chunker.chunk_large_table(table_data)
            
            indexed_tables.extend(chunks)
        
        return indexed_tables
    
    def extract_structure(self, table):
        """
        Extrae estructura compleja incluyendo headers jerárquicos
        """
        structure = {
            'headers': self.identify_hierarchical_headers(table),
            'mergedCells': self.identify_merged_cells(table)
        }
        return structure
    
    def identify_hierarchical_headers(self, table):
        """
        Identifica headers multi-nivel
        """
        headers = {'level1': [], 'level2': [], 'hierarchicalMap': []}
        
        # Analizar primeras filas para detectar headers
        for cell in table.cells:
            if cell.kind == "columnHeader":
                level = self.determine_header_level(cell)
                headers[f'level{level}'].append(cell.content)
                
                # Mapear jerarquía
                if cell.column_span > 1:
                    headers['hierarchicalMap'].append({
                        'parent': cell.content,
                        'span': cell.column_span,
                        'children': []  # Se llenarán con headers de nivel inferior
                    })
        
        return headers
```

## 3. Consultas Optimizadas para RAG

### 3.1 Patrón de Consulta Híbrida

```python
# orc/enhanced_table_retrieval.py

class EnhancedTableRetrieval:
    def __init__(self, search_client):
        self.search_client = search_client
        self.agentic_search = AgenticTableSearch(search_client)
    
    async def retrieve_for_rag(self, query, context):
        """
        Retrieval optimizado para RAG con tablas complejas
        """
        # 1. Búsqueda híbrida (keyword + vector + semantic)
        results = await self.search_client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=await self.get_query_embedding(query),
                    k_nearest_neighbors=50,
                    fields="vectorEmbeddings"
                )
            ],
            query_type="semantic",
            semantic_configuration_name="table-semantic-config",
            select=["id", "structure", "cells", "semanticContent"],
            top=10,
            # Nuevas características 2025
            facets=["structure/headers/level1", "tableMetadata/dimensions/rows"],
            highlight_fields="cells/value,semanticContent/summary"
        )
        
        # 2. Procesar facetas jerárquicas
        facet_results = results.get_facets()
        
        # 3. Si hay múltiples chunks, recuperar todos
        complete_tables = await self.reconstruct_chunked_tables(results)
        
        # 4. Formatear para LLM
        formatted_sources = self.format_for_llm(complete_tables)
        
        return {
            'sources': formatted_sources,
            'facets': facet_results,
            'total_count': results.get_count()
        }
    
    async def reconstruct_chunked_tables(self, search_results):
        """
        Reconstruye tablas que fueron divididas en chunks
        """
        complete_tables = {}
        
        for result in search_results:
            if result.get('chunkInfo'):
                parent_id = result['chunkInfo']['parentId']
                
                if parent_id not in complete_tables:
                    # Recuperar todos los chunks
                    all_chunks = await self.search_client.search(
                        filter=f"chunkInfo/parentId eq '{parent_id}'",
                        select=["cells", "chunkInfo"],
                        top=100
                    )
                    
                    # Reconstruir tabla completa
                    complete_table = self.merge_chunks(list(all_chunks))
                    complete_tables[parent_id] = complete_table
            else:
                complete_tables[result['id']] = result
        
        return list(complete_tables.values())
    
    def format_for_llm(self, tables):
        """
        Formatea tablas para consumo óptimo por LLM
        """
        formatted = []
        
        for table in tables:
            # Incluir estructura y contenido
            table_repr = {
                'metadata': table['tableMetadata'],
                'structure': {
                    'headers': table['structure']['headers'],
                    'has_merged_cells': len(table['structure'].get('mergedCells', [])) > 0
                },
                'content': self.create_markdown_representation(table),
                'semantic_summary': table['semanticContent']['summary'],
                'key_insights': table['semanticContent'].get('keyInsights', [])
            }
            formatted.append(json.dumps(table_repr, ensure_ascii=False))
        
        return "\n\n".join(formatted)
```

### 3.2 Consultas con Expresiones Lambda Avanzadas

```python
# shared/complex_table_queries.py

class ComplexTableQueries:
    """
    Consultas especializadas para tablas con estructuras complejas
    """
    
    @staticmethod
    def query_hierarchical_data(search_client, parent_header, child_header, value):
        """
        Busca datos en tablas con headers jerárquicos
        """
        filter_expr = (
            f"structure/headers/hierarchicalMap/any("
            f"h: h/parent eq '{parent_header}' and "
            f"h/children/any(c: c eq '{child_header}')"
            f") and "
            f"cells/any(cell: "
            f"cell/headers/column/any(col: col eq '{child_header}') and "
            f"cell/value eq '{value}')"
        )
        
        return search_client.search(
            search_text="*",
            filter=filter_expr,
            select=["id", "structure", "cells"],
            top=10
        )
    
    @staticmethod
    def query_merged_cells(search_client, min_span=2):
        """
        Busca tablas con celdas fusionadas
        """
        filter_expr = (
            f"structure/mergedCells/any(m: "
            f"m/rowSpan ge {min_span} or m/colSpan ge {min_span})"
        )
        
        return search_client.search(
            search_text="*",
            filter=filter_expr,
            select=["id", "structure/mergedCells"],
            top=10
        )
    
    @staticmethod
    def query_calculations(search_client, formula_type):
        """
        Busca tablas con cálculos específicos
        """
        filter_expr = (
            f"semanticContent/relationships/any(r: "
            f"r/type eq 'calculation' and "
            f"search.in(r/formula, '{formula_type}', ','))"
        )
        
        return search_client.search(
            search_text="*",
            filter=filter_expr,
            select=["id", "semanticContent/relationships"],
            top=10
        )
```

## 4. Actualización del Orchestrator

### 4.1 Modificaciones en code_orchestration.py

```python
# Agregar al inicio del archivo
from shared.agentic_table_search import AgenticTableSearch
from shared.document_intelligence_integration import DocumentIntelligenceTableExtractor
from azure.search.documents import SearchClient

# Modificar la función get_answer
async def get_answer(history, document_level, language, process, document_type, country, title):
    # ... código existente ...
    
    # Agregar después de la línea 217 (search_query = ...)
    if await is_table_query(search_query):
        # Usar búsqueda agéntica para queries de tablas
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="tables-index",
            credential=DefaultAzureCredential()
        )
        
        agentic_search = AgenticTableSearch(search_client)
        table_results = await agentic_search.execute_agentic_search(
            search_query,
            {
                'document_level': document_level,
                'language': language,
                'process': process
            }
        )
        
        # Combinar con retrieval tradicional
        sources = await merge_table_and_text_sources(
            table_results,
            traditional_sources
        )
    
    # ... resto del código ...
```

### 4.2 Nuevo Módulo de Configuración

```python
# shared/azure_search_config.py

class AzureSearchConfig:
    """
    Configuración centralizada para Azure AI Search 2025
    """
    
    # API Version
    API_VERSION = "2025-03-01-preview"
    
    # Índices
    TABLES_INDEX = "tables-index"
    DOCUMENTS_INDEX = "documents-index"
    
    # Configuraciones semánticas
    SEMANTIC_CONFIGS = {
        "table-semantic-config": {
            "prioritizedFields": {
                "titleField": {
                    "fieldName": "semanticContent/summary"
                },
                "prioritizedContentFields": [
                    {"fieldName": "cells/value"},
                    {"fieldName": "semanticContent/keyInsights"}
                ],
                "prioritizedKeywordsFields": [
                    {"fieldName": "structure/headers/level1"}
                ]
            }
        }
    }
    
    # Perfiles de búsqueda vectorial
    VECTOR_PROFILES = {
        "table-vector-profile": {
            "algorithm": "hnsw",
            "hnswParameters": {
                "metric": "cosine",
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500
            }
        }
    }
    
    # Límites y configuraciones
    MAX_CELLS_PER_DOCUMENT = 2500
    MAX_SEARCH_RESULTS = 50
    ENABLE_AGENTIC_SEARCH = True
    ENABLE_FACETED_NAVIGATION = True
    USE_SEMANTIC_RANKER = True
```

## 5. Scripts de Migración y Setup

### 5.1 Script de Creación de Índices

```python
# scripts/create_azure_search_indexes.py

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    ComplexField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)

async def create_tables_index():
    """
    Crea el índice optimizado para tablas complejas
    """
    client = SearchIndexClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        credential=DefaultAzureCredential()
    )
    
    # Definir campos
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="documentId", type=SearchFieldDataType.String, filterable=True),
        
        # Metadata de tabla
        ComplexField(
            name="tableMetadata",
            fields=[
                SimpleField(name="tableId", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="pageNumber", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
                ComplexField(
                    name="dimensions",
                    fields=[
                        SimpleField(name="rows", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
                        SimpleField(name="columns", type=SearchFieldDataType.Int32, filterable=True, facetable=True)
                    ]
                )
            ]
        ),
        
        # Estructura compleja
        ComplexField(
            name="structure",
            fields=[
                ComplexField(
                    name="headers",
                    fields=[
                        SearchableField(name="level1", type=SearchFieldDataType.Collection(SearchFieldDataType.String), facetable=True),
                        SearchableField(name="level2", type=SearchFieldDataType.Collection(SearchFieldDataType.String), facetable=True),
                        ComplexField(
                            name="hierarchicalMap",
                            collection=True,
                            fields=[
                                SearchableField(name="parent", type=SearchFieldDataType.String),
                                SimpleField(name="span", type=SearchFieldDataType.Int32),
                                SearchableField(name="children", type=SearchFieldDataType.Collection(SearchFieldDataType.String))
                            ]
                        )
                    ]
                ),
                ComplexField(
                    name="mergedCells",
                    collection=True,
                    fields=[
                        SimpleField(name="startRow", type=SearchFieldDataType.Int32),
                        SimpleField(name="startCol", type=SearchFieldDataType.Int32),
                        SimpleField(name="rowSpan", type=SearchFieldDataType.Int32),
                        SimpleField(name="colSpan", type=SearchFieldDataType.Int32),
                        SearchableField(name="content", type=SearchFieldDataType.String)
                    ]
                )
            ]
        ),
        
        # Celdas (colección compleja)
        ComplexField(
            name="cells",
            collection=True,
            fields=[
                SimpleField(name="row", type=SearchFieldDataType.Int32),
                SimpleField(name="column", type=SearchFieldDataType.Int32),
                SearchableField(name="value", type=SearchFieldDataType.String),
                SimpleField(name="dataType", type=SearchFieldDataType.String, filterable=True),
                ComplexField(
                    name="headers",
                    fields=[
                        SearchableField(name="row", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
                        SearchableField(name="column", type=SearchFieldDataType.Collection(SearchFieldDataType.String))
                    ]
                ),
                SearchableField(name="context", type=SearchFieldDataType.String)
            ]
        ),
        
        # Contenido semántico
        ComplexField(
            name="semanticContent",
            fields=[
                SearchableField(name="summary", type=SearchFieldDataType.String),
                SearchableField(name="keyInsights", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
                ComplexField(
                    name="relationships",
                    collection=True,
                    fields=[
                        SimpleField(name="type", type=SearchFieldDataType.String, filterable=True),
                        SearchableField(name="formula", type=SearchFieldDataType.String),
                        SearchableField(name="cells", type=SearchFieldDataType.Collection(SearchFieldDataType.String))
                    ]
                )
            ]
        ),
        
        # Embeddings vectoriales
        SimpleField(
            name="vectorEmbeddings",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,
            vector_search_profile_name="table-vector-profile"
        ),
        
        # Información de chunking
        ComplexField(
            name="chunkInfo",
            fields=[
                SimpleField(name="isChunked", type=SearchFieldDataType.Boolean, filterable=True),
                SimpleField(name="chunkIndex", type=SearchFieldDataType.Int32),
                SimpleField(name="totalChunks", type=SearchFieldDataType.Int32),
                SimpleField(name="parentId", type=SearchFieldDataType.String, filterable=True)
            ]
        )
    ]
    
    # Configuración de búsqueda vectorial
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algorithm",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="table-vector-profile",
                algorithm_configuration_name="hnsw-algorithm"
            )
        ]
    )
    
    # Configuración semántica
    semantic_config = SemanticConfiguration(
        name="table-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="semanticContent/summary"),
            content_fields=[
                SemanticField(field_name="cells/value"),
                SemanticField(field_name="semanticContent/keyInsights")
            ],
            keywords_fields=[
                SemanticField(field_name="structure/headers/level1")
            ]
        )
    )
    
    # Crear índice
    index = SearchIndex(
        name="tables-index",
        fields=fields,
        vector_search=vector_search,
        semantic_configurations=[semantic_config]
    )
    
    result = await client.create_or_update_index(index)
    print(f"Índice creado: {result.name}")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_tables_index())
```

## 6. Monitoreo y Métricas

### 6.1 Dashboard de Performance

```python
# shared/table_metrics.py

class TableSearchMetrics:
    """
    Métricas específicas para búsqueda de tablas
    """
    
    def __init__(self, app_insights_client):
        self.telemetry = app_insights_client
    
    def track_table_search(self, query, results_count, latency, search_type):
        """
        Registra métricas de búsqueda de tablas
        """
        self.telemetry.track_event(
            "TableSearch",
            {
                "query": query[:100],
                "resultsCount": results_count,
                "searchType": search_type,
                "useAgenticSearch": True
            },
            {
                "latency": latency,
                "averageRelevanceScore": self.calculate_avg_score(results)
            }
        )
    
    def track_chunking_operation(self, table_id, chunks_created, original_size):
        """
        Registra operaciones de chunking
        """
        self.telemetry.track_metric(
            "TableChunking",
            chunks_created,
            properties={
                "tableId": table_id,
                "originalSize": original_size,
                "compressionRatio": original_size / (chunks_created * 2500)
            }
        )
    
    def track_facet_usage(self, facets_requested, facets_returned):
        """
        Registra uso de facetas jerárquicas
        """
        self.telemetry.track_event(
            "FacetUsage",
            {
                "facetsRequested": facets_requested,
                "facetsReturned": facets_returned,
                "hierarchicalFacets": True
            }
        )
```

## 7. Conclusiones y Beneficios

### Mejoras Clave con Azure AI Search 2025

1. **Búsqueda Agéntica**: Reduce latencia mediante paralelización inteligente
2. **Tipos Complejos Nativos**: Preserva estructura jerárquica sin pérdida de contexto
3. **Facetas Jerárquicas**: Navegación intuitiva en datos tabulares complejos
4. **Expresiones Lambda**: Consultas precisas sobre colecciones complejas
5. **Integración Document Intelligence**: Extracción precisa de tablas desde documentos

### Métricas de Impacto Esperadas

- **Precisión de Retrieval**: +40% en queries sobre tablas complejas
- **Latencia**: -30% mediante búsqueda agéntica paralela
- **Cobertura**: 100% de tipos de tablas (con/sin bordes, fusionadas, jerárquicas)
- **Escalabilidad**: Manejo de tablas con >10,000 celdas mediante chunking

### Próximos Pasos Técnicos

1. Configurar Azure AI Search con API 2025-03-01-preview
2. Crear índices con esquema de tipos complejos
3. Implementar pipeline de indexación con Document Intelligence
4. Desarrollar capa de búsqueda agéntica
5. Actualizar orchestrator para usar nuevas capacidades
6. Ejecutar pruebas con datasets de tablas complejas reales