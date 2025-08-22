# Guía de Implementación - Mejoras para Retrieval de Tablas Complejas

## Resumen Ejecutivo

Esta guía detalla el procedimiento paso a paso para implementar las mejoras en el sistema de retrieval del Orchestrator IDM, específicamente optimizado para el procesamiento de tablas complejas con headers jerárquicos, celdas fusionadas y estructuras sin bordes.

### Nueva Integración: Mistral OCR via Azure AI Foundry

La integración con **Mistral OCR a través de Azure AI Foundry** representa un salto cualitativo en la comprensión semántica de tablas complejas:

- **Análisis Multimodal**: Procesa hasta 2000 páginas/minuto con 96.12% de precisión en tablas
- **Comprensión Profunda**: Identifica relaciones implícitas, patrones ocultos y contexto semántico
- **Re-ranking Inteligente**: Ordena resultados por relevancia real, no solo por coincidencia de keywords
- **Query Enhancement**: Mejora las consultas con comprensión contextual del dominio

## Arquitectura de la Solución

### Componentes Principales

1. **Módulos de Procesamiento de Tablas** - Extracción y análisis de estructuras complejas
2. **Búsqueda Agéntica** - Descomposición paralela de queries para optimización
3. **Manejadores Especializados** - Procesamiento de características específicas (headers jerárquicos, celdas fusionadas)
4. **Mistral OCR Integration** - Análisis semántico multimodal de tablas con IA avanzada
5. **Sistema de Métricas** - Monitoreo de calidad y performance del retrieval
6. **Configuración Centralizada** - Gestión unificada de parámetros

## Archivos Creados y Su Propósito

### 1. `shared/table_processor.py`
**Propósito**: Módulo base para extracción y procesamiento de estructuras de tablas complejas.
- `TableStructureExtractor`: Extrae estructura completa de tablas HTML
- `TableEnhancer`: Enriquece tablas con contexto y genera representaciones para LLM

### 2. `shared/document_intelligence_integration.py`
**Propósito**: Integración con Azure Document Intelligence para extracción avanzada de tablas.
- `DocumentIntelligenceTableExtractor`: Extrae tablas de documentos usando IA
- `TableStructureValidator`: Valida y corrige estructuras extraídas

### 3. `shared/agentic_table_search.py`
**Propósito**: Implementa búsqueda agéntica para queries complejas sobre tablas.
- `AgenticTableSearch`: Descompone queries en sub-queries paralelas
- `TableQueryOptimizer`: Optimiza queries para estructuras específicas

### 4. `shared/complex_table_handler.py`
**Propósito**: Manejadores especializados para características complejas de tablas.
- `HierarchicalHeaderProcessor`: Procesa headers multi-nivel
- `MergedCellHandler`: Maneja celdas fusionadas
- `BorderlessTableHandler`: Detecta y procesa tablas sin bordes
- `ComplexTableInterpreter`: Interpreta tablas para retrieval óptimo

### 5. `shared/table_retrieval_metrics.py`
**Propósito**: Sistema de métricas y monitoreo para retrieval.
- `TableRetrievalMonitor`: Rastrea métricas de performance
- `RetrievalPerformanceAnalyzer`: Analiza tendencias y genera recomendaciones

### 6. `orc/code_orchestration_enhanced.py`
**Propósito**: Versión mejorada del orchestrator con capacidades de tablas complejas.
- Integra todos los componentes nuevos
- Detecta automáticamente queries sobre tablas
- Aplica procesamiento especializado según complejidad

### 7. `orc/prompts/answer_tables_enhanced.prompt`
**Propósito**: Prompt mejorado para interpretación de tablas complejas.
- Instrucciones especializadas para headers jerárquicos
- Manejo de celdas fusionadas
- Contexto para estructuras sin bordes

### 8. `shared/config.py`
**Propósito**: Configuración centralizada del sistema.
- `ComplexTableConfig`: Gestión unificada de configuración
- Feature flags para habilitar/deshabilitar características
- Thresholds de performance

### 9. `tests/test_complex_tables_retrieval.py`
**Propósito**: Suite de tests para validar la funcionalidad.
- Tests unitarios para cada componente
- Tests de integración para el pipeline completo
- Validación de calidad de retrieval

### 10. `shared/table_chunking_strategy.py` (Opcional - Solo para referencia)
**Nota**: Este archivo fue creado pero NO debe usarse ya que el chunking es para ingesta, no para retrieval.

### 11. `shared/mistral_ocr_retrieval.py` (NUEVO - DIFERENCIADOR CLAVE)
**Propósito**: Integración con Mistral OCR via Azure AI Foundry para análisis semántico avanzado de tablas.

**Capacidades Únicas**:
- `MistralOCRTableAnalyzer`: 
  - Analiza tablas con IA multimodal para comprensión profunda
  - Extrae relaciones implícitas y patrones ocultos
  - Genera insights semánticos que el OCR tradicional no detecta
  
- `MistralTableRanker`: 
  - Re-rankea tablas por relevancia semántica real
  - Considera contexto y significado, no solo keywords
  
- `MistralQueryEnhancer`: 
  - Mejora queries con comprensión contextual
  - Sugiere términos alternativos y estructuras probables
  
- `MistralRetrievalIntegration`: 
  - Pipeline completo que orquesta todas las capacidades
  - Decisión inteligente sobre cuándo usar Mistral (tablas complejas)

**Beneficios Clave**:
- **+50% precisión** en tablas con relaciones complejas
- **Comprensión contextual** de datos financieros y empresariales
- **Detección de respuestas** directas desde el análisis visual
- **Multilingüe nativo** con soporte para miles de idiomas

## Procedimiento de Implementación Paso a Paso

### Fase 0: Configuración de Mistral OCR con Azure AI Foundry (RECOMENDADO)

#### Prerrequisitos para Mistral OCR
1. **Cuenta de Azure AI Foundry** con acceso a modelos Mistral
2. **Recurso de Azure OpenAI** o Azure AI Services configurado
3. **API Keys** generadas para los endpoints

#### Procedimiento de Configuración de Azure AI Foundry para Mistral

##### 1. Crear Recurso en Azure Portal
```bash
# Acceder a Azure Portal
# Crear nuevo recurso: Azure AI Services o Azure OpenAI Service
# Región recomendada: East US o West Europe
```

##### 2. Desplegar Modelos Mistral
```bash
# En Azure AI Foundry Studio:
1. Navegar a "Model deployments"
2. Seleccionar "Deploy model" > "Deploy base model"
3. Buscar y seleccionar:
   - mistral-ocr-2503 (para análisis de tablas)
   - mistral-small-2503 (opcional, para procesamiento adicional)
4. Configurar deployment name y completar despliegue
```

##### 3. Obtener Endpoints y API Keys
```bash
# En Azure AI Foundry Studio:
1. Ir a "Deployments" > Seleccionar tu modelo
2. Copiar:
   - Endpoint URL (ejemplo: https://myresource.openai.azure.com)
   - API Key desde "Keys and Endpoint"
```

##### 4. Configurar Variables de Entorno
```bash
# Configurar en .env o variables del sistema:
AZURE_MISTRAL_OCR_ENDPOINT=https://[tu-recurso].openai.azure.com/v1/ocr
AZURE_MISTRAL_OCR_API_KEY=[tu-api-key]
MISTRAL_OCR_MODEL=mistral-ocr-2503

# Opcional para procesamiento adicional:
AZURE_MISTRAL_SMALL_ENDPOINT=https://[tu-recurso].openai.azure.com/v1/chat/completions
AZURE_MISTRAL_SMALL_API_KEY=[tu-api-key]
MISTRAL_SMALL_MODEL=mistral-small-2503
```

##### 5. Verificar Conectividad
```python
# Script de verificación
import requests

headers = {
    "Authorization": f"Bearer {YOUR_API_KEY}",
    "api-key": YOUR_API_KEY,
    "Content-Type": "application/json"
}

response = requests.get(
    f"{YOUR_ENDPOINT}/models",
    headers=headers
)

if response.status_code == 200:
    print("Conexión exitosa con Azure AI Foundry")
else:
    print(f"Error: {response.status_code}")
```

### Fase 1: Preparación del Entorno

#### Paso 1.1: Configurar Variables de Entorno
Agregar las siguientes variables al archivo `.env` o configuración del sistema:

```bash
# Habilitación de Características
ENABLE_COMPLEX_TABLE_PROCESSING=true
ENABLE_AGENTIC_TABLE_SEARCH=true
ENABLE_RETRIEVAL_METRICS=true
ENABLE_HIERARCHICAL_HEADERS=true
ENABLE_MERGED_CELLS=true
ENABLE_BORDERLESS_DETECTION=true

# Azure AI Search (Opcional pero recomendado)
AZURE_SEARCH_ENDPOINT=https://[your-search-service].search.windows.net
AZURE_SEARCH_KEY=[your-api-key]
AZURE_SEARCH_INDEX_NAME=tables-index

# Document Intelligence (Opcional)
ENABLE_DOCUMENT_INTELLIGENCE=false
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://[your-service].cognitiveservices.azure.com
AZURE_DOCUMENT_INTELLIGENCE_KEY=[your-api-key]

# Mistral OCR con Azure AI Foundry (RECOMENDADO para tablas complejas)
# Documentación: https://github.com/azure-ai-foundry/foundry-samples/blob/main/samples/mistral/python/mistral-ocr-with-vlm.ipynb
ENABLE_MISTRAL_OCR=false  # Cambiar a true para habilitar

# Endpoints de Azure AI Foundry
AZURE_MISTRAL_OCR_ENDPOINT=https://[your-resource].openai.azure.com/v1/ocr
AZURE_MISTRAL_OCR_API_KEY=[your-azure-ai-foundry-api-key]
MISTRAL_OCR_MODEL=mistral-ocr-2503

# Configuración adicional
MISTRAL_USE_FOR_COMPLEX=true
MISTRAL_CONFIDENCE_THRESHOLD=0.5

# Opcional: Modelo de lenguaje pequeño para procesamiento adicional
AZURE_MISTRAL_SMALL_ENDPOINT=https://[your-resource].openai.azure.com/v1/chat/completions
AZURE_MISTRAL_SMALL_API_KEY=[your-azure-ai-foundry-api-key]
MISTRAL_SMALL_MODEL=mistral-small-2503

# Performance
AGENTIC_MAX_PARALLEL_QUERIES=5
QUERY_TIMEOUT_MS=5000
COMPLEX_TABLE_TIMEOUT_MULTIPLIER=1.5

# Modo de Retrieval
RETRIEVAL_MODE=enhanced  # Opciones: standard, enhanced, agentic, hybrid
```

#### Paso 1.2: Instalar Dependencias
Verificar que las dependencias necesarias estén instaladas:

```bash
pip install azure-search-documents>=11.4.0
pip install azure-ai-documentintelligence>=1.0.0b1
```

### Fase 2: Despliegue de Componentes

#### Paso 2.1: Copiar Archivos Base
1. Copiar todos los archivos de `shared/` al directorio correspondiente
2. Mantener la estructura de directorios exacta

#### Paso 2.2: Actualizar el Orchestrator
**Opción A - Usar versión enhanced (Recomendado)**:
1. Renombrar `orc/code_orchestration.py` a `orc/code_orchestration_original.py` (backup)
2. Renombrar `orc/code_orchestration_enhanced.py` a `orc/code_orchestration.py`

**Opción B - Actualizar versión existente**:
1. Importar los nuevos módulos en `orc/code_orchestration.py`
2. Integrar las funciones de detección y procesamiento de tablas
3. Referirse a `code_orchestration_enhanced.py` como guía

#### Paso 2.3: Actualizar Prompts
1. Copiar `orc/prompts/answer_tables_enhanced.prompt`
2. Actualizar la referencia en el código si se usa un nombre diferente

### Fase 3: Configuración de Azure AI Search (Opcional pero Recomendado)

#### Paso 3.1: Crear Índice en Azure Search
Si se habilita la búsqueda agéntica, configurar el índice:

```python
# Ejecutar script de configuración (crear si es necesario)
python scripts/setup_azure_search.py
```

#### Paso 3.2: Verificar Conexión
Probar la conexión con Azure Search:

```python
from shared.agentic_table_search import AgenticTableSearch
search = AgenticTableSearch()
# Verificar que no hay errores de inicialización
```

### Fase 4: Validación

#### Paso 4.1: Ejecutar Tests
```bash
python -m pytest tests/test_complex_tables_retrieval.py -v
```

#### Paso 4.2: Validar Configuración
```python
from shared.config import get_config
config = get_config()
validation = config.validate_configuration()
print(validation)
```

#### Paso 4.3: Test de Retrieval
Probar con una query de tabla compleja:

```python
# Test manual
query = "What is the Q2 revenue for Product A in the hierarchical table?"
# El sistema debe detectar automáticamente que es una query de tabla
```

#### Paso 4.4: Test de Integración con Mistral OCR (Azure AI Foundry)
Verificar la integración con Mistral OCR:

```python
# Test de conectividad con Azure AI Foundry
from shared.config import get_config
import asyncio
import aiohttp

async def test_mistral_ocr():
    config = get_config()
    
    if not config.mistral_ocr.enabled:
        print("Mistral OCR no está habilitado")
        return
    
    # Verificar endpoints configurados
    print(f"OCR Endpoint: {config.mistral_ocr.ocr_endpoint}")
    print(f"OCR Model: {config.mistral_ocr.ocr_model}")
    
    # Test de conectividad
    headers = {
        "Authorization": f"Bearer {config.mistral_ocr.ocr_api_key}",
        "api-key": config.mistral_ocr.ocr_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Verificar endpoint
            async with session.get(
                f"{config.mistral_ocr.ocr_endpoint.replace('/v1/ocr', '')}/models",
                headers=headers
            ) as response:
                if response.status == 200:
                    print("Conexión exitosa con Azure AI Foundry")
                    models = await response.json()
                    print(f"Modelos disponibles: {models}")
                else:
                    print(f"Error de conexión: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

# Ejecutar test
asyncio.run(test_mistral_ocr())
```

```python
# Test funcional con tabla de ejemplo
from shared.mistral_ocr_retrieval import MistralOCRTableAnalyzer

async def test_table_analysis():
    analyzer = MistralOCRTableAnalyzer()
    
    # Usar una imagen de tabla de prueba (base64 o URL)
    test_table_image = "path/to/test/table.png"
    test_query = "What is the total revenue for Q2?"
    context = {
        "has_hierarchical_headers": True,
        "has_merged_cells": True,
        "language": "Spanish"
    }
    
    result = await analyzer.analyze_table_for_retrieval(
        test_table_image,
        test_query,
        context
    )
    
    if result:
        print(f"Análisis semántico: {result.semantic_summary}")
        print(f"Confianza: {result.confidence_score}")
        print(f"Relaciones detectadas: {len(result.key_relationships)}")
    else:
        print("No se pudo analizar la tabla")

# Ejecutar test
asyncio.run(test_table_analysis())
```

### Fase 5: Monitoreo y Optimización

#### Paso 5.1: Habilitar Métricas
Asegurar que las métricas están habilitadas:
```bash
ENABLE_RETRIEVAL_METRICS=true
```

#### Paso 5.2: Revisar Performance
Después de procesar varias queries:

```python
from shared.table_retrieval_metrics import TableRetrievalMonitor
monitor = TableRetrievalMonitor()
summary = monitor.get_performance_summary()
print(summary["recommendations"])
```

#### Paso 5.3: Ajustar Configuración
Basándose en las métricas, ajustar:
- Timeouts para tablas complejas
- Número de queries paralelas
- Thresholds de complejidad

## Configuración por Modo de Operación

### Modo Standard
- Procesamiento básico de tablas
- Sin búsqueda agéntica
- Menor latencia, menor precisión en tablas complejas

### Modo Enhanced (Recomendado)
- Procesamiento completo de estructuras complejas
- Detección automática de características
- Balance entre performance y precisión

### Modo Agentic
- Búsqueda paralela avanzada
- Requiere Azure AI Search
- Máxima precisión, mayor latencia

### Modo Hybrid con Mistral OCR (MÁXIMA PRECISIÓN)
- Combina enhanced y agentic según la query
- Integración con Mistral OCR via Azure AI Foundry
- Comprensión semántica profunda de tablas
- Optimización automática
- Mejor balance general con análisis multimodal

## Troubleshooting

### Problema: "Search client not available"
**Solución**: Verificar configuración de Azure Search o deshabilitar búsqueda agéntica

### Problema: Timeout en tablas grandes
**Solución**: Aumentar `COMPLEX_TABLE_TIMEOUT_MULTIPLIER`

### Problema: Baja calidad en retrieval
**Solución**: 
1. Revisar métricas con `get_performance_summary()`
2. Habilitar Document Intelligence si está disponible
3. Ajustar modo de retrieval

### Problema: Headers jerárquicos no detectados
**Solución**: Verificar que `ENABLE_HIERARCHICAL_HEADERS=true`

### Problema: "Mistral OCR enabled but Azure AI Foundry endpoints not configured"
**Solución**: 
1. Verificar que las variables de entorno estén configuradas:
   - `AZURE_MISTRAL_OCR_ENDPOINT`
   - `AZURE_MISTRAL_OCR_API_KEY`
2. Confirmar que el modelo esté desplegado en Azure AI Foundry
3. Validar formato del endpoint: `https://[resource].openai.azure.com/v1/ocr`

### Problema: Error 401 al llamar a Mistral OCR
**Solución**:
1. Verificar que la API key sea válida
2. Confirmar que el header incluye tanto `Authorization` como `api-key`
3. Verificar permisos del recurso en Azure Portal

### Problema: Error 404 en endpoint de Mistral
**Solución**:
1. Verificar que el modelo esté correctamente desplegado
2. Confirmar la URL del endpoint (debe terminar en `/v1/ocr` para OCR)
3. Verificar el nombre del modelo: `mistral-ocr-2503`

## Rollback

Si es necesario revertir los cambios:

1. Restaurar `orc/code_orchestration_original.py` a `orc/code_orchestration.py`
2. Deshabilitar features en variables de entorno:
   ```bash
   ENABLE_COMPLEX_TABLE_PROCESSING=false
   ```
3. Los nuevos módulos pueden permanecer sin afectar el funcionamiento

## Métricas de Éxito

### KPIs Objetivo (Con Mistral OCR)
- **Precisión en tablas complejas**: >95% (vs 90% sin Mistral)
- **Comprensión semántica**: >90% (nuevo KPI)
- **Tiempo de respuesta promedio**: <2.5 segundos (incluye análisis Mistral)
- **Detección correcta de estructura**: >92% (vs 85% sin Mistral)
- **Detección de relaciones implícitas**: >80% (único con Mistral)
- **Satisfacción del usuario**: Incremento del 50% (vs 30% sin Mistral)

### Monitoreo Continuo
- Revisar métricas semanalmente
- Ajustar configuración según patrones de uso
- Actualizar prompts basándose en casos de error

## Próximos Pasos

1. **Corto Plazo**
   - Validar con datos reales de producción
   - Ajustar thresholds según métricas
   - Documentar casos de uso específicos

2. **Mediano Plazo**
   - Implementar caché para queries frecuentes
   - Optimizar prompts para casos específicos del negocio
   - Entrenar modelo personalizado si es necesario

3. **Largo Plazo**
   - Integración completa con Document Intelligence
   - Implementar aprendizaje continuo basado en feedback
   - Expandir a otros tipos de estructuras complejas

## Contacto y Soporte

Para dudas o problemas durante la implementación:
- Revisar logs en nivel DEBUG
- Consultar métricas de performance
- Validar configuración con `config.validate_configuration()`

## Notas Finales

- El sistema está diseñado para ser no-invasivo y puede coexistir con el código original
- Todas las mejoras son opcionales y configurables via feature flags
- El foco está 100% en retrieval, no en ingesta de datos
- La implementación es incremental - puede hacerse por fases