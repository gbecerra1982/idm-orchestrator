from shared.util import get_secret, get_aoai_config
from semantic_kernel.skill_definition import sk_function
from semantic_kernel.orchestration.sk_context import SKContext
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import logging
import openai
import os
import re
import requests
import time

# Azure OpenAI Integration Settings
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")

TERM_SEARCH_APPROACH='term'
VECTOR_SEARCH_APPROACH='vector'
HYBRID_SEARCH_APPROACH='hybrid'
AZURE_SEARCH_USE_SEMANTIC=os.environ.get("AZURE_SEARCH_USE_SEMANTIC") or "false"
AZURE_SEARCH_APPROACH=os.environ.get("AZURE_SEARCH_APPROACH") or HYBRID_SEARCH_APPROACH

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION", "2023-11-01")
if AZURE_SEARCH_API_VERSION < '2023-10-01-Preview':
    AZURE_SEARCH_API_VERSION = '2023-11-01'  

# Enhanced configuration for tables
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K") or "15"  # Increased from 3
AZURE_SEARCH_TABLE_BOOST = os.environ.get("AZURE_SEARCH_TABLE_BOOST") or "2.0"
ENABLE_TABLE_AWARE_SEARCH = os.environ.get("ENABLE_TABLE_AWARE_SEARCH") or "true"
TABLE_QUERY_EXPANSION = os.environ.get("TABLE_QUERY_EXPANSION") or "true"
MAX_TABLE_SOURCES = os.environ.get("MAX_TABLE_SOURCES") or "5"

AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH") or "false"
AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = True if AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH == "true" else False
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG") or "my-semantic-config"
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN") or "true"
AZURE_SEARCH_ENABLE_IN_DOMAIN = True if AZURE_SEARCH_ENABLE_IN_DOMAIN == "true" else False
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS") or "content"
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN") or "filepath"
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN") or "title"
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN") or "url"

# Set up logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def generate_embeddings(text):
    embeddings_config = get_aoai_config(AZURE_OPENAI_EMBEDDING_MODEL)
    
    openai.api_type = "azure_ad"
    openai.api_base = embeddings_config['endpoint']
    openai.api_version = embeddings_config['api_version']
    openai.api_key = embeddings_config['api_key']
    
    response = openai.Embedding.create(
        input=text, engine=embeddings_config['deployment'])
    embeddings = response['data'][0]['embedding']
    return embeddings

class RAG:
    @sk_function(
        description=re.sub('\s+', ' ', f"""
            Enhanced search for tables and complex documents in knowledge base.
            Optimized for table queries, authorization levels, and merged cells."""),
        name="Retrieval",
        input_description="The user question with query optimization",
    )
    def Retrieval(self, input: str, context: SKContext) -> str:
        search_results = []
        input_json = json.loads(input.strip("`json\n`"))
        search_query = input_json['query_string']
        
        try:
            start_time = time.time()
            
            # Detect if this is a table-related query
            is_table_query = self.detect_table_query(search_query, context)
            if is_table_query:
                logging.info(f"[sk_retrieval] Table query detected: {search_query}")
            
            # Expand query for better table retrieval
            if is_table_query and TABLE_QUERY_EXPANSION == "true":
                expanded_queries = self.expand_table_query(search_query)
                logging.info(f"[sk_retrieval] Query expanded to: {expanded_queries}")
            else:
                expanded_queries = [search_query]
            
            # Generate embeddings for the original query
            logging.info(f"[sk_retrieval] generating question embeddings. search query: {search_query}")
            embeddings_query = generate_embeddings(search_query)
            response_time = round(time.time() - start_time, 2)
            logging.info(f"[sk_retrieval] finished generating question embeddings. {response_time} seconds")
            
            azureSearchKey = get_secret('azureSearchKey')
            
            all_results = []
            
            # Execute searches for all query variations
            for query in expanded_queries:
                logging.info(f"[sk_retrieval] querying azure ai search with: {query}")
                
                # Build optimized search body
                body = self.build_search_body(
                    query,
                    embeddings_query,
                    is_table_query,
                    context
                )
                
                headers = {
                    'Content-Type': 'application/json',
                    'api-key': azureSearchKey
                }
                search_endpoint = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"
                
                start_time = time.time()
                response = requests.post(search_endpoint, headers=headers, json=body)
                status_code = response.status_code
                
                if status_code >= 400:
                    error_message = f'Status code: {status_code}.'
                    if response.text != "": 
                        error_message += f" Error: {response.text}."
                    logging.error(f"[sk_retrieval] error {status_code} when searching documents. {error_message}")
                else:
                    if response.json()['value']:
                        all_results.extend(response.json()['value'])
                
                response_time = round(time.time() - start_time, 2)
                logging.info(f"[sk_retrieval] finished querying azure ai search. {response_time} seconds")
            
            # Deduplicate and rank results
            unique_results = self.deduplicate_and_rank(all_results, is_table_query)
            
            # Process results with table awareness
            for doc in unique_results[:int(AZURE_SEARCH_TOP_K)]:
                if is_table_query and doc.get('relatedImages'):
                    # Enrich table content with structural information
                    enriched_content = self.enrich_table_content(doc)
                    search_results.append(enriched_content)
                else:
                    # Standard content processing
                    content = doc['metadata_storage_name'] + ": " + doc['content'].strip()
                    if doc.get('relatedImages'):
                        content += "\n" + f"relatedImages: {str(doc['relatedImages'])}"
                    search_results.append(content + "\n")
            
            logging.info(f"[sk_retrieval] Retrieved {len(search_results)} sources for {'table' if is_table_query else 'standard'} query")
            
        except Exception as e:
            error_message = str(e)
            logging.error(f"[sk_retrieval] error when getting the answer {error_message}")
        
        sources = '\n'.join(search_results)
        return sources
    
    def detect_table_query(self, query: str, context: SKContext) -> bool:
        """
        Detects if the query is looking for table information
        """
        table_indicators = [
            'tabla', 'table', 'columna', 'column', 'fila', 'row',
            'celda', 'cell', 'authorization level', 'approval level',
            'matriz', 'matrix', 'listado', 'list', 'comparación',
            'comparison', 'valores', 'values', 'nivel', 'level',
            'límite', 'limit', 'threshold', 'monto', 'amount'
        ]
        
        query_lower = query.lower()
        
        # Check for table indicators in query
        has_table_indicator = any(indicator in query_lower for indicator in table_indicators)
        
        # Check context variables
        is_marked_as_table = context.variables.get("is_table_query") == "true"
        
        return has_table_indicator or is_marked_as_table
    
    def expand_table_query(self, query: str) -> list:
        """
        Expands query to improve table retrieval
        """
        expanded = [query]
        query_lower = query.lower()
        
        # Handle authorization/approval level queries
        if 'authorization level' in query_lower:
            expanded.append(query.replace('authorization level', 'approval level'))
            expanded.append(query + ' table matrix authorization approval')
        elif 'approval level' in query_lower:
            expanded.append(query.replace('approval level', 'authorization level'))
            expanded.append(query + ' table matrix authorization approval')
        elif 'nivel de autorización' in query_lower:
            expanded.append(query.replace('nivel de autorización', 'nivel de aprobación'))
            expanded.append(query + ' tabla matriz autorización aprobación')
        elif 'nivel de aprobación' in query_lower:
            expanded.append(query.replace('nivel de aprobación', 'nivel de autorización'))
            expanded.append(query + ' tabla matriz autorización aprobación')
        
        # Add table context if not already present
        if 'table' not in query_lower and 'tabla' not in query_lower:
            if len(expanded) == 1:  # No expansions yet
                expanded.append(query + ' table')
                expanded.append(query + ' matrix values')
        
        # Limit to 3 query variations
        return expanded[:3]
    
    def build_search_body(self, query: str, embeddings: list, is_table: bool, context: SKContext) -> dict:
        """
        Builds optimized search body for table queries
        """
        # Enhanced field selection for tables
        select_fields = (
            "title, content, url, filepath, chunk_id, document_level, "
            "language, process, document_type, country, metadata_storage_name, "
            "chunk_headers, relatedImages"
        )
        
        # Add table-specific fields if available
        if is_table:
            select_fields += ", table_headers, merged_cells, cell_relationships"
        
        body = {
            "select": select_fields,
            "top": int(AZURE_SEARCH_TOP_K)
        }
        
        # Configure search approach
        if AZURE_SEARCH_APPROACH == TERM_SEARCH_APPROACH:
            body["search"] = query
            if is_table:
                body["searchFields"] = "content,table_headers,merged_cells"
        elif AZURE_SEARCH_APPROACH == VECTOR_SEARCH_APPROACH:
            body["vectorQueries"] = [{
                "kind": "vector",
                "vector": embeddings,
                "fields": "contentVector",
                "k": int(AZURE_SEARCH_TOP_K)
            }]
        elif AZURE_SEARCH_APPROACH == HYBRID_SEARCH_APPROACH:
            body["search"] = query
            body["vectorQueries"] = [{
                "kind": "vector",
                "vector": embeddings,
                "fields": "contentVector",
                "k": int(AZURE_SEARCH_TOP_K)
            }]
            
            # Adjust vector weight for table queries
            if is_table:
                body["vectorQueries"][0]["weight"] = 0.3  # Less weight on vector for tables
                body["searchFields"] = "content,table_headers,merged_cells"
        
        # Build metadata filters
        meta_field_list = ["document_level", "language", "process", "document_type", "country", "title"]
        meta_field_list_dict = []
        for meta_field in meta_field_list:
            if context.variables.get(meta_field):
                meta_field_list_dict.append({
                    "meta_name": meta_field, 
                    "meta_value": context.variables.get(meta_field)
                })
        
        search_filter = " and ".join(
            [f"country/any(country: country eq '{dic['meta_value']}')" 
                if dic['meta_name'] == 'country' 
                else f"{dic['meta_name']} eq '{dic['meta_value']}'" 
                for dic in meta_field_list_dict
            ]
        )
        
        if search_filter and search_filter != "":
            body["vectorFilterMode"] = "postFilter"
            body["filter"] = search_filter
            logging.info(f"[sk_retrieval] dynamic filter: {search_filter}")
        
        # Enable semantic search for better understanding
        if AZURE_SEARCH_USE_SEMANTIC == "true" and AZURE_SEARCH_APPROACH != VECTOR_SEARCH_APPROACH:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG
            if is_table:
                body["answers"] = "extractive"  # Get extractive answers for tables
                body["captions"] = "extractive"  # Get extractive captions
        
        return body
    
    def enrich_table_content(self, doc: dict) -> str:
        """
        Enriches table content with structural metadata
        """
        content_parts = []
        
        # Document name
        content_parts.append(f"{doc['metadata_storage_name']}:")
        
        # Add table headers if present
        if doc.get('table_headers'):
            content_parts.append(f"Table Headers: {doc['table_headers']}")
        
        # Add merged cells information if present
        if doc.get('merged_cells'):
            content_parts.append(f"Merged Cells Structure: {doc['merged_cells']}")
        
        # Add chunk headers for context
        if doc.get('chunk_headers'):
            content_parts.append(f"Section Context: {doc['chunk_headers']}")
        
        # Main content
        content_parts.append(doc['content'].strip())
        
        # Related images with table context
        if doc.get('relatedImages'):
            content_parts.append(f"Table Images: {str(doc['relatedImages'])}")
            content_parts.append("Note: Complex table with merged cells and hierarchical structure")
        
        # Cell relationships if available
        if doc.get('cell_relationships'):
            content_parts.append(f"Cell Relationships: {doc['cell_relationships']}")
        
        return '\n'.join(content_parts) + '\n'
    
    def deduplicate_and_rank(self, results: list, is_table_query: bool) -> list:
        """
        Deduplicates and ranks results with table preference
        """
        seen = set()
        unique = []
        
        # Custom scoring for table queries
        if is_table_query:
            for doc in results:
                # Calculate table relevance score
                table_score = 0
                if doc.get('relatedImages'):
                    table_score += len(doc['relatedImages']) * 2
                if doc.get('table_headers'):
                    table_score += 3
                if doc.get('merged_cells'):
                    table_score += 2
                if 'authorization' in doc.get('content', '').lower() or 'approval' in doc.get('content', '').lower():
                    table_score += 5
                
                doc['table_relevance_score'] = table_score
            
            # Sort by table relevance first, then by search score
            results.sort(
                key=lambda x: (
                    x.get('table_relevance_score', 0),
                    x.get('@search.score', 0)
                ),
                reverse=True
            )
        
        # Deduplicate
        for doc in results:
            doc_id = doc.get('chunk_id', doc.get('metadata_storage_name'))
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        
        logging.info(f"[sk_retrieval] Deduplicated {len(results)} results to {len(unique)} unique documents")
        
        return unique