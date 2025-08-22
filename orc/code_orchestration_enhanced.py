"""
Enhanced Code Orchestration Module
Enhanced version with complex table retrieval capabilities
Focus: Optimized retrieval for complex tables with hierarchical headers and merged cells
"""
import ast
import base64
import json
import logging
import os
import re
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from shared.util import call_semantic_function, get_chat_history_as_messages, get_message
from shared.util import get_aoai_config, get_blocked_list, chat_completion, get_secret
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse, unquote
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.core_skills import ConversationSummarySkill

# Import new table processing modules
from shared.table_processor import TableStructureExtractor, TableEnhancer
from shared.complex_table_handler import ComplexTableInterpreter
from shared.agentic_table_search import AgenticTableSearch, TableQueryOptimizer
from shared.table_retrieval_metrics import TableRetrievalMonitor, RetrievalPerformanceAnalyzer
from shared.document_intelligence_integration import DocumentIntelligenceTableExtractor
from shared.mistral_ocr_retrieval import MistralRetrievalIntegration

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'debug').upper()
logging.basicConfig(level=LOGLEVEL)
myLogger = logging.getLogger(__name__)

# Env Variables
BLOCKED_LIST_CHECK = os.environ.get("BLOCKED_LIST_CHECK", "true").lower() == "true"
GROUNDEDNESS_CHECK = os.environ.get("GROUNDEDNESS_CHECK", "true").lower() == "true"
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_CHATGPT_ANSWER_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_ANSWER_MODEL")
AZURE_OPENAI_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.17"))
AZURE_OPENAI_TOP_P = float(os.environ.get("AZURE_OPENAI_TOP_P", "0.27"))
AZURE_OPENAI_RESP_MAX_TOKENS = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "1500"))

# Table processing configuration
ENABLE_COMPLEX_TABLE_PROCESSING = os.environ.get("ENABLE_COMPLEX_TABLE_PROCESSING", "true").lower() == "true"
ENABLE_AGENTIC_SEARCH = os.environ.get("ENABLE_AGENTIC_TABLE_SEARCH", "true").lower() == "true"
ENABLE_DOCUMENT_INTELLIGENCE = os.environ.get("ENABLE_DOCUMENT_INTELLIGENCE", "false").lower() == "true"
ENABLE_RETRIEVAL_METRICS = os.environ.get("ENABLE_RETRIEVAL_METRICS", "true").lower() == "true"
ENABLE_MISTRAL_OCR = os.environ.get("ENABLE_MISTRAL_OCR", "false").lower() == "true"

SYSTEM_MESSAGE_PATH = "orc/prompts/system_message.prompt"
ANSWER_MESSAGE_PATH = "orc/prompts/answer_tables_enhanced.prompt"  # Enhanced prompt for complex tables


def initialize_kernel(model):
    kernel = sk.Kernel(log=myLogger)
    chatgpt_config = get_aoai_config(model)
    kernel.add_chat_service(
        "chat-gpt",
        sk_oai.AzureChatCompletion(
            chatgpt_config['deployment'], 
            chatgpt_config['endpoint'], 
            chatgpt_config['api_key'], 
            api_version=chatgpt_config['api_version'],
            ad_auth=True
        ), 
    )
    return kernel


def import_custom_plugins(kernel, plugins_directory, rag_plugin_name):
    rag_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, rag_plugin_name)
    native_functions = kernel.import_native_skill_from_directory(plugins_directory, rag_plugin_name)
    rag_plugin.update(native_functions)
    return rag_plugin


def create_context(kernel, system_message, ask, messages):
    context = kernel.create_new_context()
    context.variables["bot_description"] = system_message
    context.variables["ask"] = ask
    context.variables["history"] = json.dumps(messages[-5:-1], ensure_ascii=False)  # just last two interactions
    return context


def fill_template(template, variables):
    pattern = re.compile(r"\{\{\$(.*?)\}\}")
    
    def replace(match):
        var_name = match.group(1).strip()
        return variables.get(var_name, match.group(0))  # fallback to original if not found
    
    return pattern.sub(replace, template)


def get_placeholder(url):
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    return f"[IMAGE_{filename}]"


async def detect_table_query(query: str, context: Dict) -> bool:
    """
    Detect if query is related to tables.
    
    Args:
        query: User query
        context: Query context
        
    Returns:
        True if query is table-related
    """
    table_indicators = [
        "table", "tabla", "quadro",  # table in different languages
        "column", "columna", "coluna",  # column
        "row", "fila", "linha",  # row
        "cell", "celda", "célula",  # cell
        "header", "encabezado", "cabeçalho",  # header
        "merged", "fusionada", "mesclada",  # merged
        "total", "sum", "aggregate",  # calculations
        "authorization level", "approval level",  # specific business terms
        "hierarchical", "nested", "jerárquico"  # structure
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in table_indicators)


async def enhance_table_retrieval(sources: str, query: str, relatedImages: List[str]) -> Dict[str, Any]:
    """
    Enhance table retrieval with complex table processing.
    
    Args:
        sources: Retrieved source content
        query: User query
        relatedImages: Related table images
        
    Returns:
        Enhanced retrieval results
    """
    myLogger.info(f"[enhanced_retrieval] Processing complex table query: {query[:50]}")
    
    # Initialize components
    table_extractor = TableStructureExtractor()
    table_enhancer = TableEnhancer()
    complex_interpreter = ComplexTableInterpreter()
    
    # Initialize Mistral OCR if enabled
    mistral_integration = None
    if ENABLE_MISTRAL_OCR:
        mistral_integration = MistralRetrievalIntegration()
        myLogger.info("[enhanced_retrieval] Mistral OCR integration enabled")
    
    # Track performance if enabled
    if ENABLE_RETRIEVAL_METRICS:
        retrieval_monitor = TableRetrievalMonitor()
        start_time = time.time()
    
    enhanced_sources = sources
    table_context = {}
    
    try:
        # Extract table structure from HTML if present
        table_pattern = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)
        tables_found = table_pattern.findall(sources)
        
        if tables_found:
            myLogger.info(f"[enhanced_retrieval] Found {len(tables_found)} tables to process")
            
            processed_tables = []
            for idx, table_html in enumerate(tables_found):
                # Extract structure
                table_structure = await table_extractor.extract_table_structure(
                    table_html,
                    relatedImages[idx:idx+1] if idx < len(relatedImages) else []
                )
                
                # Enhance with context
                enhanced_table = table_enhancer.enhance_with_context(
                    table_structure,
                    sources[:500]  # Surrounding text for context
                )
                
                # Interpret for retrieval
                interpreted_table = complex_interpreter.interpret_table_for_retrieval(
                    enhanced_table,
                    query
                )
                
                processed_tables.append(interpreted_table)
            
            # Apply Mistral OCR enhancement if enabled
            if mistral_integration and relatedImages:
                myLogger.info("[enhanced_retrieval] Applying Mistral OCR semantic analysis")
                
                # Prepare tables with images for Mistral
                tables_with_images = []
                for i, table in enumerate(processed_tables):
                    if i < len(relatedImages):
                        table_with_image = table.copy()
                        table_with_image["image_url"] = relatedImages[i]
                        table_with_image["metadata"] = {
                            "has_merged_cells": table.get("complex_features", {}).get("has_merged_cells", False),
                            "has_hierarchical_headers": table.get("complex_features", {}).get("has_hierarchical_headers", False),
                            "complexity": table.get("complexity_assessment", "unknown")
                        }
                        tables_with_images.append(table_with_image)
                
                # Enhance with Mistral
                if tables_with_images:
                    mistral_results = await mistral_integration.enhance_table_retrieval_pipeline(
                        query,
                        tables_with_images,
                        {"language": "Spanish"}  # Can be parameterized
                    )
                    
                    if mistral_results.get("enhanced"):
                        # Update processed tables with Mistral insights
                        enhanced_tables_from_mistral = mistral_results.get("tables", [])
                        for i, enhanced_table in enumerate(enhanced_tables_from_mistral):
                            if i < len(processed_tables) and "mistral_insights" in enhanced_table:
                                processed_tables[i]["mistral_insights"] = enhanced_table["mistral_insights"]
                        
                        # Store Mistral context for later use
                        table_context["mistral_context"] = mistral_results.get("mistral_context", "")
                        table_context["mistral_enhanced"] = True
                        myLogger.info(f"[enhanced_retrieval] Mistral OCR added insights to {mistral_results.get('insights_count', 0)} tables")
            
            # Create enhanced context
            table_context = {
                "tables_processed": len(processed_tables),
                "complexity_levels": [t.get("complexity_assessment", "unknown") for t in processed_tables],
                "query_interpretations": [t.get("query_interpretation", {}) for t in processed_tables],
                "enhanced_tables": processed_tables,
                **table_context  # Include any Mistral context added above
            }
            
            # Generate enhanced source representation
            for table in processed_tables:
                # Add markdown representation
                markdown_table = table_enhancer.create_markdown_representation(table)
                
                # Add JSON representation for LLM
                json_table = table_enhancer.create_json_representation(table)
                
                # Replace original table with enhanced version
                original_table = tables_found[processed_tables.index(table)]
                enhanced_sources = enhanced_sources.replace(
                    original_table,
                    f"\n[ENHANCED_TABLE_START]\n{markdown_table}\n[TABLE_METADATA]\n{json_table}\n[ENHANCED_TABLE_END]\n"
                )
        
        # Use Document Intelligence if enabled and available
        if ENABLE_DOCUMENT_INTELLIGENCE and relatedImages:
            myLogger.info("[enhanced_retrieval] Using Document Intelligence for table extraction")
            doc_extractor = DocumentIntelligenceTableExtractor()
            
            for image_url in relatedImages[:3]:  # Process first 3 images
                extracted_tables = await doc_extractor.extract_tables_from_document(image_url)
                
                for extracted in extracted_tables:
                    # Interpret extracted table
                    interpreted = complex_interpreter.interpret_table_for_retrieval(
                        extracted,
                        query
                    )
                    
                    # Add to context
                    if "enhanced_tables" not in table_context:
                        table_context["enhanced_tables"] = []
                    table_context["enhanced_tables"].append(interpreted)
        
        # Track metrics if enabled
        if ENABLE_RETRIEVAL_METRICS:
            execution_time = time.time() - start_time
            
            # Determine table features
            table_features = {
                "has_hierarchical_headers": any(
                    t.get("complex_features", {}).get("has_hierarchical_headers", False)
                    for t in table_context.get("enhanced_tables", [])
                ),
                "has_merged_cells": any(
                    t.get("complex_features", {}).get("has_merged_cells", False)
                    for t in table_context.get("enhanced_tables", [])
                ),
                "row_count": max(
                    (t.get("metadata", {}).get("row_count", 0) for t in table_context.get("enhanced_tables", [])),
                    default=0
                ),
                "column_count": max(
                    (t.get("metadata", {}).get("column_count", 0) for t in table_context.get("enhanced_tables", [])),
                    default=0
                )
            }
            
            # Track retrieval
            metrics = await retrieval_monitor.track_retrieval(
                query=query,
                results=table_context.get("enhanced_tables", []),
                execution_time=execution_time,
                query_analysis={"intents": ["table_query"], "search_method": "enhanced"},
                table_features=table_features
            )
            
            myLogger.info(f"[enhanced_retrieval] Retrieval quality: {metrics.quality_assessment.value}")
    
    except Exception as e:
        myLogger.error(f"[enhanced_retrieval] Error processing tables: {e}")
        # Fallback to original sources
        enhanced_sources = sources
        table_context = {"error": str(e)}
    
    return {
        "enhanced_sources": enhanced_sources,
        "table_context": table_context,
        "original_sources": sources
    }


async def perform_agentic_table_search(query: str, context: Dict) -> Optional[str]:
    """
    Perform agentic search for complex table queries.
    
    Args:
        query: User query
        context: Search context
        
    Returns:
        Search results or None if not available
    """
    if not ENABLE_AGENTIC_SEARCH:
        return None
    
    try:
        myLogger.info(f"[agentic_search] Performing agentic search for: {query[:50]}")
        
        # Initialize agentic search
        agentic_search = AgenticTableSearch()
        query_optimizer = TableQueryOptimizer()
        
        # Optimize query for tables
        if context.get("has_hierarchical_headers"):
            query = query_optimizer.optimize_for_hierarchical_headers(
                query,
                context.get("header_levels", [])
            )
        
        if context.get("has_merged_cells"):
            query = query_optimizer.optimize_for_merged_cells(query)
        
        # Execute agentic search
        search_results = await agentic_search.execute_agentic_search(query, context)
        
        if search_results.get("results"):
            myLogger.info(f"[agentic_search] Found {len(search_results['results'])} results")
            
            # Format results for RAG
            formatted_results = []
            for result in search_results["results"][:10]:  # Top 10 results
                # Extract relevant content
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                
                # Add interpretation hints
                hints = result.get("interpretation_hints", [])
                if hints:
                    content += f"\n[HINTS: {', '.join(hints)}]"
                
                formatted_results.append(content)
            
            return "\n\n".join(formatted_results)
        else:
            myLogger.info("[agentic_search] No results found")
            return None
    
    except Exception as e:
        myLogger.error(f"[agentic_search] Error: {e}")
        return None


def replace_tables_enhanced(content: str, image_urls: List[str], table_context: Dict) -> str:
    """
    Enhanced table replacement with context preservation.
    
    Args:
        content: Content with tables
        image_urls: Related image URLs
        table_context: Enhanced table context
        
    Returns:
        Content with enhanced table placeholders
    """
    table_pattern = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)
    
    idx = 0
    enhanced_tables = table_context.get("enhanced_tables", [])
    
    def replacer(match):
        nonlocal idx
        
        # Get enhanced table if available
        if idx < len(enhanced_tables):
            enhanced = enhanced_tables[idx]
            placeholder = get_placeholder(image_urls[idx]) if idx < len(image_urls) else f"[TABLE_{idx}]"
            
            # Add complexity and interpretation hints
            complexity = enhanced.get("complexity_assessment", "unknown")
            hints = enhanced.get("query_interpretation", {}).get("warnings", [])
            
            replacement = f"\n{placeholder}"
            replacement += f"\n[COMPLEXITY: {complexity}]"
            
            if hints:
                replacement += f"\n[INTERPRETATION_NOTES: {'; '.join(hints)}]"
            
            replacement += "\n"
            
            idx += 1
            return replacement
        else:
            # Fallback to simple replacement
            if idx < len(image_urls):
                placeholder = get_placeholder(image_urls[idx])
                idx += 1
                return f"\n{placeholder}\n"
            else:
                return ""  # Remove table without replacement if no URL left
    
    content_with_images = table_pattern.sub(replacer, content)
    return content_with_images


async def triage_ask(kernel, rag_plugin, context):
    """
    This function is used to triage the user ask and determine the intent of the request. 
    If the ask is a Q&A question, a search query is generated to search for sources.
    If it is not a Q&A question, there's no need to retrieve sources and the answer is generated.
   
    Returns:
    dict: A dictionary containing the triage response. The response includes the intent, answer, search query, and a bypass flag.
        'intents' (str): A list of intents of the request. Defaults to ['none'] if not found.
        'answer' (str): The answer to the request. Defaults to an empty string if not found.
        'search_query' (str): The search query for the request. Defaults to an empty string if not found.
        'bypass' (bool): A flag indicating whether to bypass the the reminder flow steps (in case of an error has occurred).
    """    
    triage_response = {"intents": ["none"], "answer": "", "search_query": "", "bypass": False}
    output_context = await call_semantic_function(kernel, rag_plugin["Triage"], context)
    if context.error_occurred:
        logging.error(f"[code_orchest] error when executing RAG flow (Triage). SK error: {context.last_error_description}")
        raise Exception(f"Triage was not successful due to an error when calling semantic function: {context.last_error_description}")
    try:
        response = output_context.result.strip("`json\n`")
        response_json = json.loads(response)
    except json.JSONDecodeError:
        logging.error(f"[code_orchest] error when executing RAG flow (Triage). Invalid json: {output_context.result}")
        raise Exception(f"Triage was not successful due to a JSON error. Invalid json: {output_context.result}")

    triage_response["intents"] = response_json.get('intents', ['none'])
    triage_response["answer"] = response_json.get('answer', '')
    triage_response["search_query"] = response_json.get('query_string', '')   
    return triage_response


async def get_answer(history, document_level, language, process, document_type, country, title):
    """
    Enhanced get_answer function with complex table processing capabilities.
    """
    #############################
    # INITIALIZATION
    #############################
    
    # Initialize variables    
    answer_dict = {}
    answer = ""
    intents = "none"
    system_message = prompt = open(SYSTEM_MESSAGE_PATH, "r").read()
    search_query = ""
    sources = ""
    bypass_nxt_steps = False  # flag to bypass unnecessary steps
    blocked_list = []
    
    # Initialize table processing components if enabled
    if ENABLE_COMPLEX_TABLE_PROCESSING:
        table_interpreter = ComplexTableInterpreter()
        retrieval_monitor = TableRetrievalMonitor() if ENABLE_RETRIEVAL_METRICS else None
    
    # Get user question
    messages = get_chat_history_as_messages(history, include_last_turn=True)
    ask = messages[-1]['content']
    
    logging.info(f"[code_orchest_enhanced] starting RAG flow. {ask[:50]}")
    init_time = time.time()
    
    #############################
    # GUARDRAILS (QUESTION)
    #############################
    if BLOCKED_LIST_CHECK:
        logging.debug(f"[code_orchest_enhanced] blocked list check.")
        try:
            blocked_list = get_blocked_list()
            for blocked_word in blocked_list:
                if blocked_word in ask.lower().split():
                    logging.info(f"[code_orchest_enhanced] blocked word found in question: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    bypass_nxt_steps = True
                    break
        except Exception as e:
            logging.error(f"[code_orchest_enhanced] could not get blocked list. {e}")
        response_time = round(time.time() - init_time, 2)
        logging.info(f"[code_orchest_enhanced] finished blocked list check. {response_time} seconds.")            
    
    #############################
    # RAG-FLOW
    #############################
    
    if not bypass_nxt_steps:
        try:
            # Initialize semantic kernel
            kernel = initialize_kernel(AZURE_OPENAI_CHATGPT_MODEL)
            rag_plugin = import_custom_plugins(kernel, "orc/plugins", "RAG")
            context = create_context(kernel, system_message, ask, messages)
            
            if document_level:
                context.variables["document_level"] = document_level
            if language:
                context.variables["language"] = language
            if process:
                context.variables["process"] = process
            if document_type:
                context.variables["document_type"] = document_type
            if country:
                context.variables["country"] = country
            if title:
                context.variables["title"] = title
            
            # Import conversation summary plugin to be used by the RAG plugin
            kernel.import_skill(ConversationSummarySkill(kernel=kernel), skill_name="ConversationSummaryPlugin")
            
            # Triage (find intent and generate answer and search query when applicable)
            logging.debug(f"[code_orchest_enhanced] checking intent. ask: {ask}")
            start_time = time.time()                        
            triage_response = await triage_ask(kernel, rag_plugin, context)
            response_time = round(time.time() - start_time, 2)
            intents = triage_response['intents']
            logging.info(f"[code_orchest_enhanced] finished checking intents: {intents}. {response_time} seconds.")
            
            # Handle general intents
            if set(intents).intersection({"about_bot", "off_topic"}):
                answer = triage_response['answer']
                logging.info(f"[code_orchest_enhanced] triage answer: {answer}")
            
            # Handle question answering intent
            elif set(intents).intersection({"follow_up", "question_answering"}):         
                
                search_query = triage_response['search_query'] if triage_response['search_query'] != '' else ask
                
                # Check if this is a table-related query
                is_table_query = await detect_table_query(search_query, context.variables)
                
                if is_table_query and ENABLE_AGENTIC_SEARCH:
                    # Use agentic search for table queries
                    myLogger.info("[code_orchest_enhanced] Detected table query, using agentic search")
                    
                    agentic_results = await perform_agentic_table_search(
                        search_query,
                        {
                            "document_level": document_level,
                            "language": language,
                            "process": process,
                            "document_type": document_type,
                            "country": country,
                            "title": title
                        }
                    )
                    
                    if agentic_results:
                        sources = agentic_results
                    else:
                        # Fallback to standard retrieval
                        output_context = await kernel.run_async(
                            rag_plugin["Retrieval"],
                            input_str=search_query,
                            input_context=context
                        )
                        sources = output_context.result
                else:
                    # Standard retrieval
                    output_context = await kernel.run_async(
                        rag_plugin["Retrieval"],
                        input_str=search_query,
                        input_context=context
                    )
                    sources = output_context.result
                
                formatted_sources = sources[:100].replace('\n', ' ')
                context.variables["sources"] = sources
                logging.info(f"[code_orchest_enhanced] generating bot answer. sources: {formatted_sources}")
                
                # Handle errors
                if context.error_occurred:
                    logging.error(f"[code_orchest_enhanced] error when executing RAG flow (Retrieval). SK error: {context.last_error_description}")
                    answer = f"{get_message('ERROR_ANSWER')} (Retrieval) RAG flow: {context.last_error_description}"
                    bypass_nxt_steps = True
                
                else:
                    filter_sources_prompt = open("orc/plugins/RAG/FilterSources/skprompt.txt", "r").read()
                    
                    variables = {
                        "history": context.variables["history"],
                        "sources": sources,
                        "ask": context.variables["ask"]
                    }
                    
                    relevant_sources = await chat_completion(system_message, filter_sources_prompt, variables)
                    
                    image_urls = re.findall(r'relatedImages:\s*(\[[^\]]*\])', relevant_sources)
                    
                    # Convert and flatten the arrays
                    relatedImages = []
                    for image_url in image_urls:
                        images = ast.literal_eval(image_url)
                        relatedImages.extend(images)
                    
                    # Remove the relatedImages part from the text
                    sources = re.sub(r'relatedImages:\s*\[[^\]]*\]\s*,?\s*', '', relevant_sources)
                    
                    # Enhanced table processing
                    table_context = {}
                    if is_table_query and ENABLE_COMPLEX_TABLE_PROCESSING:
                        enhancement_result = await enhance_table_retrieval(sources, search_query, relatedImages)
                        sources = enhancement_result["enhanced_sources"]
                        table_context = enhancement_result["table_context"]
                    
                    context.variables["sources"] = sources
                    
                    # Generate the answer for the user
                    logging.info(f"[code_orchest_enhanced] generating bot answer. ask: {ask}")
                    start_time = time.time()                                                          
                    context.variables["history"] = json.dumps(messages[:-1], ensure_ascii=False)  # update context with full history
                    
                    if relatedImages: 
                        # Use enhanced prompt for complex tables
                        answer_prompt_path = ANSWER_MESSAGE_PATH if is_table_query else "orc/prompts/answer_tables.prompt"
                        answer_prompt = open(answer_prompt_path, "r").read()
                        
                        # Enhanced table replacement
                        if is_table_query and table_context:
                            sources = replace_tables_enhanced(context.variables["sources"], relatedImages, table_context)
                        else:
                            sources = replace_tables(context.variables["sources"], relatedImages)
                        
                        variables = {
                            "history": context.variables["history"],
                            "sources": sources,
                            "ask": context.variables["ask"]
                        }
                        
                        # Add table context if available
                        if table_context:
                            variables["table_context"] = json.dumps(table_context, ensure_ascii=False)
                        
                        storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
                        container_name = os.environ.get("STORAGE_CONTAINER_IMAGES")
                        storage_key = get_secret("storage-account-key")
                        connection_string = f"DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName={storage_account_name};AccountKey={storage_key}"
                        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                        container_client = blob_service_client.get_container_client(container_name)
                        
                        base64_images = []
                        
                        for image in relatedImages:
                            parsed_url = urlparse(image)
                            container_name = parsed_url.path.split("/")[1]
                            url_decoded = unquote(parsed_url.path)
                            blob_name = url_decoded[len(container_name) + 2:]
                            blob_client = container_client.get_blob_client(blob_name)
                            image_bytes = blob_client.download_blob().readall()
                            base64_image = base64.b64encode(image_bytes).decode('utf-8')                    
                            base64_images.append(base64_image)
                        
                        answer = await chat_completion(system_message, answer_prompt, variables, base64_images)
                    
                    else: 
                        answer_prompt = open("orc/plugins/RAG/Answer/skprompt.txt", "r").read()
                        
                        variables = {
                            "history": context.variables["history"],
                            "sources": sources,
                            "ask": context.variables["ask"]
                        }
                        
                        answer = await chat_completion(system_message, answer_prompt, variables)
                    
                    if context.error_occurred:
                        logging.error(f"[code_orchest_enhanced] error when executing RAG flow (get the answer). {context.last_error_description}")
                        answer = f"{get_message('ERROR_ANSWER')} (get the answer) RAG flow: {context.last_error_description}"
                        bypass_nxt_steps = True
                    
                    response_time = round(time.time() - start_time, 2)              
                    logging.info(f"[code_orchest_enhanced] finished generating bot answer. {response_time} seconds. {answer[:100]}.")
            
            elif "greeting" in intents:
                answer = triage_response['answer']
                logging.info(f"[code_orchest_enhanced] triage answer: {answer}")
            
            else:
                logging.info(f"[code_orchest_enhanced] SK did not executed, no intent found, review Triage function.")
                answer = get_message('NO_INTENT_ANSWER')
                bypass_nxt_steps = True
        
        except Exception as e:
            logging.error(f"[code_orchest_enhanced] exception when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: exception: {e}"
            bypass_nxt_steps = True
    
    #############################
    # GUARDRAILS (ANSWER)
    #############################
    
    if BLOCKED_LIST_CHECK and not bypass_nxt_steps:
        try:
            for blocked_word in blocked_list:
                if blocked_word in answer.lower().split():
                    logging.info(f"[code_orchest_enhanced] blocked word found in answer: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    break
        except Exception as e:
            logging.error(f"[code_orchest_enhanced] could not get blocked list. {e}")
    
    answer_dict["answer"] = answer
    
    # Process images in answer
    pattern = r'<img\s+src=["\']([^"\']+)["\']'
    images = re.findall(pattern, answer)
    if len(images) > 0:
        storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
        container_name = os.environ.get("STORAGE_CONTAINER_IMAGES")
        storage_key = get_secret("storage-account-key")
        connection_string = f"DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName={storage_account_name};AccountKey={storage_key}"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        answer = ""
        
        for image in images:
            parsed_url = urlparse(image)
            container_name = parsed_url.path.split("/")[1]
            url_decoded = unquote(parsed_url.path)
            blob_name = url_decoded[len(container_name) + 2:]
            blob_client = container_client.get_blob_client(blob_name)
            image_bytes = blob_client.download_blob().readall()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            data_to_insert = f"data:image/png;base64,{base64_image}"
            answer = answer + f"<img src='{data_to_insert}'>\n"
        
        answer = re.sub(
            r'(<img\s+[^>]*)(>)',  # Match <img> tags before their closing '>'
            r'\1 style="max-width:600px;height: auto;"\2',  # Insert the style attribute
            answer
        )
    
    answer_dict["prompt"] = prompt
    answer_dict["sources"] = sources.replace('[', '{').replace(']', '}')
    answer_dict["search_query"] = search_query
    answer_dict["model"] = AZURE_OPENAI_CHATGPT_MODEL    
    
    # Add performance metrics if available
    if ENABLE_RETRIEVAL_METRICS and retrieval_monitor:
        performance_summary = retrieval_monitor.get_performance_summary()
        answer_dict["performance_metrics"] = performance_summary
    
    answer_with_images = answer
    response_time = round(time.time() - init_time, 2)
    logging.info(f"[code_orchest_enhanced] finished RAG Flow. {response_time} seconds.")
    
    return answer_dict, answer_with_images


# Keep original replace_tables function for backward compatibility
def replace_tables(content, image_urls):
    table_pattern = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)
    
    idx = 0
    
    def replacer(match):
        nonlocal idx
        if idx < len(image_urls):
            placeholder = get_placeholder(image_urls[idx])
            idx += 1
            return f"\n{placeholder}\n"
        else:
            return ""  # remove table without replacement if no URL left
    
    content_with_images = table_pattern.sub(replacer, content)
    return content_with_images