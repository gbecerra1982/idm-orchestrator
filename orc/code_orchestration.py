import ast
import base64
import json
import logging
import os
import re
import requests
import time
from shared.util import call_semantic_function, get_chat_history_as_messages, get_message
from shared.util import get_aoai_config, get_blocked_list, chat_completion, get_secret
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse, unquote
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.core_skills import ConversationSummarySkill

# logging level

logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'debug').upper()
logging.basicConfig(level=LOGLEVEL)
myLogger = logging.getLogger(__name__)

# Env Variables

BLOCKED_LIST_CHECK = os.environ.get("BLOCKED_LIST_CHECK") or "true"
BLOCKED_LIST_CHECK = True if BLOCKED_LIST_CHECK.lower() == "true" else False
GROUNDEDNESS_CHECK = os.environ.get("GROUNDEDNESS_CHECK") or "true"
GROUNDEDNESS_CHECK = True if GROUNDEDNESS_CHECK.lower() == "true" else False
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_CHATGPT_ANSWER_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_ANSWER_MODEL")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TEMPERATURE = float(AZURE_OPENAI_TEMPERATURE)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_TOP_P = float(AZURE_OPENAI_TOP_P)
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1500"
AZURE_OPENAI_RESP_MAX_TOKENS = int(AZURE_OPENAI_RESP_MAX_TOKENS)

SYSTEM_MESSAGE_PATH = f"orc/prompts/system_message.prompt"
ANSWER_MESSAGE_PATH = f"orc/prompts/answer_tables.prompt"


def initialize_kernel(model):
    kernel = sk.Kernel(log=myLogger)
    chatgpt_config = get_aoai_config(model)
    kernel.add_chat_service(
        "chat-gpt",
        sk_oai.AzureChatCompletion(chatgpt_config['deployment'], 
                                    chatgpt_config['endpoint'], 
                                    chatgpt_config['api_key'], 
                                    api_version=chatgpt_config['api_version'],
                                    ad_auth=True), 
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
    context.variables["history"] = json.dumps(messages[-5:-1], ensure_ascii=False) # just last two interactions
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
    triage_response= {"intents":  ["none"], "answer": "", "search_query": "", "bypass": False}
    output_context =  await call_semantic_function(kernel, rag_plugin["Triage"], context)
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

    #############################
    # INITIALIZATION
    #############################

    #initialize variables    

    answer_dict = {}
    answer = ""
    intents = "none"
    system_message = prompt = open(SYSTEM_MESSAGE_PATH, "r").read()
    search_query = ""
    sources = ""
    bypass_nxt_steps = False  # flag to bypass unnecessary steps
    blocked_list = []

    # get user question

    messages = get_chat_history_as_messages(history, include_last_turn=True)
    ask = messages[-1]['content']

    logging.info(f"[code_orchest] starting RAG flow. {ask[:50]}")
    init_time = time.time()

    #############################
    # GUARDRAILS (QUESTION)
    #############################
    if BLOCKED_LIST_CHECK:
        logging.debug(f"[code_orchest] blocked list check.")
        try:
            blocked_list = get_blocked_list()
            for blocked_word in blocked_list:
                if blocked_word in ask.lower().split():
                    logging.info(f"[code_orchest] blocked word found in question: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    bypass_nxt_steps = True
                    break
        except Exception as e:
            logging.error(f"[code_orchest] could not get blocked list. {e}")
        response_time =  round(time.time() - init_time,2)
        logging.info(f"[code_orchest] finished blocked list check. {response_time} seconds.")            

    #############################
    # RAG-FLOW
    #############################

    if not bypass_nxt_steps:

        try:
            
            # initialize semantic kernel
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

            # import conversation summary plugin to be used by the RAG plugin
            kernel.import_skill(ConversationSummarySkill(kernel=kernel), skill_name="ConversationSummaryPlugin")
            
            # triage (find intent and generate answer and search query when applicable)
            logging.debug(f"[code_orchest] checking intent. ask: {ask}")
            start_time = time.time()                        
            triage_response = await triage_ask(kernel, rag_plugin, context)
            response_time = round(time.time() - start_time,2)
            intents = triage_response['intents']
            logging.info(f"[code_orchest] finished checking intents: {intents}. {response_time} seconds.")

            # Handle general intents
            if set(intents).intersection({"about_bot", "off_topic"}):
                answer = triage_response['answer']
                logging.info(f"[code_orchest] triage answer: {answer}")

            # Handle question answering intent
            elif set(intents).intersection({"follow_up", "question_answering"}):         
    
                search_query = triage_response['search_query'] if triage_response['search_query'] != '' else ask
                output_context = await kernel.run_async(
                    rag_plugin["Retrieval"],
                    input_str=search_query,
                    input_context=context
                )
                sources = output_context.result
                formatted_sources = sources[:100].replace('\n', ' ')
                context.variables["sources"] = sources
                logging.info(f"[code_orchest] generating bot answer. sources: {formatted_sources}")

                # Handle errors
                if context.error_occurred:
                    logging.error(f"[code_orchest] error when executing RAG flow (Retrieval). SK error: {context.last_error_description}")
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

                    context.variables["sources"] = sources
                    # Generate the answer for the user
                    logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
                    start_time = time.time()                                                          
                    context.variables["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
                    if relatedImages: 

                        answer_prompt = open(ANSWER_MESSAGE_PATH, "r").read()
                        
                        sources = replace_tables(context.variables["sources"], relatedImages)

                        variables = {
                            "history": context.variables["history"],
                            "sources": sources,
                            "ask": context.variables["ask"]
                        }

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
                        logging.error(f"[code_orchest] error when executing RAG flow (get the answer). {context.last_error_description}")
                        answer = f"{get_message('ERROR_ANSWER')} (get the answer) RAG flow: {context.last_error_description}"
                        bypass_nxt_steps = True
                    response_time =  round(time.time() - start_time,2)              
                    logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")

            elif "greeting" in intents:
                answer = triage_response['answer']
                logging.info(f"[code_orchest] triage answer: {answer}")

            else:
                logging.info(f"[code_orchest] SK did not executed, no intent found, review Triage function.")
                answer = get_message('NO_INTENT_ANSWER')
                bypass_nxt_steps = True

        except Exception as e:
            logging.error(f"[code_orchest] exception when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: exception: {e}"
            bypass_nxt_steps = True

    #############################
    # GUARDRAILS (ANSWER)
    #############################

    # if GROUNDEDNESS_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
    #     try:
    #         logging.info(f"[code_orchest] checking if it is grounded. answer: {answer[:50]}")
    #         start_time = time.time()            
    #         context.variables["answer"] = answer                      
    #         output_context = await call_semantic_function(kernel, rag_plugin["IsGrounded"], context)
    #         grounded = output_context.result
    #         logging.info(f"[code_orchest] is it grounded? {grounded}.")  
    #         if grounded.lower() == 'no':
    #             logging.info(f"[code_orchest] ungrounded answer: {answer}")
    #             output_context = await call_semantic_function(kernel, rag_plugin["NotInSourcesAnswer"], context)
    #             answer = output_context.result
    #             answer_dict['gpt_groundedness'] = 1
    #             bypass_nxt_steps = True
    #         else:
    #             answer_dict['gpt_groundedness'] = 5
    #         response_time =  round(time.time() - start_time,2)
    #         logging.info(f"[code_orchest] finished checking if it is grounded. {response_time} seconds.")
    #     except Exception as e:
    #         logging.error(f"[code_orchest] could not check answer is grounded. {e}")

    if BLOCKED_LIST_CHECK and not bypass_nxt_steps:
        try:
            for blocked_word in blocked_list:
                if blocked_word in answer.lower().split():
                    logging.info(f"[code_orchest] blocked word found in answer: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    break
        except Exception as e:
            logging.error(f"[code_orchest] could not get blocked list. {e}")
    
    answer_dict["answer"] = answer
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
            #answer = answer.replace(image,data_to_insert)
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
    # answer_dict["prompt_tokens"] = prompt_tokens
    # answer_dict["completion_tokens"] = completion_tokens
        
    answer_with_images = answer
    response_time =  round(time.time() - init_time,2)
    logging.info(f"[code_orchest] finished RAG Flow. {response_time} seconds.")

    return answer_dict, answer_with_images