import logging
import azure.functions as func
import json
import os
from . import orchestrator

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    conversation_id = req_body.get('conversation_id')
    question = req_body.get('question')
    document_level = req_body.get('document_level') 
    language = req_body.get('language') 
    process = req_body.get('process')
    document_type = req_body.get('document_type') 
    country = req_body.get('country') 
    title = req_body.get('title')
    client_principal_id = req_body.get('client_principal_id')
    client_principal_name = req_body.get('client_principal_name')
    app_principal_id = req_body.get('app_principal_id')
    app_principal_name = req_body.get('app_principal_name')
    document_level = req_body.get('document_level', None)
    language = req_body.get('language', None)
    process = req_body.get('process', None)
    document_type = req_body.get('document_type', None)
    country = req_body.get('country', None)
    if not client_principal_id or client_principal_id == '':
        client_principal_id = '00000000-0000-0000-0000-000000000000'
        client_principal_name = 'anonymous'    
    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name
    }

    if not app_principal_id or app_principal_id == '':
        app_principal_id = None
        app_principal_name = None
        logging.info('No App Registration provided by client.')
    app_principal = {
        'id': app_principal_id,
        'name': app_principal_name
    }

    if question:

        result = await orchestrator.run(conversation_id, question, document_level, language, process, document_type, country, title, client_principal, app_principal)

        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse('{"error": "no question found in json input"}', mimetype="application/json", status_code=200)
