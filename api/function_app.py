import azure.functions as func
import logging
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="message")
def message(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    response = {
        "text": "Hello from the API"
    }
    return func.HttpResponse(
        json.dumps(response),
        mimetype="application/json"
    )

#    name = req.params.get('name')
#    if not name:
#        try:
#            req_body = req.get_json()
#        except ValueError:
#            pass
#        else:
#            name = req_body.get('name')
#
#    if name:
#        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
#    else:
#        return func.HttpResponse(
#             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
#             status_code=200
#        )

@app.route(route="upload_file_old", auth_level=func.AuthLevel.ANONYMOUS)
def upload_file_old(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )


@app.route(route="upload_file", auth_level=func.AuthLevel.ANONYMOUS)
def upload_file(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get the uploaded file
        uploaded_file = req.files.get('file')
    except Exception as e:
        logging.error(f"Error during req.files.get: {str(e)}")
        return func.HttpResponse(f"Error during req.files.get: {str(e)}", status_code=500)
    
    try:
        if uploaded_file:
            # Process the file as needed
            file_content = uploaded_file.read()
            
            # Stub for additional file processing
            process_file(file_content)

            logging.info('File uploaded successfully.')
            return func.HttpResponse("File uploaded successfully.", status_code=200)
        else:
            return func.HttpResponse("No file uploaded.", status_code=400)
    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        return func.HttpResponse(f"File upload failed: {str(e)}", status_code=500)

#def process_file(file_content):
#    # Stub for additional functionality to process the uploaded file
#    logging.info('Processing file...')
#    # Add your file processing logic here


import os
import logging
from azure.cosmos import CosmosClient, PartitionKey
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain_openai import OpenAIEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_file(file_content):
    logging.info('Processing file...')

    # Load the document
    loader = TextLoader(file_content)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Compute embeddings for each chunk
    embeddings_list = [embeddings.embed_text(doc.page_content) for doc in docs]

    # Initialize Cosmos DB client
    client = CosmosClient(os.getenv("COSMOS_DB_CONNECTION_STRING"))
    database = client.get_database_client(os.getenv("COSMOS_DB_DATABASE_NAME"))
    container = database.get_container_client(os.getenv("COSMOS_DB_COLLECTION_NAME"))

    # Store chunks and their embeddings in Cosmos DB
    for doc, embedding in zip(docs, embeddings_list):
        container.upsert_item({
            'id': doc.metadata['id'],
            'content': doc.page_content,
            'embedding': embedding
        })

    logging.info('File processed and data stored in Cosmos DB.')

