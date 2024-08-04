import os
import logging
import json

from openai import OpenAI

from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain_openai import OpenAIEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

import azure.functions as func
# I probably don't need these next two imports because I am using MongoDB client for CosmosDB
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient, PartitionKey

from pymongo import MongoClient


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


openai_client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
   #text = text.replace("\n", " ")
   return openai_client.embeddings.create(input = [text], model=model).data[0].embedding


def process_file(file_content):
    logging.info('Processing file...')

    # Read the OpenAI API key from environment variable
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found")

    #try:
    if True:
        # Convert bytes to string if necessary
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')

        # Ensure the file_content is in the correct format (string)
        if not isinstance(file_content, str):
            raise ValueError("file_content must be a string containing the document content. The type of file_content is " + str(type(file_content)))

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(file_content)

        # ----- embeddings -----

        # Initialize OpenAI embeddings
        #embeddings = OpenAIEmbeddings()

        # Compute embeddings for each chunk
        #embeddings_list = [embeddings.embed_text(doc.page_content) for doc in docs]

        embeddings_list = []
        for doc in docs:
            embedding = get_embedding(doc)
            embeddings_list.append(embedding)


        # ----- CosmosDB -----

        cosmos_db_connection_string = os.getenv("COSMOS_DB_CONNECTION_STRING")
        cosmos_db_database_name = os.getenv("COSMOS_DB_DATABASE_NAME")
        cosmos_db_container_name = os.getenv("COSMOS_DB_COLLECTION_NAME")

        logging.info(f"COSMOS_DB_CONNECTION_STRING: {cosmos_db_connection_string}")
        logging.info(f"COSMOS_DB_DATABASE_NAME: {cosmos_db_database_name}")
        logging.info(f"COSMOS_DB_COLLECTION_NAME: {cosmos_db_container_name}")

        if not cosmos_db_connection_string:
            raise ValueError("COSMOS_DB_CONNECTION_STRING environment variable not found or is empty")
        if not cosmos_db_database_name:
            raise ValueError("COSMOS_DB_DATABASE_NAME environment variable not found or is empty")
        if not cosmos_db_container_name:
            raise ValueError("COSMOS_DB_COLLECTION_NAME environment variable not found or is empty")

        # Initialize Cosmos DB client
        #credential = DefaultAzureCredential()
        #client = CosmosClient(cosmos_db_connection_string, credential)
        #database = client.get_database_client(cosmos_db_database_name)
        #container = database.get_container_client(cosmos_db_container_name

        # Initialize MongoDB client
        client = MongoClient(cosmos_db_connection_string)
        database = client[cosmos_db_database_name]
        collection = database[cosmos_db_container_name]

        # Store chunks and their embeddings in Cosmos DB
        #for doc, embedding in zip(docs, embeddings_list):
        #    container.upsert_item({
        #        'id': doc.metadata['id'],
        #        'content': doc.page_content,
        #        'embedding': embedding
        #    })

        # Store chunks and their embeddings in Cosmos DB the MongoDB version
        for doc, embedding in zip(docs, embeddings_list):
            collection.update_one(
                {'id': doc.metadata['id']},
                {'$set': {
                    'content': doc.page_content,
                    'embedding': embedding
                }},
                upsert=True
            )

        logging.info('File processed and data stored in Cosmos DB.')
        
    #except Exception as e:
    #    logging.error(f"Error processing file: {e}")
