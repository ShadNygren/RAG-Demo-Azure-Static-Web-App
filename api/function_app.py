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

import hashlib

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

        # Initialize MongoDB client for CosmosDB
        client = MongoClient(cosmos_db_connection_string)
        database = client[cosmos_db_database_name]
        collection = database[cosmos_db_container_name]

        # Store chunks and their embeddings in Cosmos DB (MongoDB)
        for doc in docs:
            input_bytes = doc.encode('utf-8')
            md5_hash = hashlib.md5(input_bytes).hexdigest()
            embedding = get_embedding(doc)
            collection.update_one(
                {'id': md5_hash}, # This is used as the shard key
                {'$set': {
                    'content': doc,
                    'embedding': embedding
                }},
                upsert=True
            )

        logging.info('File processed and data stored in Cosmos DB.')
        
    #except Exception as e:
    #    logging.error(f"Error processing file: {e}")



# =================================


import time
import logging
from pymongo import MongoClient

def clear_db(batch_size=10, max_retries=3, delay=5):
    cosmos_db_connection_string = os.getenv("COSMOS_DB_CONNECTION_STRING")
    cosmos_db_database_name = os.getenv("COSMOS_DB_DATABASE_NAME")
    cosmos_db_container_name = os.getenv("COSMOS_DB_COLLECTION_NAME")

    client = MongoClient(cosmos_db_connection_string)
    database = client[cosmos_db_database_name]
    collection = database[cosmos_db_container_name]

    retries = 0

    while True:
        try:
            # Find documents to delete in batches
            documents = collection.find().limit(batch_size)
            documents_list = list(documents)

            if not documents_list:
                break

            ids_to_delete = [doc['_id'] for doc in documents_list]
            result = collection.delete_many({'_id': {'$in': ids_to_delete}})
            logging.info(f"Deleted {result.deleted_count} documents from the database.")

            if result.deleted_count == 0:
                break

            retries = 0  # Reset retries after a successful batch

        except Exception as e:
            logging.error(f"Error deleting documents: {e}")
            retries += 1
            if retries > max_retries:
                logging.error(f"Exceeded maximum retries ({max_retries}).")
                break
            logging.info(f"Retrying in {delay} seconds... (Retry {retries}/{max_retries})")
            time.sleep(delay)


# Route for clearing the database
@app.route(route="clear_db", auth_level=func.AuthLevel.ANONYMOUS)
def clear_db_route(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to clear database.')

    try:
        clear_db()
        logging.info('Database cleared successfully.')
        return func.HttpResponse("Database cleared successfully.", status_code=200)
    except Exception as e:
        logging.error(f"Error clearing database: {str(e)}")
        return func.HttpResponse(f"Error clearing database: {str(e)}", status_code=500)


# =================================

# I could not use numpy and sklearn because I got this error when I tried to include them so I have to do something smaller and simpler
#---End of Oryx build logs---
#Function Runtime Information. OS: linux, Functions Runtime: ~4, python version: 3.8
#Finished building function app with Oryx
#Zipping Api Artifacts
#Done Zipping Api Artifacts
#The content server has rejected the request with: BadRequest
#Reason: The size of the function content was too large. The limit for this Static Web App is 104857600 bytes.
#    
#import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity

#def query_mongodb(user_question, top_k=5):
#    cosmos_db_connection_string = os.getenv("COSMOS_DB_CONNECTION_STRING")
#    cosmos_db_database_name = os.getenv("COSMOS_DB_DATABASE_NAME")
#    cosmos_db_container_name = os.getenv("COSMOS_DB_COLLECTION_NAME")
#
#    client = MongoClient(cosmos_db_connection_string)
#    database = client[cosmos_db_database_name]
#    collection = database[cosmos_db_container_name]
#
#    # Get embedding for user question
#    question_embedding = get_embedding(user_question)
#
#    # Fetch all documents and their embeddings from the database
#    documents = list(collection.find({}, {'_id': 0, 'id': 1, 'content': 1, 'embedding': 1}))
#
#    # Calculate cosine similarity between user question embedding and document embeddings
#    similarities = []
#    for doc in documents:
#        doc_embedding = np.array(doc['embedding']).reshape(1, -1)
#        question_embedding_np = np.array(question_embedding).reshape(1, -1)
#        similarity = cosine_similarity(doc_embedding, question_embedding_np)[0][0]
#        similarities.append((doc, similarity))
#
#    # Sort documents by similarity score in descending order
#    similarities.sort(key=lambda x: x[1], reverse=True)
#
#    # Get top_k most similar documents
#    top_documents = [doc for doc, sim in similarities[:top_k]]
#
#    return top_documents
#
#
#@app.route(route="query_db", auth_level=func.AuthLevel.ANONYMOUS)
#def query_db_route(req: func.HttpRequest) -> func.HttpResponse:
#    logging.info('Python HTTP trigger function to query database.')
#
#    try:
#        user_question = req.params.get('question')
#        if not user_question:
#            req_body = req.get_json()
#            user_question = req_body.get('question')
#
#        if user_question:
#            results = query_mongodb(user_question)
#            logging.info('Query executed successfully.')
#            return func.HttpResponse(json.dumps(results, default=str), mimetype="application/json", status_code=200)
#        else:
#            return func.HttpResponse("Please provide a question to query.", status_code=400)
#    except Exception as e:
#        logging.error(f"Error querying database: {str(e)}")
#        return func.HttpResponse(f"Error querying database: {str(e)}", status_code=500)

# =========================================

import os
import openai

#def get_openai_api_key():
#    return os.getenv("OPENAI_API_KEY")

from openai import OpenAI

def ask_llm_question(user_prompt):
    client = OpenAI()

    system_prompt = """You are a helpful assistant that answers questions about Walt Disney and Disney characters and Disney movies and Disney theme parks and properties such as Disneyland, Disney California Adventure, Disneyworld, Epcot, Disney World Resort, Disney Hollywood Studios and Disneyland locations in Paris, Tokyo, Hong Kong Shanghai and rides and attractions and schedules at these locations.
            You use the RAG Content to assist you in answering the User Question.
            You politely decline to answer questions that are not related to Disney.
            You politely decline to answer questions that are related to politics, religion, race, violence, sex, sexism, misogyny or any other controversial topic. 
            """

    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
      ]
    )

    return completion.choices[0].message.content.strip()

# ----------------------------------

import math
import os
import logging
from pymongo import MongoClient


def cosine_similarity_manual(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm_a * norm_b)


def query_mongodb(user_question, top_k=5, minimum_similarity=0.0):
    cosmos_db_connection_string = os.getenv("COSMOS_DB_CONNECTION_STRING")
    cosmos_db_database_name = os.getenv("COSMOS_DB_DATABASE_NAME")
    cosmos_db_container_name = os.getenv("COSMOS_DB_COLLECTION_NAME")

    client = MongoClient(cosmos_db_connection_string)
    database = client[cosmos_db_database_name]
    collection = database[cosmos_db_container_name]

    # Get embedding for user question
    question_embedding = get_embedding(user_question)

    # Fetch all documents and their embeddings from the database
    documents = list(collection.find({}, {'_id': 0, 'id': 1, 'content': 1, 'embedding': 1}))

    # Calculate cosine similarity between user question embedding and document embeddings
    similarities = []
    for doc in documents:
        doc_embedding = doc['embedding']
        similarity = cosine_similarity_manual(question_embedding, doc_embedding)
        if similarity >= minimum_similarity:
            similarities.append((doc, similarity))

    # Sort documents by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top_k most similar documents
    top_documents = [doc for doc, sim in similarities[:top_k]]

    return top_documents


@app.route(route="query_db", auth_level=func.AuthLevel.ANONYMOUS)
def query_db_route(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to query database.')

    try:
        user_question = req.params.get('question')
        if not user_question:
            req_body = req.get_json()
            user_question = req_body.get('question')

        if user_question:
            results = query_mongodb(user_question, top_k=5, minimum_similarity=0.8)
            logging.info('Query executed successfully.')

            #rag_content = "This is a hard-coded response for debugging purposes. The type(results) = " + str(type(results)) + " and the len(results) = " + str(len(results)) + " here is the results list: " #+ str(results)
            rag_content = ""
            for result in results:
                rag_content += "\n\n-----\n\n" + result["content"]

            #llm_prompt = "\n\n### User Question:\n" + user_question + "\n\n### RAG Content:\n\n" + rag_content
            llm_prompt = "Who is Mickey Mouse?"

            final_answer = {
                "answer": ask_llm_question(llm_prompt)
            }
            return func.HttpResponse(json.dumps(final_answer, default=str), mimetype="application/json", status_code=200)
        else:
            return func.HttpResponse("Please provide a question to query.", status_code=400)
    except Exception as e:
        logging.error(f"Error querying database: {str(e)}")
        return func.HttpResponse(f"Error querying database: {str(e)}", status_code=500)

