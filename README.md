# Demo of RAG Retrieval Augmented Generation using the following technologies
- Azure Cloud
- Azure Static Web App
- Azure Function
- Azure Cosmos DB / MongoDB version
- OpenAI API for Embeddings and LLM

This demo provides a main page index.html and two auxiliary pages upload.html and question.html

The upload.html page is managing the RAG database and uploading files, currently only .txt text files are supported but PDF and MS Doc/DocX files can be optionally supported in the future. A text file containing a biography of Walt Disney is included in the test_files directory of the Github repo for this project.

Once uploaded these files are then split into chunks using the RecursiveCharacterTextSplitter and then vector embeddings are computed using OpenAI text-embedding-ada-002 and then these text chunks and their associated embedding vectors are stored in a CosmosDB database providing a MongoDB compatible API. This page also provides the capability to delete all of the records in the database.

The question.html page provides a basic Question and Answer capability using the information stored in the RAG database and the OpenAI gpt-4o LLM model.

This demo includes prompt to prevent answering questions about objectionable content and content otherwise not related to Disney.
