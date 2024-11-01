# Intro to Vector DBs

This project contains scripts to demonstrate the usage of vector databases with LangChain for document retrieval and question answering.

## Scripts

- **main.py**
  - Demonstrates how to use LangChain with Pinecone vector store for document retrieval and question answering. It includes examples of simple queries, retrieval chains, and custom RAG (Retrieve and Generate) chains.

- **ingestion.py**
  - Contains the script to load text documents, split them into chunks, embed them using OpenAI embeddings, and ingest them into a Pinecone vector store.

- **local.py**
  - Shows how to load documents from a PDF, split them into chunks, embed them using OpenAI embeddings, and store them in a FAISS vector store. It also demonstrates how to create a retrieval chain and query it.