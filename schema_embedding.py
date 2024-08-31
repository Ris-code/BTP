import os
import time
import pandas as pd
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import DeepInfraEmbeddings
from dotenv import load_dotenv
from schema import table_schemas
import streamlit as st

# Load environment variables
load_dotenv()

# Set environment variables
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")
# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
# os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_TOKEN")

os.environ["HF_TOKEN"] = st.secrets["HUGGING_FACE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets("PINECONE_API_KEY")
os.environ["DEEPINFRA_API_TOKEN"] = st.secrets("DEEPINFRA_API_TOKEN")

# Initialize embeddings
embeddings = DeepInfraEmbeddings(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    query_instruction="",
    embed_instruction="",
)

def convert_schemas_to_documents(schemas):
    documents = []
    for schema in schemas:
        documents.append(Document(page_content=schema.strip()))
    return documents

def pinecone_vector_store(schemas):
    index_name = "sql-schemas-1"  

    # Initialize Pinecone
    pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    existing_indexes = pinecone.list_indexes()

    # Create the index if it doesn't already exist
    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=384,  # Assuming the embedding model outputs 384-dimensional vectors
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        print("The index already exists.")

    index = pinecone.Index(index_name)
    
    # Convert schemas to documents
    documents = convert_schemas_to_documents(schemas)
    
    # Embed the documents
    document_texts = [doc.page_content for doc in documents]
    document_embeddings = embeddings.embed_documents(document_texts)
    
    # Prepare data for upsert
    # vectors = [(f"schema_{i}", embedding) for i, embedding in enumerate(document_embeddings)]
    
    # Upsert the vectors into Pinecone
    # index.upsert(vectors)

    # print(f"Stored {len(vectors)} schemas in the Pinecone index '{index_name}'.")
    PineconeVectorStore(index_name=index_name, embedding = document_embeddings)

# Store the table schemas in Pinecone
pinecone_vector_store(table_schemas)
