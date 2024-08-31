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
import streamlit as st

# Load environment variables
load_dotenv()

# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")
# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
# os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_TOKEN")

os.environ["HF_TOKEN"] = st.secrets["HUGGING_FACE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets("PINECONE_API_KEY")
os.environ["DEEPINFRA_API_TOKEN"] = st.secrets("DEEPINFRA_API_TOKEN")

embeddings = DeepInfraEmbeddings(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    query_instruction="",
    embed_instruction="",
)

def document_split(promptData):
    documents = []

    for _, row in promptData.iterrows():
        content = (
            f"prompt: {row['Prompt']}\n"
            f"json: {row['JSON']}\n"
        )

        documents.append(Document(page_content=content))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    all_splits = text_splitter.split_documents(documents)

    return all_splits

def pinecone_vector_store(promptData):
    index_name = "prompt-json-1"  
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Create the index if it doesn't already exist
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        print("The index already exists.")

    index = pc.Index(index_name)
    docs = document_split(promptData)
    # print(docs)

    document_texts = [doc.page_content for doc in docs]
    # print(document_texts)
    document_result = embeddings.embed_documents(document_texts)
    print(document_result)
    PineconeVectorStore(index_name=index_name, embedding = document_result)

# Load product data from CSV
promptData = pd.read_csv("workflow_analysis.csv")
# print(promptData.head(1))

chunk_size = 1
print(len(promptData))
for start in range(0, len(promptData), chunk_size):
    end = start + chunk_size
    print(start)
    chunk = promptData[:][start:end]
    pinecone_vector_store(chunk)
    print("DONE")
