import os
from dotenv import load_dotenv
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone as pc
from langchain_pinecone import Pinecone
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import DeepInfraEmbeddings

# Load environment variables from .env file
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_TOKEN")

embeddings = DeepInfraEmbeddings(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    query_instruction="",
    embed_instruction="",
)

def retrieve_tool(index, topic, description, pinecone_key=os.environ.get("PINECONE_API_KEY")):
    # Initialize Pinecone client
    pc_client = pc(api_key=pinecone_key)
    Index = pc_client.Index(index)

    # Initialize vector store
    vectorstore = Pinecone(Index, embedding=embeddings)
    retriever = vectorstore.as_retriever(k=1)

    # Create and return the retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        topic,
        description
    )

    return retriever_tool

# Tool 1: Product details retrieval tool
retrieve_tool_1 = retrieve_tool(
    "prompt-json-1", 
    topic="crm-workflow", 
    description="Provide the crm workflow json according to the provided criterias from the provided prompt"
)

# List of tools
tools = [retrieve_tool_1]
