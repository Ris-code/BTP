import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone as pc
from langchain_pinecone import Pinecone
from langchain.tools.retriever import create_retriever_tool

# Load environment variables from .env file
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def retrieve_tool(index, topic, description, pinecone_key=os.environ.get("PINECONE_API_KEY")):
    # Initialize Pinecone client
    pc_client = pc(api_key=pinecone_key)
    Index = pc_client.Index(index)

    # Initialize vector store
    vectorstore = Pinecone(Index, embedding=MistralAIEmbeddings())
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
    "product-index", 
    topic="product-details", 
    description="Provides all information related to a particular product asked by the user"
)

# Tool 2: Policy-related retrieval tool
retrieve_tool_2 = retrieve_tool(
    "policy", 
    topic="walmart-frequently-asked-questions", 
    description="Provides answers to all the questions related to policies, terms and conditions, return and refund policies, shopping with Walmart, Walmart+, and Walmart services"
)

# List of tools
tools = [retrieve_tool_1, retrieve_tool_2]
