import os
from dotenv import load_dotenv
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone as pc
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent

# Load environment variables from .env file
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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
retrieve_tool_2 = retrieve_tool(
    "sql-schemas-1", 
    topic="CRM-schema", 
    description="Describes the schema of the customer relationship database schema",
)

@tool
def CRMschema(question):
    """
    Query and retrieve data from an eCommerce CRM schema based on the provided question.

    The `CRMschema` function interacts with an SQL database containing various tables related to 
    customer relationship management (CRM) data for an eCommerce platform. This function allows 
    users to query the database using natural language, which is interpreted and translated into 
    SQL queries by a language model. The data retrieved can include various aspects of the eCommerce 
    operations such as customer details, order information, product data, and sales performance.

    The eCommerce CRM database includes the following tables:
    - `customers`: Information about the customers (e.g., customer IDs, names).
    - `geolocation`: Geographical data associated with customers and orders.
    - `leads_closed`: Details of sales leads that have been successfully closed.
    - `leads_qualified`: Information on leads that have been qualified for sales.
    - `order_items`: Data on individual items within customer orders.
    - `order_payments`: Payment details associated with customer orders.
    - `order_reviews`: Customer reviews and ratings of products/orders.
    - `orders`: Overall order details including status, timestamps, and customer associations.
    - `product_category_name_translation`: Mapping of product categories to their respective names.

    Schema:
    
    CREATE TABLE "product_category_name_translation" (
        "product_category_name" TEXT,
        "product_category_name_english" TEXT
    )
    CREATE TABLE "sellers" (
        "seller_id" TEXT,
        "seller_zip_code_prefix" INTEGER,
        "seller_city" TEXT,
        "seller_state" TEXT
    )
    CREATE TABLE "customers" (
        "customer_id" TEXT,
        "customer_unique_id" TEXT,
        "customer_zip_code_prefix" INTEGER,
        "customer_city" TEXT,
        "customer_state" TEXT
    )
    CREATE TABLE "geolocation" (
        "geolocation_zip_code_prefix" INTEGER,
        "geolocation_lat" REAL,
        "geolocation_lng" REAL,
        "geolocation_city" TEXT,
        "geolocation_state" TEXT
    )
    CREATE TABLE "order_items" (
        "order_id" TEXT,
        "order_item_id" INTEGER,
        "product_id" TEXT,
        "seller_id" TEXT,
        "shipping_limit_date" TEXT,
        "price" REAL,
        "freight_value" REAL
    )
    CREATE TABLE "order_payments" (
        "order_id" TEXT,
        "payment_sequential" INTEGER,
        "payment_type" TEXT,
        "payment_installments" INTEGER,
        "payment_value" REAL
    )
    CREATE TABLE "order_reviews" (
        "review_id" TEXT,
        "order_id" TEXT,
        "review_score" INTEGER,
        "review_comment_title" TEXT,
        "review_comment_message" TEXT,
        "review_creation_date" TEXT,
        "review_answer_timestamp" TEXT
    )
    CREATE TABLE "orders" (
        "order_id" TEXT,
        "customer_id" TEXT,
        "order_status" TEXT,
        "order_purchase_timestamp" TEXT,
        "order_approved_at" TEXT,
        "order_delivered_carrier_date" TEXT,
        "order_delivered_customer_date" TEXT,
        "order_estimated_delivery_date" TEXT
    )
    CREATE TABLE "products" (
        "product_id" TEXT,
        "product_category_name" TEXT,
        "product_name_lenght" REAL,
        "product_description_lenght" REAL,
        "product_photos_qty" REAL,
        "product_weight_g" REAL,
        "product_length_cm" REAL,
        "product_height_cm" REAL,
        "product_width_cm" REAL
    )
    CREATE TABLE "leads_qualified" (
        "mql_id" TEXT,
        "first_contact_date" TEXT,
        "landing_page_id" TEXT,
        "origin" TEXT
    )
    CREATE TABLE "leads_closed" (
        "mql_id" TEXT,
        "seller_id" TEXT,
        "sdr_id" TEXT,
        "sr_id" TEXT,
        "won_date" TEXT,
        "business_segment" TEXT,
        "lead_type" TEXT,
        "lead_behaviour_profile" TEXT,
        "has_company" INTEGER,
        "has_gtin" INTEGER,
        "average_stock" TEXT,
        "business_type" TEXT,
        "declared_product_catalog_size" REAL,
        "declared_monthly_revenue" REAL
    )

    Each of these tables contains specific columns and rows tailored to capture various aspects of 
    eCommerce operations, such as customer behavior, transaction history, and product information.

    Args:
    question (str): A natural language query related to the eCommerce CRM data that will be 
                    interpreted and converted into an SQL query to retrieve relevant results.

    Returns:
    str: The output from the SQL tool, containing the data retrieved from the database based 
         on the input question, formatted as a string.
    """
    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0.0,
        max_retries=2,
    )

    engine = create_engine("sqlite:///test.db")
    db = SQLDatabase(engine=engine)

    sql_tool = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
    
    result = sql_tool.invoke({"input": question})
    return result['output']

# List of tools
tools = [retrieve_tool_1, retrieve_tool_2, CRMschema]
