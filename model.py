from langchain_groq import ChatGroq
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.llms import DeepInfra
from langchain_openai import OpenAI
import streamlit as st

load_dotenv()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["DEEPINFRA_API_TOKEN"] = st.secrets["DEEPINFRA_API_TOKEN"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# Initialize language model
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.0,
    max_retries=2,
)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)

# llm = DeepInfra(model_id="google/gemma-2-9b-it")
# llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
# llm.model_kwargs = {
#     "temperature": 0.7,
#     "repetition_penalty": 1.2,
#     "max_new_tokens": 250,
#     "top_p": 0.9,
# }

# llm = OpenAI(
#     model="gpt-3.5-turbo-instruct",
#     temperature=0,
#     max_retries=2,
# )

