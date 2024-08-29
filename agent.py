import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from tool import tools
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

store = {}

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Initialize language model
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.0,
    max_retries=2,
)

# Bind the tool to the model
llm = llm.bind_tools(tools)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define agent
agent = create_tool_calling_agent(llm, tools, hub.pull("hwchase17/openai-functions-agent"))
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def agent(question):
    def create_prompt_template():
        template = """
            You are a chatbot to answer the user queries:
            
            Tools:
            1. Use the tool 'CRM-schema' for understanding the database schema and use it as reference whenever dealing with the database.
            2. Use the tool 'CRMschema' which contains all the related database tables so use this whenever require any data from the customer relation management database.
            3. Use the tool 'crm-workflow' for creating the workflow.
           

            {question}
        """
        return PromptTemplate.from_template(template=template)

    def format_prompt(prompt_template, question):
        return prompt_template.format(
            question=question,
        )

    # Prepare the prompt
    prompt_template = create_prompt_template()
    formatted_prompt = format_prompt(prompt_template, question)

    # Asynchronously invoke the agent
    response = agent_with_chat_history.invoke(
        {"input": formatted_prompt},
        {"configurable": {"session_id": "<foo>"}}
    )
    # print(response)
    with open("output.txt", "w") as file:
        file.write(response['output'])

    return response['output']

question = input("Ask a question:")
output = agent(question)
print(output)

# with open("output.txt", "w") as file:
#     file.write(output)