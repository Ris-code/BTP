import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from tool import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

store = {}

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Initialize language model
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.0,
    max_retries=2,
)

# Bind the tool to the model
llm = llm.bind_tools(tool)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define agent
agent = create_tool_calling_agent(llm, tool, hub.pull("hwchase17/openai-functions-agent"))
agent_executor = AgentExecutor(agent=agent, tools=tool)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

async def agent(question):
    def create_prompt_template():
        template = """
            You are a Walmart customer care agent. Your primary goal is to assist customers efficiently and accurately, using the provided tools to address their inquiries about Walmart products, services, and policies.

            Context:
            Walmart Inc. is a multi-channel retailer operating physical stores (including grocery stores, supermarkets, hypermarkets, department stores, discount stores, and neighborhood markets), warehouse clubs (Sam's Clubs), and eCommerce platforms (websites and mobile applications).

            Available Tools:
            1. 'walmart-frequently-asked-questions': Use for general Walmart policies and information.
            2. 'product-details': Use for specific product information and inquiries.

            Instructions:
            1. Analyze the customer's question carefully.
            2. Determine which tool is most appropriate to address the query.
            3. Use the selected tool to gather relevant information.
            4. Formulate a clear, concise, and helpful response.

            Input:
            Question: {question}

            Response Format:
            1. Greeting: Brief and friendly acknowledgment of the customer's query.
            2. Answer: Directly address the question using information from the appropriate tool.
            3. Additional Information: Provide relevant details that may be helpful, if applicable.
            4. Next Steps or Conclusion: Offer guidance on what to do next or conclude the interaction positively.

            Guidelines:
            - Maintain a courteous and empathetic tone throughout.
            - Provide clear, jargon-free explanations.
            - Be concise while ensuring all necessary information is conveyed.
            - If more information is needed, ask specific, relevant follow-up questions.
            - For complex issues, offer more detailed explanations; for simple queries, be brief.
            - Adhere strictly to Walmart's policies and procedures.
            - If unable to fully answer, suggest next steps or offer to connect with a human representative.
            - Prioritize customer satisfaction and trust-building in every interaction.

            Remember: Your role is to efficiently resolve customer concerns while representing Walmart positively. Analyze each query carefully and use the appropriate tool to provide the most accurate and helpful response.
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
    response = await asyncio.to_thread(agent_with_chat_history.invoke, {"input": formatted_prompt}, {"configurable": {"session_id": "<foo>"}})

    return response['output']