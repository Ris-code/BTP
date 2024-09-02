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
import pandas as pd
import streamlit as st
from model import llm

store = {}

load_dotenv()

# # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]


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
def remove_code_block_markers(text):
    lines = text.strip().splitlines()
    filtered_lines = [line for line in lines if not line.startswith('```')]
    return '\n'.join(filtered_lines)

def agent(question):
    def create_prompt_template():
        template = """
            You are a CRM workflow generator. Create a json CRM workflow provided in the prompt below based on the following criteria provided by the user in the {question}:
            
            Tools:
            1. Use the tool 'CRM-schema' for understanding the database schema and use it as reference whenever dealing with the database.
            2. Use the tool 'CRMschema' which contains all the related database tables so use this whenever require any data from the customer relation management database.
            3. Use the tool 'crm-workflow' for creating the workflow.

            Use the following json format for defining the flow: {json}

            Explanation : 

            # Explanation of JSON Schema Fields Based on Prompts

            ## Top-Level Fields

            1. **ID**: 
            - Value: "random"
                Explanation: This seems to be a placeholder for a unique identifier for each workflow.

            2. **RuleName**: 
            - Values from prompts: 
                    Explanation: This field contains the name of the workflow as specified in each prompt.

            3. **Status**: 
            - Value:
                Explanation: All prompts specify that the workflows should be created in an inactive state.

            ## Module

            4. **ModuleObject**: 
            - Values from prompts:
                        Explanation: This field specifies the type of module the workflow applies to.

            5. **ModuleName**: 
                Explanation: This would be the same as ModuleObject in these cases.

            6. **ModuleEvaluationCondition**: 
            - Values from prompts:
                    Explanation: This field indicates when the workflow should be triggered.

            ## Criteria

            7. **LogicalCondition**: 
                Explanation: This represents the overall logical structure of the criteria. The `Value` and `Text` subfields would contain the composite logical conditions from the prompts.

            8. **Criteria**: 
            - Explanation: This is an array of individual criteria mentioned in the prompts. Each criterion has the following subfields:
            - ** Name ** - (eg : criteria1, criteria2 etc.)
            -**Logical Condition**: (The logical condition to be applied over the subcriteria)
            Example: (##1## AND ##2## AND ##3##) OR (##4## AND ##5##) → The logical operator should be applied to the criteria (##x## refers to sub-criteria number x).

            -**SubCritera** : (subcriteria to be satisfied , following are the fields of subcriteria ) 
                - **Operator**: The comparison operation (e.g., "equals", "contains", "is not empty")
                    - **TextSearch**: The value being searched for or compared against
                    - **LogicalOperator**: How this criterion combines with others (e.g., "and", "or")
                    - **Attribute**: 
                    - **Name**: The field name being evaluated 
                    - **Label**: A human-readable label for the field
                    - **SearchValue**: The value being compared against

            ## Task

            9. **Task**: 
            - Explanation: This array represents immediate actions to be taken when criteria are met. Each task has the following subfields:
                - **Name**: The name or title of the task (e.g., "User Approval Email", "New Prospect From Reg Page")
                - **Template**: The email template to be used (e.g., "User Registration Confirmation Mail For Partner")
                - **AdditionalEmails**: Extra email recipients mentioned in the prompts
                - **RecipientType**: Who should receive the email (e.g., "Related User", "Specific Email")
                - **TaskType**: The type of action (e.g., "Email", "Field Update")
                - **TaskCategory**: When the task should be executed (e.g., "immediate")
                - **CriteriaName**: A reference to the criteria that triggers this task

            ## ScheduledTask

            10. **ScheduledTask**: 
                - Explanation: This array represents actions that are scheduled to occur at a later time. Each scheduled task has:
                - **CriteriaName**: The name of the criteria that triggers this task
                - **ActionName**: A name for the scheduled action
                - **SubCriteria**: Similar structure to the top-level subCriteria, but specific to this scheduled task
                - **Task**: An array of tasks to be performed, similar to the top-level Task structure
                - **DurationOffSet**: How long to wait before executing the task (e.g., "15 minutes", "2 days")
                - **DurationFrequency**: The unit of time for the duration (e.g., "minutes", "days")
                - **SourceTime**: The reference point for the schedule (e.g., "Date Modified", "Date Created")

            This schema provides a flexible structure to represent the various workflows described in the prompts, capturing the conditions, immediate actions, and scheduled actions for each workflow.

            GuideLines:
                1. Extract the entities mentioned in the json format and fill the values.
                2. Strictly follow the json format provided.
                3. Properly understand the tasks, scheduled tasks as mentioned in the prompt in the given order.
                4. Add the criterias and the subcriterias in the json files based on the explanation provided after extracting the entities from the prompt.        
        """
        return PromptTemplate.from_template(template=template)

    
    def format_prompt(prompt_template, question):
        json_template = {
                "ID": "random",
                "RuleName": "",
                "Status": "Inactive",
                "Module": {
                    "ModuleObject": "",
                    "ModuleName": "",
                    "ModuleEvaluationCondition": ""
                },
                "Criteria": {
                    "LogicalCondition": {
                    "Value": "",
                    "Text": ""
                    },
                    "Criteria": [
                    {
                        "Operator": "",
                        "TextSearch": "",
                        "LogicalOperator": "",
                        "Attribute": {
                        "Name": "",
                        "Label": ""
                        },
                        "SearchValue": ""
                    }
                    ]
                },
                "Task": [
                    {
                    "Name": "",
                    "Template": "",
                    "AdditionalEmails": "",
                    "RecipientType": "",
                    "TaskType": "",
                    "TaskCategory": "",
                    "CriteriaName": ""
                    }
                ],
                "ScheduledTask": [
                    {
                    "CriteriaName": "",
                    "ActionName": "",
                    "Criteria": {
                        "LogicalCondition": {
                        "Value": "",
                        "Text": ""
                        },
                        "Criteria": {
                        "Operator": "",
                        "TextSearch": "",
                        "LogicalOperator": "",
                        "Attribute": {
                            "Name": "",
                            "Label": ""
                        }
                        }
                    },
                    "Task": [],
                    "DurationOffSet": "",
                    "DurationFrequency": "",
                    "SourceTime": ""
                    }
                ]
            }
        return prompt_template.format(
            question=question,
            json=json_template,
        )

    # Prepare the prompt
    prompt_template = create_prompt_template()
    formatted_prompt = format_prompt(prompt_template, question)

    # Asynchronously invoke the agent
    response = agent_executor.invoke(
        {"input": formatted_prompt},
        {"configurable": {"session_id": "<foo>"}}
    )

    # print(response)
    output = remove_code_block_markers(response['output'])
    # with open("output.json", "w") as file:
    #     file.write(output)

    return output

if __name__ == '__main__':
    question = input("Ask a question:")
    output = agent(question)

    try:
        df = pd.read_csv("result.csv")
    except:
        df = pd.DataFrame(columns = ["Prompt" , "JSON"])

    df.loc[len(df.index)] = [question, output]
    df.to_csv("result.csv", index=False)

    print(output)
