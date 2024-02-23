from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents import Tool

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file


"""

import sys
import os

# Get the root directory of your project
root_dir = os.path.dirname(os.path.abspath(__file__))

# Add the root directory to the Python path
sys.path.append(root_dir)

def build_csv_agent(llm, file_path):
    assert isinstance(file_path, list)
    if len(file_path) == 1:
         file_path = file_path[0]

    csv_agent = create_csv_agent(
        llm,
        file_path,
        verbose=True,
        handle_parsing_errors=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return csv_agent


    

def csv_as_tool(agent):
     return Tool.from_function(
                    name = "csv_retrieval_tool",
                    func= agent.run,
                    description= 'This tool useful for statistics, calculations, plotting and as well as data aggregation'
                )