from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
import os

# Mock OCR Tool
def mock_ocr(image_path: str) -> str:
    return "Extracted text from image: Drinking bleach cures COVID-19."

# Mock Medical DB Tool
def medical_db_lookup(query: str) -> str:
    if "bleach" in query.lower():
        return "WHO: Do NOT drink bleach. It is dangerous and does not cure COVID-19."
    return "No specific medical record found."

class LangGraphAdapter:
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, input_dict):
        query = input_dict.get("input", "")
        messages = [HumanMessage(content=query)]
        result = self.graph.invoke({"messages": messages})
        # Extract the last message content
        last_message = result["messages"][-1]
        return {"output": last_message.content}

def get_agent():
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        tools = [
            DuckDuckGoSearchRun(),
            Tool(
                name="Medical DB",
                func=medical_db_lookup,
                description="Useful for looking up verified medical facts."
            ),
            Tool(
                name="OCR",
                func=mock_ocr,
                description="Useful for extracting text from images."
            )
        ]

        # Create LangGraph agent
        graph = create_react_agent(llm, tools)
        return LangGraphAdapter(graph)

    except Exception as e:
        print(f"Warning: Failed to create LangChain/LangGraph agent (likely missing API key). Using mock agent. Error: {e}")
        return MockAgentExecutor()

class MockAgentExecutor:
    def invoke(self, input_dict):
        query = input_dict.get("input", "")
        return {
            "output": f"MOCK ANALYSIS: The claim '{query}' was analyzed. \n\nVerdict: Unverified (Mock Mode)\n\nExplanation: The system is running in mock mode because the OpenAI API key is missing or invalid. In a real scenario, this would use LangChain to verify the claim against medical databases.\n\nSources: [Mock Source 1], [Mock Source 2]"
        }
